"""
runtime/agent.py
Native runtime harness for SmolLM2-360M / NanoHat Fedora Linux Agent.
Directly aligns with the fine-tuned dataset format:
- Canonical 37-token system prompt
- Zero prompt overhead / no smolagents bloat
- Native <thought> and <tool_call> tag extraction
- Direct Python execution of safe Fedora tools
- Deterministic sampling (temperature=0.0)
- Loop & repetition guard
"""

import os
import sys
import re
import json
import argparse
import urllib.request
import urllib.error
from typing import List, Dict, Any, Tuple, Optional

from runtime.tools import (
    calculator,
    web_search,
    user_memory,
    scheduler,
    system_health,
    system_action,
    start_notification_worker,
)

try:
    from generator.schemas import CANONICAL_SYSTEM_PROMPT, LEGACY_TOOL_ALIASES
except ImportError:  # standalone fallback; keep in sync with generator/schemas.py
    CANONICAL_SYSTEM_PROMPT = (
        "You are NanoHat, a helpful AI agent running on Fedora Linux. "
        "Available tools: [calculator, web_search, user_memory, scheduler, system_health, system_action]. "
        "Use tools when necessary by reasoning inside <thought> tags, then outputting a <tool_call> JSON block."
    )
    LEGACY_TOOL_ALIASES = {
        "calc": "calculator", "math": "calculator", "evaluate": "calculator",
        "search": "web_search", "web": "web_search", "google": "web_search", "ddg": "web_search",
        "memory": "user_memory", "store_memory": "user_memory", "get_memory": "user_memory",
        "remind": "scheduler", "schedule_reminder": "scheduler", "notify": "scheduler",
        "reminder": "scheduler",
        "health": "system_health", "sys_health": "system_health", "diagnostics": "system_health",
        "check_system": "system_health",
        "action": "system_action", "sys_action": "system_action", "os_action": "system_action",
    }

TOOL_REGISTRY = {
    "calculator": calculator,
    "web_search": web_search,
    "user_memory": user_memory,
    "scheduler": scheduler,
    "system_health": system_health,
    "system_action": system_action,
}

# Context management: training window is MAX_SEQ_LENGTH=1536 tokens (~6000 chars).
# Keep total rendered history under this budget so inference stays in-distribution.
MAX_HISTORY_CHARS = 6000
TOOL_MSG_CHAR_LIMIT = 700
OLLAMA_NUM_CTX = 2048  # headroom over the 1536-token training window


def _repair_and_parse_json(json_str: str) -> Optional[Dict[str, Any]]:
    """Resiliently parses and recovers tool call JSON blobs from small LLM outputs."""
    raw = json_str.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    # Attempt 1: Direct JSON load
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Attempt 2: Quote unescaping (\" -> ")
    try:
        fixed = raw.replace('\\"', '"')
        data = json.loads(fixed)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Attempt 3: Single quotes to double quotes replacement
    try:
        fixed = re.sub(r"\'([a-zA-Z0-9_\-\.]+)\'\s*:", r'"\1":', raw)
        fixed = re.sub(r":\s*\'([^\']*?)\'", r': "\1"', fixed)
        data = json.loads(fixed)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Attempt 4: Regex key extraction + balanced-brace arguments extraction
    name_match = re.search(r'["\']?name["\']?\s*:\s*["\']([a-zA-Z0-9_]+)["\']', raw)
    if name_match:
        tool_name = name_match.group(1)
        args_match = re.search(
            r'["\']?(?:arguments|parameters|params|args)["\']?\s*:\s*', raw)
        if args_match:
            blob = _extract_balanced_braces(raw, args_match.end())
            if blob:
                try:
                    args = json.loads(blob)
                    if isinstance(args, dict):
                        return {"name": tool_name, "arguments": args}
                except Exception:
                    pass
        return {"name": tool_name, "arguments": {}}

    return None


def _extract_balanced_braces(text: str, start: int) -> Optional[str]:
    """Returns the first balanced {...} block at or after `start` (handles nesting)."""
    begin = text.find("{", start)
    if begin == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(begin, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[begin:i + 1]
    return None


def _normalize_tool_call(tool_name: str, args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Normalizes tool names and argument aliases produced by small models."""
    name = str(tool_name).strip().lower()

    # Tool name aliases (legacy + common LLM misspellings)
    if name in LEGACY_TOOL_ALIASES:
        name = LEGACY_TOOL_ALIASES[name]

    # Argument normalization & auto-routing
    if name == "system_action":
        action = str(args.get("action", "")).strip().lower()
        target = args.get("target", "")

        # Auto-route health checks accidentally sent to system_action
        if action in {"check_cpu", "check_ram", "check_disk", "cpu", "ram", "disk",
                      "network", "battery", "top_processes", "all"}:
            name = "system_health"
            args = {"target": action.replace("check_", "")}
        else:
            args = {"action": action, "target": str(target)}

    elif name == "scheduler":
        # Legacy reminder-style argument adaptation
        if "due" not in args:
            legacy_time = (args.get("time_or_delay") or args.get("delay")
                           or args.get("time") or "")
            if legacy_time:
                args["due"] = legacy_time
        clean = {"action": str(args.get("action", "set")).strip().lower()}
        for k in ("task", "due", "id", "status"):
            if str(args.get(k, "")).strip():
                clean[k] = str(args[k])
        rng = str(args.get("range", "")).strip()
        if rng:
            clean["range"] = rng
        args = clean

    elif name == "user_memory":
        action = str(args.get("action", "")).strip().lower() or "get"
        clean = {"action": action}
        if str(args.get("key", "")).strip():
            clean["key"] = str(args["key"])
        if action == "store" and str(args.get("value", "")) != "":
            clean["value"] = str(args["value"])
        args = clean

    return name, args


def _execute_tool(tool_name: str, args: Dict[str, Any]) -> str:
    """Invokes the appropriate tool function with parameter mapping."""
    if tool_name not in TOOL_REGISTRY:
        return f"ERROR[unknown_tool]: Unknown tool '{tool_name}'. Available tools: {sorted(TOOL_REGISTRY.keys())}"

    func = TOOL_REGISTRY[tool_name]
    try:
        if tool_name == "calculator":
            expr = args.get("expression") or args.get("expr") or args.get("query") or ""
            return func(expression=str(expr))
        if tool_name == "web_search":
            query = args.get("query") or args.get("q") or args.get("keywords") or ""
            return func(query=str(query))
        if tool_name == "user_memory":
            return func(
                action=str(args.get("action", "get")),
                key=str(args.get("key", "")),
                value=str(args.get("value", "")),
            )
        if tool_name == "scheduler":
            kwargs = {"action": str(args.get("action", "set"))}
            for key in ("task", "due", "id", "status", "range"):
                if str(args.get(key, "")).strip():
                    kwargs[key] = str(args[key])
            return func(**kwargs)
        if tool_name == "system_health":
            return func(target=str(args.get("target", "all")))
        if tool_name == "system_action":
            return func(action=str(args.get("action", "")), target=str(args.get("target", "")))
    except Exception as e:
        return f"ERROR[exception]: executing {tool_name} failed: {e}"

    return f"ERROR[execution_failed]: Tool execution failed for {tool_name}."


class NanoHatAgent:
    """
    Lightweight, native multi-turn agent runtime for SmolLM2-360M / NanoHat Fedora models.
    """
    def __init__(self, backend: str = "ollama", model_name: str = "nanohat:360m", max_steps: int = 5):
        self.backend = backend
        self.model_name = model_name
        self.max_steps = max_steps
        self.api_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/api/chat")
        
        # Internal conversation state
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": CANONICAL_SYSTEM_PROMPT}
        ]

        self.transformers_model = None
        self.transformers_tokenizer = None

        if self.backend == "transformers":
            self._init_transformers()

        # Persistent tasks must ring even if they came due while we were closed.
        try:
            start_notification_worker()
        except Exception:
            pass

    def _log_session(self, user_input: str, final_text: str, trace: list):
        """Appends a compact JSONL trace for later eval/dataset mining. Opt-out via env."""
        if os.environ.get("NANOHAT_DISABLE_LOGGING", "").strip() in {"1", "true", "yes"}:
            return
        try:
            import datetime as _dt
            log_path = os.path.join(
                os.path.expanduser("~/.config/smollm2-fedora-agent"), "sessions.jsonl")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            entry = {
                "ts": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "user": user_input[:500],
                "steps": [{"tool": s[0], "args": s[1], "output": s[2][:400]} for s in trace],
                "final": final_text[:1000],
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _init_transformers(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"Loading Transformers model: {self.model_name}...")
            self.transformers_tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.transformers_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                trust_remote_code=True
            )
            print("✓ Transformers model loaded successfully.")
        except Exception as e:
            print(f"[Error loading Transformers model]: {e}")
            sys.exit(1)

    def _query_llm(self, messages: List[Dict[str, str]]) -> str:
        """Sends conversation turns to the selected backend with greedy sampling (temp=0.0)."""
        view = self._build_context_view(messages)
        if self.backend == "ollama":
            payload = {
                "model": self.model_name,
                "messages": view,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "stop": ["<|im_end|>", "<|endoftext|>"],
                    "num_ctx": OLLAMA_NUM_CTX,
                }
            }
            req = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("message", {}).get("content", "").strip()
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"Ollama returned HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:200]}")
            except urllib.error.URLError as e:
                raise RuntimeError(
                    f"Failed to connect to Ollama at {self.api_url}. Is Ollama running? (ollama serve). Details: {e}"
                )

        else: # Transformers
            import torch
            formatted_prompt = self.transformers_tokenizer.apply_chat_template(
                view,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.transformers_tokenizer(formatted_prompt, return_tensors="pt").to(self.transformers_model.device)
            with torch.no_grad():
                outputs = self.transformers_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.transformers_tokenizer.eos_token_id
                )
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            return self.transformers_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def run(self, user_input: str) -> str:
        """Executes a full user turn with multi-step tool execution and observation feedback."""
        self.messages.append({"role": "user", "content": user_input})
        self._prune_history()

        step = 0
        executed_calls = set()
        repeat_count = 0
        trace = []

        while step < self.max_steps:
            step += 1
            raw_response = self._query_llm(self.messages)

            if not raw_response:
                return "I did not receive a response from the model."

            # 1. Parse <thought>
            thought_match = re.search(r'<thought>(.*?)</thought>', raw_response, re.DOTALL)
            if thought_match:
                thought_text = thought_match.group(1).strip()
                print(f"  💭 [Thought]: {thought_text}")

            # 2. Check for <tool_call> (or fallback <tool>)
            tool_match = re.search(r'<tool_call>(.*?)</tool_call>', raw_response, re.DOTALL)
            if not tool_match:
                tool_match = re.search(r'<tool>(.*?)</tool>', raw_response, re.DOTALL)

            # Scenario A: No tool called -> Pure conversational answer or Final resolution
            if not tool_match:
                # Clean up any leftover XML tags if present
                clean_response = re.sub(r'<thought>.*?</thought>', '', raw_response, flags=re.DOTALL).strip()
                clean_response = re.sub(r'</?(?:thought|tool_call|tool)>', '', clean_response).strip()

                final_text = clean_response if clean_response else raw_response
                self.messages.append({"role": "assistant", "content": final_text})
                self._prune_history()
                self._log_session(user_input, final_text, trace)
                return final_text

            # Scenario B: Tool Call requested
            tool_payload = tool_match.group(1).strip()
            parsed_json = _repair_and_parse_json(tool_payload)

            if not parsed_json or "name" not in parsed_json:
                print(f"  ⚠️ [Invalid JSON in tool call]: {tool_payload}")
                # Provide immediate self-correction feedback turn
                self.messages.append({"role": "assistant", "content": raw_response})
                self.messages.append({
                    "role": "tool",
                    "content": 'ERROR[invalid_format]: Malformed tool call. Must be exactly '
                               '<tool_call>{"name": "...", "arguments": {...}}</tool_call> with valid JSON.'
                })
                continue

            tool_name = parsed_json.get("name", "")
            raw_args = parsed_json.get("arguments", {})
            if not isinstance(raw_args, dict):
                raw_args = {}

            # Normalize & Auto-route
            tool_name, tool_args = _normalize_tool_call(tool_name, raw_args)

            # Repetition / Loop Guard: one finalize nudge, then hard stop
            call_signature = (tool_name, json.dumps(tool_args, sort_keys=True))
            if call_signature in executed_calls:
                repeat_count += 1
                print(f"  ⚠️ [Loop Detected]: Repeated call {tool_name}({tool_args}).")
                if repeat_count >= 2:
                    apology = ("I got stuck repeating the same action and stopped to avoid a loop. "
                               "Nothing else was changed - could you rephrase that?")
                    self.messages.append({"role": "assistant", "content": apology})
                    self._log_session(user_input, apology, trace)
                    return apology
                self.messages.append({"role": "assistant", "content": raw_response})
                self.messages.append({
                    "role": "tool",
                    "content": "ERROR[repeated_call]: You already ran this exact call and got the result above. "
                               "Do NOT call the same tool again. Summarize all collected information for "
                               "the user now in plain text."
                })
                continue

            executed_calls.add(call_signature)

            # Execute tool safely
            print(f"  ⚡ [Tool Call]: {tool_name}({tool_args})")
            tool_output = _execute_tool(tool_name, tool_args)
            print(f"  📥 [Tool Output]: {tool_output}")
            trace.append((tool_name, tool_args, tool_output))

            # Append assistant step and capped tool observation to history
            self.messages.append({"role": "assistant", "content": raw_response})
            self.messages.append({
                "role": "tool",
                "content": _cap_tool_output(tool_output),
            })
            self._prune_history()

        final = ("I hit my maximum number of steps for this request without fully resolving it. "
                 "Here is what I know so far - could you narrow the request down?")
        self._log_session(user_input, final, trace)
        return final


    # ------------------------------------------------------------------
    # Context management: keep rendered history within the training window
    # ------------------------------------------------------------------
    def _build_context_view(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Returns a budgeted view of the conversation for the LLM.

        - Always keeps the system prompt (messages[0]).
        - Walks newest-first until MAX_HISTORY_CHARS is exhausted, then snaps
          the start boundary forward to the next 'user' message so the view
          never begins mid-exchange.
        """
        if not messages:
            return messages
        system = messages[0] if messages[0].get("role") == "system" else None
        body = messages[1:] if system else messages

        total = len(system["content"]) if system else 0
        kept = []
        for msg in reversed(body):
            cost = len(msg.get("content", "")) + 12
            if total + cost > MAX_HISTORY_CHARS and kept:
                break
            total += cost
            kept.append(msg)
        kept.reverse()

        # Snap start to a 'user' turn so the view begins at an exchange boundary
        while kept and kept[0]["role"] != "user":
            kept.pop(0)

        view = ([system] if system else []) + kept
        return view

    def _prune_history(self) -> None:
        """Hard-prunes persistent history to the same budget used for the LLM view."""
        if len(self.messages) <= 1:
            return
        self.messages = self._build_context_view(self.messages)

    def reset_session(self) -> None:
        """Clears session history back to the initial system prompt."""
        self.messages = [{"role": "system", "content": CANONICAL_SYSTEM_PROMPT}]


def _cap_tool_output(output: str, limit: int = TOOL_MSG_CHAR_LIMIT) -> str:
    """Caps oversized tool observations so they cannot flood the tiny context."""
    if len(output) <= limit:
        return output
    head = output[:limit]
    # Keep whole lines when possible
    nl = head.rfind("\n")
    if nl > limit // 2:
        head = head[:nl]
    return head + "\n…[output truncated]"


def main():
    parser = argparse.ArgumentParser(
        description="NanoHat 2.1 — Autonomous Fedora Linux Agent (SmolLM2-360M)",
        epilog="Examples:\n  nanohat 'What is 15 percent of 800?'\n  nanohat 'What is my current RAM usage?'\n  nanohat 'Check battery status'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="*", help="Instruction or question to execute")
    parser.add_argument("-q", "--query", dest="named_query", type=str, default=None, help="Explicit query string")
    parser.add_argument("--backend", choices=["ollama", "transformers"], default="ollama", help="Inference backend (default: ollama)")
    parser.add_argument("--model", default="nanohat2.1:360m", help="Model tag or path (default: nanohat2.1:360m)")
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum tool execution steps per turn (default: 5)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive multi-turn chat session")
    args = parser.parse_args()

    query_str = args.named_query if args.named_query is not None else (" ".join(args.query).strip() if args.query else "")

    if not query_str and not args.interactive:
        print("""🎩 NanoHat 2.1 — Autonomous Fedora Linux Agent (SmolLM2-360M)

Usage:
  nanohat "<instruction or question>"

Examples:
  nanohat "What is 15 percent of 800?"
  nanohat "What is my current RAM usage?"
  nanohat "Check battery status"
  nanohat "Search for Fedora 41 release schedule"

Options:
  --backend {ollama,transformers}  Inference backend (default: ollama)
  --model MODEL                    Model tag (default: nanohat2.1:360m)
  --interactive                    Start interactive chat session
  -h, --help                       Show this help message
""")
        return

    agent = NanoHatAgent(backend=args.backend, model_name=args.model, max_steps=args.max_steps)

    if query_str:
        try:
            response = agent.run(query_str)
            print(response)
        except Exception as e:
            print(f"[Runtime Error]: {e}", file=sys.stderr)
            sys.exit(1)
        return

    print("\n" + "="*60)
    print("🤖 NanoHat-360M Fedora Agent Online")
    print("   Backend:", args.backend, "| Model:", args.model)
    print("   Type 'exit' or 'quit' to close.")
    print("="*60 + "\n")

    while True:
        try:
            query = input("User > ").strip()
            if not query or query.lower() in {"exit", "quit"}:
                print("\nGoodbye!")
                break
            response = agent.run(query)
            print(f"\nAgent > {response}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Runtime Error]: {e}\n")


if __name__ == "__main__":
    main()
