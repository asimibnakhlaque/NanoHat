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
    reminder,
    system_health,
    system_action,
)

CANONICAL_SYSTEM_PROMPT = (
    "You are a helpful AI agent running on Fedora Linux. "
    "Available tools: [calculator, web_search, user_memory, reminder, system_health, system_action]. "
    "Use tools when necessary by reasoning inside <thought> tags, then outputting a <tool_call> JSON block."
)

TOOL_REGISTRY = {
    "calculator": calculator,
    "web_search": web_search,
    "user_memory": user_memory,
    "reminder": reminder,
    "system_health": system_health,
    "system_action": system_action,
}


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

    # Attempt 4: Regex key-value extraction for name & arguments
    name_match = re.search(r'["\']?name["\']?\s*:\s*["\']([a-zA-Z0-9_]+)["\']', raw)
    if name_match:
        tool_name = name_match.group(1)
        args_match = re.search(r'["\']?(?:arguments|parameters|params|args)["\']?\s*:\s*({.*?})', raw, re.DOTALL)
        if args_match:
            try:
                args = json.loads(args_match.group(1))
                return {"name": tool_name, "arguments": args}
            except Exception:
                pass
        return {"name": tool_name, "arguments": {}}

    return None


def _normalize_tool_call(tool_name: str, args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Normalizes tool names and argument aliases produced by small models."""
    name = tool_name.strip().lower()
    
    # Tool name aliases
    if name in {"calc", "math", "add_calculator", "evaluate"}:
        name = "calculator"
    elif name in {"search", "web", "google", "ddg"}:
        name = "web_search"
    elif name in {"memory", "store_memory", "get_memory"}:
        name = "user_memory"
    elif name in {"remind", "schedule_reminder", "notify"}:
        name = "reminder"
    elif name in {"health", "sys_health", "diagnostics", "check_system"}:
        name = "system_health"
    elif name in {"action", "sys_action", "os_action"}:
        name = "system_action"

    # Argument normalization & auto-routing
    if name == "system_action":
        action = str(args.get("action", "")).strip().lower()
        target = str(args.get("target", "")).strip()

        # Auto-route health checks accidentally sent to system_action
        if action in {"check_cpu", "cpu", "ram", "disk", "network", "battery", "top_processes", "all"}:
            name = "system_health"
            args = {"target": action.replace("check_", "")}
        elif action == "check_ram":
            name = "system_health"
            args = {"target": "ram"}
        elif action == "check_disk":
            name = "system_health"
            args = {"target": "disk"}

    return name, args


def _execute_tool(tool_name: str, args: Dict[str, Any]) -> str:
    """Invokes the appropriate tool function with parameter mapping."""
    if tool_name not in TOOL_REGISTRY:
        return f"Error: Unknown tool '{tool_name}'. Available tools: {list(TOOL_REGISTRY.keys())}"

    func = TOOL_REGISTRY[tool_name]
    try:
        if tool_name == "calculator":
            expr = args.get("expression") or args.get("expr") or args.get("query") or ""
            return func(expression=str(expr))
        elif tool_name == "web_search":
            query = args.get("query") or args.get("q") or args.get("keywords") or ""
            return func(query=str(query))
        elif tool_name == "user_memory":
            action = args.get("action", "get")
            key = args.get("key", "")
            value = args.get("value", "")
            return func(action=str(action), key=str(key), value=str(value))
        elif tool_name == "reminder":
            task = args.get("task", "")
            time_or_delay = args.get("time_or_delay") or args.get("delay") or args.get("time") or "1m"
            return func(task=str(task), time_or_delay=str(time_or_delay))
        elif tool_name == "system_health":
            target = args.get("target", "all")
            return func(target=str(target))
        elif tool_name == "system_action":
            action = args.get("action", "")
            target = args.get("target", "")
            return func(action=str(action), target=str(target))
    except Exception as e:
        return f"Error executing {tool_name}: {e}"

    return f"Error: Tool execution failed for {tool_name}."


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
        if self.backend == "ollama":
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "stop": ["<|im_end|>", "<|endoftext|>"]
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
            except urllib.error.URLError as e:
                raise RuntimeError(
                    f"Failed to connect to Ollama at {self.api_url}. Is Ollama running? (ollama serve). Details: {e}"
                )

        else: # Transformers
            import torch
            formatted_prompt = self.transformers_tokenizer.apply_chat_template(
                messages,
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
        
        step = 0
        executed_calls = set()

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
                    "content": "Error: Invalid JSON formatting in tool call. Must be <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>"
                })
                continue

            tool_name = parsed_json.get("name", "")
            raw_args = parsed_json.get("arguments", {})
            if not isinstance(raw_args, dict):
                raw_args = {}

            # Normalize & Auto-route
            tool_name, tool_args = _normalize_tool_call(tool_name, raw_args)

            # Repetition / Loop Guard
            call_signature = (tool_name, json.dumps(tool_args, sort_keys=True))
            if call_signature in executed_calls:
                print(f"  ⚠️ [Loop Detected]: Repeated call {tool_name}({tool_args}). Forcing final answer.")
                self.messages.append({"role": "assistant", "content": raw_response})
                self.messages.append({
                    "role": "tool",
                    "content": "Observation complete. Summarize all collected information for the user now."
                })
                continue

            executed_calls.add(call_signature)

            # Execute tool safely
            print(f"  ⚡ [Tool Call]: {tool_name}({tool_args})")
            tool_output = _execute_tool(tool_name, tool_args)
            print(f"  📥 [Tool Output]: {tool_output}")

            # Append assistant step and tool observation to conversation history
            self.messages.append({"role": "assistant", "content": raw_response})
            self.messages.append({"role": "tool", "content": tool_output})

        return "Reached maximum execution steps without final resolution."


def main():
    parser = argparse.ArgumentParser(description="SmolLM2-360M / NanoHat Fedora Agent Runtime")
    parser.add_argument("--backend", choices=["ollama", "transformers"], default="ollama", help="Inference backend")
    parser.add_argument("--model", default="nanohat:360m", help="Model name or weights path")
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum tool execution steps per turn")
    args = parser.parse_args()

    agent = NanoHatAgent(backend=args.backend, model_name=args.model, max_steps=args.max_steps)

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
