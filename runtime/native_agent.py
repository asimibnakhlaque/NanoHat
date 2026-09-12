"""
runtime/native_agent.py
Production Native Ollama Fedora Linux Assistant.
Utilizes the official Ollama Native Tool Calling standard (model-level JSON schema constraints)
with multi-turn session memory and parallel tool execution.
"""

import os
import sys
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

from runtime.tools import (
    calculator,
    web_search,
    user_memory,
    scheduler,
    system_health,
    system_action,
    start_notification_worker,
)
from generator.schemas import TOOL_DEFINITIONS

DEFAULT_SYSTEM_PROMPT = (
    "You are NanoHat, a helpful AI agent running on Fedora Linux. "
    "You have access to safe system tools for calculations, web searches, persistent user memory, "
    "scheduled task management, system resource health checks, and desktop operations. "
    "Always invoke tool functions directly through the function calling API whenever a task "
    "(such as scheduling a task, storing memory, or checking system metrics) is requested. "
    "Never print tool call JSON code blocks in your text response-always execute the tool directly."
)

# Legacy reminder-style calls route into the scheduler transparently.
def _legacy_reminder(task: str = "", time_or_delay: str = "", **_ignored) -> str:
    return scheduler(action="set", task=task, due=time_or_delay)


TOOL_DISPATCHER = {
    "calculator": calculator,
    "web_search": web_search,
    "user_memory": user_memory,
    "scheduler": scheduler,
    "reminder": _legacy_reminder,
    "system_health": system_health,
    "system_action": system_action,
}


class NativeOllamaAgent:
    """Unified desktop AI assistant powered by native Ollama tool calling."""

    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        api_url: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_steps: int = 5,
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.api_url = api_url or os.getenv("OLLAMA_API_BASE", "http://localhost:11434/api/chat")
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_steps = max_steps
        self.verbose = verbose

        # Format schemas for Ollama
        self.tools = [{"type": "function", "function": t} for t in TOOL_DEFINITIONS]

        # Multi-turn conversation state
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def reset_conversation(self):
        """Clears session history back to the initial system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def _call_ollama(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calls the local Ollama chat endpoint with function schemas."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "tools": self.tools,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("message", {})
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to connect to Ollama ({self.model_name}) at {self.api_url}. Is Ollama running? (ollama serve). Details: {e}"
            )

    def run(self, user_input: str) -> str:
        """
        Executes a user turn. Handles multi-step and parallel tool calling loops
        until the model returns its final conversational response.
        """
        self.messages.append({"role": "user", "content": user_input})
        step = 0
        executed_signatures = set()

        while step < self.max_steps:

            step += 1
            asst_message = self._call_ollama(self.messages)
            self.messages.append(asst_message)

            tool_calls = asst_message.get("tool_calls", [])

            # Scenario A: No tool calls -> Model produced its final response
            if not tool_calls:
                final_content = asst_message.get("content", "").strip()
                return final_content if final_content else "Task completed."

            # Scenario B: One or more tool calls produced
            for tc in tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                fn_args = fn.get("arguments", {})

                if not isinstance(fn_args, dict):
                    fn_args = {}

                call_sig = (fn_name, json.dumps(fn_args, sort_keys=True))
                if call_sig in executed_signatures:
                    if self.verbose:
                        print(f"  ⚠️ [Loop Guard]: Skipping repeated call {fn_name}({fn_args}).")
                    self.messages.append({
                        "role": "tool",
                        "content": "Observation complete. Summarize the findings for the user.",
                    })
                    continue

                executed_signatures.add(call_sig)

                if self.verbose:
                    print(f"  ⚡ [Tool Call]: {fn_name}({fn_args})")

                tool_func = TOOL_DISPATCHER.get(fn_name)
                if tool_func:
                    try:
                        tool_output = str(tool_func(**fn_args))
                    except Exception as e:
                        tool_output = f"Error executing {fn_name}: {e}"
                else:
                    tool_output = f"Error: Unknown tool '{fn_name}'."

                if self.verbose:
                    print(f"  📥 [Tool Result]: {tool_output}")

                # Feed observation back to the model as role: tool
                self.messages.append({
                    "role": "tool",
                    "content": tool_output,
                })

        return "Reached maximum tool execution turns."


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fedora Linux AI Assistant (Native Ollama Tool Calling)")
    parser.add_argument("query", nargs="*", help="Optional query to execute in one-shot mode")
    parser.add_argument("-q", "--query", dest="named_query", type=str, default=None, help="Explicit single-shot query string")
    parser.add_argument("--model", default="nanohat2.1:360m", help="Ollama model tag (default: nanohat2.1:360m)")
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum tool execution steps per turn")
    parser.add_argument("--quiet", action="store_true", help="Suppress tool execution logs")
    args = parser.parse_args()

    agent = NativeOllamaAgent(
        model_name=args.model,
        max_steps=args.max_steps,
        verbose=not args.quiet,
    )

    query_str = args.named_query if args.named_query is not None else (" ".join(args.query).strip() if args.query else "")

    if query_str:
        try:
            response = agent.run(query_str)
            print(response)
        except Exception as e:
            print(f"[Runtime Error]: {e}", file=sys.stderr)
            sys.exit(1)
        return

    print("\n" + "=" * 65)
    print("🤖 Fedora Linux AI Assistant Online")
    print(f"   Model:     {args.model}")
    print("   Protocol:  Native Ollama Tool Calling (/api/chat)")
    print(f"   Tools:     calculator, web_search, user_memory, scheduler,")
    print(f"              system_health, system_action")
    print("   Type 'exit' or 'quit' to close. Type 'clear' to reset chat.")
    print("=" * 65 + "\n")

    while True:
        try:
            query = input("User > ").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit"}:
                print("\nGoodbye!")
                break
            if query.lower() == "clear":
                agent.reset_conversation()
                print("\n🧹 Conversation memory cleared.\n")
                continue

            response = agent.run(query)
            print(f"\nAgent > {response}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Runtime Error]: {e}\n")


if __name__ == "__main__":
    main()
