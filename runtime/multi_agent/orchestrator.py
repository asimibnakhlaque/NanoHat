"""
runtime/multi_agent/orchestrator.py
Master Multi-Agent Orchestrator for Fedora Linux.
Coordinates Router, Planner, Executor, and Chitchat agents with Two-Layer Memory isolation.
"""

import sys
import argparse
from typing import List, Dict, Optional

from runtime.multi_agent.router import RouterAgent
from runtime.multi_agent.planner import PlannerAgent
from runtime.multi_agent.executor import ExecutorAgent
from runtime.multi_agent.chitchat import ChitchatAgent


class NanoHatOrchestrator:
    """Coordinates specialized sub-agents with strict context sanitization and sliding-window memory."""

    def __init__(
        self,
        router_model: str = "smollm2:135m",
        planner_model: str = "smollm2:360m",
        executor_model: str = "nanohat2:360m",
        chitchat_model: str = "smollm2:360m",
        max_history_turns: int = 6,
        max_planner_retries: int = 2,
        verbose: bool = True,
        api_url: Optional[str] = None,
    ):
        self.router = RouterAgent(model_name=router_model, api_url=api_url)
        self.planner = PlannerAgent(model_name=planner_model, api_url=api_url)
        self.executor = ExecutorAgent(model_name=executor_model, api_url=api_url)
        self.chitchat = ChitchatAgent(model_name=chitchat_model, api_url=api_url)

        self.max_history_turns = max_history_turns
        self.max_planner_retries = max_planner_retries
        self.verbose = verbose

        # Two-Layer Memory
        self.clean_history: List[Dict[str, str]] = []
        self._tool_scratchpad: Dict[str, any] = {}

    def _trim_history(self):
        """Maintains clean history within the sliding window limit (2 messages per turn)."""
        max_messages = self.max_history_turns * 2
        if len(self.clean_history) > max_messages:
            self.clean_history = self.clean_history[-max_messages:]

    def _update_clean_history(self, user_query: str, assistant_response: str):
        """Appends sanitized user/assistant text pairs into long-term clean memory."""
        self.clean_history.append({"role": "user", "content": user_query})
        self.clean_history.append({"role": "assistant", "content": assistant_response})
        self._trim_history()

    def run(self, user_query: str) -> str:
        """Executes a full user turn across the multi-agent pipeline."""
        self._tool_scratchpad = {}

        # 1. Routing
        intent = self.router.classify(user_query)
        if self.verbose:
            print(f"\n[Router] Intent -> {intent}")

        # 2A. Conversational Path
        if intent == "CHAT":
            response = self.chitchat.chat(self.clean_history, user_query)
            self._update_clean_history(user_query, response)
            return response

        # 2B. Tool Path: Planning
        plan_text = self.planner.create_plan(user_query)
        steps = self.planner.parse_steps(plan_text)
        if self.verbose:
            print(f"[Planner] Generated Plan:\n  " + "\n  ".join([f"{i}. {s}" for i, s in enumerate(steps, 1)]))

        if not steps:
            # Fallback to chat if no actionable steps were produced
            response = self.chitchat.chat(self.clean_history, user_query)
            self._update_clean_history(user_query, response)
            return response

        # 3. Execution in Rolling Session
        observations, error = self.executor.run_plan(steps, verbose=self.verbose)

        # 4. Self-Healing Diagnostician Loop (if execution failed)
        retries = 0
        current_plan_text = plan_text
        while error and retries < self.max_planner_retries:
            retries += 1
            if self.verbose:
                print(f"⚠️ [Self-Healing] Step failed ({error}). Requesting Planner rediagnosis (attempt {retries}/{self.max_planner_retries})...")
            new_plan_text = self.planner.diagnose_and_replan(user_query, current_plan_text, error)
            new_steps = self.planner.parse_steps(new_plan_text)
            current_plan_text = new_plan_text
            if new_steps:
                if self.verbose:
                    print(f"[Planner] Corrected Plan:\n  " + "\n  ".join([f"{i}. {s}" for i, s in enumerate(new_steps, 1)]))
                new_observations, new_error = self.executor.run_plan(new_steps, verbose=self.verbose)
                if new_observations:
                    observations = new_observations
                error = new_error
                if not error:
                    break
            else:
                break

        # 5. Synthesis
        all_obs = observations if observations else [f"Tool execution encountered an error: {error}"]
        response = self.chitchat.synthesize(user_query, all_obs)
        self._update_clean_history(user_query, response)
        self._tool_scratchpad = {}  # Wipe transient scratchpad
        return response


def main():
    parser = argparse.ArgumentParser(description="NanoHat Multi-Agent Fedora Linux Runtime")
    parser.add_argument("--router-model", default="smollm2:135m", help="Router model tag")
    parser.add_argument("--planner-model", default="smollm2:360m", help="Planner model tag")
    parser.add_argument("--executor-model", default="nanohat2:360m", help="Executor fine-tuned model tag")
    parser.add_argument("--chitchat-model", default="smollm2:360m", help="Chitchat & Synthesizer model tag")
    parser.add_argument("--max-history", type=int, default=6, help="Sliding window size (turns)")
    parser.add_argument("--quiet", action="store_true", help="Suppress intermediate thought/tool logs")
    args = parser.parse_args()

    orchestrator = NanoHatOrchestrator(
        router_model=args.router_model,
        planner_model=args.planner_model,
        executor_model=args.executor_model,
        chitchat_model=args.chitchat_model,
        max_history_turns=args.max_history,
        verbose=not args.quiet,
    )

    print("\n" + "=" * 65)
    print("🤖 NanoHat Multi-Agent Fedora System Online")
    print(f"   Router:   {args.router_model} | Planner:  {args.planner_model}")
    print(f"   Executor: {args.executor_model} | Chitchat: {args.chitchat_model}")
    print("   Type 'exit' or 'quit' to close.")
    print("=" * 65 + "\n")

    while True:
        try:
            query = input("User > ").strip()
            if not query or query.lower() in {"exit", "quit"}:
                print("\nGoodbye!")
                break
            response = orchestrator.run(query)
            print(f"\nAgent > {response}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Runtime Error]: {e}\n")


if __name__ == "__main__":
    main()
