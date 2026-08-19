"""
main.py
Unified entry point for the SmolLM2 / NanoHat Fedora Linux Agent.

Usage:
  python main.py                     # Multi-Agent Pipeline (Default)
  python main.py --legacy            # Legacy single-agent runtime
  python main.py --quiet             # Run with clean output without intermediate traces
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="NanoHat Fedora Linux AI Agent")
    parser.add_argument("--legacy", action="store_true", help="Run the legacy single-agent runtime")
    parser.add_argument("--router-model", default="smollm2:135m", help="Router model tag in Ollama")
    parser.add_argument("--planner-model", default="smollm2:360m", help="Planner model tag in Ollama")
    parser.add_argument("--executor-model", default="nanohat2:360m", help="Executor fine-tuned model tag in Ollama")
    parser.add_argument("--chitchat-model", default="smollm2:360m", help="Chitchat & Synthesizer model tag in Ollama")
    parser.add_argument("--max-history", type=int, default=6, help="Sliding window size (turns)")
    parser.add_argument("--quiet", action="store_true", help="Suppress intermediate thought and tool logs")
    args, unknown = parser.parse_known_args()

    if args.legacy:
        from runtime.agent import main as legacy_main
        sys.argv = [sys.argv[0]] + unknown
        legacy_main()
    else:
        from runtime.multi_agent.orchestrator import NanoHatOrchestrator
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
        print(f"   Memory:   Sliding Window ({args.max_history} turns) + Clean Quarantine")
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
