"""
runtime/agent.py
Hugging Face smolagents runtime harness for SmolLM2-360M-Instruct on Fedora Linux.
Includes FedoraFormatBridge to seamlessly translate custom <thought>/<tool_call> tags.
"""

import os
import sys
import re
import argparse
from runtime.tools import (
    calculator, web_search, user_memory, reminder, system_health, system_action
)

# Flexible imports
try:
    from smolagents import ToolCallingAgent, OpenAIServerModel, TransformersModel
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False
    OpenAIServerModel = object
    TransformersModel = object
    ToolCallingAgent = None

class FedoraFormatBridge(OpenAIServerModel):
    """
    Bridges custom-trained <thought> and <tool_call> model outputs
    to the JSON structure parsed by smolagents.ToolCallingAgent.
    """
    def __call__(self, messages, stop_sequences=None, grammar=None):
        raw_output = super().__call__(messages, stop_sequences, grammar)
        # Intercept and extract JSON inside <tool_call> tags
        match = re.search(r'<tool_call>(.*?)</tool_call>', str(raw_output), re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw_output

def build_agent(backend="ollama", model_path="smollm2-360m-fedora-agent"):
    if not SMOLAGENTS_AVAILABLE:
        print("Notice: 'smolagents' is not installed in the current Python environment.")
        print("To install all dependencies, run: pip install -r training/requirements.txt")
        return None

    if backend == "ollama":
        model = FedoraFormatBridge(
            model_id=model_path,
            api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1"),
            api_key="ollama"
        )
    else:
        # Load directly in PyTorch memory via Transformers
        model = TransformersModel(model_id=model_path, device_map="auto")
        
    tools = [calculator, web_search, user_memory, reminder, system_health, system_action]
    
    agent = ToolCallingAgent(
        tools=tools,
        model=model,
        max_steps=5
    )
    return agent

def main():
    parser = argparse.ArgumentParser(description="SmolLM2-360M Fedora Agent via smolagents")
    parser.add_argument("--backend", choices=["ollama", "transformers"], default="ollama", help="Inference backend")
    parser.add_argument("--model", default="smollm2-360m-fedora-agent", help="Model name or weights path")
    args = parser.parse_args()

    agent = build_agent(backend=args.backend, model_path=args.model)
    if agent is None:
        sys.exit(1)

    print("\n" + "="*60)
    print("🤖 SmolLM2-360M Fedora Agent (smolagents) Online")
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
