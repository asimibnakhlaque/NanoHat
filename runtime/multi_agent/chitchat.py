"""
runtime/multi_agent/chitchat.py
Agent 2: Conversational Responder & Response Synthesizer
Model: smollm2:360m (base)
Role:
  1. CHAT mode: Provides natural, friendly replies to casual conversational turns using clean history.
  2. SYNTH mode: Converts raw tool observations into concise, grounded user-facing prose.
"""

from typing import List, Dict, Optional
from runtime.multi_agent.base_agent import BaseAgent

CHAT_SYSTEM_PROMPT = (
    "You are a friendly, helpful AI assistant running on Fedora Linux. "
    "Answer the user naturally, concisely, and accurately. "
    "Do NOT output XML tags or JSON tool calls."
)

SYNTH_SYSTEM_PROMPT = (
    "You are a helpful assistant running on Fedora Linux. "
    "Given the user's initial request and the collected tool execution results, "
    "write a concise, clear, and friendly answer. "
    "Only use facts provided in the tool results. Do not guess or invent data. "
    "Do NOT output XML tags, JSON blocks, or tool call syntax."
)


class ChitchatAgent(BaseAgent):
    """Conversational assistant and tool observation synthesizer."""

    def __init__(self, model_name: str = "smollm2:360m", api_url: Optional[str] = None):
        super().__init__(
            model_name=model_name,
            system_prompt=CHAT_SYSTEM_PROMPT,
            temperature=0.3,
            api_url=api_url,
        )

    def chat(self, clean_history: List[Dict[str, str]], user_query: str) -> str:
        """Generates a natural conversational reply using the sanitized multi-turn history."""
        messages = self._build_messages(clean_history, user_query)
        return self._call(messages, override_temperature=0.3)

    def synthesize(self, user_query: str, observations: List[str]) -> str:
        """Synthesizes raw tool outputs into a natural response for the user."""
        formatted_obs = "\n".join([f"- {obs}" for obs in observations if obs])
        if not formatted_obs:
            formatted_obs = "- (No output generated)"

        synth_content = (
            f"User Request: {user_query}\n\n"
            f"Tool Execution Results:\n{formatted_obs}\n\n"
            "Provide the final response for the user:"
        )

        messages = [
            {"role": "system", "content": SYNTH_SYSTEM_PROMPT},
            {"role": "user", "content": synth_content},
        ]
        return self._call(messages, override_temperature=0.2)
