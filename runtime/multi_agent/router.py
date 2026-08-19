"""
runtime/multi_agent/router.py
Agent 1: Fast Intent Classifier
Model: smollm2:135m
Role: Classifies incoming user messages into either 'CHAT' (conversational) or 'TOOL' (system actions/queries).
"""

from typing import Literal, Optional
from runtime.multi_agent.base_agent import BaseAgent

ROUTER_SYSTEM_PROMPT = (
    "Classify the user message. Reply with exactly one word: CHAT or TOOL.\n"
    "TOOL = needs system info, CPU, RAM, disk, battery, network, processes, calculations, reminders, desktop actions, web search, memory, or real-time data.\n"
    "CHAT = greetings, jokes, general knowledge, creative writing, opinions, follow-up banter, or casual conversation.\n"
    "Output only: CHAT or TOOL"
)


class RouterAgent(BaseAgent):
    """Ultra-fast intent classifier running on SmolLM2-135M."""

    def __init__(self, model_name: str = "smollm2:135m", api_url: Optional[str] = None):
        super().__init__(
            model_name=model_name,
            system_prompt=ROUTER_SYSTEM_PROMPT,
            temperature=0.0,
            api_url=api_url,
        )

    def classify(self, user_query: str) -> Literal["CHAT", "TOOL"]:
        """
        Classifies the user query into CHAT or TOOL.
        Leans towards TOOL on ambiguity to ensure actions/data lookups are executed.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]
        raw_output = self._call(messages).strip().upper()

        if "CHAT" in raw_output and "TOOL" not in raw_output:
            return "CHAT"
        if "TOOL" in raw_output:
            return "TOOL"

        # Fallback keyword heuristic for small model edge cases
        lowered = user_query.lower()
        tool_indicators = [
            "cpu", "ram", "memory", "disk", "battery", "wifi", "bluetooth",
            "kill", "restart", "launch", "app", "screenshot", "trash",
            "time", "date", "remind", "calculate", "search", "remember",
            "lock", "process", "status", "+", "-", "*", "/"
        ]
        if any(indicator in lowered for indicator in tool_indicators):
            return "TOOL"

        return "CHAT"
