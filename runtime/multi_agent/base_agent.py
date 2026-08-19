"""
runtime/multi_agent/base_agent.py
Shared base class for all multi-agent pipeline components communicating with Ollama.
"""

import os
import json
import urllib.request
import urllib.error
from typing import List, Dict, Optional, Any

OLLAMA_API_URL = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/api/chat")


class BaseAgent:
    """Base client for an agent invoking a local Ollama model."""

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        temperature: float = 0.0,
        api_url: Optional[str] = None,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.api_url = api_url or OLLAMA_API_URL

    def _call(
        self,
        messages: List[Dict[str, str]],
        override_temperature: Optional[float] = None,
        override_model: Optional[str] = None,
        timeout: int = 60,
    ) -> str:
        """Sends messages to Ollama and returns the raw assistant reply."""
        temp = self.temperature if override_temperature is None else override_temperature
        model = override_model or self.model_name

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temp,
                "stop": ["<|im_end|>", "<|endoftext|>"],
            },
        }

        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("message", {}).get("content", "").strip()
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to connect to Ollama ({model}) at {self.api_url}. Is Ollama running? Details: {e}"
            )

    def _build_messages(
        self,
        history: List[Dict[str, str]],
        user_input: str,
        system_override: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Constructs a message list with the system prompt, history, and current user input."""
        sys_prompt = system_override if system_override is not None else self.system_prompt
        return (
            [{"role": "system", "content": sys_prompt}]
            + history
            + [{"role": "user", "content": user_input}]
        )
