"""
runtime/multi_agent/executor.py
Agent 4: NanoHat2 Executor
Model: nanohat2:360m (fine-tuned)
Role: Executes tool steps in a single rolling conversation per task, allowing dynamic step dependencies
(e.g., get PID in Step 1 -> kill PID in Step 2) to resolve naturally via rolling tool observations.
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from runtime.multi_agent.base_agent import BaseAgent
from runtime.agent import (
    CANONICAL_SYSTEM_PROMPT,
    _repair_and_parse_json,
    _normalize_tool_call,
    _validate_tool_call,
    _execute_tool,
)


class ExecutorAgent(BaseAgent):
    """Executes discrete tool steps in a rolling session with tool validation and error feedback."""

    def __init__(
        self,
        model_name: str = "nanohat2:360m",
        system_prompt: str = CANONICAL_SYSTEM_PROMPT,
        api_url: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            temperature=0.0,
            api_url=api_url,
        )

    def run_plan(
        self,
        steps: List[str],
        max_step_retries: int = 2,
        verbose: bool = True,
    ) -> Tuple[List[str], Optional[str]]:
        """
        Executes a sequence of steps in a single persistent rolling conversation.
        Returns: (observations_list, error_summary_or_None)
        """
        session_messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        observations: List[str] = []
        executed_signatures = set()

        for step_idx, step in enumerate(steps, start=1):
            session_messages.append({"role": "user", "content": step})
            step_success = False
            step_error: Optional[str] = None

            for attempt in range(max_step_retries + 1):
                raw_response = self._call(session_messages)
                if not raw_response:
                    step_error = f"Step {step_idx} ('{step}') returned empty response from model."
                    continue

                # 1. Inspect <thought>
                thought_match = re.search(r"<thought>(.*?)</thought>", raw_response, re.DOTALL)
                if thought_match and verbose:
                    print(f"    💭 [NanoHat2 Thought]: {thought_match.group(1).strip()}")

                # 2. Extract <tool_call> or <tool>
                tool_match = re.search(r"<tool_call>(.*?)</tool_call>", raw_response, re.DOTALL)
                if not tool_match:
                    tool_match = re.search(r"<tool>(.*?)</tool>", raw_response, re.DOTALL)

                if not tool_match:
                    # Model responded conversationally or didn't produce XML
                    step_error = f"No <tool_call> JSON block produced for step: '{step}'."
                    session_messages.append({"role": "assistant", "content": raw_response})
                    session_messages.append({
                        "role": "tool",
                        "content": "Error: You must output a valid <tool_call> JSON block for this step."
                    })
                    continue

                tool_payload = tool_match.group(1).strip()
                parsed = _repair_and_parse_json(tool_payload)

                if not parsed or "name" not in parsed:
                    step_error = f"Malformed JSON in tool call: '{tool_payload}'"
                    session_messages.append({"role": "assistant", "content": raw_response})
                    session_messages.append({
                        "role": "tool",
                        "content": "Error: Invalid JSON formatting in <tool_call>. Must be JSON with 'name' and 'arguments'."
                    })
                    continue

                tool_name = parsed.get("name", "")
                raw_args = parsed.get("arguments", {})
                if not isinstance(raw_args, dict):
                    raw_args = {}

                # Normalize aliases & validate
                tool_name, tool_args = _normalize_tool_call(tool_name, raw_args)
                is_valid, validation_error = _validate_tool_call(tool_name, tool_args)

                if not is_valid:
                    step_error = f"Validation failed: {validation_error}"
                    session_messages.append({"role": "assistant", "content": raw_response})
                    session_messages.append({"role": "tool", "content": f"Error: {validation_error}"})
                    continue

                # Repetition check
                call_sig = (tool_name, json.dumps(tool_args, sort_keys=True))
                if call_sig in executed_signatures:
                    if verbose:
                        print(f"    ⚠️ [Loop Guard]: Repeated call {tool_name}({tool_args}).")
                    step_success = True
                    break

                executed_signatures.add(call_sig)

                # Execute tool
                if verbose:
                    print(f"    ⚡ [Executing Tool]: {tool_name}({tool_args})")
                tool_output = _execute_tool(tool_name, tool_args)
                if verbose:
                    print(f"    📥 [Tool Output]: {tool_output}")

                observations.append(tool_output)
                session_messages.append({"role": "assistant", "content": raw_response})
                session_messages.append({"role": "tool", "content": tool_output})
                step_success = True
                break

            if not step_success:
                return observations, step_error or f"Step {step_idx} ('{step}') failed after retries."

        return observations, None
