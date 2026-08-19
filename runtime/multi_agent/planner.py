"""
runtime/multi_agent/planner.py
Agent 3: Planner and Failure Diagnostician
Model: smollm2:360m
Role: Generates high-level numbered tool execution roadmaps for NanoHat2, and diagnoses failures to provide replans.
"""

import re
from typing import List, Optional
from runtime.multi_agent.base_agent import BaseAgent

TOOL_LIST_DOC = (
    "1. calculator(expression): Evaluate math (e.g. '120 * 0.15')\n"
    "2. web_search(query): Real-time web search keywords\n"
    "3. user_memory(action, key, value): Memory actions ('store', 'get', 'delete')\n"
    "4. reminder(task, time_or_delay): Desktop notification (e.g. '5m', '1h')\n"
    "5. system_health(target): Check metrics ('cpu', 'ram', 'disk', 'network', 'battery', 'top_processes', 'all')\n"
    "6. system_action(action, target): Fedora actions ('kill_process', 'restart_service', 'toggle_wifi', 'toggle_bluetooth', 'empty_trash', 'lock_screen', 'take_screenshot', 'launch_app', 'get_datetime')"
)

PLANNER_SYSTEM_PROMPT = f"""You are a task planner for a Fedora Linux AI tool executor.
Available tools:
{TOOL_LIST_DOC}

Given a user request, output ONLY a numbered list of tool calls using the available tool names.
Format strictly as:
1. <tool_name>(<arguments>)
2. <tool_name>(<arguments>)

Examples:
User: check my CPU and RAM
1. system_health(target=cpu)
2. system_health(target=ram)

User: remember my favorite editor is vim
1. user_memory(action=store, key=favorite_editor, value=vim)

User: what is 50 * 12?
1. calculator(expression=50 * 12)

Output ONLY the numbered steps. No markdown fences. No explanations. No extra text."""


REPLAN_SYSTEM_PROMPT = f"""You are a failure analyst and plan repair specialist for a Fedora AI tool executor.
Available tools:
{TOOL_LIST_DOC}

You will receive: the original user request, the failed plan, and the exact error encountered.
Output a corrected, numbered list of steps that avoids the previous error.
Format strictly as:
1. <step instruction>
2. <step instruction>

Be minimal and direct. Output ONLY the corrected numbered steps."""


class PlannerAgent(BaseAgent):
    """Generates structured step-plans and diagnoses execution errors."""

    def __init__(self, model_name: str = "smollm2:360m", api_url: Optional[str] = None):
        super().__init__(
            model_name=model_name,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            temperature=0.0,
            api_url=api_url,
        )

    def create_plan(self, user_query: str) -> str:
        """Generates a numbered step-by-step plan for the user request."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]
        return self._call(messages)

    def diagnose_and_replan(self, user_query: str, failed_plan: str, error: str) -> str:
        """Diagnoses execution failure and outputs a corrected step-plan."""
        content = (
            f"User Request: {user_query}\n\n"
            f"Failed Plan:\n{failed_plan}\n\n"
            f"Execution Error:\n{error}\n\n"
            "Provide the corrected numbered steps:"
        )
        messages = [
            {"role": "system", "content": REPLAN_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        return self._call(messages)

    @staticmethod
    def parse_steps(plan_text: str) -> List[str]:
        """Extracts individual step strings from a numbered or multi-line plan."""
        lines = [line.strip() for line in plan_text.splitlines() if line.strip()]
        steps: List[str] = []
        for line in lines:
            # Match "1. ...", "1) ...", "- ..." or plain line
            match = re.match(r"^(?:\d+[\.\)]|\-|\*)\s*(.+)$", line)
            if match:
                steps.append(match.group(1).strip())
            elif line.startswith("Step"):
                step_body = re.sub(r"^Step\s*\d+[:\-\.]?\s*", "", line, flags=re.IGNORECASE)
                steps.append(step_body.strip())
            elif len(lines) == 1 and not line.startswith(("<", "{", "[")):
                steps.append(line)

        if not steps and plan_text.strip():
            steps.append(plan_text.strip())

        return steps
