"""
generator/prompts.py
Curriculum prompt templates for Phase 1 (Single-Tool), Phase 2 (No-Tool),
Phase 3 (Multi-Tool Chaining), and Phase 4 (Edge Cases).
"""

import json
from generator.schemas import CANONICAL_SYSTEM_PROMPT, TOOL_DEFINITIONS

RUN2_NOVELTY_INSTRUCTION = """
### CRITICAL NOVELTY RULE (THIS IS RUN 2):
You are generating NEW data for a second training run. The following scenario 
types have ALREADY been generated and you MUST NOT repeat them:
- "fans are roaring / laptop is hot" → system_health
- "check my disk usage / running low on space" → system_health(disk)
- "remind me to water plants" → reminder
- "what's 15% of 2300" → calculator
- "when is fedora 40/44 coming out" → web_search
- "launch firefox / open gnome-calculator" → system_action(launch_app)
- "sudo rm -rf" refusal
- DNF vs Flatpak vs RPM explanation
- GNOME workspace shortcuts

### INSTEAD, generate scenarios from these UNTAPPED themes:
- Audio/video issues: "no sound coming out", "mic not working in zoom"
- Bluetooth pairing problems
- Printer/scanner troubleshooting
- Container issues: "podman container keeps crashing"
- Git workflow problems: "my repo is in detached head state"
- Font rendering issues in specific apps
- Monitor/display: "second monitor not detected", "screen flickering"
- Keyboard layout issues: "my keyboard is typing in the wrong language"
- Network: "vpn keeps dropping", "can't access local NAS"
- Battery: "battery drains fast even when idle"
- Snap/Flatpak permission issues
- Cron job / timer failures
- Clipboard manager not working
- Night light / color profile issues

Generate ONLY fresh, unique scenarios. Any conversation resembling the 
"already generated" list above will be REJECTED by the validator.
"""

def _get_schema_str(tool_name=None):
    if tool_name:
        tools = [t for t in TOOL_DEFINITIONS if t["name"] == tool_name]
    else:
        tools = TOOL_DEFINITIONS
    return json.dumps(tools, indent=2)

def build_single_tool_prompt(tool_name: str, real_outputs: dict) -> str:
    schema_str = _get_schema_str(tool_name)
    fixtures = real_outputs.get(tool_name, {})
    fixtures_str = json.dumps(fixtures, indent=2)

    return f"""You are an expert synthetic data generator producing high-precision multi-turn training data for a 360M parameter Fedora Linux AI agent.

{RUN2_NOVELTY_INSTRUCTION}

### OBJECTIVE: PHASE 1 - SINGLE-TOOL MASTERY ({tool_name.upper()})
Generate a JSON object containing exactly 5 multi-turn conversations. Every conversation MUST utilize the `{tool_name}` tool.

Enforce this exact scenario distribution across the 5 conversations:
1. Conversation 1 (Direct Request): User explicitly asks for an action requiring `{tool_name}`.
2. Conversation 2 (Implicit Intent): User describes a symptom or goal without naming the tool (e.g. "my fans are roaring" -> check CPU/top_processes).
3. Conversation 3 (Tool Failure & Graceful Recovery): The tool returns an error/failure, and the assistant politely explains the situation and offers a next step.
4. Conversation 4 (Mid-Conversation Correction): User changes their mind or updates parameters mid-turn.
5. Conversation 5 (Follow-up / Chaining within same tool): User asks a follow-up query requiring a second call to `{tool_name}`.

NEVER output JSON `null` or the string `"None"`. If a parameter is not applicable, use an empty string `""`.

### TOOL SCHEMA:
{schema_str}

### REAL FEDORA SYSTEM OUTPUT FIXTURES (Reference for 'role': 'tool' messages):
{fixtures_str}

### STRICT CONSTRAINTS & FORMAT RULES:
- The first message in every conversation MUST have role "system" with exact text:
  "{CANONICAL_SYSTEM_PROMPT}"
- Assistant reasoning traces MUST be concise inside <thought>...</thought> tags (maximum 25 words).
- Assistant tool calls MUST immediately follow the thought inside <tool_call>{{"name": "{tool_name}", "arguments": {{...}}}}</tool_call>.
- The next message MUST be role "tool" with the output string.
- Assistant final response follows immediately after the tool response.
- No consecutive assistant turns.
- Output ONLY valid JSON in format: {{"conversations": [ {{"messages": [ {{"role": "system", ...}}, ... ]}}, ... 5 conversations ... ]}}
"""

def build_no_tool_prompt() -> str:
    return f"""You are an expert synthetic data generator for a 360M parameter Fedora Linux AI agent.

{RUN2_NOVELTY_INSTRUCTION}

### OBJECTIVE: PHASE 2 - NO-TOOL / PURE CONVERSATION
Generate a JSON object containing exactly 5 high-quality conversations where NO tools should be called.

Topics to cover:
1. Fedora Linux package management knowledge (DNF, Flatpak, RPM).
2. General Linux desktop advice (GNOME shortcut keys, Wayland vs X11).
3. Greetings, capabilities overview, and role explanation.
4. Out-of-scope refusal (e.g. user asking for dangerous root commands, agent politely explains it only runs safe unprivileged user actions).
5. Troubleshooting advice (conceptual explanations without executing tools).

### STRICT CONSTRAINTS & FORMAT RULES:
- First message MUST be role "system" with exact text:
  "{CANONICAL_SYSTEM_PROMPT}"
- Assistant responds directly and concisely without <thought> or <tool_call> tags.
- Output ONLY valid JSON: {{"conversations": [ {{"messages": [...]}}, ... 5 conversations ... ]}}
"""

def build_multi_tool_prompt(real_outputs: dict) -> str:
    tools_str = _get_schema_str()
    fixtures_str = json.dumps(real_outputs, indent=2)

    return f"""You are an expert synthetic data generator for a 360M parameter Fedora Linux AI agent.

{RUN2_NOVELTY_INSTRUCTION}

### OBJECTIVE: PHASE 3 - MULTI-TOOL CHAINING
Generate a JSON object containing exactly 5 multi-turn conversations demonstrating multi-tool workflows.

Scenario distribution:
1. Diagnostic Triage Chain: User reports machine slowdown -> Assistant calls `system_health(target="top_processes")` -> tool returns rogue process -> Assistant calls `system_action(action="kill_process", target="...")` -> confirms termination.
2. Contextual Information Flow: `system_action(action="get_datetime")` -> `web_search` -> `user_memory(action="store")`.
3. Memory Recall & Calculation: `user_memory(action="get")` retrieves past rate -> `calculator` computes invoice.
4. Resource Monitoring & Reminder: `system_health(target="battery")` -> checks low battery -> `reminder` schedules charging check.
5. Search & System Check: `web_search` queries Fedora update info -> `system_health(target="disk")` checks available root space.

NEVER output JSON `null` or the string `"None"`. If a parameter is not applicable, use an empty string `""`.

### ALL TOOL SCHEMAS:
{tools_str}

### REAL FEDORA CLI FIXTURES:
{fixtures_str}

### STRICT CONSTRAINTS & FORMAT RULES:
- First message MUST be role "system" with exact text:
  "{CANONICAL_SYSTEM_PROMPT}"
- Thoughts MUST be under 25 words: <thought>Concise intent → choose tool → choose argument.</thought>
- Tool calls formatted as <tool_call>{{"name": "...", "arguments": {{...}}}}</tool_call>
- Output ONLY valid JSON: {{"conversations": [ {{"messages": [...]}}, ... 5 conversations ... ]}}
"""

def build_edge_case_prompt(real_outputs: dict) -> str:
    tools_str = _get_schema_str()
    return f"""You are an expert synthetic data generator for a 360M parameter Fedora Linux AI agent.

{RUN2_NOVELTY_INSTRUCTION}

### OBJECTIVE: PHASE 4 - EDGE CASES & AMBIGUITY
Generate a JSON object containing exactly 5 edge-case conversations:
1. Ambiguous Command: User gives a vague request ("make it faster"), assistant asks a clarifying question or inspects `top_processes`.
2. Multiple Intent Disambiguation: User asks two things at once, assistant sequentially addresses each with appropriate tool calls.
3. Argument Correction: User mentions the wrong process name, tool returns failure, user provides correct PID/name, assistant succeeds.
4. Out-of-Bounds Parameter: User asks to kill a system-critical process or root service, assistant refuses safely without crashing.
5. Misspelled/Chaotic User Input: User query contains typos ("chekc my batry"), assistant maps intent accurately to `system_health(target="battery")`.

NEVER output JSON `null` or the string `"None"`. If a parameter is not applicable, use an empty string `""`.

### ALL TOOL SCHEMAS:
{tools_str}

### STRICT CONSTRAINTS & FORMAT RULES:
- First message MUST be role "system" with exact text:
  "{CANONICAL_SYSTEM_PROMPT}"
- Thoughts under 25 words.
- Output ONLY valid JSON: {{"conversations": [ {{"messages": [...]}}, ... 5 conversations ... ]}}
"""
