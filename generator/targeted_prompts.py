"""
generator/targeted_prompts.py
Targeted rebalanced prompt generation engine for SmolLM2 Fedora Agent (Run 4).
Generates high-precision synthetic conversations focused on under-represented actions,
complex 3+ step tool chaining, realistic tool error recovery, and safe refusals.
"""

import json
from generator.schemas import CANONICAL_SYSTEM_PROMPT, TOOL_DEFINITIONS

RUN4_BASE_RULES = f"""
### STRICT FORMAT & SLM CAPACITY RULES:
- First message MUST be role "system" with exact text:
  "{CANONICAL_SYSTEM_PROMPT}"
- Thoughts MUST be under 25 words inside <thought>...</thought> tags.
- Tool calls formatted strictly as: <tool_call>{{"name": "...", "arguments": {{...}}}}</tool_call> (use "arguments", NEVER "parameters").
- NEVER output JSON null, None, or "None". Use empty string "" if a parameter is not applicable.
- Return a single valid JSON object: {{"conversations": [ ... 5 unique conversation objects ... ]}}
"""

def _get_schema_str(tool_name: str = None) -> str:
    """Returns formatted JSON string of tool schemas."""
    if tool_name:
        tools = [t for t in TOOL_DEFINITIONS if t["name"] == tool_name]
    else:
        tools = TOOL_DEFINITIONS
    return json.dumps(tools, indent=2)

def build_targeted_prompt(task_category: str, fixtures: dict = None) -> str:
    """
    Builds a specialized prompt for one of the 6 targeted rebalancing categories.
    """
    fixtures = fixtures or {}
    tools_str = _get_schema_str()
    fix_str = json.dumps(fixtures, indent=2)

    if task_category == "starved_desktop_actions":
        return f"""You are generating training data for a 360M Fedora Linux agent.
{RUN4_BASE_RULES}
### TARGET: UNDER-REPRESENTED DESKTOP ACTIONS
Generate 5 conversations heavily focusing on these specific actions:
1. `system_action(action="empty_trash")`: User cleaning disk, deleting temp clutter, emptying trash bin.
2. `system_action(action="take_screenshot")`: Capturing desktop errors, full screen captures, visual bug reports.
3. `system_action(action="toggle_bluetooth")`: Bluetooth headset/mouse connect/disconnect, power toggles.
4. `system_action(action="lock_screen")`: Stepping away from desk, privacy locking, workstation security.
5. `system_action(action="launch_app", target="...")`: Launching desktop applications (e.g. 'vlc', 'gimp', 'gnome-control-center', 'inkscape', 'libreoffice-calc').

### ALL TOOL SCHEMAS:
{tools_str}
"""

    elif task_category == "starved_network_and_diagnostics":
        return f"""You are generating training data for a 360M Fedora Linux agent.
{RUN4_BASE_RULES}
### TARGET: NETWORK DIAGNOSTICS & HARDWARE HEALTH
Generate 5 conversations focusing on:
1. `system_health(target="network")`: Checking Wi-Fi/Ethernet link status, default gateway, nmcli interface states.
2. Multi-step network triage: `system_health(target="network")` -> `system_action(action="toggle_wifi")` -> `system_health(target="network")`.
3. Process & resource degradation: `system_health(target="top_processes")` -> identifying rogue memory leak -> `system_action(action="kill_process")`.
4. Battery health & power save: `system_health(target="battery")` -> checking discharge rate.
5. Storage & root inspection: `system_health(target="disk")` -> identifying full partition -> `system_action(action="empty_trash")`.

### ALL TOOL SCHEMAS:
{tools_str}
### FIXTURES:
{fix_str}
"""

    elif task_category == "rich_reminders_and_calculator":
        return f"""You are generating training data for a 360M Fedora Linux agent.
{RUN4_BASE_RULES}
### TARGET: REMINDERS & ARITHMETIC REASONING
Generate 5 conversations focusing on `reminder` and `calculator`:
1. Time conversions & Math: `calculator` computing gigabyte conversions (e.g. '15 * 1024'), battery time to seconds, or percentages.
2. Delayed Reminders: `reminder(task="Check long-running backup", time_or_delay="45m")`.
3. Fast Notifications: `reminder(task="Meeting in 2 minutes", time_or_delay="2m")` or `30s`.
4. Chained flow: `web_search` finding conference time -> `reminder` scheduling attendance -> `user_memory` saving note.
5. Memory Recall & Calculation: `user_memory(action="get")` retrieving past storage limit -> `calculator` computing remaining quota.

### ALL TOOL SCHEMAS:
{tools_str}
"""

    elif task_category == "complex_multi_tool_chains":
        return f"""You are generating training data for a 360M Fedora Linux agent.
{RUN4_BASE_RULES}
### TARGET: 3+ STEP MULTI-TOOL WORKFLOWS
Generate 5 deep multi-step conversations with 2 to 4 sequential tool calls:
1. Full System Audit: `system_action(get_datetime)` -> `system_health(cpu)` -> `system_health(ram)` -> `system_health(disk)`.
2. Developer Rescue: `web_search(git error)` -> `user_memory(store)` -> `system_action(restart_service)`.
3. Performance Triage: `system_health(top_processes)` -> `system_action(take_screenshot)` -> `system_action(kill_process)`.
4. Storage Optimization: `system_health(disk)` -> `calculator(free space %)` -> `system_action(empty_trash)`.
5. Peripheral Setup: `system_health(network)` -> `system_action(toggle_bluetooth)` -> `reminder(test audio in 5m)`.

### ALL TOOL SCHEMAS:
{tools_str}
"""

    elif task_category == "tool_failure_and_recovery":
        return f"""You are generating training data for a 360M Fedora Linux agent.
{RUN4_BASE_RULES}
### TARGET: TOOL ERROR RECOVERY & PARAMETER CORRECTIONS
Generate 5 conversations where tools return realistic Linux failures and the agent recovers gracefully:
1. App launch failure: `system_action(launch_app, target="chrome")` returns "Application not found" -> agent suggests Flatpak/Firefox alternative.
2. Service failure: `system_action(restart_service, target="httpd")` fails -> agent checks `systemctl --user` context.
3. Memory key not found: `user_memory(get, target="api_key")` returns not found -> agent asks user to provide value and stores it.
4. Process kill failure: process already exited -> agent confirms process is no longer active.
5. Mid-turn correction: User says "Check RAM... wait no, check my network instead".

### ALL TOOL SCHEMAS:
{tools_str}
"""

    else:  # negative_pure_chat
        return f"""You are generating training data for a 360M Fedora Linux agent.
{RUN4_BASE_RULES}
### TARGET: PURE CONVERSATION & SAFE REFUSALS (NO TOOLS)
Generate 5 conversations where NO tools are called:
1. Fedora DNF vs Flatpak vs RPM conceptual architecture.
2. GNOME desktop shortcut advice (Super key, Wayland gestures).
3. Safe refusal for dangerous root/sudo commands (`sudo rm -rf`, `chmod -R 777 /`).
4. General Linux troubleshooting explanations without touching system state.
5. Agent capabilities and privacy explanation (local SLM model running on device).
"""
