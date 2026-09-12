"""
generator/schemas.py
Canonical tool schemas, allowed action/target enums, and system prompt definitions.
Single source of truth — imported by the validator, generator, and runtimes.

V2 ("Beast Mode") changes:
- Canonical identity: NanoHat
- `reminder` replaced by `scheduler` (persistent tasks, absolute due times,
  backend-minted ids, status lifecycle, temporal ranges incl. yesterday/week)
- `user_memory` gains the `list` action (enumerable memory)
- Uniform tool-output contract: every tool returns "OK: ..." or "ERROR[reason]: ..."
"""

CANONICAL_SYSTEM_PROMPT = (
    "You are NanoHat, a helpful AI agent running on Fedora Linux. "
    "Available tools: [calculator, web_search, user_memory, scheduler, system_health, system_action]. "
    "Use tools when necessary by reasoning inside <thought> tags, then outputting a <tool_call> JSON block."
)

ALLOWED_TOOLS = {
    "calculator",
    "web_search",
    "user_memory",
    "scheduler",
    "system_health",
    "system_action",
}

ALLOWED_HEALTH_TARGETS = {
    "cpu",
    "ram",
    "disk",
    "network",
    "battery",
    "top_processes",
    "all",
}

ALLOWED_SYSTEM_ACTIONS = {
    "kill_process",
    "restart_service",
    "toggle_wifi",
    "toggle_bluetooth",
    "empty_trash",
    "lock_screen",
    "take_screenshot",
    "launch_app",
    "get_datetime",
}

ALLOWED_MEMORY_ACTIONS = {"store", "get", "list", "delete"}

ALLOWED_SCHEDULER_ACTIONS = {"set", "list", "update", "delete"}
ALLOWED_SCHEDULER_RANGES = {"today", "tomorrow", "yesterday", "week", "all"}
ALLOWED_SCHEDULER_STATUSES = {"pending", "done", "cancelled", "any"}

# Legacy tool name accepted only at the runtime normalization layer (mapped to
# scheduler); it is NOT valid in freshly generated training conversations.
LEGACY_TOOL_ALIASES = {
    "calc": "calculator", "math": "calculator", "evaluate": "calculator",
    "search": "web_search", "web": "web_search", "google": "web_search", "ddg": "web_search",
    "memory": "user_memory", "store_memory": "user_memory", "get_memory": "user_memory",
    "remind": "scheduler", "schedule_reminder": "scheduler", "notify": "scheduler",
    "reminder": "scheduler",
    "health": "system_health", "sys_health": "system_health", "diagnostics": "system_health",
    "check_system": "system_health",
    "action": "system_action", "sys_action": "system_action", "os_action": "system_action",
}

TOOL_DEFINITIONS = [
    {
        "name": "calculator",
        "description": "Evaluates a mathematical expression safely.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical expression string (e.g. '120 * 0.15')"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "web_search",
        "description": "Searches the web for real-time information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query keywords"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "user_memory",
        "description": "Manages persistent user key-value memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["store", "get", "list", "delete"],
                            "description": "Memory operation"},
                "key": {"type": "string", "description": "Memory key name (not required for 'list')"},
                "value": {"type": "string", "description": "Value to store (leave empty for get/list/delete)"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "scheduler",
        "description": "Creates, lists, updates, or deletes persistent scheduled tasks with due dates.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["set", "list", "update", "delete"],
                            "description": "Task operation"},
                "task": {"type": "string", "description": "Task description (required for 'set')"},
                "due": {"type": "string", "description": "Absolute 'YYYY-MM-DD HH:MM' or relative delay like '10m', '1h30m', 'in 20 minutes' (required for 'set')"},
                "id": {"type": "string", "description": "Backend-provided task id (required for 'update'/'delete')"},
                "status": {"type": "string", "enum": ["pending", "done", "cancelled", "any"],
                           "description": "'any'/filter for 'list'; new lifecycle state for 'update'"},
                "range": {"type": "string", "enum": ["today", "tomorrow", "yesterday", "week", "all"],
                          "description": "Time window for 'list'"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "system_health",
        "description": "Inspects Fedora Linux subsystem metrics and resource consumption.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["cpu", "ram", "disk", "network", "battery", "top_processes", "all"],
                    "description": "Subsystem to inspect"
                }
            },
            "required": ["target"]
        }
    },
    {
        "name": "system_action",
        "description": "Safely executes predefined OS desktop actions or queries system state.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string",
                           "enum": ["kill_process", "restart_service", "toggle_wifi", "toggle_bluetooth",
                                    "empty_trash", "lock_screen", "take_screenshot", "launch_app", "get_datetime"],
                           "description": "Safe action keyword"},
                "target": {"type": "string", "description": "Target process/service/app name or an https:// URL to open (e.g. 'firefox', 'NetworkManager', 'https://youtube.com')"}
            },
            "required": ["action"]
        }
    },
]
