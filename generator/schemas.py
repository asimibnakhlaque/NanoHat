"""
generator/schemas.py
Canonical tool schemas, allowed action/target enums, and system prompt definitions.
"""

CANONICAL_SYSTEM_PROMPT = (
    "You are a helpful AI agent running on Fedora Linux. "
    "Available tools: [calculator, web_search, user_memory, reminder, system_health, system_action]. "
    "Use tools when necessary by reasoning inside <thought> tags, then outputting a <tool_call> JSON block."
)

ALLOWED_TOOLS = {
    "calculator",
    "web_search",
    "user_memory",
    "reminder",
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

ALLOWED_MEMORY_ACTIONS = {"store", "get", "delete"}

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
                "action": {"type": "string", "enum": ["store", "get", "delete"]},
                "key": {"type": "string", "description": "Memory key name"},
                "value": {"type": "string", "description": "Value to store (optional for get/delete)"}
            },
            "required": ["action", "key"]
        }
    },
    {
        "name": "reminder",
        "description": "Schedules a desktop notification.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Reminder task description"},
                "time_or_delay": {"type": "string", "description": "Delay string, e.g., '15m', '1h', '30s'"}
            },
            "required": ["task", "time_or_delay"]
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
                    "enum": ["cpu", "ram", "disk", "network", "battery", "top_processes", "all"]
                }
            },
            "required": ["target"]
        }
    },
    {
        "name": "system_action",
        "description": "Executes predefined safe Fedora system and desktop operations.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "kill_process",
                        "restart_service",
                        "toggle_wifi",
                        "toggle_bluetooth",
                        "empty_trash",
                        "lock_screen",
                        "take_screenshot",
                        "launch_app",
                        "get_datetime"
                    ]
                },
                "target": {"type": "string", "description": "Target app/service/process name"}
            },
            "required": ["action"]
        }
    }
]
