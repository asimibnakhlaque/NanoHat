"""
validator/canonicalizer.py
Robust canonicalizer with JS bareword repair, parameter alias mapping,
action inference, argument unpacking, and consecutive turn merging.
"""
import ast
import json
import re
from typing import Dict, Any, Optional, List

# Expanded parameter name aliases
PARAM_ALIASES = {
    "calculator": {
        "expr": "expression", "calc": "expression", "math": "expression",
        "input": "expression", "equation": "expression", "formula": "expression",
        "calculation": "expression", "evaluate": "expression", "eval": "expression",
        "problem": "expression", "query": "expression", "text": "expression"
    },
    "web_search": {
        "search": "query", "q": "query", "search_query": "query", "terms": "query",
        "term": "query", "text": "query", "prompt": "query", "input": "query",
        "keyword": "query", "keywords": "query", "topic": "query", "question": "query",
        "search_term": "query", "search_terms": "query"
    },
    "reminder": {
        "message": "task", "text": "task", "msg": "task", "reminder": "task",
        "event": "task", "title": "task", "content": "task", "note": "task",
        "todo": "task", "memo": "task", "action": "task",
        "time": "time_or_delay", "delay": "time_or_delay", "when": "time_or_delay",
        "date": "time_or_delay", "timestamp": "time_or_delay", "schedule": "time_or_delay",
        "timer": "time_or_delay", "duration": "time_or_delay", "at": "time_or_delay"
    },
    "user_memory": {
        "key_name": "key", "val": "value", "data": "value", "content": "value",
        "item": "key", "target": "key"
    },
    "system_health": {
        "subsystem": "target", "metric": "target", "type": "target", "resource": "target"
    },
    "system_action": {
        "app": "target", "process": "target", "service": "target", "name": "target",
        "cmd": "action", "operation": "action", "command": "action", "type": "action"
    },
}

# Subsystem target synonyms
TARGET_SYNONYMS = {
    "disc": "disk", "disck": "disk", "dsk": "disk", "storage": "disk", "harddrive": "disk", "hdd": "disk", "ssd": "disk",
    "memory": "ram", "mem": "ram", "processor": "cpu", "load": "cpu",
    "processes": "top_processes", "procs": "top_processes", "process": "top_processes", "top": "top_processes",
    "power": "battery", "bat": "battery", "wifi": "network", "net": "network",
    "overall": "all", "system": "all",
}

# Action verb mapping for system_action
ACTION_SYNONYMS = {
    "open": "launch_app", "launch": "launch_app", "start": "launch_app", "run": "launch_app",
    "exec": "launch_app", "open_app": "launch_app",
    "kill": "kill_process", "terminate": "kill_process", "stop": "kill_process", "close": "kill_process",
    "end": "kill_process", "pkill": "kill_process",
    "restart": "restart_service", "reboot_service": "restart_service", "reload": "restart_service",
    "wifi": "toggle_wifi", "wifi_toggle": "toggle_wifi",
    "bluetooth": "toggle_bluetooth", "bt": "toggle_bluetooth", "bluetooth_toggle": "toggle_bluetooth",
    "trash": "empty_trash", "clean_trash": "empty_trash", "clear_trash": "empty_trash",
    "lock": "lock_screen", "lock_desktop": "lock_screen",
    "screenshot": "take_screenshot", "capture_screen": "take_screenshot", "screen": "take_screenshot",
    "time": "get_datetime", "date": "get_datetime", "datetime": "get_datetime", "now": "get_datetime",
    "current_time": "get_datetime",
}

# Action verb mapping for user_memory
MEMORY_ACTION_SYNONYMS = {
    "set": "store", "save": "store", "write": "store", "add": "store", "put": "store",
    "update": "store", "remember": "store", "insert": "store",
    "read": "get", "retrieve": "get", "find": "get", "fetch": "get", "recall": "get",
    "lookup": "get", "load": "get", "search": "get",
    "remove": "delete", "clear": "delete", "forget": "delete", "erase": "delete", "drop": "delete",
}

ALLOWED_SYSTEM_ACTIONS = {
    "kill_process", "restart_service", "toggle_wifi", "toggle_bluetooth",
    "empty_trash", "lock_screen", "take_screenshot", "launch_app", "get_datetime"
}


def _repair_tool_json(raw_json: str) -> Optional[dict]:
    """Parses JSON with multi-layered fallbacks for bareword keys, quotes, and strings."""
    cleaned = raw_json.strip()

    # Strip markdown backticks, comments, and prefixes
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"//.*?\n", "\n", cleaned)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)

    # 0. Unescape double-escaped quotes (e.g. {\"name\": ...})
    if r'\"' in cleaned:
        unescaped = cleaned.replace(r'\"', '"')
        try:
            return json.loads(unescaped)
        except Exception:
            pass

    # 1. Strict JSON
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Python literal eval
    try:
        obj = ast.literal_eval(cleaned)
        if isinstance(obj, dict):
            return obj
    except (ValueError, SyntaxError):
        pass

    # 3. Regex repair: Quote bareword keys and normalize literals
    try:
        target_str = cleaned.replace(r'\"', '"') if r'\"' in cleaned else cleaned
        # Quote unquoted keys: {key: "val"} or { key: "val" } or , key: "val"
        repaired = re.sub(r'([{\s,])([a-zA-Z_][a-zA-Z0-9_\-\.]*)\s*:', r'\1"\2":', target_str)
        # Convert single quotes around strings
        repaired = re.sub(r"(?<![\\])'", '"', repaired)
        # Convert Python literals
        repaired = re.sub(r'\bTrue\b', 'true', repaired)
        repaired = re.sub(r'\bFalse\b', 'false', repaired)
        repaired = re.sub(r'\bNone\b', 'null', repaired)
        # Remove trailing commas
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
        return json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        pass

    # 4. Fallback: Structural regex extraction if JSON is heavily malformed
    target_str = cleaned.replace(r'\"', '"') if r'\"' in cleaned else cleaned
    name_match = re.search(r'["\']?name["\']?\s*:\s*["\']([a-zA-Z0-9_]+)["\']', target_str)
    if name_match:
        tool_name = name_match.group(1)
        args_match = re.search(r'["\']?(?:arguments|parameters|params|args|inputs|input)["\']?\s*:\s*(\{.*?\}|".*?"|\'.*?\')', target_str, re.DOTALL)
        if args_match:
            try:
                args_val = _repair_tool_json(args_match.group(1))
                if args_val is not None:
                    return {"name": tool_name, "arguments": args_val}
            except Exception:
                pass
        return {"name": tool_name, "arguments": {}}

    return None


def _sanitize_and_alias_arguments(call_obj: dict) -> dict:
    """Unpacks arguments, normalizes aliases, and infers missing schema fields."""
    name = call_obj.get("name", "")

    # Look for arguments under multiple key variations
    args = None
    for arg_key in ["arguments", "parameters", "params", "args", "inputs", "input"]:
        if arg_key in call_obj and call_obj[arg_key] is not None:
            args = call_obj.pop(arg_key)
            break
    if args is None:
        args = {}

    # Clean up any lingering alternative keys
    for extra_k in ["parameters", "params", "args", "inputs", "input"]:
        call_obj.pop(extra_k, None)

    # Convert plain string/list arguments to standard parameter dictionaries
    if isinstance(args, str):
        if name == "calculator":
            args = {"expression": args}
        elif name == "web_search":
            args = {"query": args}
        elif name == "system_health":
            args = {"target": args}
        elif name == "system_action":
            args = {"action": args, "target": ""}
        elif name == "reminder":
            args = {"task": args, "time_or_delay": "10m"}
        else:
            args = {"key": args}
    elif isinstance(args, list) and len(args) > 0:
        val = str(args[0])
        if name == "calculator":
            args = {"expression": val}
        elif name == "web_search":
            args = {"query": val}
        elif name == "system_health":
            args = {"target": val}
        else:
            args = {"target": val}
    elif not isinstance(args, dict):
        args = {}

    # 1. Parameter name alias mapping
    aliases = PARAM_ALIASES.get(name, {})
    normalized_args = {}
    for k, v in args.items():
        canonical_key = aliases.get(k, k)
        normalized_args[canonical_key] = v

    # 2. Tool-specific semantic normalization & recovery
    if name == "calculator":
        if "expression" not in normalized_args:
            for k in ["query", "task", "target", "val", "value"]:
                if k in normalized_args:
                    normalized_args["expression"] = str(normalized_args.pop(k))
                    break

    elif name == "web_search":
        if "query" not in normalized_args:
            for k in ["expression", "task", "target", "key", "val", "value"]:
                if k in normalized_args:
                    normalized_args["query"] = str(normalized_args.pop(k))
                    break

    elif name == "system_health":
        target = normalized_args.get("target")
        if target in [None, "None", "null", ""]:
            normalized_args["target"] = "all"
        elif isinstance(target, str):
            cleaned_target = target.lower().strip()
            normalized_args["target"] = TARGET_SYNONYMS.get(cleaned_target, cleaned_target)

    elif name == "system_action":
        action = normalized_args.get("action")
        target = normalized_args.get("target", "")

        # Clean string "None"
        if action in [None, "None", "null"]:
            action = ""
        if target in [None, "None", "null"]:
            target = ""

        action_str = str(action).lower().strip()
        target_str = str(target).strip()

        # Swap if target contains the action keyword and action is empty/invalid
        if action_str not in ALLOWED_SYSTEM_ACTIONS and target_str in ALLOWED_SYSTEM_ACTIONS:
            action_str = target_str
            target_str = ""

        # Map action synonyms
        action_str = ACTION_SYNONYMS.get(action_str, action_str)

        # Infer missing action from target context
        if action_str not in ALLOWED_SYSTEM_ACTIONS:
            if target_str.endswith(".service"):
                action_str = "restart_service"
            elif target_str:
                action_str = "launch_app"
            else:
                action_str = "get_datetime"

        normalized_args["action"] = action_str
        normalized_args["target"] = target_str

    elif name == "user_memory":
        action = normalized_args.get("action")
        if action in [None, "None", "null"]:
            action = ""
        action_str = str(action).lower().strip()
        action_str = MEMORY_ACTION_SYNONYMS.get(action_str, action_str)

        # Infer action if missing
        if action_str not in {"store", "get", "delete"}:
            if normalized_args.get("value"):
                action_str = "store"
            else:
                action_str = "get"

        normalized_args["action"] = action_str
        if "key" not in normalized_args:
            normalized_args["key"] = "general_note"

    elif name == "reminder":
        task = normalized_args.get("task", "")
        time_delay = normalized_args.get("time_or_delay", "")

        if not time_delay and task:
            # Extract time string embedded inside task description
            time_match = re.search(r'\b(?:at|in)\s+([0-9]{1,2}(?::[0-9]{2})?(?:\s*[ap]m)?|[0-9]+\s*(?:m|min|mins|minutes|h|hours|s|seconds))\b', str(task), re.IGNORECASE)
            if time_match:
                normalized_args["time_or_delay"] = time_match.group(1).strip()
            else:
                normalized_args["time_or_delay"] = "10m"
        elif not task and time_delay:
            normalized_args["task"] = "Reminder notification"

    call_obj["arguments"] = normalized_args
    return call_obj


def _merge_consecutive_assistant_turns(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Merges consecutive assistant messages into a single coherent turn."""
    if not messages:
        return []

    merged: List[Dict[str, str]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "").strip()

        if merged and merged[-1].get("role") == "assistant" and role == "assistant":
            prev_content = merged[-1]["content"]
            merged[-1]["content"] = f"{prev_content}\n{content}".strip()
        else:
            merged.append({"role": role, "content": content})

    return merged


def canonicalize_conversation(conv_dict: Any) -> Optional[Dict[str, Any]]:
    """Normalizes tags, repairs JSON syntax, and enforces uniform surface format."""
    if isinstance(conv_dict, list):
        raw_messages = conv_dict
    elif isinstance(conv_dict, dict):
        raw_messages = (
            conv_dict.get("messages")
            or conv_dict.get("conversations")
            or conv_dict.get("conversation")
            or conv_dict.get("dialog")
            or conv_dict.get("turns")
            or []
        )
    else:
        return None

    if not isinstance(raw_messages, list) or not raw_messages:
        return None

    messages = _merge_consecutive_assistant_turns(raw_messages)
    cleaned_messages = []

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", "")

        # Auto-rescue: if a message following a tool call is labeled "system", normalize to "tool"
        if idx > 0 and role == "system":
            prev_msg = cleaned_messages[-1] if cleaned_messages else {}
            if prev_msg.get("role") == "assistant" and "<tool_call>" in prev_msg.get("content", ""):
                role = "tool"

        if role == "assistant":
            content = re.sub(r'<\s*thought\s*>', '<thought>', content)
            content = re.sub(r'<\s*/\s*thought\s*>', '</thought>', content)
            content = re.sub(r'<\s*tool_call\s*>', '<tool_call>', content)
            content = re.sub(r'<\s*/\s*tool_call\s*>', '</tool_call>', content)

            content = re.sub(r'<tool_call>\s*```(?:json)?\s*', '<tool_call>', content)
            content = re.sub(r'\s*```\s*</tool_call>', '</tool_call>', content)

            # Process every tool call inside the turn in reverse index order
            matches = list(re.finditer(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL))
            for match in reversed(matches):
                raw_json = match.group(1).strip()
                call_obj = _repair_tool_json(raw_json)

                if call_obj is not None:
                    call_obj = _sanitize_and_alias_arguments(call_obj)
                    canonical_json = json.dumps(call_obj, separators=(',', ':'))
                    content = (
                        content[:match.start()]
                        + f"<tool_call>{canonical_json}</tool_call>"
                        + content[match.end():]
                    )

        cleaned_messages.append({
            "role": role,
            "content": content.strip()
        })

    return {"messages": cleaned_messages}