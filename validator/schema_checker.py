"""
validator/schema_checker.py
Strict syntactic and semantic validator for synthetic conversation turns.
"""

import json
import re
from typing import Tuple, Dict, Any
from generator.schemas import (
    CANONICAL_SYSTEM_PROMPT,
    ALLOWED_TOOLS,
    ALLOWED_HEALTH_TARGETS,
    ALLOWED_SYSTEM_ACTIONS,
    ALLOWED_MEMORY_ACTIONS,
    ALLOWED_SCHEDULER_ACTIONS,
    ALLOWED_SCHEDULER_RANGES,
    ALLOWED_SCHEDULER_STATUSES,
)

MAX_THOUGHT_WORDS = 25

def validate_tool_call(call_dict: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(call_dict, dict):
        return False, "Tool call is not a JSON object"

    name = call_dict.get("name")
    if name not in ALLOWED_TOOLS:
        return False, f"Hallucinated tool name: '{name}'"

    args = call_dict.get("arguments", {})
    if not isinstance(args, dict):
        return False, f"Tool '{name}' arguments must be a dictionary"

    if name == "calculator":
        if "expression" not in args or not isinstance(args["expression"], str) or not args["expression"].strip():
            return False, "calculator missing required non-empty string 'expression'"

    elif name == "web_search":
        if "query" not in args or not isinstance(args["query"], str) or not args["query"].strip():
            return False, "web_search missing required non-empty string 'query'"

    elif name == "user_memory":
        action = args.get("action")
        if action not in ALLOWED_MEMORY_ACTIONS:
            return False, f"Invalid user_memory action '{action}'"
        if action == "list":
            pass  # key intentionally omitted for enumeration
        elif "key" not in args or not isinstance(args["key"], str) or not args["key"].strip():
            return False, f"user_memory {action} requires a non-empty 'key'"
        if action == "store" and (args.get("value") is None or not str(args.get("value", "")).strip()):
            return False, "user_memory store requires a non-empty 'value'"
        if action in ("get", "delete", "list") and str(args.get("value", "") or "") != "":
            # value must be empty/omitted unless storing
            if action != "store" and str(args.get("value", "")) != "":
                return False, f"user_memory {action} must leave 'value' empty"

    elif name == "scheduler":
        sched_action = args.get("action")
        if sched_action not in ALLOWED_SCHEDULER_ACTIONS:
            return False, f"Invalid scheduler action '{sched_action}'"
        if sched_action == "set":
            if not str(args.get("task", "")).strip():
                return False, "scheduler set requires non-empty 'task'"
            if not str(args.get("due", "")).strip():
                return False, "scheduler set requires non-empty 'due'"
        elif sched_action in ("update", "delete"):
            if not str(args.get("id", "")).strip():
                return False, f"scheduler {sched_action} requires non-empty 'id'"
        elif sched_action == "list":
            rng = args.get("range", "")
            if rng and rng not in ALLOWED_SCHEDULER_RANGES:
                return False, f"Invalid scheduler range '{rng}'"
            stat = args.get("status", "")
            if stat and stat not in ALLOWED_SCHEDULER_STATUSES:
                return False, f"Invalid scheduler status filter '{stat}'"
        # update: optional task/due/status change fields
        upd_status = args.get("status", "")
        if sched_action == "update" and upd_status and \
                upd_status not in ("pending", "done", "cancelled"):
            return False, f"Invalid scheduler target status '{upd_status}'"

    elif name == "system_health":
        target = args.get("target")
        if target not in ALLOWED_HEALTH_TARGETS:
            return False, f"Invalid system_health target '{target}'. Allowed: {ALLOWED_HEALTH_TARGETS}"

    elif name == "system_action":
        action = args.get("action")
        if action not in ALLOWED_SYSTEM_ACTIONS:
            return False, f"Invalid system_action keyword '{action}'. Allowed: {ALLOWED_SYSTEM_ACTIONS}"
        if action in {"kill_process", "restart_service", "launch_app"} and \
                not str(args.get("target", "")).strip():
            return False, f"system_action {action} requires a non-empty 'target'"

    return True, "OK"

def validate_conversation(conv_dict: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(conv_dict, dict):
        return False, "Conversation is not a dictionary"
        
    messages = conv_dict.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False, f"Invalid messages list (len={len(messages) if isinstance(messages, list) else 0})"
        
    # 1. First message must be system role with canonical prompt
    first_msg = messages[0]
    if first_msg.get("role") != "system":
        return False, f"First message role is '{first_msg.get('role')}', expected 'system'"
    if first_msg.get("content", "").strip() != CANONICAL_SYSTEM_PROMPT.strip():
        return False, "System prompt does not match CANONICAL_SYSTEM_PROMPT"
        
    # 2. Check for illegal consecutive assistant messages
    for i in range(len(messages) - 1):
        if messages[i].get("role") == "assistant" and messages[i + 1].get("role") == "assistant":
            return False, f"Illegal consecutive assistant turns at indices {i} and {i+1}"
            
    # 3. Validate assistant turns and tool calls
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", "")
        
        if role not in {"system", "user", "assistant", "tool"}:
            return False, f"Unknown message role '{role}' at index {idx}"
            
        if role == "assistant":
            # Check thought length if present
            thought_match = re.search(r'<thought>(.*?)</thought>', content, re.DOTALL)
            if thought_match:
                thought_text = thought_match.group(1).strip()
                word_count = len(thought_text.split())
                if word_count > MAX_THOUGHT_WORDS:
                    return False, f"Thought block too long ({word_count} words > {MAX_THOUGHT_WORDS}) at index {idx}"
                    
            # Check tool_call if present
            if "<tool_call>" in content:
                call_match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
                if not call_match:
                    return False, f"Malformed or unclosed <tool_call> tag at index {idx}"
                    
                call_raw = call_match.group(1).strip()
                try:
                    call_dict = json.loads(call_raw)
                except Exception as e:
                    return False, f"Invalid JSON inside <tool_call> at index {idx}: {e}"
                    
                is_valid, reason = validate_tool_call(call_dict)
                if not is_valid:
                    return False, f"Tool call validation failed at index {idx}: {reason}"
                    
                # Next message should ideally be a tool response if there is a next message
                if idx + 1 < len(messages) and messages[idx + 1].get("role") not in {"tool", "user"}:
                    return False, f"Expected 'tool' or 'user' role following assistant tool call at index {idx+1}"
                    
    return True, "OK"
