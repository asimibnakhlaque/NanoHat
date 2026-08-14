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
        if "expression" not in args or not isinstance(args["expression"], str):
            return False, "calculator missing required string 'expression'"
            
    elif name == "web_search":
        if "query" not in args or not isinstance(args["query"], str):
            return False, "web_search missing required string 'query'"
            
    elif name == "user_memory":
        action = args.get("action")
        if action not in ALLOWED_MEMORY_ACTIONS:
            return False, f"Invalid user_memory action '{action}'"
        if "key" not in args or not isinstance(args["key"], str):
            return False, "user_memory missing required string 'key'"
            
    elif name == "reminder":
        if "task" not in args or "time_or_delay" not in args:
            return False, "reminder missing required 'task' or 'time_or_delay'"
            
    elif name == "system_health":
        target = args.get("target")
        if target not in ALLOWED_HEALTH_TARGETS:
            return False, f"Invalid system_health target '{target}'. Allowed: {ALLOWED_HEALTH_TARGETS}"
            
    elif name == "system_action":
        action = args.get("action")
        if action not in ALLOWED_SYSTEM_ACTIONS:
            return False, f"Invalid system_action keyword '{action}'. Allowed: {ALLOWED_SYSTEM_ACTIONS}"
            
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
