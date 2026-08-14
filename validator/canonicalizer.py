"""
validator/canonicalizer.py
Cleans formatting drift, ensures byte-identical tag styling, and re-serializes JSON.
"""

import json
import re
from typing import Dict, Any, Optional

def canonicalize_conversation(conv_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Cleans up whitespace, strips markdown backticks around tool calls,
    and re-serializes tool call JSON to ensure a uniform surface format.
    """
    if not isinstance(conv_dict, dict) or "messages" not in conv_dict:
        return None
        
    cleaned_messages = []
    
    for msg in conv_dict["messages"]:
        role = msg.get("role")
        content = msg.get("content", "")
        
        if role == "assistant":
            # 1. Normalize thought tags
            content = re.sub(r'<\s*thought\s*>', '<thought>', content)
            content = re.sub(r'<\s*/\s*thought\s*>', '</thought>', content)
            
            # 2. Normalize tool_call tags
            content = re.sub(r'<\s*tool_call\s*>', '<tool_call>', content)
            content = re.sub(r'<\s*/\s*tool_call\s*>', '</tool_call>', content)
            
            # 3. Strip any accidental markdown json fences inside tool_call
            content = re.sub(r'<tool_call>\s*```(?:json)?\s*', '<tool_call>', content)
            content = re.sub(r'\s*```\s*</tool_call>', '</tool_call>', content)
            
            # 4. Extract and re-serialize JSON inside tool_call canonically
            match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
            if match:
                raw_json = match.group(1).strip()
                try:
                    call_obj = json.loads(raw_json)
                    canonical_json = json.dumps(call_obj, separators=(',', ':'))
                    content = content[:match.start()] + f"<tool_call>{canonical_json}</tool_call>" + content[match.end():]
                except Exception:
                    # If JSON cannot be parsed, leave as is; schema_checker will reject it
                    pass
                    
        cleaned_messages.append({
            "role": role,
            "content": content.strip()
        })
        
    return {"messages": cleaned_messages}
