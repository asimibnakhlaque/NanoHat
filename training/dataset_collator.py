"""
training/dataset_collator.py
Custom data collator and preprocessing logic for fine-tuning SmolLM2-360M-Instruct.
Tokenizes the full conversation in a single pass with character offset mapping
to compute loss exclusively on assistant turns (labels = -100 for non-assistant tokens).
"""

from typing import Dict, List, Any

def mask_non_assistant_labels(example: Dict[str, Any], tokenizer, max_length: int = 2048) -> Dict[str, List[int]]:
    """
    Applies ChatML chat template to the entire conversation at once, then maps
    character spans to token boundaries to assign -100 to all non-assistant tokens.
    """
    messages = example.get("messages", [])
    if not messages:
        return {"input_ids": [], "attention_mask": [], "labels": []}
        
    # 1. Render full conversation text using tokenizer's native chat template
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    # 2. Tokenize full conversation with character offset mapping
    full_encoding = tokenizer(
        full_text,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True
    )
    
    input_ids = full_encoding["input_ids"]
    attention_mask = full_encoding.get("attention_mask", [1] * len(input_ids))
    offsets = full_encoding["offset_mapping"]
    
    # 3. Build character-level boolean mask: True = assistant span, False = other
    char_is_assistant = [False] * len(full_text)
    
    for i, msg in enumerate(messages):
        # Text up to this turn
        prefix_text = tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=False) if i > 0 else ""
        # Text up to and including this turn
        through_this_text = tokenizer.apply_chat_template(messages[:i+1], tokenize=False, add_generation_prompt=False)
        
        start_char = len(prefix_text)
        end_char = len(through_this_text)
        
        if msg.get("role") == "assistant":
            for ci in range(start_char, min(end_char, len(char_is_assistant))):
                char_is_assistant[ci] = True
                
    # 4. Map character mask to token-level labels
    labels = []
    for tok_idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            # Special / boundary token without character length
            labels.append(-100)
        else:
            # Token belongs to assistant turn if any character overlaps with assistant span
            is_asst = any(char_is_assistant[c] for c in range(tok_start, min(tok_end, len(char_is_assistant))))
            labels.append(input_ids[tok_idx] if is_asst else -100)
            
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
