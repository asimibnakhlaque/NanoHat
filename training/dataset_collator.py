"""
training/dataset_collator.py
Dataset loading and ChatML formatting logic for fine-tuning SmolLM2-360M-Instruct.
Formats multi-turn conversations using SmolLM2 ChatML template for response-only loss masking.
"""

import os
import json
from datasets import Dataset


def load_dataset_as_text(json_path: str, tokenizer) -> Dataset:
    """Loads JSON multi-turn conversations and formats them using SmolLM2 ChatML template."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset path '{json_path}' does not exist.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Standardize data whether it's a list or a dictionary wrapper
    if isinstance(data, dict) and "conversations" in data:
        conversations = data["conversations"]
    else:
        conversations = data

    formatted_texts = []
    for conv in conversations:
        msgs = conv.get("messages", [])
        if msgs:
            # Render native ChatML text: <|im_start|>system...<|im_end|><|im_start|>user...
            rendered = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False
            )
            formatted_texts.append(rendered)

    return Dataset.from_dict({"text": formatted_texts})
