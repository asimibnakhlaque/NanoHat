"""
validator/dedup.py
Semantic deduplication engine to prevent repetitive conversations
across multiple generation runs.
"""
import json
import re
import hashlib
from typing import List, Dict, Any, Set, Tuple


def extract_conversation_fingerprint(conv: Dict[str, Any]) -> str:
    """
    Creates a structural fingerprint of a conversation.
    Two conversations with the same tool sequence and similar
    user intents will produce similar fingerprints.
    """
    messages = conv.get("messages", [])
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant" and "<tool_call>" in content:
            # Extract just the tool name and action/target
            match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
            if match:
                try:
                    call = json.loads(match.group(1))
                    name = call.get("name", "")
                    args = call.get("arguments", {})
                    # Fingerprint = tool name + sorted arg keys + key values
                    arg_sig = sorted(
                        [(k, str(v)[:30]) for k, v in args.items()]
                    )
                    parts.append(f"TC:{name}:{arg_sig}")
                except json.JSONDecodeError:
                    parts.append(f"TC:parse_error")
        elif role == "user":
            # Normalize user text: lowercase, strip punctuation, take first 8 words
            words = re.sub(r'[^\w\s]', '', content.lower()).split()[:8]
            parts.append(f"U:{' '.join(words)}")
        elif role == "assistant" and "<tool_call>" not in content:
            words = re.sub(r'[^\w\s]', '', content.lower()).split()[:6]
            parts.append(f"A:{' '.join(words)}")
    return "|".join(parts)


def build_existing_fingerprints(dataset_path: str) -> Set[str]:
    """Load existing golden dataset and build fingerprint set."""
    fingerprints = set()
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for conv in existing:
            fp = extract_conversation_fingerprint(conv)
            fingerprints.add(fp)
        print(f"  [Dedup] Loaded {len(fingerprints)} existing fingerprints")
    except FileNotFoundError:
        print(f"  [Dedup] No existing dataset found at {dataset_path}. Starting fresh.")
    return fingerprints


def is_duplicate(
    conv: Dict[str, Any],
    existing_fps: Set[str],
    strict: bool = True
) -> bool:
    """
    Check if a conversation is a duplicate.
    strict=True  → exact fingerprint match
    strict=False → prefix match (catches same-sequence conversations)
    """
    fp = extract_conversation_fingerprint(conv)
    if strict:
        return fp in existing_fps
    # Loose check: compare first 3 structural elements
    prefix = "|".join(fp.split("|")[:3])
    for existing_fp in existing_fps:
        if existing_fp.startswith(prefix):
            return True
    return False


def deduplicate_batch(
    conversations: List[Dict[str, Any]],
    existing_fps: Set[str]
) -> Tuple[List[Dict[str, Any]], int]:
    """Filter a batch against existing fingerprints. Returns (kept, rejected_count)."""
    kept = []
    rejected = 0
    for conv in conversations:
        if is_duplicate(conv, existing_fps, strict=True):
            rejected += 1
        else:
            kept.append(conv)
            # Add to set so we also dedup within this batch
            fp = extract_conversation_fingerprint(conv)
            existing_fps.add(fp)
    return kept, rejected