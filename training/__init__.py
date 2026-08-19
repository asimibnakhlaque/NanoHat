"""
training module for SmolLM2 Fedora Agent (NanoHat-360M).
Provides dataset collation, Unsloth 16-bit LoRA training, and merged/GGUF export.
"""

from .dataset_collator import load_dataset_as_text
from .merge_adapter import export_merged_and_gguf, merge_from_adapter
from .train_unsloth import run_unsloth_training

__all__ = [
    "load_dataset_as_text",
    "export_merged_and_gguf",
    "merge_from_adapter",
    "run_unsloth_training",
]
