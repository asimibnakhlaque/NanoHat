"""
training/merge_adapter.py
Merges fine-tuned LoRA adapter into base SmolLM2-360M-Instruct and exports GGUF formats using Unsloth.
"""

import os
import argparse
from unsloth import FastLanguageModel

BASE_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
MAX_SEQ_LENGTH = 1536


def export_merged_and_gguf(
    model,
    tokenizer,
    merged_dir: str = "training/merged_model",
    gguf_output_dir: str = "NanoHat-360m"
):
    """Direct 16-Bit Merge & GGUF Export."""
    print("\n📦 Exporting Merged Standalone Model & GGUFs...")

    # Export 16-bit Hugging Face Directory
    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"✓ Exported 16-bit merged weights to: {merged_dir}")

    # Direct GGUF Conversions
    os.makedirs(gguf_output_dir, exist_ok=True)

    # Export F16 GGUF
    model.save_pretrained_gguf(gguf_output_dir, tokenizer, quantization_method="f16")
    print(f"✓ Exported F16 GGUF to: {gguf_output_dir}/nanohat-360m-f16.gguf")

    # Export Q4_K_M GGUF (Ultra-compact for fast CPU / Fedora local agent inference)
    model.save_pretrained_gguf(gguf_output_dir, tokenizer, quantization_method="q4_k_m")
    print(f"✓ Exported Q4_K_M GGUF to: {gguf_output_dir}/nanohat-360m-q4_k_m.gguf")


def merge_from_adapter(
    adapter_path: str = "training/adapter_unsloth",
    merged_dir: str = "training/merged_model",
    gguf_output_dir: str = "NanoHat-360m",
    max_seq_length: int = MAX_SEQ_LENGTH
):
    """Loads adapter with FastLanguageModel and exports merged weights + GGUF binaries."""
    print(f"🚀 Loading model & adapter from '{adapter_path}'...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    export_merged_and_gguf(model, tokenizer, merged_dir=merged_dir, gguf_output_dir=gguf_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter & export GGUF binaries")
    parser.add_argument("--adapter", default="training/adapter_unsloth", help="Path to saved LoRA adapter")
    parser.add_argument("--merged-dir", default="training/merged_model", help="Path to output 16-bit merged directory")
    parser.add_argument("--gguf-dir", default="NanoHat-360m", help="Path to output GGUF directory")
    args = parser.parse_args()

    merge_from_adapter(
        adapter_path=args.adapter,
        merged_dir=args.merged_dir,
        gguf_output_dir=args.gguf_dir
    )
