"""
training/merge_adapter.py
Merges the fine-tuned LoRA adapter into the base SmolLM2-360M-Instruct model
and exports full 16-bit weights ready for Hugging Face or GGUF conversion.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

def merge(adapter_path="training/adapter_output", output_path="training/merged_model"):
    print(f"Loading Base Model: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    )

    print(f"Loading Peft Adapter from: {adapter_path}...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter weights into base model...")
    merged_model = peft_model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    print(f"Saving merged model to: {output_path}...")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("✓ Merge complete! Model is ready for standalone deployment or GGUF conversion.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="training/adapter_output")
    parser.add_argument("--output", default="training/merged_model")
    args = parser.parse_args()
    merge(adapter_path=args.adapter, output_path=args.output)
