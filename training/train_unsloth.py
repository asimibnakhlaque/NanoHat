#!/usr/bin/env python3
"""
training/train_unsloth.py
Ultra-fast 16-bit LoRA fine-tuning for SmolLM2-360M-Instruct using Unsloth.
Includes native ChatML response loss masking and direct GGUF export.
"""

import os
import argparse
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

try:
    from training.dataset_collator import load_dataset_as_text
    from training.merge_adapter import export_merged_and_gguf
except ImportError:
    from dataset_collator import load_dataset_as_text
    from merge_adapter import export_merged_and_gguf

BASE_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
MAX_SEQ_LENGTH = 1536


def run_unsloth_training(
    train_path: str = "dataset/validated/train.json",
    eval_path: str = "dataset/validated/eval.json",
    output_dir: str = "training/adapter_unsloth",
    gguf_output_dir: str = "NanoHat-360m",
    epochs: int = 5,
    lr: float = 2e-4,
    batch_size: int = 4,
    grad_accum: int = 4,
):
    print("=" * 60)
    print(f"🚀 Initializing Unsloth FastLanguageModel ({BASE_MODEL_NAME})")
    print("=" * 60)

    # 1. Load Model & Tokenizer (Full 16-bit LoRA without 4-bit degradation)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,               # Auto-detects bfloat16 / float16
        load_in_4bit=False,        # 360M model fits easily in 16-bit (< 1.5 GB VRAM)
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Attach Optimized LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,           # Unsloth supports 0 dropout for optimized kernel speed
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth intelligent checkpointing
        random_state=3407,
    )

    # 3. Load & Format Datasets
    print(f"\n📂 Loading datasets: train='{train_path}', eval='{eval_path}'...")
    train_dataset = load_dataset_as_text(train_path, tokenizer)
    eval_dataset = load_dataset_as_text(eval_path, tokenizer) if os.path.exists(eval_path) else None

    # 4. Training Configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=0.05,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=3407,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # 5. Mask Non-Assistant Tokens (Computes loss ONLY on agent responses & tool calls)
    print("\n🎯 Applying Unsloth Response-Only Loss Masking...")
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # 6. Execute Fine-Tuning
    print("\n⚡ Starting Training Run...")
    trainer.train()
    print("✓ Fine-tuning complete!")

    # 7. Save LoRA Adapter
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Saved LoRA adapter to: {output_dir}")

    # 8. Direct 16-Bit Merge & GGUF Export
    export_merged_and_gguf(
        model=model,
        tokenizer=tokenizer,
        merged_dir="training/merged_model",
        gguf_output_dir=gguf_output_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsloth Fine-Tuning for NanoHat-360M")
    parser.add_argument("--train-data", default="dataset/validated/train.json")
    parser.add_argument("--eval-data", default="dataset/validated/eval.json")
    parser.add_argument("--output-dir", default="training/adapter_unsloth")
    parser.add_argument("--gguf-dir", default="NanoHat-360m")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    run_unsloth_training(
        train_path=args.train_data,
        eval_path=args.eval_data,
        output_dir=args.output_dir,
        gguf_output_dir=args.gguf_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum
    )
