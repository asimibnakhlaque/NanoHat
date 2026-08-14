#!/usr/bin/env python3
"""
training/train_qlora.py
Fine-tunes HuggingFaceTB/SmolLM2-360M-Instruct using QLoRA with custom loss masking.
"""

import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from training.dataset_collator import mask_non_assistant_labels

BASE_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

def load_json_dataset(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} does not exist.")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list([{"messages": conv["messages"]} for conv in data])

def train(
    train_path="dataset/validated/train.json",
    eval_path="dataset/validated/eval.json",
    output_dir="training/adapter_output",
    epochs=3,
    lr=2e-4,
    batch_size=4,
    grad_accum=4,
    use_4bit=True
):
    print(f"Loading Base Model & Tokenizer: {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit Quantization Config for QLoRA
    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True
        )

    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
        trust_remote_code=True
    )

    if use_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    # LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and map datasets with loss masking
    print(f"Loading dataset splits: train='{train_path}', eval='{eval_path}'...")
    raw_train = load_json_dataset(train_path)
    raw_eval = load_json_dataset(eval_path) if os.path.exists(eval_path) else None

    train_dataset = raw_train.map(
        lambda ex: mask_non_assistant_labels(ex, tokenizer),
        remove_columns=raw_train.column_names
    )
    eval_dataset = raw_eval.map(
        lambda ex: mask_non_assistant_labels(ex, tokenizer),
        remove_columns=raw_eval.column_names
    ) if raw_eval else None

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting QLoRA Fine-Tuning...")
    trainer.train()

    print(f"Saving fine-tuned adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✓ Training finished successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for SmolLM2-360M-Instruct")
    parser.add_argument("--train-data", default="dataset/validated/train.json")
    parser.add_argument("--eval-data", default="dataset/validated/eval.json")
    parser.add_argument("--output-dir", default="training/adapter_output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization (run in fp16/fp32)")
    args = parser.parse_args()

    train(
        train_path=args.train_data,
        eval_path=args.eval_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        use_4bit=not args.no_4bit
    )
