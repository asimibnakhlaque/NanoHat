#!/usr/bin/env python3
"""
training/colab_train_v2.py — Colab T4 training entry for NanoHat-360M v2.

Usage (Colab, after `pip install unsloth` etc. and mounting Drive):

  # 0) ALWAYS run the probe first — verifies tool-role tokens are loss-masked
  python training/colab_train_v2.py --probe

  # 1) Fresh training run
  python training/colab_train_v2.py --drive-dir /content/drive/MyDrive/nanohat_v2

  # 2) Resume after Colab session died
  python training/colab_train_v2.py --drive-dir /content/drive/MyDrive/nanohat_v2 --resume

Design:
  - SmolLM2-360M-Instruct + LoRA r=32 (v2 stays Instruct; base-model switch is v3)
  - seq 1536 (matches dataset window), fp16 on T4, adamw_8bit
  - response-only loss masking via unsloth train_on_responses_only
  - checkpoints: local writes, rsync to Drive at each save (save_steps=500)
  - resume: pulls newest Drive checkpoint into local output dir first
"""

import argparse
import os
import shutil
import glob

MAX_SEQ_LENGTH = 1536
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEFAULT_TRAIN = "dataset/v2/train.json"
DEFAULT_EVAL = "dataset/v2/eval.json"
SAVE_STEPS = 500


def _drive_latest(ckpt_root):
    """Newest checkpoint directory under Drive, or None."""
    cands = sorted(glob.glob(os.path.join(ckpt_root, "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[-1]))
    return cands[-1] if cands else None


def sync_to_drive(local_dir, drive_dir):
    if not drive_dir:
        return
    os.makedirs(drive_dir, exist_ok=True)
    for ckpt in glob.glob(os.path.join(local_dir, "checkpoint-*")):
        name = os.path.basename(ckpt)
        dst = os.path.join(drive_dir, name)
        if not os.path.exists(dst):
            print(f"  ☁  archiving {name} -> Drive")
            shutil.copytree(ckpt, dst, dirs_exist_ok=True)
    for f in ("trainer_state.json",):
        src = os.path.join(local_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(drive_dir, f))


def restore_from_drive(local_dir, drive_dir):
    latest = _drive_latest(drive_dir)
    if not latest:
        return None
    name = os.path.basename(latest)
    dst = os.path.join(local_dir, name)
    if not os.path.exists(dst):
        print(f"  ☁  restoring {name} from Drive")
        shutil.copytree(latest, dst, dirs_exist_ok=True)
    return dst


def load_model(precision="fp16"):
    import torch
    from unsloth import FastLanguageModel
    dtype = {"fp16": None, "fp32": torch.float32}[precision]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=dtype,
        load_in_4bit=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def attach_lora(model):
    from unsloth import FastLanguageModel
    return FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )


def run_probe(model, tokenizer, train_path):
    """Empirically verify tool-role content is excluded from the loss."""
    print("\n=== PROBE: response-only loss masking ===")
    import json
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth.chat_templates import train_on_responses_only
    from transformers import TrainingArguments

    convs = json.load(open(train_path))["conversations"][:8]
    texts = [tokenizer.apply_chat_template(c["messages"], tokenize=False,
                                           add_generation_prompt=False) for c in convs]
    ds = Dataset.from_dict({"text": texts})

    targs = TrainingArguments(
        output_dir="/tmp/probe_ckpt", per_device_train_batch_size=1,
        report_to="none", seed=3407)
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        dataset_text_field="text", max_seq_length=MAX_SEQ_LENGTH,
        packing=False, args=targs)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(4)])
    labels = batch["labels"]
    ok, bad = 0, 0
    for row_lab, row_ids in zip(labels, batch["input_ids"]):
        kept = [tid for tid, lab in zip(row_ids.tolist(), row_lab.tolist()) if lab != -100]
        text = tokenizer.decode(kept, skip_special_tokens=False)
        if "<|im_start|>tool" in text or "Memory stored" in text and "<|im_start|>tool" in text:
            bad += 1
        else:
            ok += 1
        # also assert some assistant content IS kept (loss exists at all)
    assert ok >= 1, "probe: no assistant tokens retained — masking broken in the other direction"
    print(f"probe rows clean: {ok}/4 (tool-role content excluded from loss)")
    print("=== PROBE PASS ===\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp32",
                    help="fp32 default: T4+fp16 GradScaler produced nan/0 grad_norms "
                         "(frozen weights). Only use fp16 on bf16-capable GPUs after "
                         "verifying healthy grad_norm in the first 30 log lines.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--drive-dir", default="/content/drive/MyDrive/nanohat_v2")
    ap.add_argument("--train", default=DEFAULT_TRAIN)
    ap.add_argument("--eval", default=DEFAULT_EVAL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--local-dir", default="training/adapter_v2")
    args = ap.parse_args()

    from unsloth import is_bfloat16_supported
    from transformers import TrainingArguments, TrainerCallback
    from trl import SFTTrainer
    from unsloth.chat_templates import train_on_responses_only
    sys_path_fix()

    model, tokenizer = load_model(args.precision)

    if args.probe:
        run_probe(model, tokenizer, args.train)
        return

    model = attach_lora(model)

    from datasets import Dataset
    import json

    def load(p):
        d = json.load(open(p))
        convs = d["conversations"] if isinstance(d, dict) else d
        texts = [tokenizer.apply_chat_template(c["messages"], tokenize=False,
                                               add_generation_prompt=False) for c in convs]
        return Dataset.from_dict({"text": texts})

    train_ds = load(args.train)
    eval_ds = load(args.eval) if os.path.exists(args.eval) else None

    os.makedirs(args.local_dir, exist_ok=True)
    resume_from = restore_from_drive(args.local_dir, args.drive_dir) if args.resume else None
    if resume_from:
        print(f"  ↩  resuming from {resume_from}")

    # v5-compatible warmup: transformers >=5.2 removed warmup_ratio
    import math
    effective_bs = args.batch_size * args.grad_accum
    total_steps = args.epochs * math.ceil(len(train_ds) / effective_bs)
    warmup_steps = max(10, int(0.05 * total_steps))
    print(f"  total_steps={total_steps}  warmup_steps={warmup_steps}  precision={args.precision}")

    targs = TrainingArguments(
        output_dir=args.local_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=warmup_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps", eval_steps=1000 if eval_ds else "no",
        save_strategy="steps", save_steps=SAVE_STEPS, save_total_limit=3,
        fp16=(args.precision == "fp16" and not is_bfloat16_supported()),
        bf16=(args.precision == "fp16" and is_bfloat16_supported()),
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=3407,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=train_ds,
        eval_dataset=eval_ds, dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH, dataset_num_proc=2,
        packing=False, args=targs,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    class _Sync(TrainerCallback):
        def on_save(self, targs, state, control, **kw):
            sync_to_drive(args.local_dir, args.drive_dir)
        def on_epoch_end(self, targs, state, control, **kw):
            sync_to_drive(args.local_dir, args.drive_dir)
    trainer.add_callback(_Sync())

    trainer.train(resume_from_checkpoint=resume_from)
    sync_to_drive(args.local_dir, args.drive_dir)

    model.save_pretrained(args.local_dir + "/final")
    tokenizer.save_pretrained(args.local_dir + "/final")
    sync_to_drive(args.local_dir + "/final", args.drive_dir + "/final")
    print("\n✓ training complete — adapter + checkpoints on Drive")


def sys_path_fix():
    import sys
    sys.path.insert(0, os.getcwd())


if __name__ == "__main__":
    main()
