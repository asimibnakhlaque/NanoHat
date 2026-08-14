#!/usr/bin/env python3
"""
dataset/train_eval_split.py
Splits dataset/validated/golden_dataset.json into train.json and eval.json (90% / 10%).
"""

import os
import json
import random
import argparse

def create_splits(input_path="dataset/validated/golden_dataset.json", output_dir="dataset/validated", split_ratio=0.9, seed=42):
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return
        
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if not isinstance(data, list) or len(data) == 0:
        print(f"Error: {input_path} contains 0 conversations.")
        return
        
    random.seed(seed)
    shuffled = list(data)
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * split_ratio)
    train_data = shuffled[:split_idx]
    eval_data = shuffled[split_idx:]
    
    train_path = os.path.join(output_dir, "train.json")
    eval_path = os.path.join(output_dir, "eval.json")
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)
        
    print(f"✓ Split dataset ({len(data)} total):")
    print(f"  - Train: {len(train_data)} samples -> {train_path}")
    print(f"  - Eval:  {len(eval_data)} samples -> {eval_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/validated/golden_dataset.json")
    parser.add_argument("--output-dir", default="dataset/validated")
    parser.add_argument("--ratio", type=float, default=0.9)
    args = parser.parse_args()
    create_splits(input_path=args.input, output_dir=args.output_dir, split_ratio=args.ratio)
