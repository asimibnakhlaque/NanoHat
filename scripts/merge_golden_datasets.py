"""
scripts/merge_golden_datasets.py
Merges run1 and run2 golden datasets into a single unified training file.
"""
import json
import os

RUN1_PATH = "dataset/validated/golden_dataset_run1_backup.json"
RUN2_PATH = "dataset/validated/golden_dataset_run2_backup.json"
OUTPUT_PATH = "dataset/validated/golden_dataset_merged.json"

def merge():
    conversations = []
    
    for path, label in [(RUN1_PATH, "Run1"), (RUN2_PATH, "Run2")]:
        if not os.path.exists(path):
            print(f"  ⚠ {label} file not found at {path}, skipping.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  ✓ Loaded {len(data)} conversations from {label}")
        conversations.extend(data)
    
    print(f"\n  Total merged: {len(conversations)} conversations")
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2)
    print(f"  ✓ Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    merge()