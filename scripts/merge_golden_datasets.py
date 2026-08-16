#!/usr/bin/env python3
"""
scripts/merge_golden_datasets.py
Consolidates and merges golden datasets across Runs 1, 2, 3, and 4 into
the final master training dataset with cross-run deduplication.
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator.dedup import extract_conversation_fingerprint

SEARCH_PATHS = [
    ("dataset/validated/golden_dataset_run1_backup.json", "Run 1 (Baseline)"),
    ("dataset/validated/golden_dataset_run2_backup.json", "Run 2 (Novel Scenarios)"),
    ("dataset/validated/golden_dataset_run3_backup.json", "Run 3 (Final Phase)"),
    ("dataset/validated/golden_dataset_run4.json", "Run 4 (Targeted Rebalance)"),
]

PRIMARY_OUTPUT_PATH = "dataset/validated/golden_dataset_merged.json"
FINAL_OUTPUT_PATH = "dataset/validated/golden_dataset_final.json"

def merge_all_runs():
    print("\n" + "="*70)
    print("CONSOLIDATING & MERGING GOLDEN DATASETS ACROSS ALL RUNS")
    print("="*70 + "\n")

    merged_conversations = []
    seen_fingerprints = set()
    total_loaded = 0
    dupes_filtered = 0

    for path, label in SEARCH_PATHS:
        if not os.path.exists(path):
            print(f"  ⚠ {label}: not found at '{path}', skipping.")
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                print(f"  ⚠ {label}: Unexpected JSON format (expected list), skipping.")
                continue

            count = len(data)
            total_loaded += count
            added = 0

            for conv in data:
                fp = extract_conversation_fingerprint(conv)
                if fp in seen_fingerprints:
                    dupes_filtered += 1
                    continue
                seen_fingerprints.add(fp)
                merged_conversations.append(conv)
                added += 1

            print(f"  ✓ {label}: Loaded {count} convos (+{added} unique, {count - added} duplicates skipped)")

        except Exception as e:
            print(f"  ✗ Error reading {path}: {e}")

    print("\n" + "-"*50)
    print(f"Total Scanned: {total_loaded} conversations")
    print(f"Duplicates Removed: {dupes_filtered}")
    print(f"Final Unique Master Total: {len(merged_conversations)} conversations")
    print("-"*50 + "\n")

    for out_path in [PRIMARY_OUTPUT_PATH, FINAL_OUTPUT_PATH]:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged_conversations, f, indent=2)
        print(f"  ✓ Saved final master dataset to: {out_path}")

    print("\n🎉 Consolidation complete! Dataset is ready for training.\n")

if __name__ == "__main__":
    merge_all_runs()