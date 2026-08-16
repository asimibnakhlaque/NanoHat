"""
validator/validate_dataset.py
Batch validation runner that scans raw batch JSON files, validates each conversation against
the canonical schema, auto-rescues malformed formats via canonicalizer, optionally deduplicates
against prior runs, and writes the validated golden dataset to disk.
"""

import os
import sys
import json
import argparse
from validator.canonicalizer import canonicalize_conversation
from validator.schema_checker import validate_conversation
from validator.dedup import (
    build_existing_fingerprints,
    is_duplicate,
    extract_conversation_fingerprint,
)

def run_validation(
    raw_dir: str = "dataset/raw_batches",
    output_file: str = "dataset/validated/golden_dataset.json",
    prefix_filter: str = None,
    dedup: bool = False,
    existing_dataset_path: str = "dataset/validated/golden_dataset_merged.json",
):
    if not os.path.exists(raw_dir):
        print(f"Directory '{raw_dir}' does not exist.")
        return []

    raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
    if prefix_filter:
        raw_files = [f for f in raw_files if f.startswith(prefix_filter)]

    if not raw_files:
        filter_msg = f" matching prefix '{prefix_filter}'" if prefix_filter else ""
        print(f"No JSON batch files found in '{raw_dir}'{filter_msg}.")
        return []

    print(f"Scanning {len(raw_files)} batch files in {raw_dir} (Prefix filter: {prefix_filter or 'None'})...")

    existing_fps = set()
    if dedup:
        if os.path.exists(existing_dataset_path):
            existing_fps = build_existing_fingerprints(existing_dataset_path)
            print(f"  ✓ Loaded {len(existing_fps)} baseline fingerprints for deduplication")
        else:
            print(f"  ⚠ Existing dataset '{existing_dataset_path}' not found. Starting fresh fingerprint set.")

    total_found = 0
    passed_conversations = []
    rejected_log = []
    duplicate_count = 0

    for f in sorted(raw_files):
        file_path = os.path.join(raw_dir, f)
        try:
            with open(file_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            convos = data.get("conversations", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            total_found += len(convos)

            for idx, raw_conv in enumerate(convos):
                # 1. Canonicalize & Auto-Rescue
                clean_conv = canonicalize_conversation(raw_conv)
                if not clean_conv:
                    rejected_log.append((f, idx, "Canonicalization returned None"))
                    continue

                # 2. Strict Schema Validation
                is_valid, reason = validate_conversation(clean_conv)
                if not is_valid:
                    rejected_log.append((f, idx, reason))
                    continue

                # 3. Deduplication Check
                if dedup:
                    if is_duplicate(clean_conv, existing_fps, strict=True):
                        duplicate_count += 1
                        continue
                    fp_sig = extract_conversation_fingerprint(clean_conv)
                    existing_fps.add(fp_sig)

                passed_conversations.append(clean_conv)

        except Exception as e:
            print(f"  Error reading {file_path}: {e}")

    print("\n" + "="*50)
    print("VALIDATION & CANONICALIZATION SUMMARY")
    print("="*50)
    print(f"Total Raw Conversations Scanned: {total_found}")
    print(f"Passed & Canonicalized: {len(passed_conversations)} ({len(passed_conversations)/max(total_found, 1)*100:.1f}%)")
    if dedup:
        print(f"Duplicates Filtered: {duplicate_count} ({duplicate_count/max(total_found, 1)*100:.1f}%)")
    print(f"Rejected Invalid: {len(rejected_log)} ({len(rejected_log)/max(total_found, 1)*100:.1f}%)")

    if rejected_log:
        print("\nRejection Reasons (first 10):")
        for r_file, r_idx, r_reason in rejected_log[:10]:
            print(f"  - [{r_file} #{r_idx}]: {r_reason}")
        if len(rejected_log) > 10:
            print(f"  ... and {len(rejected_log) - 10} more.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(passed_conversations, out, indent=2)

    print(f"\n✓ Saved clean golden dataset to: {output_file}")
    return passed_conversations

def main():
    parser = argparse.ArgumentParser(description="Validate, canonicalize, and deduplicate synthetic dataset batches")
    parser.add_argument("--raw-dir", default="dataset/raw_batches", help="Path to raw batch directory")
    parser.add_argument("--output", default="dataset/validated/golden_dataset.json", help="Path to output golden dataset")
    parser.add_argument("--prefix", default=None, help="Optional batch filename prefix filter (e.g. 'run4')")
    parser.add_argument("--run-id", default=None, help="Shortcut for --prefix (e.g. 'run4')")
    parser.add_argument("--dedup", action="store_true", help="Enable deduplication against master dataset and within batch")
    parser.add_argument("--existing-dataset", default="dataset/validated/golden_dataset_merged.json", help="Path to existing master dataset for deduplication")
    args = parser.parse_args()

    prefix = args.prefix or args.run_id
    output = args.output
    if args.run_id and output == "dataset/validated/golden_dataset.json":
        output = f"dataset/validated/golden_dataset_{args.run_id}.json"

    run_validation(
        raw_dir=args.raw_dir,
        output_file=output,
        prefix_filter=prefix,
        dedup=args.dedup,
        existing_dataset_path=args.existing_dataset
    )

if __name__ == "__main__":
    main()
