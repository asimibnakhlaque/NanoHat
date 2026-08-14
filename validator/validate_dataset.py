"""
validator/validate_dataset.py
Batch validation runner that scans dataset/raw_batches/, validates each conversation,
applies canonicalization, filters out malformed rows, and writes dataset/validated/golden_dataset.json.
"""

import os
import json
import argparse
from validator.canonicalizer import canonicalize_conversation
from validator.schema_checker import validate_conversation

def run_validation(raw_dir="dataset/raw_batches", output_file="dataset/validated/golden_dataset.json"):
    if not os.path.exists(raw_dir):
        print(f"Directory '{raw_dir}' does not exist.")
        return []
        
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
    if not raw_files:
        print(f"No JSON batch files found in '{raw_dir}'.")
        return []
        
    print(f"Scanning {len(raw_files)} batch files in {raw_dir}...")
    
    total_found = 0
    passed_conversations = []
    rejected_log = []
    
    for f in sorted(raw_files):
        file_path = os.path.join(raw_dir, f)
        try:
            with open(file_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                
            convos = data.get("conversations", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            total_found += len(convos)
            
            for idx, raw_conv in enumerate(convos):
                # 1. Canonicalize
                clean_conv = canonicalize_conversation(raw_conv)
                if not clean_conv:
                    rejected_log.append((f, idx, "Canonicalization returned None"))
                    continue
                    
                # 2. Validate
                is_valid, reason = validate_conversation(clean_conv)
                if is_valid:
                    passed_conversations.append(clean_conv)
                else:
                    rejected_log.append((f, idx, reason))
                    
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
            
    print("\n--- VALIDATION SUMMARY ---")
    print(f"Total Raw Conversations: {total_found}")
    print(f"Passed & Canonicalized: {len(passed_conversations)} ({len(passed_conversations)/max(total_found, 1)*100:.1f}%)")
    print(f"Rejected: {len(rejected_log)} ({len(rejected_log)/max(total_found, 1)*100:.1f}%)")
    
    if rejected_log:
        print("\nTop 5 Rejection Reasons:")
        for r_file, r_idx, r_reason in rejected_log[:5]:
            print(f"  - [{r_file} #{r_idx}]: {r_reason}")
            
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(passed_conversations, out, indent=2)
        
    print(f"\n✓ Saved clean golden dataset to: {output_file}")
    return passed_conversations

def main():
    parser = argparse.ArgumentParser(description="Validate and canonicalize raw synthetic dataset batches")
    parser.add_argument("--raw-dir", default="dataset/raw_batches", help="Path to raw batch directory")
    parser.add_argument("--output", default="dataset/validated/golden_dataset.json", help="Path to output golden dataset")
    args = parser.parse_args()
    
    run_validation(raw_dir=args.raw_dir, output_file=args.output)

if __name__ == "__main__":
    main()
