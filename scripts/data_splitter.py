#!/usr/bin/env python3
"""
split_json.py
Reads a JSON file containing a list under a given key and splits it into
3 roughly equal parts, writing each to a separate output file.

Usage:
    python split_json.py input.json                       # splits "conversations"
    python split_json.py input.json --key items           # custom key
    python split_json.py input.json --out-prefix chunk    # custom output prefix
"""
import argparse
import json
import math
import os


def split_into_n(lst, n):
    """Split `lst` into `n` roughly equal chunks (last chunk may be smaller)."""
    if n <= 0:
        raise ValueError("n must be positive")
    size = math.ceil(len(lst) / n)
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def main():
    ap = argparse.ArgumentParser(description="Split a JSON list into 3 equal parts.")
    ap.add_argument("input", help="Input JSON file")
    ap.add_argument("--key", default="conversations",
                    help="Top-level key holding the list (default: conversations)")
    ap.add_argument("--parts", type=int, default=3, help="Number of parts (default: 3)")
    ap.add_argument("--out-prefix", default=None,
                    help="Output filename prefix (default: <input>_part<N>.json)")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or args.key not in data:
        raise SystemExit(f"Error: '{args.key}' key not found in {args.input}")

    items = data[args.key]
    if not isinstance(items, list):
        raise SystemExit(f"Error: data['{args.key}'] is not a list.")

    chunks = split_into_n(items, args.parts)
    base = args.out_prefix or os.path.splitext(args.input)[0]

    for i, chunk in enumerate(chunks, start=1):
        out_path = f"{base}_part{i}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({args.key: chunk}, f, indent=2, ensure_ascii=False)
        print(f"[{i}/{len(chunks)}] wrote {len(chunk)} items -> {out_path}")

    print(f"\nTotal: {len(items)} items split into {len(chunks)} files.")


if __name__ == "__main__":
    main()