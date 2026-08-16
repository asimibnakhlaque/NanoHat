"""
generator/run_generator.py
Main orchestrator for generating curriculum synthetic training data with DeepSeek V4 Pro.
"""

import os
import sys
import json
import time
import re
import argparse
from openai import OpenAI

from generator.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    TEACHER_MODEL,
    TEMPERATURE,
    TOP_P,
    OUTPUT_RAW_DIR,
    GROUNDING_FIXTURES_PATH,
    PHASE_QUOTAS,
    EXISTING_DATASET_PATH
)
from generator.schemas import ALLOWED_TOOLS
from generator.prompts import (
    build_single_tool_prompt,
    build_no_tool_prompt,
    build_multi_tool_prompt,
    build_edge_case_prompt,
)
from validator.dedup import build_existing_fingerprints, deduplicate_batch

# Tracker to guarantee strictly equal representation across all 6 tools in Phase 1
TOOLS = sorted(list(ALLOWED_TOOLS))
TOOL_BATCH_COUNT = {tool: 0 for tool in TOOLS}

def get_next_single_tool():
    """Selects the next tool with the minimum batch count to maintain perfect balance."""
    min_count = min(TOOL_BATCH_COUNT.values())
    eligible = [t for t, c in TOOL_BATCH_COUNT.items() if c == min_count]
    chosen = eligible[0]
    TOOL_BATCH_COUNT[chosen] += 1
    return chosen

def load_grounding_fixtures(path=GROUNDING_FIXTURES_PATH):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"Warning: Fixture file {path} not found. Using default empty fixtures.")
    return {}

def call_deepseek_batch(client: OpenAI, prompt: str, batch_label: str, retries=3):
    for attempt in range(attempt_count := retries):
        try:
            print(f"  [API Call] Requesting batch '{batch_label}' ({TEACHER_MODEL})...")
            response = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Generate the JSON object with the 5 conversations now."}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                response_format={"type": "json_object"}
            )
            
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            comp_tokens = usage.completion_tokens if usage else 0
            print(f"  [Tokens] Prompt: {prompt_tokens} | Completion: {comp_tokens} | Total: {prompt_tokens + comp_tokens}")
            
            content = response.choices[0].message.content
            # Extract JSON block
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
                
            data = json.loads(content)
            convos = data.get("conversations", [])
            
            if not convos or not isinstance(convos, list):
                print(f"  [Warning] Batch returned 0 conversations. Retrying (attempt {attempt+1}/{retries})...")
                time.sleep(2)
                continue
                
            # Save raw batch immediately
            os.makedirs(OUTPUT_RAW_DIR, exist_ok=True)
            batch_file = os.path.join(OUTPUT_RAW_DIR, f"{batch_label}.json")
            with open(batch_file, "w", encoding="utf-8") as f:
                json.dump({"batch_label": batch_label, "tokens": {"prompt": prompt_tokens, "completion": comp_tokens}, "conversations": convos}, f, indent=2)
                
            print(f"  ✓ Saved {len(convos)} conversations to {batch_file}")
            return convos, prompt_tokens, comp_tokens
            
        except Exception as e:
            print(f"  ✗ Attempt {attempt+1} failed with error: {e}")
            time.sleep(3)
            
    print(f"  ✗ Failed to generate batch '{batch_label}' after {retries} attempts.")
    return [], 0, 0

def run_calibration_probe(client: OpenAI, fixtures: dict):
    print("\n======================================================")
    print("RUNNING 3-BATCH CALIBRATION PROBE (15 CONVERSATIONS)")
    print("======================================================")
    
    total_prompt_tok = 0
    total_comp_tok = 0
    total_samples = 0
    
    # 1 Single-Tool, 1 No-Tool, 1 Multi-Tool
    probe_tasks = [
        ("probe_single_system_health", build_single_tool_prompt("system_health", fixtures)),
        ("probe_notool_general", build_no_tool_prompt()),
        ("probe_multi_triage", build_multi_tool_prompt(fixtures))
    ]
    
    for label, prompt in probe_tasks:
        convos, p_tok, c_tok = call_deepseek_batch(client, prompt, label)
        total_prompt_tok += p_tok
        total_comp_tok += c_tok
        total_samples += len(convos)
        time.sleep(1)
        
    print("\n--- CALIBRATION RESULTS ---")
    print(f"Total Conversations Generated: {total_samples}")
    print(f"Total Prompt Tokens: {total_prompt_tok} | Total Completion Tokens: {total_comp_tok}")
    # Pricing estimate: assuming ~$0.453/M in, $0.87/M out (typical V4 pricing)
    est_cost = (total_prompt_tok / 1_000_000 * 0.453) + (total_comp_tok / 1_000_000 * 0.87)
    print(f"Estimated 3-Batch Spend: ${est_cost:.4f} USD")
    print(f"Estimated Cost per Conversation: ${(est_cost / max(total_samples, 1)):.4f} USD")
    print("========================================================\n")

def run_full_pipeline(client: OpenAI, fixtures: dict, run_id: str = "run2", enable_dedup: bool = True):
    print("\n========================================================")
    print(" RUNNING FULL CURRICULUM SYNTHETIC GENERATION PIPELINE")
    print("========================================================")

    existing_fps = set()
    if enable_dedup:
        existing_fps = build_existing_fingerprints(EXISTING_DATASET_PATH)

    total_dupes_removed = 0
    
    # Phase 1: Single-Tool Mastery
    print("\n--- Phase 1: Single-Tool Mastery ---")
    batches_per_tool = PHASE_QUOTAS["single_tool_batches_per_tool"]
    total_single_batches = len(TOOLS) * batches_per_tool
    
    for i in range(total_single_batches):
        tool = get_next_single_tool()
        batch_label = f"{run_id}_phase1_{tool}_batch{TOOL_BATCH_COUNT[tool]}"
        prompt = build_single_tool_prompt(tool, fixtures)
        convos, p_tok, c_tok = call_deepseek_batch(client, prompt, batch_label)

        if enable_dedup and convos:
            convos, dupes = deduplicate_batch(convos, existing_fps)
            total_dupes_removed += dupes
            if dupes:
                print(f" [Dedup] Removed {dupes} duplicated convesations")

        time.sleep(1)
        
    # Phase 2: No-Tool Chat
    print("\n--- Phase 2: No-Tool Pure Conversation ---")
    for i in range(PHASE_QUOTAS["no_tool_batches"]):
        batch_label = f"{run_id}_phase2_notool_batch{i+1}"
        prompt = build_no_tool_prompt()
        convos, p_tok, c_tok = call_deepseek_batch(client, prompt, batch_label)

        if enable_dedup and convos:
            convos, dupes = deduplicate_batch(convos, existing_fps)
            total_dupes_removed += dupes
            if dupes:
                print(f" [Dedup] Removed {dupes} duplicated convesations")

        time.sleep(1)
        
    # Phase 3: Multi-Tool Chaining
    print("\n--- Phase 3: Multi-Tool Chaining ---")
    for i in range(PHASE_QUOTAS["multi_tool_batches"]):
        batch_label = f"{run_id}_phase3_multitool_batch{i+1}"
        prompt = build_multi_tool_prompt(fixtures)
        convos, p_tok, c_tok = call_deepseek_batch(client, prompt, batch_label)

        if enable_dedup and convos:
            convos, dupes = deduplicate_batch(convos, existing_fps)
            total_dupes_removed += dupes
            if dupes:
                print(f" [Dedup] Removed {dupes} duplicated convesations")

        time.sleep(1)
        
    # Phase 4: Edge Cases
    print("\n--- Phase 4: Edge Cases & Recovery ---")
    for i in range(PHASE_QUOTAS["edge_case_batches"]):
        batch_label = f"{run_id}_phase4_edgecase_batch{i+1}"
        prompt = build_edge_case_prompt(fixtures)
        convos, p_tok, c_tok = call_deepseek_batch(client, prompt, batch_label)

        if enable_dedup and convos:
            convos, dupes = deduplicate_batch(convos, existing_fps)
            total_dupes_removed += dupes
            if dupes:
                print(f" [Dedup] Removed {dupes} duplicated convesations")

        time.sleep(1)
        
    print("\n✓ Full generation run completed. Raw batches stored in:", OUTPUT_RAW_DIR)

def main():
    parser = argparse.ArgumentParser(description="Curriculum Synthetic Dataset Generator for SmolLM2 Fedora Agent")
    parser.add_argument("--probe", action="store_true", help="Run a 3-batch calibration probe (15 conversations)")
    parser.add_argument("--api-key", default=DEEPSEEK_API_KEY, help="DeepSeek API Key")
    parser.add_argument("--run-id", default="run2", help="Run identifier for batch filenames")
    parser.add_argument("--dedup", action="store_true", help="Enable deduplication against existing dataset")
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Notice: DEEPSEEK_API_KEY environment variable is not set.")
        print("To run live generation against DeepSeek V4 Pro, set: export DEEPSEEK_API_KEY='sk-...'")
        print("Exiting generator gracefully.")
        sys.exit(0)
        
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    fixtures = load_grounding_fixtures()
    
    if args.probe:
        run_calibration_probe(client, fixtures)
    else:
        run_full_pipeline(client, fixtures, run_id=args.run_id)

if __name__ == "__main__":
    main()
