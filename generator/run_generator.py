#!/usr/bin/env python3
"""
generator/run_generator.py
Unified concurrent synthetic dataset generator for SmolLM2-360M Fedora Agent.

Features:
- High-throughput multi-threading with ThreadPoolExecutor workers
- Thread-safe telemetry, cost tracking, and budget circuit-breaker
- Rate-limit exponential backoff (HTTP 429) & graceful stop on balance exhaustion (HTTP 402)
- Supports Run 4 Targeted Rebalancing mode (default) and Classical 4-Phase Curriculum mode
- Fast calibration probe and category-level slicing via CLI arguments
"""

import os
import sys
import json
import time
import re
import random
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from generator.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    TEACHER_MODEL,
    TEMPERATURE,
    TOP_P,
    DEFAULT_MAX_WORKERS,
    DEFAULT_BUDGET_USD,
    ESTIMATED_INPUT_PRICE_PER_M,
    ESTIMATED_OUTPUT_PRICE_PER_M,
    OUTPUT_RAW_DIR,
    GROUNDING_FIXTURES_PATH,
    TARGETED_RUN4_QUOTAS,
    CURRICULUM_PHASE_QUOTAS,
)
from generator.schemas import ALLOWED_TOOLS
from generator.targeted_prompts import build_targeted_prompt
from generator.prompts import (
    build_single_tool_prompt,
    build_no_tool_prompt,
    build_multi_tool_prompt,
    build_edge_case_prompt,
)

# Thread-safe telemetry state
state_lock = threading.Lock()
total_prompt_tokens = 0
total_completion_tokens = 0
total_conversations_saved = 0
total_batches_completed = 0
total_estimated_cost = 0.0
stop_generation_flag = False

def load_grounding_fixtures(path: str = GROUNDING_FIXTURES_PATH) -> dict:
    """Loads grounding output fixtures from disk if available."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load fixtures from {path}: {e}")
    return {}

def generate_worker_batch(
    worker_id: int,
    batch_index: int,
    task_label: str,
    prompt: str,
    run_id: str,
    output_dir: str,
    budget_limit: float,
    api_key: str,
    base_url: str,
) -> None:
    """Worker task that queries DeepSeek V4 Pro and saves raw JSON batch to disk."""
    global total_prompt_tokens, total_completion_tokens, total_conversations_saved
    global total_batches_completed, total_estimated_cost, stop_generation_flag

    with state_lock:
        if stop_generation_flag or total_estimated_cost >= budget_limit:
            return

    client = OpenAI(api_key=api_key, base_url=base_url)
    batch_filename = f"{run_id}_b{batch_index:03d}_{task_label}.json"
    batch_path = os.path.join(output_dir, batch_filename)

    for attempt in range(4):
        with state_lock:
            if stop_generation_flag or total_estimated_cost >= budget_limit:
                return

        try:
            response = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Generate the JSON object containing 5 unique conversations now."}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                response_format={"type": "json_object"}
            )

            usage = response.usage
            p_tok = usage.prompt_tokens if usage else 0
            c_tok = usage.completion_tokens if usage else 0
            call_cost = (p_tok / 1_000_000 * ESTIMATED_INPUT_PRICE_PER_M) + (c_tok / 1_000_000 * ESTIMATED_OUTPUT_PRICE_PER_M)

            content = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)

            data = json.loads(content)
            convos = data.get("conversations", [])

            if not convos or not isinstance(convos, list):
                time.sleep(2)
                continue

            # Save raw batch directly to disk
            os.makedirs(output_dir, exist_ok=True)
            with open(batch_path, "w", encoding="utf-8") as f:
                json.dump({
                    "batch_label": f"{run_id}_b{batch_index:03d}_{task_label}",
                    "tokens": {"prompt": p_tok, "completion": c_tok},
                    "conversations": convos
                }, f, indent=2)

            with state_lock:
                total_prompt_tokens += p_tok
                total_completion_tokens += c_tok
                total_conversations_saved += len(convos)
                total_batches_completed += 1
                total_estimated_cost += call_cost

                print(
                    f"[Worker {worker_id:02d}] ✓ {batch_filename} | "
                    f"+{len(convos)} convos | "
                    f"Spend: ${total_estimated_cost:.4f} / ${budget_limit:.2f} | "
                    f"Total Convos: {total_conversations_saved}"
                )

                if total_estimated_cost >= budget_limit:
                    print(f"\n⚠️  [Budget Limit Reached] Total spend reached ${total_estimated_cost:.4f} (Limit: ${budget_limit:.2f}). Stopping generation safely.")
                    stop_generation_flag = True

            return

        except Exception as e:
            err_str = str(e)
            if "402" in err_str or "insufficient" in err_str.lower() or "quota" in err_str.lower():
                with state_lock:
                    print(f"\n🛑 [Account Balance Consumed / HTTP 402] {err_str}")
                    stop_generation_flag = True
                return
            elif "429" in err_str or "rate" in err_str.lower():
                backoff = (2 ** attempt) + random.uniform(0.5, 2.0)
                time.sleep(backoff)
            else:
                time.sleep(2)

    print(f"[Worker {worker_id:02d}] ✗ Failed to generate batch '{batch_filename}' after 4 attempts.")

def run_calibration_probe(api_key: str, base_url: str, fixtures: dict, output_dir: str):
    """Runs a quick 3-batch calibration probe across 3 concurrent worker threads."""
    print("\n" + "="*70)
    print("🔬 RUNNING 3-BATCH CALIBRATION PROBE (15 CONVERSATIONS)")
    print("="*70 + "\n")

    probe_tasks = [
        ("probe_starved_desktop", build_targeted_prompt("starved_desktop_actions", fixtures)),
        ("probe_rich_reminders", build_targeted_prompt("rich_reminders_and_calculator", fixtures)),
        ("probe_complex_chains", build_targeted_prompt("complex_multi_tool_chains", fixtures)),
    ]

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for idx, (label, prompt) in enumerate(probe_tasks, start=1):
            fut = executor.submit(
                generate_worker_batch,
                worker_id=idx,
                batch_index=idx,
                task_label=label,
                prompt=prompt,
                run_id="probe",
                output_dir=output_dir,
                budget_limit=0.10,
                api_key=api_key,
                base_url=base_url,
            )
            futures.append(fut)

        for fut in as_completed(futures):
            pass

    elapsed = time.time() - start_time
    print("\n--- CALIBRATION RESULTS ---")
    print(f"Time Elapsed: {elapsed:.2f}s")
    print(f"Total Conversations: {total_conversations_saved}")
    print(f"Prompt Tokens: {total_prompt_tokens} | Completion Tokens: {total_completion_tokens}")
    print(f"Estimated Spend: ${total_estimated_cost:.4f} USD")
    print(f"Estimated Cost Per Conversation: ${(total_estimated_cost / max(total_conversations_saved, 1)):.4f} USD")
    print("="*70 + "\n")

def run_targeted_pipeline(
    api_key: str,
    base_url: str,
    fixtures: dict,
    run_id: str,
    max_workers: int,
    budget_limit: float,
    output_dir: str,
    category_filter: str = None
):
    """Runs concurrent targeted rebalancing generation sprint for Run 4."""
    print("\n" + "="*70)
    print(f"⚡ SMOLM2-360M FEDORA AGENT: TARGETED REBALANCED SPRINT ({run_id.upper()}) ⚡")
    print(f"   Teacher: {TEACHER_MODEL} | Workers: {max_workers} | Budget Limit: ${budget_limit:.2f}")
    print(f"   Raw Output Directory: {output_dir}")
    if category_filter:
        print(f"   Category Filter: {category_filter}")
    print("="*70 + "\n")

    task_queue = []
    batch_idx = 1

    quotas = TARGETED_RUN4_QUOTAS
    if category_filter:
        if category_filter not in quotas:
            print(f"Error: Unknown category '{category_filter}'. Allowed categories: {list(quotas.keys())}")
            sys.exit(1)
        quotas = {category_filter: quotas[category_filter]}

    for category, count in quotas.items():
        prompt = build_targeted_prompt(category, fixtures)
        for _ in range(count):
            task_queue.append((batch_idx, category, prompt))
            batch_idx += 1

    random.shuffle(task_queue)
    print(f"Queued {len(task_queue)} targeted batches (~{len(task_queue)*5} conversations) across {max_workers} concurrent threads.\n")

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for worker_num, (b_idx, cat, prompt) in enumerate(task_queue):
            if stop_generation_flag:
                break
            fut = executor.submit(
                generate_worker_batch,
                worker_id=(worker_num % max_workers) + 1,
                batch_index=b_idx,
                task_label=cat,
                prompt=prompt,
                run_id=run_id,
                output_dir=output_dir,
                budget_limit=budget_limit,
                api_key=api_key,
                base_url=base_url,
            )
            futures.append(fut)

        for fut in as_completed(futures):
            pass

    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("🏁 TARGETED SPRINT FINISHED!")
    print(f"   Time Elapsed: {elapsed/60:.1f} minutes")
    print(f"   Batches Completed: {total_batches_completed} / {len(task_queue)}")
    print(f"   Total Conversations Generated: {total_conversations_saved}")
    print(f"   Total Estimated Spend: ${total_estimated_cost:.4f} USD")
    print(f"   Raw Output Batches Saved: {output_dir}")
    print("="*70 + "\n")

def run_curriculum_pipeline(
    api_key: str,
    base_url: str,
    fixtures: dict,
    run_id: str,
    max_workers: int,
    budget_limit: float,
    output_dir: str
):
    """Runs concurrent classical 4-phase curriculum generation."""
    print("\n" + "="*70)
    print(f"📚 SMOLM2-360M FEDORA AGENT: CLASSICAL 4-PHASE CURRICULUM ({run_id.upper()}) 📚")
    print(f"   Teacher: {TEACHER_MODEL} | Workers: {max_workers} | Budget Limit: ${budget_limit:.2f}")
    print(f"   Raw Output Directory: {output_dir}")
    print("="*70 + "\n")

    task_queue = []
    batch_idx = 1

    # Phase 1: Single-Tool
    tools = sorted(list(ALLOWED_TOOLS))
    batches_per_tool = CURRICULUM_PHASE_QUOTAS["single_tool_batches_per_tool"]
    for tool in tools:
        prompt = build_single_tool_prompt(tool, fixtures)
        for _ in range(batches_per_tool):
            task_queue.append((batch_idx, f"phase1_{tool}", prompt))
            batch_idx += 1

    # Phase 2: No-Tool
    no_tool_prompt = build_no_tool_prompt()
    for _ in range(CURRICULUM_PHASE_QUOTAS["no_tool_batches"]):
        task_queue.append((batch_idx, "phase2_notool", no_tool_prompt))
        batch_idx += 1

    # Phase 3: Multi-Tool
    multi_tool_prompt = build_multi_tool_prompt(fixtures)
    for _ in range(CURRICULUM_PHASE_QUOTAS["multi_tool_batches"]):
        task_queue.append((batch_idx, "phase3_multitool", multi_tool_prompt))
        batch_idx += 1

    # Phase 4: Edge Cases
    edge_case_prompt = build_edge_case_prompt(fixtures)
    for _ in range(CURRICULUM_PHASE_QUOTAS["edge_case_batches"]):
        task_queue.append((batch_idx, "phase4_edgecase", edge_case_prompt))
        batch_idx += 1

    print(f"Queued {len(task_queue)} curriculum batches (~{len(task_queue)*5} conversations) across {max_workers} concurrent threads.\n")

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for worker_num, (b_idx, label, prompt) in enumerate(task_queue):
            if stop_generation_flag:
                break
            fut = executor.submit(
                generate_worker_batch,
                worker_id=(worker_num % max_workers) + 1,
                batch_index=b_idx,
                task_label=label,
                prompt=prompt,
                run_id=run_id,
                output_dir=output_dir,
                budget_limit=budget_limit,
                api_key=api_key,
                base_url=base_url,
            )
            futures.append(fut)

        for fut in as_completed(futures):
            pass

    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("🏁 CURRICULUM RUN FINISHED!")
    print(f"   Time Elapsed: {elapsed/60:.1f} minutes")
    print(f"   Batches Completed: {total_batches_completed} / {len(task_queue)}")
    print(f"   Total Conversations Generated: {total_conversations_saved}")
    print(f"   Total Estimated Spend: ${total_estimated_cost:.4f} USD")
    print(f"   Raw Output Batches Saved: {output_dir}")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Unified Concurrent Synthetic Dataset Generator for SmolLM2 Fedora Agent")
    parser.add_argument("--mode", choices=["targeted", "curriculum"], default="targeted", help="Generation mode: 'targeted' for Run 4 rebalancing (default) or 'curriculum' for 4-phase classical")
    parser.add_argument("--run-id", default="run4", help="Run identifier for batch file naming (e.g. run4)")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS, help=f"Concurrent worker threads (default: {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD, help=f"Budget cap in USD (default: ${DEFAULT_BUDGET_USD:.2f})")
    parser.add_argument("--output-dir", default=OUTPUT_RAW_DIR, help=f"Directory to save raw batch files (default: {OUTPUT_RAW_DIR})")
    parser.add_argument("--category", default=None, help="Generate only a single targeted category (e.g. starved_desktop_actions)")
    parser.add_argument("--probe", action="store_true", help="Run a fast 3-batch calibration probe (15 conversations)")
    parser.add_argument("--api-key", default=DEEPSEEK_API_KEY, help="DeepSeek API Key (or set DEEPSEEK_API_KEY environment variable)")
    parser.add_argument("--base-url", default=DEEPSEEK_BASE_URL, help=f"DeepSeek Base URL (default: {DEEPSEEK_BASE_URL})")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Notice: DEEPSEEK_API_KEY environment variable is not set.")
        print("To run live generation against DeepSeek V4 Pro, set: export DEEPSEEK_API_KEY='sk-...' or pass --api-key")
        print("Exiting generator gracefully.")
        sys.exit(0)

    fixtures = load_grounding_fixtures()

    if args.probe:
        run_calibration_probe(
            api_key=api_key,
            base_url=args.base_url,
            fixtures=fixtures,
            output_dir=args.output_dir
        )
    elif args.mode == "targeted":
        run_targeted_pipeline(
            api_key=api_key,
            base_url=args.base_url,
            fixtures=fixtures,
            run_id=args.run_id,
            max_workers=args.workers,
            budget_limit=args.budget,
            output_dir=args.output_dir,
            category_filter=args.category
        )
    else:
        run_curriculum_pipeline(
            api_key=api_key,
            base_url=args.base_url,
            fixtures=fixtures,
            run_id=args.run_id,
            max_workers=args.workers,
            budget_limit=args.budget,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main()
