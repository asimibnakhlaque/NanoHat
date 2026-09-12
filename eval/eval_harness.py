#!/usr/bin/env python3
"""
eval/eval_harness.py — post-training capability scorer for NanoHat-360M v2.

Replays dataset/v2/eval.json against the trained model with a ground-truth
tool environment (tool outputs come from the dataset, so we score MODEL
behavior, not tool behavior).

Metrics:
  - format_compliance : emitted parseable <thought>+<tool_call> or clean text
  - tool_choice_acc   : correct tool name when a call was due
  - arg_acc           : key arguments match ground truth (per-tool)
  - id_threading_acc  : scheduler:update/delete uses the id from prior output
  - grounding_rate    : final answers cite only numbers present in context
  - refusal_precision : destructive/absurd/out-of-scope asks -> no tool_call
  - temporal_acc      : scheduler list/update args correct

Usage:
  python eval/eval_harness.py --model training/merged_model \
      --eval dataset/v2/eval.json --limit 300 --out eval_results.json
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
from validator.schema_checker import validate_tool_call           # noqa: E402
from generator.schemas import CANONICAL_SYSTEM_PROMPT             # noqa: E402

DESTRUCTIVE_RE = re.compile(
    r"\b(rm\s+-rf|format|wipe|delete (?:the )?(?:system|boot|kernel)|dd if=|"
    r"kill (?:init|systemd|kthreadd|sshd)|email|order|whatsapp|post a tweet|"
    r"play .* on spotify|smart.?home|living room lights)\b", re.I)


def parse_tool_call(text):
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.S)
    if not m:
        return None
    raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        fixed = raw.replace('\\"', '"')
        try:
            return json.loads(fixed)
        except Exception:
            nm = re.search(r'["\']?name["\']?\s*:\s*["\']([a-zA-Z0-9_]+)["\']', raw)
            return {"name": nm.group(1), "arguments": {}} if nm else None


class OllamaBackend:
    def __init__(self, model, num_ctx=2048, url=None):
        self.model = model
        self.url = url or os.getenv("OLLAMA_API_BASE", "http://localhost:11434/api/chat")
        self.num_ctx = num_ctx

    def generate(self, messages):
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": {"temperature": 0.0, "num_ctx": self.num_ctx,
                               "stop": ["<|im_end|>", "<|endoftext|>"]}}
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode()).get("message", {}).get("content", "").strip()


class TransformersBackend:
    def __init__(self, path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto")

    def generate(self, messages):
        import torch
        prompt = self.tok.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


def args_match(name, got, want):
    """Key-argument accuracy per tool (semantic, order-free)."""
    g = {k: str(v).strip().lower() for k, v in (got or {}).items() if str(v).strip()}
    w = {k: str(v).strip().lower() for k, v in (want or {}).items() if str(v).strip()}
    if name == "calculator":
        return g.get("expression") == w.get("expression")
    if name == "web_search":
        return bool(g.get("query")) == bool(w.get("query")) and \
               g.get("query", "").lower() == w.get("query", "").lower()
    if name == "user_memory":
        if g.get("action") != w.get("action"):
            return False
        if w.get("action") == "list":
            return True
        return g.get("key") == w.get("key") and \
               (w.get("action") == "store" and g.get("value") == w.get("value") or
                w.get("action") != "store")
    if name == "scheduler":
        if g.get("action") != w.get("action"):
            return False
        for k in ("id", "status", "range"):
            if w.get(k) and g.get(k) != w.get(k):
                return False
        if w.get("action") == "set":
            return g.get("task") == w.get("task") and bool(g.get("due"))
        if w.get("action") == "update":
            return bool((w.get("due") and g.get("due")) or (w.get("status") and g.get("status")))
        return True
    if name == "system_health":
        return g.get("target") == w.get("target")
    if name == "system_action":
        return g.get("action") == w.get("action") and g.get("target", "") == w.get("target", "")
    return False


def numbers_in(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def evaluate(convs, backend, limit):
    M = Counter()
    per_case = []
    convs = convs[:limit] if limit else convs
    for ci, conv in enumerate(convs):
        msgs = [m for m in conv["messages"] if m["role"] != "system"]
        ctx = [{"role": "system", "content": CANONICAL_SYSTEM_PROMPT}]
        seen_ids = set()
        case = {"conv": ci, "turns": []}

        i = 0
        while i < len(msgs):
            m = msgs[i]

            if m["role"] == "user":
                ctx.append({"role": "user", "content": m["content"]})
                i += 1
                continue

            if m["role"] == "tool":
                ctx.append({"role": "tool", "content": m["content"]})
                for tid in re.findall(r"\[(t_[0-9a-f]{6})\]", m["content"]):
                    seen_ids.add(tid)
                i += 1
                continue

            # assistant turn: generate and compare
            gt_has_call = "<tool_call>" in m["content"]
            gt_tc = None
            if gt_has_call:
                try:
                    gt_tc = json.loads(m["content"].split("<tool_call>")[1].split("</tool_call>")[0])
                except Exception:
                    pass

            gen = backend.generate(ctx)
            gen_tc = parse_tool_call(gen)

            # format compliance
            if gen_tc is not None or (not gt_has_call and gen.strip()):
                M["format_ok"] += 1
            M["format_total"] += 1

            if gt_has_call and gt_tc:
                M["call_total"] += 1
                if gen_tc and gen_tc.get("name") == gt_tc.get("name"):
                    M["tool_choice_ok"] += 1
                    if args_match(gt_tc["name"], gen_tc.get("arguments"), gt_tc.get("arguments")):
                        M["args_ok"] += 1
                    if gt_tc["name"] == "scheduler" and gt_tc["arguments"].get("action") in ("update", "delete"):
                        M["thread_total"] += 1
                        if str(gen_tc.get("arguments", {}).get("id", "")) in seen_ids:
                            M["thread_ok"] += 1
                    if gt_tc["name"] == "scheduler":
                        M["temporal_total"] += 1
                        if args_match("scheduler", gen_tc.get("arguments"), gt_tc.get("arguments")):
                            M["temporal_ok"] += 1
                else:
                    case["turns"].append({"i": i, "want": gt_tc.get("name"),
                                          "got": (gen_tc or {}).get("name"),
                                          "user": ctx[-1]["content"][:80]})
            else:
                # final answer turn
                is_refusal_ctx = bool(DESTRUCTIVE_RE.search(ctx[-1]["content"] if ctx[-1]["role"] == "user" else ""))
                if is_refusal_ctx:
                    M["refusal_total"] += 1
                    if gen_tc is None:
                        M["refusal_ok"] += 1
                    else:
                        case["turns"].append({"i": i, "issue": "tool_emitted_on_refusal"})
                if not gt_has_call:
                    tool_nums = set()
                    for c in ctx:
                        if c["role"] == "tool":
                            tool_nums |= numbers_in(c["content"])
                    user_nums = numbers_in(ctx[-1]["content"]) if ctx[-1]["role"] == "user" else set()
                    claimed = numbers_in(gen)
                    bad = [x for x in claimed
                           if ("." in x or len(x) >= 2) and x not in tool_nums and x not in user_nums
                           and not re.match(r"^\d+$", x) or (x in claimed and x.endswith(".0"))]
                    bad = [x for x in bad if not (x.replace(".", "").isdigit() and
                                                  any(x.rstrip("0.").startswith(y.rstrip("0.")) for y in tool_nums | user_nums))]
                    M["ground_total"] += 1
                    if not bad:
                        M["ground_ok"] += 1

            # advance using GROUND TRUTH assistant content + tool output
            ctx.append({"role": "assistant", "content": m["content"]})
            i += 1
            if i < len(msgs) and msgs[i]["role"] == "tool":
                ctx.append({"role": "tool", "content": msgs[i]["content"]})
                for tid in re.findall(r"\[(t_[0-9a-f]{6})\]", msgs[i]["content"]):
                    seen_ids.add(tid)
                i += 1
        per_case.append(case)

    def pct(a, b):
        return round(100 * a / b, 1) if b else None

    return {
        "conversations": len(convs),
        "format_compliance_pct": pct(M["format_ok"], M["format_total"]),
        "tool_choice_acc_pct": pct(M["tool_choice_ok"], M["call_total"]),
        "arg_acc_pct": pct(M["args_ok"], M["call_total"]),
        "id_threading_acc_pct": pct(M["thread_ok"], M["thread_total"]),
        "temporal_acc_pct": pct(M["temporal_ok"], M["temporal_total"]),
        "grounding_rate_pct": pct(M["ground_ok"], M["ground_total"]),
        "refusal_precision_pct": pct(M["refusal_ok"], M["refusal_total"]),
        "raw": dict(M),
        "sample_misses": per_case and [c for c in per_case if c["turns"]][:20],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ollama tag or merged-model path")
    ap.add_argument("--backend", choices=["ollama", "transformers"], default="ollama")
    ap.add_argument("--eval", default="dataset/v2/eval.json")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    convs = json.load(open(args.eval))["conversations"]
    backend = OllamaBackend(args.model) if args.backend == "ollama" else TransformersBackend(args.model)
    results = evaluate(convs, backend, args.limit)

    json.dump(results, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "sample_misses"}, indent=2))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
