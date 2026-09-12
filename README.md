# NanoHat: Sub-1B Fedora Linux OS Agent

A lightweight, autonomous AI agent powered by **SmolLM2-360M-Instruct** designed specifically for **Fedora Linux**, powered by a high-performance native CLI runtime.

## Key Features
- **6 Consolidated Core Tools**: `calculator`, `web_search`, `user_memory`, `scheduler`, `system_health`, and `system_action`.
- **Fail-Closed Safety**: destructive actions require interactive confirmation; headless sessions refuse by default (`NANOHAT_AUTO_APPROVE_DESTRUCTIVE=1` overrides for trusted daemons). Persistent tasks with absolute due times survive restarts.
- **Zero Shell Interpolation Security**: All system executions run through `subprocess.run(shell=False)` with argument arrays and process sanitization.
- **Zero-Overhead Native Harness**: Minimal prompt overhead (~100 tokens), preserving context for sub-1B language models.
- **FedoraFormatBridge**: Intercepts custom `<thought>` and `<tool_call>` tags from fine-tuned weights and bridges them to the native tool-calling engine.
- **Curriculum Synthetic Data Generator**: 4 curriculum phases with balanced tool tracking, ground-truth CLI fixtures, and canonical syntax validation.
- **Loss Masking Collator**: Full-conversation single-pass tokenization with character offset mapping (labels = -100 for non-assistant tokens).

---

## Directory Structure

```
~/nanohat/
├── grounding/
│   ├── capture_tool_outputs.py      # Captures live Fedora CLI fixtures
│   └── real_tool_outputs.json       # Ground-truth CLI output snippets
├── generator/
│   ├── config.py                    # Generator parameters & budget limits
│   ├── schemas.py                   # 6-tool JSON schema & allowed enums
│   ├── prompts.py                   # Curriculum prompt templates (Phases 1-4)
│   └── run_generator.py             # Generator orchestrator with balanced tool tracking
├── validator/
│   ├── canonicalizer.py             # Re-serializes tool calls into canonical JSON syntax
│   ├── schema_checker.py            # Enforces role order, exact system prompt, concise thought
│   └── validate_dataset.py          # Batch validator filtering raw files into golden dataset
├── dataset/
│   ├── raw_batches/                 # Raw DeepSeek generation batches
│   ├── validated/                   # Clean canonicalized dataset (500–800 samples)
│   └── train_eval_split.py          # Split generator (90% train / 10% eval)
├── training/
│   ├── dataset_collator.py          # Dataset loading & ChatML template formatting
│   ├── train_unsloth.py             # Ultra-fast 16-bit LoRA fine-tuning & response loss masking via Unsloth
│   └── merge_adapter.py             # Standalone 16-bit merger & direct GGUF (F16/Q4_K_M) export
│   
├── runtime/
│   ├── tools.py                     # 6 consolidated @tool definitions with confirmation gates
│   ├── agent.py                     # Native OS agent runtime with FedoraFormatBridge
│   └── test_agent.py                # Local integration & security test suite
├── README.md
└── requirements.txt             # Python dependencies
```

---

## Quickstart Guide

### 1. One-Line Installer (Recommended)
```bash
./install.sh
```
Or install directly via pip:
```bash
pip install -e .
```

### 2. Run Single-Shot OS Commands
Execute any query directly from your terminal:
```bash
nanohat "What is 15 percent of 800?"
nanohat "What is my current RAM usage?"
nanohat "Check battery status"
nanohat "Search for Fedora 41 release schedule"
```

For interactive multi-turn chat:
```bash
nanohat --interactive
```

### 3. Generate Synthetic Training Data (DeepSeek V4 Pro)
```bash
# 1. Run 3-batch calibration probe (15 samples)
export DEEPSEEK_API_KEY="sk-..."
python -m generator.run_generator --probe

# 2. Run full curriculum generation
python -m generator.run_generator
```

### 4. Validate & Canonicalize Dataset
```bash
# Validate raw batches and build golden dataset
python -m validator.validate_dataset

# Create train/eval splits
python dataset/train_eval_split.py
```

### 5. Fast Unsloth Fine-Tuning & Export
```bash
python -m training.train_unsloth --train-data dataset/validated/train.json --eval-data dataset/validated/eval.json --epochs 5
```

### 6. Interactive Local Agent
```bash
# Run using local Ollama model
python -m runtime.agent --backend ollama --model nanohat2.1:360m

# Or load directly into PyTorch in-memory weights
python -m runtime.agent --backend transformers --model training/merged_model
```

---

## Running Unit & Security Tests
```bash
python -m unittest runtime.test_agent
```
