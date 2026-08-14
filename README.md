# SmolLM2-360M Fedora Agent

A lightweight, autonomous AI agent powered by **SmolLM2-360M-Instruct** designed specifically for **Fedora Linux**, orchestrated via the **Hugging Face `smolagents`** framework.

## Key Features
- **6 Consolidated Core Tools**: `calculator`, `web_search`, `user_memory`, `reminder`, `system_health`, and `system_action`.
- **Zero Shell Interpolation Security**: All system executions run through `subprocess.run(shell=False)` with argument arrays and process sanitization.
- **Hugging Face `smolagents` Harness**: Minimal prompt overhead (~100 tokens), preserving context for small language models.
- **FedoraFormatBridge**: Intercepts custom `<thought>` and `<tool_call>` tags from fine-tuned weights and bridges them to the `smolagents` tool-calling engine.
- **Curriculum Synthetic Data Generator**: 4 curriculum phases with balanced tool tracking, ground-truth CLI fixtures, and canonical syntax validation.
- **Loss Masking Collator**: Full-conversation single-pass tokenization with character offset mapping (labels = -100 for non-assistant tokens).

---

## Directory Structure

```
smollm2-fedora-agent/
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
│   ├── dataset_collator.py          # Single-pass offset-mapped loss masking
│   ├── train_qlora.py               # TRL SFTTrainer / Unsloth QLoRA script for SmolLM2-360M
│   └── merge_adapter.py             # LoRA weights merger for local inference / GGUF export
│   
├── runtime/
│   ├── tools.py                     # 6 consolidated @tool definitions with confirmation gates
│   ├── agent.py                     # smolagents.ToolCallingAgent with FedoraFormatBridge
│   └── test_agent.py                # Local integration & security test suite
├── README.md
└── requirements.txt             # Python dependencies
```

---

## Quickstart Guide

### 1. Environment Installation
```bash
python3 -m venv .venv 
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Capture Real Fedora CLI Outputs
```bash
python grounding/capture_tool_outputs.py
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

### 5. QLoRA Fine-Tuning
```bash
python -m training.train_qlora --train-data dataset/validated/train.json --eval-data dataset/validated/eval.json --epochs 3
```

### 6. Interactive Local Agent (smolagents)
```bash
# Run using local Ollama model
python -m runtime.agent --backend ollama --model smollm2-360m-fedora-agent

# Or load directly into PyTorch in-memory weights
python -m runtime.agent --backend transformers --model training/merged_model
```

---

## Running Unit & Security Tests
```bash
python -m unittest runtime.test_agent
```
