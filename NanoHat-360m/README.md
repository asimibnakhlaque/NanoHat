---

license: apache-2.0
base_model: HuggingFaceTB/SmolLM2-360M-Instruct
tags:
- agent
- tool-use
- fedora
- linux
- qlora
- smollm2
- desktop-agent
- function-calling

---

# 🤖 NanoHat-360M

A lightweight **Fedora Linux desktop AI agent** fine-tuned from `SmolLM2-360M-Instruct` for multi-turn system interaction, structured tool calling, and Linux desktop assistance.

> **Project:** [NanoHat](https://github.com/asimibnakhlaque/NanoHat)

The model is designed to operate through the NanoHat runtime, which provides the `smolagents` execution harness, FedoraFormatBridge, six consolidated tools, and security-hardened system execution.

## ✨ Highlights

* Fedora/Linux-focused desktop assistance
* Structured `<thought>` and `<tool_call>` generation
* Multi-tool diagnostic and system workflows
* Advanced Fedora/Linux sysadmin coverage
* Semantic cross-run dataset deduplication
* Canonical JSON validation and auto-rescue
* Argument alias and synonym normalization
* Assistant-only loss masking during fine-tuning
* Security-focused subprocess execution

## 🛠️ Supported Tools

| Tool                              | Purpose                                                                                                  |
| --------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `calculator(expression)`          | Mathematical evaluations                                                                                 |
| `web_search(query)`               | Live web search queries                                                                                  |
| `user_memory(action, key, value)` | Persistent user preference storage                                                                       |
| `reminder(task, time_or_delay)`   | Desktop notifications and delayed reminders                                                              |
| `system_health(target)`           | CPU, RAM, disk, process, battery, and system telemetry                                                   |
| `system_action(action, target)`   | Controlled OS actions including Wi-Fi, Bluetooth, application launching, screenshots, and screen locking |

## 🐧 Fedora/Linux Coverage

The training curriculum covers a broad range of desktop and system-administration workflows, including:

* SELinux and `ausearch`
* `firewalld` configuration
* SSH keys, forwarding, and host aliases
* Git rebase/stash conflict recovery
* Python PEP 668 / virtual environments
* Btrfs snapshots and LVM storage
* GRUB, suspend, and power troubleshooting
* Wayland, PipeWire, OBS Studio, and fractional scaling
* Wacom tablets, USB-C docks, and webcams
* DNS, SMB/NFS, and NetworkManager configuration
* `dnf5`, system upgrades, COPR, and `rpm-ostree`
* Bluetooth, audio, VPN, Podman, multi-monitor, and other desktop troubleshooting scenarios

## 🧠 Dataset & Training

The model was fine-tuned on a curated **1,094-sample golden dataset** containing validated multi-turn Fedora/Linux interaction trajectories.

The dataset was generated through a structured curriculum covering:

* Single-tool interactions
* No-tool conversations
* Multi-tool workflows
* Edge cases and recovery scenarios
* Advanced Fedora/Linux system-administration workflows

Multiple generation runs were consolidated into a unified golden dataset using semantic cross-run deduplication and canonical validation.

The validation pipeline includes Fedora-grounded tool outputs, novelty prompting, JSON syntax auto-rescue, argument synonym normalization, turn-structure correction, and deterministic JSON re-serialization.

## 🔐 Runtime Safety

NanoHat uses a security-hardened subprocess execution model:

* No `os.system()`
* No `shell=True`
* Discrete subprocess argument lists
* Process-name validation
* Confirmation gates for destructive operations
* Explicit Bluetooth state checks before power changes

The model itself should **not** be treated as a security boundary. Runtime-side validation and authorization remain responsible for enforcing safe system actions.

## 🚀 Quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "blackpirates/NanoHat-360M"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=(
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float32
    ),
    device_map="auto",
)

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful AI agent running on Fedora Linux. "
            "Available tools: [calculator, web_search, user_memory, "
            "reminder, system_health, system_action]."
        ),
    },
    {
        "role": "user",
        "content": "Check my CPU usage and set a reminder to drink water in 20 minutes.",
    },
]

input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.2,
)

print(
    tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
)
```

## 📊 Training Details

* **Base Model:** `HuggingFaceTB/SmolLM2-360M-Instruct`
* **Method:** QLoRA
* **LoRA Rank:** 16
* **LoRA Alpha:** 32
* **Target Modules:** All linear projections
* **Loss Masking:** Loss computed exclusively on assistant-generated turns (`<thought>`, `<tool_call>`, and assistant responses)
* **Epochs:** 3
* **Final Evaluation Loss:** 0.8567

## ⚠️ Limitations

This is a **360M-parameter specialized agent model**, not a general-purpose frontier language model.

Performance may degrade on:

* Unseen operating systems
* Complex general reasoning tasks
* Tasks outside the trained toolset
* Unsupported tool arguments or workflows
* Long-horizon autonomous tasks

For real system operations, always validate model-generated actions at the runtime/tool layer before execution.

