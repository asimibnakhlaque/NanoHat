"""
generator/config.py
Configuration parameters for synthetic data generation using DeepSeek V4 Pro.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# DeepSeek API Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
TEACHER_MODEL = "deepseek-v4-pro"

# Sampling Parameters (Strict JSON & schema compliance)
TEMPERATURE = 0.7
TOP_P = 0.95
SAMPLES_PER_BATCH = 5

# Concurrency & Budget Settings
DEFAULT_MAX_WORKERS = 6
DEFAULT_BUDGET_USD = 1.50
TOTAL_BUDGET_USD = DEFAULT_BUDGET_USD

# Token Pricing in USD per Million Tokens (DeepSeek V4)
ESTIMATED_INPUT_PRICE_PER_M = 0.27
ESTIMATED_OUTPUT_PRICE_PER_M = 1.10
ESTIMATED_COST_PER_BATCH_USD = 0.006

# Targeted Run 4 Batch Quotas (250 batches = 1,250 conversations)
# Cures under-represented actions while reinforcing multi-tool flows and error recovery
TARGETED_RUN4_QUOTAS = {
    "starved_desktop_actions": 60,
    "starved_network_and_diagnostics": 50,
    "rich_reminders_and_calculator": 45,
    "complex_multi_tool_chains": 45,
    "tool_failure_and_recovery": 30,
    "negative_pure_chat": 20,
}

# Classical 4-Phase Curriculum Quotas
CURRICULUM_PHASE_QUOTAS = {
    "single_tool_batches_per_tool": 8,
    "no_tool_batches": 12,
    "multi_tool_batches": 30,
    "edge_case_batches": 30,
}
# Backward compatibility alias
PHASE_QUOTAS = CURRICULUM_PHASE_QUOTAS

# Output & Grounding Paths
OUTPUT_RAW_DIR = "dataset/raw_batches"
OUTPUT_VALIDATED_DIR = "dataset/validated"
GROUNDING_FIXTURES_PATH = "grounding/real_tool_outputs.json"
EXISTING_DATASET_PATH = "dataset/validated/golden_dataset_merged.json"
