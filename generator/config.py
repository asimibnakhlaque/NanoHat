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

# Budget Guard Settings
TOTAL_BUDGET_USD = 1.00
# Estimated cost per batch (5 conversations) in USD
ESTIMATED_COST_PER_BATCH_USD = 0.006

# Quotas across Curriculum Phases
PHASE_QUOTAS = {
    # Phase 1: Single-Tool Mastery (6 tools * 6 batches * 5 convos = 180 samples)
    "single_tool_batches_per_tool": 6,
    # Phase 2: No-Tool Pure Chat (8 batches * 5 convos = 40 samples)
    "no_tool_batches": 8,
    # Phase 3: Multi-Tool Chaining (12 batches * 5 convos = 60 samples)
    "multi_tool_batches": 12,
    # Phase 4: Edge Cases & Recovery (10 batches * 5 convos = 50 samples)
    "edge_case_batches": 10,
}

OUTPUT_RAW_DIR = "dataset/raw_batches"
OUTPUT_VALIDATED_DIR = "dataset/validated"
GROUNDING_FIXTURES_PATH = "grounding/real_tool_outputs.json"

# NEW: Path to existing golden dataset for deduplication
EXISTING_DATASET_PATH = "dataset/validated/golden_dataset.json"
