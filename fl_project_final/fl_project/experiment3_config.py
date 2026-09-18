"""Centralized Configuration for Experiment 3.

Defines all parameters for the 10-client, 20-round, 8-seed benchmark:
  - Dynamic Heterogeneity Regimes (alphas 100.0, 1.0, 0.2)
  - Byzantine Schedule (Rounds 16-20 with 2 attackers)
  - Attack configuration (-3.0 * clean update, post-training)
  - 7 Canonical Methods (FixedFedAvg, FixedMedian, FixedTrimmedMean, FixedKrum, RuleBased, SingleShotLLM, ReflectiveAgent)
  - Oracle and Validation split parameters
  - Groq and Model settings (strict same-model comparison)
"""

import os
import sys
import platform
import numpy as np
import torch

# ==============================================================================
# EXPERIMENT 3 TOPOLOGY & HORIZON
# ==============================================================================
NUM_CLIENTS = 10
NUM_ROUNDS = 20
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]

# ==============================================================================
# DYNAMIC HETEROGENEITY & BYZANTINE REGIME SCHEDULE
# ==============================================================================
# Rounds 1–5:   alpha = 100.0 (Near-IID)
# Rounds 6–10:  alpha = 1.0   (Moderate heterogeneity)
# Rounds 11–15: alpha = 0.2   (Severe heterogeneity)
# Rounds 16–20: alpha = 0.2 + 2 Byzantine attackers
REGIME_SCHEDULE = [
    {"rounds": range(1, 6),   "alpha": 100.0, "attack_active": False, "num_attackers": 0, "name": "near_iid"},
    {"rounds": range(6, 11),  "alpha": 1.0,   "attack_active": False, "num_attackers": 0, "name": "moderate_non_iid"},
    {"rounds": range(11, 16), "alpha": 0.2,   "attack_active": False, "num_attackers": 0, "name": "severe_non_iid"},
    {"rounds": range(16, 21), "alpha": 0.2,   "attack_active": True,  "num_attackers": 2, "name": "severe_non_iid_byzantine"},
]

def get_regime_for_round(round_num: int) -> dict:
    """Return the regime configuration for a given round number."""
    for regime in REGIME_SCHEDULE:
        if round_num in regime["rounds"]:
            return regime
    raise ValueError(f"Round {round_num} outside defined schedule (1-{NUM_ROUNDS})")

# ==============================================================================
# BYZANTINE ATTACK CONFIGURATION
# ==============================================================================
ATTACK_TYPE = "scaled_opposite_update"
ATTACK_SCALE = -3.0  # malicious_update = -3.0 * clean_update
NUM_BYZANTINE = 2
BYZANTINE_BUDGET = 2  # Krum constraint: n >= 2m + 3 -> 10 >= 7

def get_attacker_ids(seed: int, num_clients: int = NUM_CLIENTS, num_attackers: int = NUM_BYZANTINE) -> list[int]:
    """Deterministically select attacker IDs matched across methods for the same seed."""
    rng = np.random.default_rng(seed + 7777)
    ids = list(range(num_clients))
    rng.shuffle(ids)
    return sorted(ids[:num_attackers])

# ==============================================================================
# CANONICAL METHODS (Baselines + Agentic AI)
# ==============================================================================
METHODS = [
    "FixedFedAvg",
    "FixedMedian",
    "FixedTrimmedMean",
    "FixedKrum",
    "RuleBased",
    "SingleShotLLM",
    "ReflectiveAgent",
    "AgenticAI",
]

# ==============================================================================
# ALGORITHMIC PARAMETERS
# ==============================================================================
HISTORY_LENGTH = 3
MAX_CANDIDATES = 10
TRIM_RATIO = 0.2
LOCAL_EPOCHS = 2
LOCAL_LR = 0.01
LOCAL_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 1024

# ==============================================================================
# VALIDATION SPLIT (TRAINING-DERIVED)
# ==============================================================================
VAL_SPLIT_SIZE = 5000  # 5,000 samples reserved from 50k train set for Oracle evaluation
VAL_SPLIT_SEED = 42

# Always force-load from .env file to ensure the latest API key is used
for env_candidate in [
    os.path.join(os.path.dirname(__file__), ".env"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
    ".env",
]:
    if os.path.exists(env_candidate):
        with open(env_candidate, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ[k.strip()] = v.strip()

DEFAULT_LLM_PROVIDER = os.environ.get("FL_LLM_PROVIDER", "groq")
DEFAULT_LLM_MODEL = os.environ.get("FL_LLM_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ==============================================================================
# GROQ QUOTA & SAFETY MARGIN CONFIGURATION
# ==============================================================================
GROQ_DAILY_TOKEN_LIMIT = int(os.environ.get("GROQ_DAILY_TOKEN_LIMIT", 200000))
GROQ_DAILY_REQUEST_LIMIT = int(os.environ.get("GROQ_DAILY_REQUEST_LIMIT", 1000))
GROQ_TOKEN_SAFETY_MARGIN = float(os.environ.get("GROQ_TOKEN_SAFETY_MARGIN", 0.90))
GROQ_REQUEST_SAFETY_MARGIN = float(os.environ.get("GROQ_REQUEST_SAFETY_MARGIN", 0.90))
GROQ_MIN_REMAINING_TOKENS = int(os.environ.get("GROQ_MIN_REMAINING_TOKENS", 1000))
GROQ_MIN_REMAINING_REQUESTS = int(os.environ.get("GROQ_MIN_REMAINING_REQUESTS", 20))

# ==============================================================================
# DATABASE
# ==============================================================================
DB_PATH = "fl_metrics_experiment3.db"

# ==============================================================================
# REPRODUCIBILITY METADATA
# ==============================================================================
def get_environment_metadata() -> dict:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "numpy_version": np.__version__,
        "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS,
        "seeds": SEEDS,
        "methods": METHODS,
        "trim_ratio": TRIM_RATIO,
        "byzantine_budget": BYZANTINE_BUDGET,
        "history_length": HISTORY_LENGTH,
        "attack_scale": ATTACK_SCALE,
        "llm_model": DEFAULT_LLM_MODEL,
        "llm_provider": DEFAULT_LLM_PROVIDER,
    }
