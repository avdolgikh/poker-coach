from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# LLM provider: "openai" or "anthropic"
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# Two-tier model config: cheap model for coaching, expensive for judge/optimizer
PROD_MODEL: str = os.getenv("PROD_MODEL", "gpt-5-mini")
TUNER_MODEL: str = os.getenv("TUNER_MODEL", "gpt-5")

# Evaluation
NUMERICAL_TOLERANCE: float = 0.02  # +-2% for equity checks

HEURISTIC_WEIGHT: float = 0.65
JUDGE_WEIGHT: float = 0.35

# Full eval heuristic weights (annotation-based, used on holdout)
HEURISTIC_WEIGHTS = {
    "concept_mention": 0.5,
    "contradiction": 0.3,
    "format_score": 0.2,
}

# Cheap eval heuristic weights (solver-derived, used on training)
CHEAP_HEURISTIC_WEIGHTS = {
    "format_score": 0.25,
    "ev_awareness": 0.25,
    "frequency_awareness": 0.25,
    "position_awareness": 0.25,
}

# Soft-gate weights (DSPy engines — partial credit)
SOFT_GATE_WEIGHT = 0.30
SOFT_HEURISTIC_WEIGHT = 0.45
SOFT_JUDGE_WEIGHT = 0.25
SOFT_GATE_WEIGHT_NO_JUDGE = 0.35
SOFT_HEURISTIC_WEIGHT_NO_JUDGE = 0.65

DSPY_PROGRAMS_DIR = Path(__file__).resolve().parent.parent / "dspy_programs"

# Hard gates — all must pass or composite = 0.0
GATES = [
    {"name": "legality", "rule": "recommended action must be in the scenario's legal_actions list (case-insensitive)"},
    {"name": "action_alignment", "rule": "recommended action must match the solver's recommended_action"},
    {"name": "numerical_accuracy", "rule": f"equity within +-{NUMERICAL_TOLERANCE*100:.0f}% of solver_output.equity, pot odds must appear verbatim (skipped when N/A)"},
    {"name": "required_evidence", "rule": "explanation must mention equity (always) and pot odds (when solver_output.pot_odds != N/A)"},
]

# Format scoring: (operator, threshold_chars, penalty)
FORMAT_LENGTH_PENALTIES = [
    ("<", 50, 0.5),
    ("<", 100, 0.3),
    (">", 1500, 0.5),
    (">", 800, 0.3),
    (">", 500, 0.1),
]
FORMAT_MISSING_ACTION_PENALTY: float = 0.2
FORMAT_TARGET_RANGE = (200, 500)

# Data splits
DSPY_TRAIN_FRAC: float = 0.6    # DSPy engines: fraction of annotated data for training
EXAMPLES_POOL_FRAC: float = 0.1 # Manual engine: fraction of annotated data for examples pool

# Pipeline
NUM_ITERATIONS: int = 3
MAX_EDITS_PER_ITERATION: int = 3

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
