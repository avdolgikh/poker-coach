from pydantic import BaseModel


# --- Scenario data (two-tier: solver-only base + annotated extension) ---

class GameState(BaseModel):
    hand: str  # e.g. "Ah Kd"
    board: str  # e.g. "Jd 7d 2c"
    pot: float
    stack: float
    position: str  # e.g. "BTN", "BB", "CO"
    street: str  # "preflop", "flop", "turn", "river"
    villain_action: str  # e.g. "bets 50"


class SolverOutput(BaseModel):
    recommended_action: str
    action_frequencies: dict[str, float]  # e.g. {"fold": 0.1, "call": 0.6, "raise": 0.3}
    equity: float  # 0.0 - 1.0
    pot_odds: str  # e.g. "3:1" or "N/A" when no bet
    ev: float


class Scenario(BaseModel):
    """Solver-only base — cheap to generate, used for training."""
    id: str
    game_state: GameState
    legal_actions: list[str]
    solver_output: SolverOutput


class ScenarioAnnotated(Scenario):
    """Extends Scenario with human annotations — expensive, used for holdout."""
    required_concepts: list[str]
    key_numbers: dict[str, str]
    contradiction_rules: list[str]
    reference_reasoning: str = ""


# --- Coaching output ---

class CoachingOutput(BaseModel):
    recommended_action: str
    explanation: str


# --- Evaluation ---

class GateResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class GateResults(BaseModel):
    gates: list[GateResult]
    all_passed: bool


class RuleBasedScores(BaseModel):
    """Heuristic scores — cheap fields are None during full eval, and vice versa."""
    # Full eval (annotation-based)
    concept_mention: float | None = None
    contradiction: float | None = None
    # Cheap eval (solver-derived)
    ev_awareness: float | None = None
    frequency_awareness: float | None = None
    position_awareness: float | None = None
    # Always present
    format_score: float | None = None
    composite: float


class JudgeScores(BaseModel):
    """LLM-as-a-judge scores (1-5 each)."""
    coherence: int
    readability: int
    coaching_tone: int

    def normalized_avg(self) -> float:
        return (self.coherence + self.readability + self.coaching_tone) / 3.0 / 5.0


class EvalResult(BaseModel):
    scenario_id: str
    gates: GateResults
    rule_based: RuleBasedScores | None = None  # None if gates failed
    judge: JudgeScores | None = None  # None if gates failed or judge not called
    composite: float  # 0.0 if gates failed


# --- Optimizer ---

class PromptEdit(BaseModel):
    diagnosis: str
    change: str
    expected_effect: str


class OptimizerOutput(BaseModel):
    edits: list[PromptEdit]


# --- Prompt versioning ---

class PromptVersion(BaseModel):
    version: str
    prompt_text: str
    metadata: dict = {}


# --- Failure Detector ---

class ToolCall(BaseModel):
    """A single tool invocation in the DA pipeline trace."""
    tool_name: str
    input: dict
    output: dict


class ToolTrace(BaseModel):
    """Ordered sequence of tool calls for one hand."""
    calls: list[ToolCall]


class Claim(BaseModel):
    """A factual claim extracted from coaching output."""
    text: str
    claim_type: str  # "numeric", "categorical", "action", "conceptual"
    topic: str  # maps to evidence contract key


class ClaimVerification(BaseModel):
    """Result of verifying a single claim against a tool output."""
    claim: Claim
    matched_tool: str | None
    verdict: str  # "supported", "contradicted", "unsupported", "unverifiable"
    reasoning: str
    severity: str  # "hard_fail", "soft_fail", "info"


class FailureLabel(BaseModel):
    """A single labeled failure (for ground truth or prediction)."""
    failure_type: str  # "contradiction", "skipped_tool", "concept_misapplication"
    claim_text: str | None = None
    tool_name: str | None = None
    detail: str


class FailureVerdict(BaseModel):
    """Complete failure detection result for one hand."""
    claims: list[Claim]
    verifications: list[ClaimVerification]
    skipped_tools: list[str]
    failures: list[FailureLabel]
    has_hard_failure: bool
    overall_pass: bool
    summary: str


class HandPackage(BaseModel):
    """A hand with coaching output, tool trace, and optional ground truth."""
    scenario_id: str
    coaching_output: CoachingOutput
    tool_trace: ToolTrace
    labels: list[FailureLabel] | None = None
