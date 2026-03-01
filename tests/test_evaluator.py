import pytest

from src.evaluator import (
    _score_concept_mention,
    _score_contradiction,
    _score_ev_awareness,
    _score_format,
    _score_frequency_awareness,
    _score_position_awareness,
    compute_rule_based_cheap,
    compute_rule_based_full,
    evaluate,
    run_gates,
)
from src.models import CoachingOutput, JudgeScores, SolverOutput
from tests.conftest import make_scenario, make_solver_scenario


def _gate(gates, name):
    return next(g for g in gates.gates if g.name == name)


# --- Gates ---

def test_gate_legality_pass(scenario, good_output):
    gate = _gate(run_gates(scenario, good_output), "legality")
    assert gate.passed is True


def test_gate_legality_fail(scenario, good_output):
    output = good_output.model_copy(deep=True)
    output.recommended_action = "bet"
    gate = _gate(run_gates(scenario, output), "legality")
    assert gate.passed is False


def test_gate_action_alignment_pass(scenario, good_output):
    gate = _gate(run_gates(scenario, good_output), "action_alignment")
    assert gate.passed is True


def test_gate_action_alignment_fail(scenario, wrong_action_output):
    gate = _gate(run_gates(scenario, wrong_action_output), "action_alignment")
    assert gate.passed is False


def test_gate_numerical_accuracy_pass(scenario, good_output):
    gate = _gate(run_gates(scenario, good_output), "numerical_accuracy")
    assert gate.passed is True


def test_gate_numerical_accuracy_fail(scenario, bad_numbers_output):
    gate = _gate(run_gates(scenario, bad_numbers_output), "numerical_accuracy")
    assert gate.passed is False


def test_gate_numerical_accuracy_pot_odds(scenario, good_output):
    output = good_output.model_copy(deep=True)
    output.explanation = output.explanation.replace("3:1", "2:1")
    gate = _gate(run_gates(scenario, output), "numerical_accuracy")
    assert gate.passed is False


def test_gate_required_evidence_pass(scenario, good_output):
    gate = _gate(run_gates(scenario, good_output), "required_evidence")
    assert gate.passed is True


def test_gate_required_evidence_fail(scenario, missing_evidence_output):
    gate = _gate(run_gates(scenario, missing_evidence_output), "required_evidence")
    assert gate.passed is False


def test_gates_work_on_solver_only_scenario(good_output):
    s = make_solver_scenario()
    gates = run_gates(s, good_output)
    assert gates.all_passed is True


def test_gate_pot_odds_na_skips_check():
    s = make_solver_scenario(
        solver_output={
            "recommended_action": "check",
            "action_frequencies": {"check": 0.7, "bet": 0.3},
            "equity": 0.34,
            "pot_odds": "N/A",
            "ev": 0.0,
        },
        legal_actions=["check", "bet"],
    )
    output = CoachingOutput(
        recommended_action="check",
        explanation="Check with 34% equity. No bet to call so pot odds are irrelevant.",
    )
    gate = _gate(run_gates(s, output), "numerical_accuracy")
    assert gate.passed is True


def test_any_gate_fail_composite_zero(scenario, wrong_action_output):
    result = evaluate(scenario, wrong_action_output)
    assert result.composite == 0.0
    assert result.rule_based is None


# --- Annotation-based heuristics ---

def test_concept_mention_all_found(scenario, good_output):
    assert _score_concept_mention(scenario, good_output) == 1.0


def test_concept_mention_partial(scenario, good_output):
    output = good_output.model_copy(deep=True)
    output.explanation = "Call because equity is 34%."
    assert _score_concept_mention(scenario, output) == pytest.approx(1 / 3, rel=1e-3)


def test_contradiction_none_triggered(scenario, good_output):
    assert _score_contradiction(scenario, good_output) == 1.0


def test_contradiction_all_triggered(scenario, contradiction_output):
    assert _score_contradiction(scenario, contradiction_output) == 0.0


def test_format_good_length():
    output = CoachingOutput(
        recommended_action="call",
        explanation=("call " + ("x" * 195)).strip(),
    )
    assert _score_format(output) == 1.0


def test_format_too_short():
    output = CoachingOutput(recommended_action="call", explanation="call now")
    assert _score_format(output) == 0.5


def test_format_missing_action():
    output = CoachingOutput(recommended_action="call", explanation="x" * 200)
    assert _score_format(output) == 0.8


# --- Solver-derived heuristics ---

def test_ev_awareness_significant_mentioned():
    s = make_solver_scenario(solver_output=SolverOutput(
        recommended_action="call", action_frequencies={"call": 1.0},
        equity=0.34, pot_odds="3:1", ev=15.0,
    ))
    output = CoachingOutput(recommended_action="call",
        explanation="Call. The EV of this play is +15.0, clearly profitable.")
    assert _score_ev_awareness(s, output) == 1.0


def test_ev_awareness_significant_missing():
    s = make_solver_scenario(solver_output=SolverOutput(
        recommended_action="call", action_frequencies={"call": 1.0},
        equity=0.34, pot_odds="3:1", ev=15.0,
    ))
    output = CoachingOutput(recommended_action="call",
        explanation="Call because equity is 34%.")
    assert _score_ev_awareness(s, output) == 0.0


def test_frequency_awareness_mixed_mentioned():
    s = make_solver_scenario(solver_output=SolverOutput(
        recommended_action="call", action_frequencies={"fold": 0.3, "call": 0.5, "raise": 0.2},
        equity=0.34, pot_odds="3:1", ev=4.2,
    ))
    output = CoachingOutput(recommended_action="call",
        explanation="Call. This is a mixed strategy spot — sometimes we fold too.")
    assert _score_frequency_awareness(s, output) == 1.0


def test_frequency_awareness_mixed_missing():
    s = make_solver_scenario(solver_output=SolverOutput(
        recommended_action="call", action_frequencies={"fold": 0.3, "call": 0.5, "raise": 0.2},
        equity=0.34, pot_odds="3:1", ev=4.2,
    ))
    output = CoachingOutput(recommended_action="call",
        explanation="Call because equity is 34%.")
    assert _score_frequency_awareness(s, output) == 0.0


def test_position_awareness_mentioned():
    s = make_solver_scenario()
    output = CoachingOutput(recommended_action="call",
        explanation="Call from the BTN with 34% equity.")
    assert _score_position_awareness(s, output) == 1.0


def test_position_awareness_missing():
    s = make_solver_scenario()
    output = CoachingOutput(recommended_action="call",
        explanation="Call because equity is 34%.")
    assert _score_position_awareness(s, output) == 0.0


# --- Composite scoring and eval modes ---

def test_rule_based_composite_weights(scenario, good_output):
    scores = compute_rule_based_full(scenario, good_output)
    expected = 0.5 * scores.concept_mention + 0.3 * scores.contradiction + 0.2 * scores.format_score
    assert scores.composite == pytest.approx(expected)


def test_composite_with_judge(scenario, good_output):
    judge = JudgeScores(coherence=3, readability=4, coaching_tone=5)
    result = evaluate(scenario, good_output, judge_scores=judge, full_eval=True)
    assert result.rule_based is not None
    expected = 0.65 * result.rule_based.composite + 0.35 * judge.normalized_avg()
    assert result.composite == pytest.approx(expected)


def test_compute_rule_based_cheap():
    s = make_solver_scenario()
    output = CoachingOutput(recommended_action="call",
        explanation="Call from the BTN. Mixed strategy — sometimes fold. EV is +4.2.")
    scores = compute_rule_based_cheap(s, output)
    assert scores.format_score is not None
    assert scores.ev_awareness is not None
    assert scores.frequency_awareness is not None
    assert scores.position_awareness is not None
    assert scores.concept_mention is None
    assert scores.contradiction is None


def test_compute_rule_based_full():
    s = make_scenario()
    output = CoachingOutput(recommended_action="call",
        explanation="Call with 34% equity and pot odds of 3:1. Flush draw.")
    scores = compute_rule_based_full(s, output)
    assert scores.concept_mention is not None
    assert scores.contradiction is not None
    assert scores.format_score is not None
    assert scores.ev_awareness is None
    assert scores.frequency_awareness is None
    assert scores.position_awareness is None


def test_evaluate_default_is_cheap(good_output):
    s = make_solver_scenario()
    result = evaluate(s, good_output)
    assert result.rule_based is not None
    assert result.rule_based.concept_mention is None
