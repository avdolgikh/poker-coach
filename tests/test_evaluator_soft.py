"""Tests for soft-gate evaluation (partial credit when gates fail)."""

import pytest

from src import config
from src.evaluator import evaluate_soft, compute_composite_soft
from src.models import (
    CoachingOutput, GateResult, GateResults, JudgeScores, RuleBasedScores,
)
from tests.conftest import make_scenario


def _all_pass_gates(n=4):
    return GateResults(
        gates=[GateResult(name=f"g{i}", passed=True, detail="") for i in range(n)],
        all_passed=True,
    )


def _partial_gates(n_pass, n_total=4):
    gates = ([GateResult(name=f"g{i}", passed=True, detail="") for i in range(n_pass)]
             + [GateResult(name=f"g{i}", passed=False, detail="fail") for i in range(n_pass, n_total)])
    return GateResults(gates=gates, all_passed=(n_pass == n_total))


def _rb(composite=0.8):
    return RuleBasedScores(format_score=0.9, ev_awareness=0.8, composite=composite)


def test_soft_all_gates_pass_no_judge():
    gates = _all_pass_gates()
    rb = _rb(0.8)
    result = compute_composite_soft(gates, rb, judge=None)
    expected = config.SOFT_GATE_WEIGHT_NO_JUDGE * 1.0 + config.SOFT_HEURISTIC_WEIGHT_NO_JUDGE * 0.8
    assert abs(result - expected) < 1e-9


def test_soft_all_gates_pass_with_judge():
    gates = _all_pass_gates()
    rb = _rb(0.8)
    judge = JudgeScores(coherence=4, readability=4, coaching_tone=4)
    result = compute_composite_soft(gates, rb, judge=judge)
    expected = (config.SOFT_GATE_WEIGHT * 1.0
                + config.SOFT_HEURISTIC_WEIGHT * 0.8
                + config.SOFT_JUDGE_WEIGHT * judge.normalized_avg())
    assert abs(result - expected) < 1e-9


def test_soft_partial_gate_failure_nonzero():
    gates = _partial_gates(2, 4)  # 2/4 pass
    rb = _rb(0.8)
    result = compute_composite_soft(gates, rb, judge=None)
    assert result > 0.0
    # gate_fraction = 0.5
    expected = config.SOFT_GATE_WEIGHT_NO_JUDGE * 0.5 + config.SOFT_HEURISTIC_WEIGHT_NO_JUDGE * 0.8
    assert abs(result - expected) < 1e-9


def test_soft_all_gates_fail_floor():
    gates = _partial_gates(0, 4)  # 0/4 pass
    rb = _rb(0.0)
    result = compute_composite_soft(gates, rb, judge=None)
    assert result == 0.01


def test_soft_proportional():
    """More gates passing → higher score."""
    rb = _rb(0.5)
    score_3of4 = compute_composite_soft(_partial_gates(3, 4), rb)
    score_1of4 = compute_composite_soft(_partial_gates(1, 4), rb)
    assert score_3of4 > score_1of4


def test_evaluate_soft_always_computes_heuristics():
    """Even when gates fail, heuristics are computed (unlike hard-gate evaluate)."""
    scenario = make_scenario()
    # Wrong action → action_alignment gate fails
    output = CoachingOutput(
        recommended_action="fold",
        explanation="Fold with 34% equity and 3:1 pot odds in BTN position.",
    )
    result = evaluate_soft(scenario, output, full_eval=False)
    assert not result.gates.all_passed
    assert result.rule_based is not None  # heuristics computed despite gate failure
    assert result.composite > 0.0  # partial credit, not zeroed out
