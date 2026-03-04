"""Tests for ManualEngine (moved from test_pipeline.py + new tests)."""

import pytest

from src import config, prompt_store
from src import engine_manual
from src.engine_manual import ManualEngine
from src.engine_protocol import EngineResult
from src.models import (
    CoachingOutput,
    EvalResult,
    GateResult,
    GateResults,
    RuleBasedScores,
    JudgeScores,
    OptimizerOutput,
    PromptEdit,
)
from tests.conftest import make_scenario, make_solver_scenario


def _mk_eval(scenario_id: str, composite: float, all_passed: bool = True) -> EvalResult:
    gates = GateResults(
        gates=[GateResult(name="legality", passed=all_passed, detail="")],
        all_passed=all_passed,
    )
    rule_based = None
    if all_passed:
        rule_based = RuleBasedScores(
            concept_mention=1.0,
            contradiction=1.0,
            format_score=1.0,
            composite=composite,
        )
    return EvalResult(
        scenario_id=scenario_id,
        gates=gates,
        rule_based=rule_based,
        composite=composite if all_passed else 0.0,
    )


def _setup_engine_mocks(monkeypatch, marker_to_append: str):
    def fake_run_batch(scenarios, prompt):
        marker = "BASE"
        if "IMPROVED" in prompt:
            marker = "IMPROVED"
        elif "WORSE" in prompt:
            marker = "WORSE"
        return [
            CoachingOutput(
                recommended_action="call",
                explanation=f"Call with 34% equity and 3:1 pot odds. {marker}",
            )
            for _ in scenarios
        ]

    def fake_evaluate(scenario, output, judge_scores=None, full_eval=False):
        if "WORSE" in output.explanation:
            score = 0.2
        elif "IMPROVED" in output.explanation:
            score = 0.9
        else:
            score = 0.6
        return _mk_eval(scenario.id, composite=score, all_passed=True)

    def fake_judge_coaching(game_state, solver_output, legal_actions, coaching_output):
        return JudgeScores(coherence=4, readability=4, coaching_tone=4)

    def fake_propose_edits(current_prompt, metrics, gate_failure_breakdown,
                           representative_failures, example=None):
        return OptimizerOutput(
            edits=[PromptEdit(diagnosis="d", change="c", expected_effect="e")]
        )

    def fake_apply_edits(prompt, edits):
        return f"{prompt}\n{marker_to_append}"

    def fake_generate_explanation(scenario, prompt):
        marker = "IMPROVED" if "IMPROVED" in prompt else "BASE"
        return CoachingOutput(
            recommended_action="call",
            explanation=f"Call with 34% equity and 3:1 pot odds. {marker}",
        )

    monkeypatch.setattr(engine_manual, "run_batch", fake_run_batch)
    monkeypatch.setattr(engine_manual, "evaluate", fake_evaluate)
    monkeypatch.setattr(engine_manual, "judge_coaching", fake_judge_coaching)
    monkeypatch.setattr(engine_manual, "propose_edits", fake_propose_edits)
    monkeypatch.setattr(engine_manual, "apply_edits", fake_apply_edits)
    monkeypatch.setattr(engine_manual, "generate_explanation", fake_generate_explanation)


def _mk_train():
    return [make_solver_scenario() for _ in range(3)]


@pytest.fixture(autouse=True)
def _isolated_prompt_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_store, "PROMPTS_DIR", tmp_path / "prompts")


# --- Moved from test_pipeline.py ---

def test_manual_training_uses_cheap_eval(monkeypatch):
    train = _mk_train()
    eval_calls = []

    def tracking_evaluate(scenario, output, judge_scores=None, full_eval=False):
        eval_calls.append({"scenario_id": scenario.id, "full_eval": full_eval})
        return _mk_eval(scenario.id, composite=0.7, all_passed=True)

    _setup_engine_mocks(monkeypatch, marker_to_append="IMPROVED")
    monkeypatch.setattr(engine_manual, "evaluate", tracking_evaluate)
    ManualEngine().optimize(train, "v0 prompt")

    train_ids = {s.id for s in train}
    train_evals = [c for c in eval_calls if c["scenario_id"] in train_ids]
    assert all(c["full_eval"] is False for c in train_evals)


def test_manual_optimizer_receives_failure_data(monkeypatch):
    train = _mk_train()
    propose_calls = []

    def failing_run_batch(scenarios, prompt):
        return [
            CoachingOutput(
                recommended_action="call",
                explanation="Call with 34% equity and 3:1 pot odds.",
            )
            for _ in scenarios
        ]

    def mixed_evaluate(scenario, output, judge_scores=None, full_eval=False):
        if scenario is train[0]:
            return _mk_eval(scenario.id, composite=0.0, all_passed=False)
        return _mk_eval(scenario.id, composite=0.4, all_passed=True)

    def capturing_propose_edits(current_prompt, metrics, gate_failure_breakdown,
                                representative_failures, example=None):
        propose_calls.append({
            "metrics": metrics,
            "gate_failure_breakdown": gate_failure_breakdown,
            "representative_failures": representative_failures,
        })
        return OptimizerOutput(
            edits=[PromptEdit(diagnosis="d", change="c", expected_effect="e")]
        )

    monkeypatch.setattr(engine_manual, "run_batch", failing_run_batch)
    monkeypatch.setattr(engine_manual, "evaluate", mixed_evaluate)
    monkeypatch.setattr(engine_manual, "judge_coaching",
                        lambda gs, so, la, co: JudgeScores(coherence=4, readability=4, coaching_tone=4))
    monkeypatch.setattr(engine_manual, "propose_edits", capturing_propose_edits)
    monkeypatch.setattr(engine_manual, "apply_edits", lambda p, e: p + "\n## Constraints\n- c")
    monkeypatch.setattr(engine_manual, "generate_explanation",
                        lambda s, p: CoachingOutput(recommended_action="call", explanation="test"))

    ManualEngine().optimize(train, "v0 prompt")

    assert len(propose_calls) > 0
    call = propose_calls[0]
    assert "gate_pass_rate" in call["metrics"]
    assert len(call["gate_failure_breakdown"]) > 0
    assert len(call["representative_failures"]) > 0


def test_manual_passes_example_to_optimizer(monkeypatch):
    train = _mk_train()
    examples = [make_scenario() for _ in range(3)]
    propose_calls = []

    def tracking_propose(current_prompt, metrics, gate_failure_breakdown,
                         representative_failures, example=None):
        propose_calls.append({"example": example})
        return OptimizerOutput(
            edits=[PromptEdit(diagnosis="d", change="c", expected_effect="e")]
        )

    _setup_engine_mocks(monkeypatch, marker_to_append="IMPROVED")
    monkeypatch.setattr(engine_manual, "propose_edits", tracking_propose)
    monkeypatch.setattr(engine_manual.random, "random", lambda: 0.01)

    ManualEngine().optimize(train, "v0 prompt", examples_pool=examples)

    assert all(c["example"] is not None for c in propose_calls)


def test_manual_solver_only_omits_annotations(monkeypatch):
    train = [make_solver_scenario() for _ in range(2)]
    propose_calls = []

    def tracking_propose(current_prompt, metrics, gate_failure_breakdown,
                         representative_failures, example=None):
        propose_calls.append(representative_failures)
        return OptimizerOutput(
            edits=[PromptEdit(diagnosis="d", change="c", expected_effect="e")]
        )

    _setup_engine_mocks(monkeypatch, marker_to_append="IMPROVED")
    monkeypatch.setattr(engine_manual, "propose_edits", tracking_propose)

    ManualEngine().optimize(train, "v0 prompt")

    assert len(propose_calls) > 0
    for failure in propose_calls[0]:
        assert "required_concepts" not in failure
        assert "key_numbers" not in failure
        assert "contradiction_rules" not in failure


# --- New tests ---

def test_manual_returns_engine_result(monkeypatch):
    train = _mk_train()
    _setup_engine_mocks(monkeypatch, marker_to_append="IMPROVED")
    result = ManualEngine().optimize(train, "v0 prompt")
    assert isinstance(result, EngineResult)
    assert result.best_label in ("v0", "v1", "v2", "v3")
    assert callable(result.coach_fn)


def test_manual_coach_fn_generates_output(monkeypatch):
    train = _mk_train()
    _setup_engine_mocks(monkeypatch, marker_to_append="IMPROVED")
    result = ManualEngine().optimize(train, "v0 prompt")
    output = result.coach_fn(train[0])
    assert output is not None
    assert output.recommended_action == "call"


def test_manual_no_judge_skips_judge(monkeypatch):
    train = _mk_train()
    judge_calls = []

    def tracking_judge(game_state, solver_output, legal_actions, coaching_output):
        judge_calls.append(True)
        return JudgeScores(coherence=4, readability=4, coaching_tone=4)

    _setup_engine_mocks(monkeypatch, marker_to_append="IMPROVED")
    monkeypatch.setattr(engine_manual, "judge_coaching", tracking_judge)

    ManualEngine().optimize(train, "v0 prompt", no_judge=True)

    assert len(judge_calls) == 0
