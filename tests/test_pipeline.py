import pytest

from src import config, pipeline, prompt_store
from src.models import (
    CoachingOutput,
    EvalResult,
    GateResult,
    GateResults,
    RuleBasedScores,
    JudgeScores,
    OptimizerOutput,
    PromptEdit,
    ScenarioAnnotated,
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


def _setup_pipeline_mocks(monkeypatch, marker_to_append: str):
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

    monkeypatch.setattr(pipeline, "run_batch", fake_run_batch)
    monkeypatch.setattr(pipeline, "evaluate", fake_evaluate)
    monkeypatch.setattr(pipeline, "judge_coaching", fake_judge_coaching)
    monkeypatch.setattr(pipeline, "propose_edits", fake_propose_edits)
    monkeypatch.setattr(pipeline, "apply_edits", fake_apply_edits)


def _mk_train_holdout():
    train = [make_solver_scenario() for _ in range(3)]
    holdout = [make_scenario() for _ in range(2)]
    return train, holdout


@pytest.fixture(autouse=True)
def _isolated_prompt_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_store, "PROMPTS_DIR", tmp_path / "prompts")


def test_pipeline_runs_3_iterations(monkeypatch):
    train, holdout = _mk_train_holdout()
    _setup_pipeline_mocks(monkeypatch, marker_to_append="IMPROVED")
    result = pipeline.run_pipeline(train, holdout, initial_prompt="v0 prompt")
    assert len(result["iterations"]) == config.NUM_ITERATIONS


def test_pipeline_accepts_improvement(monkeypatch):
    train, holdout = _mk_train_holdout()
    _setup_pipeline_mocks(monkeypatch, marker_to_append="IMPROVED")
    result = pipeline.run_pipeline(train, holdout, initial_prompt="v0 prompt")
    assert result["accepted"] is True


def test_pipeline_rejects_regression(monkeypatch):
    train, holdout = _mk_train_holdout()
    _setup_pipeline_mocks(monkeypatch, marker_to_append="WORSE")
    result = pipeline.run_pipeline(train, holdout, initial_prompt="v0 prompt")
    assert result["accepted"] is False


def test_pipeline_optimizer_receives_failure_data(monkeypatch):
    train, holdout = _mk_train_holdout()
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

    monkeypatch.setattr(pipeline, "run_batch", failing_run_batch)
    monkeypatch.setattr(pipeline, "evaluate", mixed_evaluate)
    monkeypatch.setattr(pipeline, "judge_coaching",
                        lambda gs, so, la, co: JudgeScores(coherence=4, readability=4, coaching_tone=4))
    monkeypatch.setattr(pipeline, "propose_edits", capturing_propose_edits)
    monkeypatch.setattr(pipeline, "apply_edits", lambda p, e: p + "\n## Constraints\n- c")

    pipeline.run_pipeline(train, holdout, initial_prompt="v0 prompt")

    assert len(propose_calls) > 0
    call = propose_calls[0]
    assert "gate_pass_rate" in call["metrics"]
    assert len(call["gate_failure_breakdown"]) > 0
    assert len(call["representative_failures"]) > 0


def test_pipeline_training_uses_cheap_eval(monkeypatch):
    train, holdout = _mk_train_holdout()
    eval_calls = []

    def tracking_evaluate(scenario, output, judge_scores=None, full_eval=False):
        eval_calls.append({"scenario_id": scenario.id, "full_eval": full_eval})
        return _mk_eval(scenario.id, composite=0.7, all_passed=True)

    _setup_pipeline_mocks(monkeypatch, marker_to_append="IMPROVED")
    monkeypatch.setattr(pipeline, "evaluate", tracking_evaluate)
    pipeline.run_pipeline(train, holdout, initial_prompt="v0 prompt")

    train_ids = {s.id for s in train}
    train_evals = [c for c in eval_calls if c["scenario_id"] in train_ids]
    assert all(c["full_eval"] is False for c in train_evals)


def test_pipeline_holdout_uses_full_eval(monkeypatch):
    train, holdout = _mk_train_holdout()
    eval_calls = []

    def tracking_evaluate(scenario, output, judge_scores=None, full_eval=False):
        eval_calls.append({"scenario_id": scenario.id, "full_eval": full_eval})
        return _mk_eval(scenario.id, composite=0.7, all_passed=True)

    _setup_pipeline_mocks(monkeypatch, marker_to_append="IMPROVED")
    monkeypatch.setattr(pipeline, "evaluate", tracking_evaluate)
    pipeline.run_pipeline(train, holdout, initial_prompt="v0 prompt")

    holdout_ids = {s.id for s in holdout}
    holdout_evals = [c for c in eval_calls if c["scenario_id"] in holdout_ids]
    assert all(c["full_eval"] is True for c in holdout_evals)


def test_pipeline_passes_example_to_optimizer(monkeypatch):
    train, holdout = _mk_train_holdout()
    examples = [make_scenario() for _ in range(3)]
    propose_calls = []

    def tracking_propose(current_prompt, metrics, gate_failure_breakdown,
                         representative_failures, example=None):
        propose_calls.append({"example": example})
        return OptimizerOutput(
            edits=[PromptEdit(diagnosis="d", change="c", expected_effect="e")]
        )

    _setup_pipeline_mocks(monkeypatch, marker_to_append="IMPROVED")
    monkeypatch.setattr(pipeline, "propose_edits", tracking_propose)
    monkeypatch.setattr(pipeline.random, "random", lambda: 0.01)

    pipeline.run_pipeline(train, holdout, initial_prompt="v0 prompt",
                          examples_pool=examples)

    assert all(c["example"] is not None for c in propose_calls)


def test_pipeline_solver_only_omits_annotations(monkeypatch):
    train = [make_solver_scenario() for _ in range(2)]
    holdout = [make_scenario() for _ in range(2)]
    propose_calls = []

    def tracking_propose(current_prompt, metrics, gate_failure_breakdown,
                         representative_failures, example=None):
        propose_calls.append(representative_failures)
        return OptimizerOutput(
            edits=[PromptEdit(diagnosis="d", change="c", expected_effect="e")]
        )

    _setup_pipeline_mocks(monkeypatch, marker_to_append="IMPROVED")
    monkeypatch.setattr(pipeline, "propose_edits", tracking_propose)

    pipeline.run_pipeline(train, holdout, initial_prompt="v0 prompt")

    assert len(propose_calls) > 0
    for failure in propose_calls[0]:
        assert "required_concepts" not in failure
        assert "key_numbers" not in failure
        assert "contradiction_rules" not in failure
