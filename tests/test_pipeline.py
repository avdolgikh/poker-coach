import pytest

from src import config, pipeline, prompt_store
from src.models import (
    CoachingOutput,
    EvalResult,
    GateResult,
    GateResults,
    RuleBasedScores,
    JudgeScores,
)
from src.engine_protocol import EngineResult
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


class FakeEngine:
    """Canned engine that returns a fixed EngineResult."""

    def __init__(self, coach_marker="IMPROVED", best_label="v1",
                 best_train_composite=0.8, iterations=None):
        self.coach_marker = coach_marker
        self.best_label = best_label
        self.best_train_composite = best_train_composite
        self._iterations = iterations or [
            {"version": f"v{i+1}", "gate_pass_rate": 1.0,
             "avg_composite": 0.8, "edits": ["c"]}
            for i in range(config.NUM_ITERATIONS)
        ]

    def optimize(self, train, initial_prompt, **kwargs):
        marker = self.coach_marker

        def coach_fn(scenario):
            return CoachingOutput(
                recommended_action="call",
                explanation=f"Call with 34% equity and 3:1 pot odds. {marker}",
            )

        return EngineResult(
            coach_fn=coach_fn,
            iterations=self._iterations,
            best_train_composite=self.best_train_composite,
            best_label=self.best_label,
        )


def _setup_holdout_mocks(monkeypatch):
    """Mock holdout-related functions (run_batch, evaluate, judge for pipeline)."""

    def fake_run_batch(scenarios, prompt):
        return [
            CoachingOutput(
                recommended_action="call",
                explanation="Call with 34% equity and 3:1 pot odds. BASE",
            )
            for _ in scenarios
        ]

    def fake_evaluate(scenario, output, judge_scores=None, full_eval=False):
        return _mk_eval(scenario.id, composite=0.6, all_passed=True)

    def fake_judge_coaching(game_state, solver_output, legal_actions, coaching_output):
        return JudgeScores(coherence=4, readability=4, coaching_tone=4)

    monkeypatch.setattr(pipeline, "run_batch", fake_run_batch)
    monkeypatch.setattr(pipeline, "evaluate", fake_evaluate)
    monkeypatch.setattr(pipeline, "judge_coaching", fake_judge_coaching)


def _mk_train_holdout():
    train = [make_solver_scenario() for _ in range(3)]
    holdout = [make_scenario() for _ in range(2)]
    return train, holdout


@pytest.fixture(autouse=True)
def _isolated_prompt_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_store, "PROMPTS_DIR", tmp_path / "prompts")


def test_pipeline_runs_3_iterations(monkeypatch):
    train, holdout = _mk_train_holdout()
    _setup_holdout_mocks(monkeypatch)
    engine = FakeEngine()
    result = pipeline.run_pipeline(engine, train, holdout, initial_prompt="v0 prompt")
    assert len(result["iterations"]) == config.NUM_ITERATIONS


def test_pipeline_accepts_improvement(monkeypatch):
    train, holdout = _mk_train_holdout()

    def improving_evaluate(scenario, output, judge_scores=None, full_eval=False):
        if "IMPROVED" in output.explanation:
            return _mk_eval(scenario.id, composite=0.9, all_passed=True)
        return _mk_eval(scenario.id, composite=0.6, all_passed=True)

    _setup_holdout_mocks(monkeypatch)
    monkeypatch.setattr(pipeline, "evaluate", improving_evaluate)

    engine = FakeEngine(coach_marker="IMPROVED")
    result = pipeline.run_pipeline(engine, train, holdout, initial_prompt="v0 prompt")
    assert result["accepted"] is True


def test_pipeline_rejects_regression(monkeypatch):
    train, holdout = _mk_train_holdout()

    def worsening_evaluate(scenario, output, judge_scores=None, full_eval=False):
        if "WORSE" in output.explanation:
            return _mk_eval(scenario.id, composite=0.2, all_passed=True)
        return _mk_eval(scenario.id, composite=0.6, all_passed=True)

    _setup_holdout_mocks(monkeypatch)
    monkeypatch.setattr(pipeline, "evaluate", worsening_evaluate)

    engine = FakeEngine(coach_marker="WORSE")
    result = pipeline.run_pipeline(engine, train, holdout, initial_prompt="v0 prompt")
    assert result["accepted"] is False


def test_pipeline_holdout_uses_full_eval(monkeypatch):
    train, holdout = _mk_train_holdout()
    eval_calls = []

    def tracking_evaluate(scenario, output, judge_scores=None, full_eval=False):
        eval_calls.append({"scenario_id": scenario.id, "full_eval": full_eval})
        return _mk_eval(scenario.id, composite=0.7, all_passed=True)

    _setup_holdout_mocks(monkeypatch)
    monkeypatch.setattr(pipeline, "evaluate", tracking_evaluate)

    engine = FakeEngine()
    pipeline.run_pipeline(engine, train, holdout, initial_prompt="v0 prompt")

    holdout_ids = {s.id for s in holdout}
    holdout_evals = [c for c in eval_calls if c["scenario_id"] in holdout_ids]
    assert all(c["full_eval"] is True for c in holdout_evals)
