"""Tests for DSPy engines (GEPA + MIPROv2). Skipped until dspy is installed."""

import pytest

dspy = pytest.importorskip("dspy")

from src.engine_dspy import (
    PokerCoaching, PokerCoach, _to_dspy_example, _make_coach_fn,
    _build_metric, _build_mipro_metric,
    GEPAEngine, MIPROEngine,
)
from src.engine_protocol import EngineResult
from src.models import CoachingOutput
from tests.conftest import make_scenario, make_solver_scenario


# --- Shared infrastructure ---

def test_poker_coaching_signature_fields():
    sig = PokerCoaching
    assert "scenario" in sig.input_fields
    assert "recommended_action" in sig.output_fields
    assert "explanation" in sig.output_fields


def test_to_dspy_example_fields():
    s = make_scenario()
    ex = _to_dspy_example(s)
    assert hasattr(ex, "scenario")
    assert hasattr(ex, "scenario_id")
    assert hasattr(ex, "recommended_action")


def test_to_dspy_example_with_inputs():
    s = make_scenario()
    ex = _to_dspy_example(s)
    assert "scenario" in ex.inputs()


def test_make_coach_fn_returns_coaching_output():
    class FakeModule:
        def __call__(self, scenario):
            class Pred:
                recommended_action = "call"
                explanation = "Test explanation"
            return Pred()

    fn = _make_coach_fn(FakeModule())
    result = fn(make_solver_scenario())
    assert isinstance(result, CoachingOutput)
    assert result.recommended_action == "call"


# --- GEPA metric ---

def test_gepa_metric_returns_float_without_pred_name():
    """dspy.Evaluate calls metric without pred_name — must return float."""
    s = make_scenario()
    lookup = {s.id: s}
    metric = _build_metric(lookup, no_judge=True)

    class Gold:
        scenario_id = s.id
    class Pred:
        recommended_action = "call"
        explanation = "Call with 34% equity and 3:1 pot odds. Flush draw equity."

    result = metric(Gold(), Pred())
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_gepa_metric_returns_dict_with_pred_name():
    """GEPA's feedback_fn_creator calls metric with pred_name — must return dict."""
    s = make_scenario()
    lookup = {s.id: s}
    metric = _build_metric(lookup, no_judge=True)

    class Gold:
        scenario_id = s.id
    class Pred:
        recommended_action = "call"
        explanation = "Call with 34% equity and 3:1 pot odds. Flush draw equity."

    result = metric(Gold(), Pred(), pred_name="coach")
    assert "score" in result
    assert "feedback" in result


def test_gepa_metric_feedback_includes_gate_failures():
    s = make_scenario()
    lookup = {s.id: s}
    metric = _build_metric(lookup, no_judge=True)

    class Gold:
        scenario_id = s.id
    class Pred:
        recommended_action = "fold"  # wrong action → gate fails
        explanation = "Fold here."

    result = metric(Gold(), Pred(), pred_name="coach")
    assert "GATE FAIL" in result["feedback"]


# --- MIPROv2 metric ---

def test_mipro_metric_returns_float():
    s = make_scenario()
    lookup = {s.id: s}
    metric = _build_mipro_metric(lookup, no_judge=True)

    class Gold:
        scenario_id = s.id
    class Pred:
        recommended_action = "call"
        explanation = "Call with 34% equity and 3:1 pot odds."

    result = metric(Gold(), Pred(), trace=None)
    assert isinstance(result, float)


def test_mipro_metric_trace_returns_bool():
    s = make_scenario()
    lookup = {s.id: s}
    metric = _build_mipro_metric(lookup, no_judge=True)

    class Gold:
        scenario_id = s.id
    class Pred:
        recommended_action = "call"
        explanation = "Call with 34% equity and 3:1 pot odds."

    result = metric(Gold(), Pred(), trace="bootstrap")
    assert isinstance(result, bool)


# --- Engine integration ---

def test_gepa_engine_returns_engine_result(monkeypatch):
    s = make_scenario()
    train = [s]

    class FakeCompiled:
        class coach:
            class signature:
                instructions = "optimized prompt"
            demos = []
        def __call__(self, scenario):
            class Pred:
                recommended_action = "call"
                explanation = "test"
            return Pred()
        def save(self, path):
            pass

    class FakeGEPA:
        def __init__(self, **kwargs):
            pass
        def compile(self, module, trainset):
            return FakeCompiled()

    monkeypatch.setattr(dspy, "GEPA", FakeGEPA)
    monkeypatch.setattr(dspy, "configure", lambda **kw: None)

    engine = GEPAEngine()
    result = engine.optimize(train, "initial prompt", no_judge=True)
    assert isinstance(result, EngineResult)
    assert result.best_label == "gepa_best"


def test_mipro_engine_returns_engine_result(monkeypatch):
    s = make_scenario()
    train = [s]

    class FakeCompiled:
        class coach:
            class signature:
                instructions = "optimized prompt"
            demos = []
        def __call__(self, scenario):
            class Pred:
                recommended_action = "call"
                explanation = "test"
            return Pred()
        def save(self, path):
            pass

    class FakeMIPRO:
        def __init__(self, **kwargs):
            pass
        def compile(self, module, trainset, **kwargs):
            return FakeCompiled()

    monkeypatch.setattr(dspy, "MIPROv2", FakeMIPRO)
    monkeypatch.setattr(dspy, "configure", lambda **kw: None)

    engine = MIPROEngine()
    result = engine.optimize(train, "initial prompt", no_judge=True)
    assert isinstance(result, EngineResult)
    assert result.best_label == "mipro_best"
