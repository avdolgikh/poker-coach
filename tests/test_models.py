from src.models import Scenario, ScenarioAnnotated
from tests.conftest import make_scenario, make_solver_scenario


def test_scenario_has_only_solver_fields():
    s = make_solver_scenario()
    assert hasattr(s, "id")
    assert hasattr(s, "game_state")
    assert hasattr(s, "legal_actions")
    assert hasattr(s, "solver_output")
    assert not hasattr(s, "required_concepts")
    assert not hasattr(s, "key_numbers")
    assert not hasattr(s, "contradiction_rules")


def test_annotated_extends_scenario():
    a = make_scenario()
    assert isinstance(a, ScenarioAnnotated)
    assert isinstance(a, Scenario)
    assert hasattr(a, "required_concepts")
    assert hasattr(a, "key_numbers")
    assert hasattr(a, "contradiction_rules")
