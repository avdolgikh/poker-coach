from unittest.mock import patch

from pydantic import BaseModel

from src.generate_data import generate_scenarios, validate_scenario
from src.models import Scenario, ScenarioAnnotated
from tests.conftest import make_scenario, make_solver_scenario


def test_valid_scenario_passes(scenario):
    assert validate_scenario(scenario) == []


def test_action_not_in_legal_actions(scenario):
    s = scenario.model_copy(deep=True)
    s.solver_output.recommended_action = "bet"
    assert any("not in legal_actions" in error for error in validate_scenario(s))


def test_equity_out_of_range(scenario):
    s = scenario.model_copy(deep=True)
    s.solver_output.equity = 1.2
    assert any("out of [0,1] range" in error for error in validate_scenario(s))


def test_freq_sum_off(scenario):
    s = scenario.model_copy(deep=True)
    s.solver_output.action_frequencies = {"fold": 0.4, "call": 0.4, "raise": 0.4}
    assert any("sum to" in error for error in validate_scenario(s))


def test_hand_format_single_card(scenario):
    s = scenario.model_copy(deep=True)
    s.game_state.hand = "Ah"
    assert any("exactly 2 space-separated cards" in error for error in validate_scenario(s))


def test_card_overlap(scenario):
    s = scenario.model_copy(deep=True)
    s.game_state.board = "Ah 7d 2c"
    assert any("card overlap" in error for error in validate_scenario(s))


def test_board_length_mismatch(scenario):
    s = scenario.model_copy(deep=True)
    s.game_state.street = "turn"
    s.game_state.board = "Jd 7d 2c"
    assert any("expects 4" in error for error in validate_scenario(s))


def test_check_facing_bet(scenario):
    s = scenario.model_copy(deep=True)
    s.game_state.villain_action = "bets 20"
    s.legal_actions = ["fold", "call", "check"]
    assert any("includes 'check'" in error for error in validate_scenario(s))


def test_validate_solver_only_skips_annotation_checks():
    s = make_solver_scenario()
    assert validate_scenario(s) == []


@patch("src.generate_data.generate_structured")
def test_generate_scenarios_returns_requested_count(mock_llm):
    valid = [make_scenario(), make_scenario(), make_scenario()]

    class ScenarioList(BaseModel):
        scenarios: list[ScenarioAnnotated]

    mock_llm.return_value = ScenarioList(scenarios=valid)
    result = generate_scenarios(3)
    assert len(result) == 3
    for s in result:
        assert validate_scenario(s) == []
