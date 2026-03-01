from unittest.mock import patch

from src.judge_agent import judge_coaching
from src.models import JudgeScores


@patch("src.judge_agent.generate_structured")
def test_judge_returns_scores_1_to_5(mock_llm, scenario, good_output):
    mock_llm.return_value = JudgeScores(coherence=3, readability=4, coaching_tone=5)
    result = judge_coaching(
        scenario.game_state, scenario.solver_output,
        scenario.legal_actions, good_output,
    )
    assert 1 <= result.coherence <= 5
    assert 1 <= result.readability <= 5
    assert 1 <= result.coaching_tone <= 5


def test_judge_normalized_avg():
    scores = JudgeScores(coherence=3, readability=4, coaching_tone=5)
    assert scores.normalized_avg() == 0.8
