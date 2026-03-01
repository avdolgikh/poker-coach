from unittest.mock import patch

from src import config
from src.coaching_agent import _format_context, generate_explanation, run_batch
from src.models import CoachingOutput
from tests.conftest import make_scenario


def test_format_context_contains_all_fields(scenario):
    text = _format_context(scenario)
    assert "Hand: Ah Kd" in text
    assert "Board: Jd 7d 2c" in text
    assert "Pot: 50.0" in text
    assert "Stack: 300.0" in text
    assert "Equity: 34.0%" in text
    assert "Pot odds: 3:1" in text
    assert "Action frequencies: fold: 10%, call: 60%, raise: 30%" in text
    # Annotation fields must NOT leak to the coaching LLM
    assert "Key numbers" not in text
    assert "Required concepts" not in text
    assert "contradiction" not in text


@patch("src.coaching_agent.generate_structured")
def test_generate_explanation_calls_llm(mock_llm, scenario, v0_prompt):
    mock_llm.return_value = CoachingOutput(recommended_action="call", explanation="test")
    generate_explanation(scenario, v0_prompt)
    assert mock_llm.call_count == 1
    assert mock_llm.call_args.kwargs["model"] == config.PROD_MODEL


@patch("src.coaching_agent.generate_explanation")
def test_run_batch_returns_none_on_error(mock_generate, good_output):
    scenarios = [make_scenario(), make_scenario()]
    mock_generate.side_effect = [good_output, RuntimeError("boom")]
    result = run_batch(scenarios, "prompt")
    assert result[0] is good_output
    assert result[1] is None
