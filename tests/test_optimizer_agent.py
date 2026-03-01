from unittest.mock import patch

from src import config
from src.models import OptimizerOutput, PromptEdit
from src.optimizer_agent import _format_context, apply_edits, propose_edits
from tests.conftest import make_scenario


def _edit(change: str) -> PromptEdit:
    return PromptEdit(
        diagnosis="diagnosis",
        change=change,
        expected_effect="effect",
    )


def test_apply_edits_appends_constraints():
    prompt = "Base prompt text."
    updated = apply_edits(prompt, [_edit("Always cite equity and pot odds.")])
    assert "## Constraints" in updated
    assert "- Always cite equity and pot odds." in updated


def test_apply_edits_multiple_edits():
    prompt = "Base prompt text."
    edits = [_edit("A"), _edit("B"), _edit("C")]
    updated = apply_edits(prompt, edits)
    assert updated.count("\n- ") == 3


def test_apply_edits_bloat_cap():
    base = "B" * 1960
    prompt = f"{base}\n\n## Constraints\n- oldest\n- newer"
    updated = apply_edits(prompt, [_edit("latest")])
    assert len(updated) <= 2000
    assert "oldest" not in updated


@patch("src.optimizer_agent.generate_structured")
def test_optimizer_max_3_edits(mock_llm):
    mock_llm.return_value = OptimizerOutput(
        edits=[_edit("1"), _edit("2"), _edit("3"), _edit("4")]
    )
    result = propose_edits(
        current_prompt="prompt",
        metrics={"gate_pass_rate": 0.5},
        gate_failure_breakdown={"action_alignment": 2},
        representative_failures=[],
    )
    assert len(result.edits) <= config.MAX_EDITS_PER_ITERATION


def test_format_context_with_example():
    example = make_scenario(reference_reasoning="Expert says call here.")
    text = _format_context(
        current_prompt="prompt",
        metrics={"gate_pass_rate": 0.5, "avg_composite": 0.3},
        gate_failure_breakdown={},
        representative_failures=[],
        example=example,
    )
    assert "Gold-Standard Example" in text
    assert "Expert says call here." in text
