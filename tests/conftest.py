import itertools
from pathlib import Path

import pytest

from src.models import CoachingOutput, GameState, Scenario, ScenarioAnnotated, SolverOutput


_id_counter = itertools.count()

_SOLVER_DEFAULTS = dict(
    game_state=GameState(
        hand="Ah Kd",
        board="Jd 7d 2c",
        pot=50.0,
        stack=300.0,
        position="BTN",
        street="flop",
        villain_action="bets 25",
    ),
    legal_actions=["fold", "call", "raise"],
    solver_output=SolverOutput(
        recommended_action="call",
        action_frequencies={"fold": 0.1, "call": 0.6, "raise": 0.3},
        equity=0.34,
        pot_odds="3:1",
        ev=4.2,
    ),
)


def make_scenario(**overrides) -> ScenarioAnnotated:
    """Factory for annotated test scenarios. Backward compat — all existing tests use this."""
    idx = next(_id_counter)
    defaults = dict(
        id=f"test-{idx}",
        **_SOLVER_DEFAULTS,
        reference_reasoning="Call due to equity and pot odds alignment.",
        required_concepts=["equity", "pot odds", "flush draw"],
        key_numbers={"equity": "34%", "pot_odds": "3:1"},
        contradiction_rules=["must fold", "no outs"],
    )
    defaults.update(overrides)
    return ScenarioAnnotated(**defaults)


def make_solver_scenario(**overrides) -> Scenario:
    """Factory for solver-only test scenarios (no annotations)."""
    idx = next(_id_counter)
    defaults = dict(
        id=f"test-solver-{idx}",
        **_SOLVER_DEFAULTS,
    )
    defaults.update(overrides)
    return Scenario(**defaults)


@pytest.fixture
def scenario() -> ScenarioAnnotated:
    return make_scenario()


@pytest.fixture
def good_output() -> CoachingOutput:
    return CoachingOutput(
        recommended_action="call",
        explanation=(
            "You should call here because your equity is 34% with a flush draw, "
            "and the pot odds are 3:1, which supports continuing."
        ),
    )


@pytest.fixture
def wrong_action_output() -> CoachingOutput:
    return CoachingOutput(
        recommended_action="fold",
        explanation="Fold despite the equity and pot odds.",
    )


@pytest.fixture
def bad_numbers_output() -> CoachingOutput:
    return CoachingOutput(
        recommended_action="call",
        explanation=(
            "Call because your equity is 55% and pot odds are 3:1, "
            "so this is clearly profitable."
        ),
    )


@pytest.fixture
def missing_evidence_output() -> CoachingOutput:
    return CoachingOutput(
        recommended_action="call",
        explanation="Call because this hand has enough playability and pressure potential.",
    )


@pytest.fixture
def contradiction_output() -> CoachingOutput:
    return CoachingOutput(
        recommended_action="call",
        explanation=(
            "You should call. But you must fold because you have no outs, "
            "which conflicts with the recommendation."
        ),
    )


@pytest.fixture
def v0_prompt() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "prompts" / "v0.txt").read_text(encoding="utf-8")
