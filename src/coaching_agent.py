"""Coaching agent: generates poker coaching explanations via LLM."""

from src import config
from src.llm import generate_structured
from src.models import Scenario, CoachingOutput


def _format_context(scenario: Scenario) -> str:
    """Format scenario into the user message for the coaching LLM."""
    gs = scenario.game_state
    so = scenario.solver_output
    freqs = ", ".join(f"{k}: {v:.0%}" for k, v in so.action_frequencies.items())
    return (
        f"Game State:\n"
        f"- Hand: {gs.hand}\n"
        f"- Board: {gs.board or '(preflop)'}\n"
        f"- Street: {gs.street}\n"
        f"- Position: {gs.position}\n"
        f"- Pot: {gs.pot}\n"
        f"- Stack: {gs.stack}\n"
        f"- Villain action: {gs.villain_action}\n\n"
        f"Legal actions: {', '.join(scenario.legal_actions)}\n\n"
        f"Solver Output:\n"
        f"- Recommended action: {so.recommended_action}\n"
        f"- Equity: {so.equity:.1%}\n"
        f"- Pot odds: {so.pot_odds}\n"
        f"- EV: {so.ev}\n"
        f"- Action frequencies: {freqs}"
    )


def generate_explanation(scenario: Scenario, system_prompt: str) -> CoachingOutput:
    user_msg = _format_context(scenario)
    return generate_structured(
        system_prompt, user_msg, CoachingOutput,
        model=config.PROD_MODEL,
    )


def run_batch(
    scenarios: list[Scenario], prompt: str,
) -> list[CoachingOutput | None]:
    """Run coaching on all scenarios. Returns None for failures."""
    results = []
    for i, s in enumerate(scenarios):
        try:
            output = generate_explanation(s, prompt)
            print(f"  [{i+1}/{len(scenarios)}] {s.id}: {output.recommended_action}")
            results.append(output)
        except Exception as e:
            print(f"  [{i+1}/{len(scenarios)}] {s.id}: ERROR - {e}")
            results.append(None)
    return results
