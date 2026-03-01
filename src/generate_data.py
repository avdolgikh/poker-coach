"""Scenario generation: LLM-powered with validation and retry."""

import json
import re
from pathlib import Path

from src import config
from src.llm import generate_structured
from src.models import GameState, SolverOutput, Scenario, ScenarioAnnotated

SYSTEM_PROMPT = """\
You are a poker scenario generator for a coaching system. Generate realistic Texas Hold'em \
heads-up poker scenarios with complete data for evaluating coaching explanations.

Requirements:
- Vary streets (preflop, flop, turn, river), positions, hand strengths, stack depths, board textures
- Include realistic equity values, pot odds, and action frequencies
- Write reference_reasoning as a gold-standard coaching explanation (2-4 sentences)
- required_concepts: list poker concepts a good explanation MUST mention (e.g. "flush draw", "pot odds")
- key_numbers: numbers the explanation must reference accurately (equity as "XX%", pot_odds as "X:1")
- contradiction_rules: phrases that would indicate factual errors if present in an explanation
- recommended_action must be in legal_actions
- Equity must be between 0.0 and 1.0. Note: turn flush draws are ~20%, flop ~35-45%.
- pot_odds must be a string like "2.5:1". If villain checks, use "N/A".
- action_frequencies must sum to approximately 1.0
- Cards on the board must not overlap with hole cards
- Board card count must match street: preflop=0, flop=3, turn=4, river=5
- Card format: rank + suit letter, e.g. "Ah", "Kd", "Tc", "9s". Ranks: 2-9, T, J, Q, K, A. Suits: c, d, h, s.
- Hand must be exactly 2 space-separated cards, e.g. "Ah Kd"
- villain_action should specify the amount if it's a bet/raise, e.g. "bets 50" or "shoves for 100".
"""


def _generation_prompt(batch_num: int, count: int) -> str:
    streets = ["preflop", "flop", "turn", "river"]
    positions = ["BTN", "SB", "BB", "CO", "MP", "UTG"]
    actions = ["fold", "call", "raise", "check", "bet"]

    return f"""\
Generate scenario batch #{batch_num} with {count} unique poker scenarios.

DIVERSITY IS MANDATORY — distribute evenly across all categories:
- Streets: {streets} — use each roughly equally, do NOT over-represent any single street
- Positions: {positions} — rotate through all positions, avoid defaulting to BTN
- Recommended actions: {actions} — MUST include at least one "check" recommendation (villain checked, hero checks back)
- Hand strengths: mix of strong made hands, draws, marginal hands, bluffs
- Stack depths: vary from short (15-30 BB) to medium (50-80 BB) to deep (100+ BB)
- Board textures: dry, wet, paired, monotone, rainbow, connected

Note: Do NOT provide an 'id' field. I will assign IDs automatically.
Return each scenario as a JSON object matching the Scenario schema (excluding 'id').
"""


def _get_bet_size(action: str) -> float:
    match = re.search(r'(\d+\.?\d*)', action)
    return float(match.group(1)) if match else 0.0


def validate_scenario(s: Scenario) -> list[str]:
    errors = []

    if s.solver_output.recommended_action.lower() not in [a.lower() for a in s.legal_actions]:
        errors.append(f"recommended_action '{s.solver_output.recommended_action}' not in legal_actions")

    if not 0.0 <= s.solver_output.equity <= 1.0:
        errors.append(f"equity {s.solver_output.equity} out of [0,1] range")

    freq_sum = sum(s.solver_output.action_frequencies.values())
    if abs(freq_sum - 1.0) > 0.05:
        errors.append(f"action_frequencies sum to {freq_sum}, expected ~1.0")

    if isinstance(s, ScenarioAnnotated):
        if not s.required_concepts:
            errors.append("required_concepts is empty")
        if not s.key_numbers:
            errors.append("key_numbers is empty")

    hand_parts = s.game_state.hand.split()
    if len(hand_parts) != 2:
        errors.append(f"hand must be exactly 2 space-separated cards, got: '{s.game_state.hand}'")

    card_pattern = re.compile(r'^[2-9TJQKA][cdhs]$')
    all_cards = list(hand_parts)
    if s.game_state.board:
        all_cards += s.game_state.board.split()
    bad_cards = [c for c in all_cards if not card_pattern.match(c)]
    if bad_cards:
        errors.append(f"invalid card format: {bad_cards} (expected e.g. 'Ah', 'Td', '9s')")

    hand_cards = set(s.game_state.hand.split())
    board_cards = set(s.game_state.board.split()) if s.game_state.board else set()
    overlap = hand_cards & board_cards
    if overlap:
        errors.append(f"card overlap between hand and board: {overlap}")

    street = s.game_state.street.lower()
    expected_board = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
    if street in expected_board:
        actual = len(board_cards)
        if actual != expected_board[street]:
            errors.append(f"board has {actual} cards but street '{street}' expects {expected_board[street]}")

    villain = s.game_state.villain_action.lower()
    bet_size = _get_bet_size(s.game_state.villain_action)
    is_facing_bet = bet_size > 0 or any(w in villain for w in ["bet", "raise", "all-in", "allin", "shove"])

    if is_facing_bet and "check" in [a.lower() for a in s.legal_actions]:
        errors.append("legal_actions includes 'check' but villain bet/raised")

    no_bet = any(w in villain for w in ["none", "check"])
    if no_bet and "call" in [a.lower() for a in s.legal_actions]:
        errors.append("legal_actions includes 'call' but there is no bet to call")

    if is_facing_bet and bet_size > 0 and s.game_state.stack < bet_size:
        errors.append(f"stack ({s.game_state.stack}) < villain bet ({bet_size})")

    if is_facing_bet:
        if s.solver_output.pot_odds == "N/A":
            errors.append("pot_odds is 'N/A' but villain bet/raised")
        elif bet_size > 0:
            calc_ratio = (s.game_state.pot + bet_size) / bet_size
            match = re.search(r'(\d+\.?\d*):1', s.solver_output.pot_odds)
            if match:
                llm_ratio = float(match.group(1))
                if abs(llm_ratio - calc_ratio) > 0.5:
                    errors.append(f"pot_odds hallucination: LLM says {s.solver_output.pot_odds}, math says {calc_ratio:.2f}:1")
    else:
        if s.solver_output.pot_odds != "N/A":
            errors.append(f"zombie pot_odds: villain checked but pot_odds is '{s.solver_output.pot_odds}'")

    if isinstance(s, ScenarioAnnotated):
        concepts = [c.lower() for c in s.required_concepts]
        if "flush draw" in concepts and "made flush" not in concepts:
            if street == "turn" and s.solver_output.equity > 0.40:
                errors.append(f"Equity {s.solver_output.equity} too high for turn flush draw (max ~35%)")
            if street == "flop" and s.solver_output.equity > 0.60:
                errors.append(f"Equity {s.solver_output.equity} too high for flop flush draw (max ~55%)")

    return errors


def generate_scenarios(
    count: int,
    existing: list[Scenario] | None = None,
    solver_only: bool = False,
) -> list[Scenario]:
    from pydantic import BaseModel

    if solver_only:
        class ScenarioGen(BaseModel):
            game_state: GameState
            legal_actions: list[str]
            solver_output: SolverOutput

        model_cls = Scenario
    else:
        class ScenarioGen(BaseModel):
            game_state: GameState
            legal_actions: list[str]
            solver_output: SolverOutput
            reference_reasoning: str
            required_concepts: list[str]
            key_numbers: dict[str, str]
            contradiction_rules: list[str]

        model_cls = ScenarioAnnotated

    class ScenarioListGen(BaseModel):
        scenarios: list[ScenarioGen]

    existing = existing or []
    next_id_num = len(existing) + 1
    all_scenarios: list[Scenario] = []
    batch_num = 1
    max_retries = 3

    while len(all_scenarios) < count and max_retries > 0:
        remaining = count - len(all_scenarios)
        prompt = _generation_prompt(batch_num, remaining)

        try:
            result = generate_structured(SYSTEM_PROMPT, prompt, ScenarioListGen, model=config.PROD_MODEL)
            for s_gen in result.scenarios:
                scenario_id = f"s{next_id_num:03d}"
                s = model_cls(id=scenario_id, **s_gen.model_dump(exclude={"id"}))

                errors = validate_scenario(s)
                if errors:
                    print(f"  [WARN] Scenario candidate failed validation: {errors}")
                else:
                    all_scenarios.append(s)
                    print(f"  [OK] Scenario {s.id}")
                    next_id_num += 1
        except Exception as e:
            print(f"  [ERROR] Generation failed: {e}")
            max_retries -= 1

        batch_num += 1

    return all_scenarios[:count]


def save_dataset(scenarios: list[Scenario], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [s.model_dump() for s in scenarios]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved {len(scenarios)} scenarios to {path}")


def load_dataset(path: Path, solver_only: bool = False) -> list[Scenario]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    cls = Scenario if solver_only else ScenarioAnnotated
    return [cls.model_validate(d) for d in data]


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.generate_data <count> [--solver-only]")
        sys.exit(1)

    count = int(sys.argv[1])
    solver_only = "--solver-only" in sys.argv

    if solver_only:
        path = config.DATA_DIR / "scenarios.json"
    else:
        path = config.DATA_DIR / "scenarios-annotated.json"

    existing = load_dataset(path, solver_only=solver_only)
    mode_label = "solver-only" if solver_only else "annotated"
    print(f"Generating {count} {mode_label} scenarios (have {len(existing)} existing)...")

    new_scenarios = generate_scenarios(count, existing=existing, solver_only=solver_only)
    all_scenarios = existing + new_scenarios

    save_dataset(all_scenarios, path)
    print(f"Done: {len(new_scenarios)} new, {len(all_scenarios)} total")


if __name__ == "__main__":
    main()
