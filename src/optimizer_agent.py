"""Optimizer agent: analyzes evaluation failures, proposes prompt edits."""

import json

from src import config
from src.evaluator import get_eval_rules
from src.llm import generate_structured
from src.models import OptimizerOutput, PromptEdit

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert prompt optimizer for a poker coaching system. Your job is to analyze \
coaching evaluation results and propose specific, targeted edits to improve the coaching prompt.

You will receive:
1. The current coaching system prompt being used by the coaching agent
2. Aggregate metrics (gate pass rate, average composite score)
3. Gate failure breakdown (which hard gates fail most often)
4. Up to 5 representative failures with FULL context:
   - The poker scenario (hand, board, pot, stack, position, villain action)
   - The solver's recommendation (action, equity, pot odds, EV, action frequencies)
   - What the coaching agent actually produced (action + explanation)
   - Hard gate results (legality, action_alignment, numerical_accuracy, required_evidence)
   - Rule-based heuristic scores (concept_mention, contradiction, format)
   - LLM judge scores (coherence, readability, coaching_tone)

Your task: diagnose WHY the coaching prompt is producing these failures and propose up to 3 \
concrete edits to fix them.

Scoring rubric (what the evaluator measures — ALL gates must pass or composite = 0.0):
{rubric}

Rules:
- Study each failure carefully: compare the solver data against what the coach wrote
- Each edit must address a specific, observed failure pattern — NO vague improvements
- Focus on the most impactful failures first (gate failures before quality improvements)
- Be specific: "Always state equity as a percentage matching the solver output" NOT "improve numbers"
- Each edit has: diagnosis (what's wrong), change (what to add/modify), expected_effect (measurable improvement)
- Edits are appended as constraints to the prompt, so phrase changes as clear directives\
"""

BLOAT_CAP = 2000


def _format_failure(i: int, f: dict) -> str:
    lines = [f"\n--- Failure {i} (scenario {f.get('scenario_id', '?')}, composite: {f.get('composite', 0):.2f}) ---"]

    gs = f.get("game_state", {})
    if gs:
        lines.append(
            f"  Game: {gs.get('hand')} on {gs.get('board') or '(preflop)'} | "
            f"{gs.get('street')} | {gs.get('position')} | "
            f"pot {gs.get('pot')} stack {gs.get('stack')} | villain: {gs.get('villain_action')}"
        )

    so = f.get("solver_output", {})
    if so:
        freqs = so.get("action_frequencies", {})
        freq_str = ", ".join(f"{k}: {v:.0%}" for k, v in freqs.items()) if freqs else ""
        lines.append(
            f"  Solver: {so.get('recommended_action')} | "
            f"equity {so.get('equity', 0):.1%} | odds {so.get('pot_odds')} | "
            f"EV {so.get('ev')} | freqs: {freq_str}"
        )

    if f.get("legal_actions"):
        lines.append(f"  Legal actions: {f['legal_actions']}")
    if f.get("required_concepts"):
        lines.append(f"  Required concepts: {f['required_concepts']}")
    if f.get("key_numbers"):
        lines.append(f"  Key numbers: {f['key_numbers']}")
    if f.get("contradiction_rules"):
        lines.append(f"  Contradiction rules: {f['contradiction_rules']}")

    lines.append(f"  Coach action: {f.get('coaching_action', '?')}")
    lines.append(f"  Coach explanation: {f.get('coaching_explanation', '?')}")

    gate_results = f.get("gate_results", [])
    if gate_results:
        gate_strs = []
        for g in gate_results:
            status = "PASS" if g["passed"] else "FAIL"
            detail = f" ({g['detail']})" if g.get("detail") else ""
            gate_strs.append(f"{g['name']}: {status}{detail}")
        lines.append(f"  Gates: {' | '.join(gate_strs)}")

    rb = f.get("rule_based_scores")
    if rb:
        is_full = "concept_mention" in rb
        if is_full:
            lines.append(
                f"  Heuristics (Annotated): concept={rb['concept_mention']:.2f} "
                f"contra={rb['contradiction']:.2f} "
                f"fmt={rb['format_score']:.2f} "
                f"(composite={rb['composite']:.2f})"
            )
        else:
            lines.append(
                f"  Heuristics (Solver-Only): ev={rb.get('ev_awareness', 0):.2f} "
                f"freq={rb.get('frequency_awareness', 0):.2f} "
                f"pos={rb.get('position_awareness', 0):.2f} "
                f"fmt={rb.get('format_score', 0):.2f} "
                f"(composite={rb.get('composite', 0):.2f})"
            )

    js = f.get("judge_scores")
    if js:
        lines.append(
            f"  Judge: coherence={js['coherence']}/5 "
            f"readability={js['readability']}/5 "
            f"coaching_tone={js['coaching_tone']}/5"
        )

    return "\n".join(lines)


def _format_gold_example(example) -> str:
    gs = example.game_state
    so = example.solver_output
    lines = [
        "--- Example ---",
        f"  Game: {gs.hand} on {gs.board or '(preflop)'} | {gs.street} | {gs.position} | "
        f"pot {gs.pot} stack {gs.stack} | villain: {gs.villain_action}",
        f"  Solver: {so.recommended_action} | equity {so.equity:.0%} | "
        f"odds {so.pot_odds} | EV {so.ev:+.1f}",
    ]
    if hasattr(example, "reference_reasoning") and example.reference_reasoning:
        lines.append(f"  Expert coaching:\n    {example.reference_reasoning}")
    if hasattr(example, "required_concepts") and example.required_concepts:
        lines.append(f"  Why this is good: covers {', '.join(example.required_concepts)}")
    return "\n".join(lines)


def _format_context(
    current_prompt: str,
    metrics: dict,
    gate_failure_breakdown: dict,
    representative_failures: list,
    example=None,
) -> str:
    breakdown = "\n".join(
        f"- {gate}: {count} failures"
        for gate, count in gate_failure_breakdown.items()
    )

    failures_text = "\n".join(
        _format_failure(i, f)
        for i, f in enumerate(representative_failures[:5], 1)
    )

    text = (
        f"Current Coaching System Prompt:\n{current_prompt}\n\n"
        f"Aggregate Metrics:\n"
        f"- Gate pass rate: {metrics.get('gate_pass_rate', 0):.1%}\n"
        f"- Average composite score: {metrics.get('avg_composite', 0):.2f}\n\n"
        f"Gate Failure Breakdown:\n{breakdown}\n\n"
        f"Representative Failures (worst cases with full context):\n{failures_text}"
    )

    if example:
        text += f"\n\nGold-Standard Example (target quality):\n{_format_gold_example(example)}"

    return text


def propose_edits(
    current_prompt: str,
    metrics: dict,
    gate_failure_breakdown: dict,
    representative_failures: list,
    example=None,
) -> OptimizerOutput:
    user_msg = _format_context(
        current_prompt, metrics, gate_failure_breakdown, representative_failures,
        example=example,
    )
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(rubric=json.dumps(get_eval_rules(), indent=2))
    result = generate_structured(
        system_prompt, user_msg, OptimizerOutput,
        model=config.TUNER_MODEL,
    )
    result.edits = result.edits[:config.MAX_EDITS_PER_ITERATION]
    return result


def apply_edits(prompt: str, edits: list[PromptEdit]) -> str:
    new_items = [f"- {e.change}" for e in edits]

    if "## Constraints" in prompt:
        idx = prompt.index("## Constraints")
        base = prompt[:idx].rstrip("\n")
        constraints_text = prompt[idx + len("## Constraints"):]
        existing = [
            line for line in constraints_text.strip().split("\n")
            if line.strip().startswith("- ")
        ]
    else:
        base = prompt
        existing = []

    all_constraints = existing + new_items

    def _build(constraints: list[str]) -> str:
        block = "## Constraints\n" + "\n".join(constraints)
        return f"{base}\n\n{block}"

    result = _build(all_constraints)

    while len(result) > BLOAT_CAP and len(all_constraints) > len(new_items):
        all_constraints.pop(0)
        result = _build(all_constraints)

    return result
