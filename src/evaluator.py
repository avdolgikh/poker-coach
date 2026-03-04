"""Evaluator: hard gates + rule-based heuristics + LLM-as-a-judge composite score."""

import re

from src import config
from src.models import (
    Scenario, CoachingOutput, GateResult, GateResults,
    RuleBasedScores, JudgeScores, EvalResult,
)


# --- Hard gates (all must pass or composite = 0.0) ---

def _gate_legality(scenario: Scenario, output: CoachingOutput) -> GateResult:
    """Action must be in legal_actions."""
    legal = [a.lower() for a in scenario.legal_actions]
    passed = output.recommended_action.lower() in legal
    detail = "" if passed else f"'{output.recommended_action}' not in {scenario.legal_actions}"
    return GateResult(name="legality", passed=passed, detail=detail)


def _gate_action_alignment(scenario: Scenario, output: CoachingOutput) -> GateResult:
    """Action must match the solver's recommendation."""
    expected = scenario.solver_output.recommended_action.lower()
    actual = output.recommended_action.lower()
    passed = actual == expected
    detail = "" if passed else f"expected '{expected}', got '{actual}'"
    return GateResult(name="action_alignment", passed=passed, detail=detail)


def _gate_numerical_accuracy(scenario: Scenario, output: CoachingOutput) -> GateResult:
    """Equity and pot odds in the explanation must match solver_output."""
    errors = []
    text = output.explanation

    # Equity check: find closest percentage in text, compare to solver
    target = scenario.solver_output.equity * 100
    found_pcts = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*%", text)]
    if found_pcts:
        closest = min(found_pcts, key=lambda x: abs(x - target))
        tol = config.NUMERICAL_TOLERANCE * 100
        if abs(closest - target) > tol:
            errors.append(
                f"equity: closest {closest}% vs expected {target}% (>{tol}% tolerance)"
            )

    # Pot odds check: must appear verbatim (skip when N/A)
    if scenario.solver_output.pot_odds != "N/A":
        expected_odds = scenario.solver_output.pot_odds
        if expected_odds not in text:
            errors.append(f"pot odds '{expected_odds}' not found")

    passed = len(errors) == 0
    return GateResult(name="numerical_accuracy", passed=passed, detail="; ".join(errors))


def _gate_required_evidence(scenario: Scenario, output: CoachingOutput) -> GateResult:
    """Explanation must reference equity (always) and pot odds (when applicable)."""
    missing = []
    lower = output.explanation.lower()

    has_equity = "equity" in lower or bool(re.search(r"\d+%", output.explanation))
    if not has_equity:
        missing.append("equity")

    if scenario.solver_output.pot_odds != "N/A":
        has_odds = "pot odds" in lower or bool(re.search(r"\d+(\.\d+)?:\d+", output.explanation))
        if not has_odds:
            missing.append("pot odds")

    passed = len(missing) == 0
    detail = f"missing: {missing}" if missing else ""
    return GateResult(name="required_evidence", passed=passed, detail=detail)


def run_gates(scenario: Scenario, output: CoachingOutput) -> GateResults:
    gates = [
        _gate_legality(scenario, output),
        _gate_action_alignment(scenario, output),
        _gate_numerical_accuracy(scenario, output),
        _gate_required_evidence(scenario, output),
    ]
    return GateResults(gates=gates, all_passed=all(g.passed for g in gates))


# --- Rule-based heuristic scores ---

def _normalize(text: str) -> str:
    return text.lower().replace("-", " ")


def _score_concept_mention(scenario: Scenario, output: CoachingOutput) -> float:
    """Fraction of required_concepts found in explanation."""
    if not scenario.required_concepts:
        return 1.0
    normalized = _normalize(output.explanation)
    found = sum(1 for c in scenario.required_concepts if _normalize(c) in normalized)
    return found / len(scenario.required_concepts)


def _score_contradiction(scenario: Scenario, output: CoachingOutput) -> float:
    """1.0 minus fraction of contradiction_rules triggered."""
    if not scenario.contradiction_rules:
        return 1.0
    lower = output.explanation.lower()
    triggered = sum(1 for r in scenario.contradiction_rules if r.lower() in lower)
    return 1.0 - triggered / len(scenario.contradiction_rules)


def _score_format(output: CoachingOutput) -> float:
    """Penalizes too-short, too-long, or missing action mention."""
    score = 1.0
    length = len(output.explanation)

    short_applied = False
    long_applied = False
    for op, threshold, penalty in config.FORMAT_LENGTH_PENALTIES:
        if op == "<" and not short_applied and length < threshold:
            score -= penalty
            short_applied = True
        elif op == ">" and not long_applied and length > threshold:
            score -= penalty
            long_applied = True

    if output.recommended_action.lower() not in output.explanation.lower():
        score -= config.FORMAT_MISSING_ACTION_PENALTY

    return max(0.0, score)


# --- Solver-derived heuristics (cheap eval, no annotations needed) ---

def _score_ev_awareness(scenario: Scenario, output: CoachingOutput) -> float:
    """1.0 if EV is mentioned when significant (|EV| > 5), auto-pass otherwise."""
    if abs(scenario.solver_output.ev) <= 5:
        return 1.0
    lower = output.explanation.lower()
    return 1.0 if ("ev" in lower or "expected value" in lower) else 0.0


def _score_frequency_awareness(scenario: Scenario, output: CoachingOutput) -> float:
    """1.0 if mixed strategy acknowledged when no action dominates (max_freq < 0.8)."""
    freqs = scenario.solver_output.action_frequencies
    max_freq = max(freqs.values()) if freqs else 1.0
    if max_freq >= 0.8:
        return 1.0  # dominant action — no need to discuss mixed strategy
    lower = output.explanation.lower()
    has_freq = any(kw in lower for kw in ["frequenc", "mix", "sometimes", "portion", "split"])
    return 1.0 if has_freq else 0.0


def _score_position_awareness(scenario: Scenario, output: CoachingOutput) -> float:
    """1.0 if explanation mentions position or positional concepts."""
    lower = output.explanation.lower()
    pos = scenario.game_state.position.lower()
    has_position = (pos in lower or "position" in lower or "in position" in lower
                    or "out of position" in lower)
    return 1.0 if has_position else 0.0


# --- Eval rules as structured data (fed to the optimizer's system prompt) ---

def get_eval_rules() -> dict:
    """Single source of truth — the evaluator executes these, the optimizer reads them."""
    format_info = {
        "length_penalties": [
            {"op": op, "threshold_chars": t, "penalty": p}
            for op, t, p in config.FORMAT_LENGTH_PENALTIES
        ],
        "missing_action_penalty": config.FORMAT_MISSING_ACTION_PENALTY,
        "target_range_chars": list(config.FORMAT_TARGET_RANGE),
    }
    return {
        "gates": config.GATES,
        "cheap_heuristic_weights": config.CHEAP_HEURISTIC_WEIGHTS,
        "cheap_heuristic_scores": {
            "format_score": format_info,
            "ev_awareness": "1.0 if EV mentioned when |EV|>5, else 0.0; auto 1.0 when insignificant",
            "frequency_awareness": "1.0 if mixed strategy mentioned when max_freq<0.8; auto 1.0 when dominant",
            "position_awareness": "1.0 if position/IP/OOP mentioned, else 0.0",
        },
        "full_heuristic_weights": config.HEURISTIC_WEIGHTS,
        "full_heuristic_scores": {
            "concept_mention": "fraction of required_concepts found in explanation (0-1)",
            "contradiction": "1.0 minus fraction of contradiction_rules triggered (0-1)",
            "format_score": format_info,
        },
        "composite": {
            "heuristic_weight": config.HEURISTIC_WEIGHT,
            "judge_weight": config.JUDGE_WEIGHT,
            "judge_dimensions": ["coherence", "readability", "coaching_tone"],
            "judge_scale": "1-5",
        },
    }


# --- Composite scoring ---

def compute_rule_based_cheap(scenario: Scenario, output: CoachingOutput) -> RuleBasedScores:
    """Solver-derived heuristics only (no annotations needed)."""
    fmt = _score_format(output)
    ev = _score_ev_awareness(scenario, output)
    freq = _score_frequency_awareness(scenario, output)
    pos = _score_position_awareness(scenario, output)
    w = config.CHEAP_HEURISTIC_WEIGHTS
    composite = (
        w["format_score"] * fmt
        + w["ev_awareness"] * ev
        + w["frequency_awareness"] * freq
        + w["position_awareness"] * pos
    )
    return RuleBasedScores(
        format_score=fmt,
        ev_awareness=ev,
        frequency_awareness=freq,
        position_awareness=pos,
        composite=composite,
    )


def compute_rule_based_full(scenario: Scenario, output: CoachingOutput) -> RuleBasedScores:
    """Annotation-based heuristics (requires ScenarioAnnotated)."""
    concept = _score_concept_mention(scenario, output)
    contradiction = _score_contradiction(scenario, output)
    fmt = _score_format(output)
    w = config.HEURISTIC_WEIGHTS
    composite = (
        w["concept_mention"] * concept
        + w["contradiction"] * contradiction
        + w["format_score"] * fmt
    )
    return RuleBasedScores(
        concept_mention=concept,
        contradiction=contradiction,
        format_score=fmt,
        composite=composite,
    )


def evaluate(
    scenario: Scenario,
    output: CoachingOutput,
    judge_scores: JudgeScores | None = None,
    full_eval: bool = False,
) -> EvalResult:
    """Gates → heuristics → composite. full_eval=True uses annotation-based heuristics."""
    gates = run_gates(scenario, output)

    if not gates.all_passed:
        return EvalResult(
            scenario_id=scenario.id,
            gates=gates,
            composite=0.0,
        )

    if full_eval:
        rule_based = compute_rule_based_full(scenario, output)
    else:
        rule_based = compute_rule_based_cheap(scenario, output)

    if judge_scores is not None:
        composite = (
            config.HEURISTIC_WEIGHT * rule_based.composite
            + config.JUDGE_WEIGHT * judge_scores.normalized_avg()
        )
    else:
        composite = rule_based.composite

    return EvalResult(
        scenario_id=scenario.id,
        gates=gates,
        rule_based=rule_based,
        judge=judge_scores,
        composite=composite,
    )


# --- Soft-gate scoring (for DSPy engines — partial credit when gates fail) ---

def compute_composite_soft(gates: GateResults, rule_based: RuleBasedScores,
                           judge: JudgeScores | None = None) -> float:
    """Weighted composite with fractional gate credit instead of hard zeroing."""
    gate_fraction = sum(1 for g in gates.gates if g.passed) / len(gates.gates)

    if judge is not None:
        result = (config.SOFT_GATE_WEIGHT * gate_fraction
                  + config.SOFT_HEURISTIC_WEIGHT * rule_based.composite
                  + config.SOFT_JUDGE_WEIGHT * judge.normalized_avg())
    else:
        result = (config.SOFT_GATE_WEIGHT_NO_JUDGE * gate_fraction
                  + config.SOFT_HEURISTIC_WEIGHT_NO_JUDGE * rule_based.composite)

    return max(0.01, result)


def evaluate_soft(
    scenario: Scenario,
    output: CoachingOutput,
    judge_scores: JudgeScores | None = None,
    full_eval: bool = False,
) -> EvalResult:
    """Soft-gate evaluation: always computes heuristics, partial credit for gate failures."""
    gates = run_gates(scenario, output)

    if full_eval:
        rule_based = compute_rule_based_full(scenario, output)
    else:
        rule_based = compute_rule_based_cheap(scenario, output)

    # Judge scores only applied when all gates pass
    judge = judge_scores if gates.all_passed else None
    composite = compute_composite_soft(gates, rule_based, judge)

    return EvalResult(
        scenario_id=scenario.id,
        gates=gates,
        rule_based=rule_based,
        judge=judge,
        composite=composite,
    )
