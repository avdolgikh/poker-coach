"""Main optimization loop: train → evaluate → optimize → repeat."""

import random

from src import config
from src import prompt_store
from src.coaching_agent import run_batch
from src.evaluator import evaluate
from src.judge_agent import judge_coaching
from src.models import EvalResult, GateResult, GateResults, ScenarioAnnotated
from src.optimizer_agent import propose_edits, apply_edits


def _failed_eval(scenario_id: str) -> EvalResult:
    """Zero-score result for when LLM call itself fails."""
    return EvalResult(
        scenario_id=scenario_id,
        gates=GateResults(
            gates=[GateResult(name="generation_error", passed=False, detail="LLM call failed")],
            all_passed=False,
        ),
        composite=0.0,
    )


def _eval_set(scenarios, prompt, full_eval=False):
    """Run coaching + judge + evaluate on a set of scenarios.

    Returns (avg_composite, gate_pass_rate, results_with_context).
    Failed outputs count as 0.0. Judge is only called when gates pass.
    """
    outputs = run_batch(scenarios, prompt)
    results_with_context = []
    for scenario, output in zip(scenarios, outputs):
        if output is None:
            results_with_context.append((scenario, None, _failed_eval(scenario.id)))
            continue
        result = evaluate(scenario, output, full_eval=full_eval)
        if result.gates.all_passed:
            try:
                js = judge_coaching(
                    scenario.game_state, scenario.solver_output,
                    scenario.legal_actions, output,
                )
                result = evaluate(scenario, output, judge_scores=js, full_eval=full_eval)
            except Exception as e:
                print(f"    [WARN] Judge failed for {scenario.id}: {e}")
        results_with_context.append((scenario, output, result))

    n = len(results_with_context)
    avg = sum(r.composite for _, _, r in results_with_context) / n if n else 0.0
    gpr = sum(1 for _, _, r in results_with_context if r.gates.all_passed) / n if n else 0.0
    return avg, gpr, results_with_context


def _print_eval_details(results_with_context):
    for s, o, r in results_with_context:
        status = "PASS" if r.gates.all_passed else "FAIL"
        print(f"    {s.id}: [{status}] composite={r.composite:.3f}")
        if not r.gates.all_passed:
            for g in r.gates.gates:
                if not g.passed:
                    print(f"      gate:{g.name} — {g.detail}")
        elif r.rule_based:
            rb = r.rule_based
            parts = []
            if rb.ev_awareness is not None:
                parts.append(f"ev={rb.ev_awareness:.2f}")
            if rb.frequency_awareness is not None:
                parts.append(f"freq={rb.frequency_awareness:.2f}")
            if rb.position_awareness is not None:
                parts.append(f"pos={rb.position_awareness:.2f}")
            if rb.concept_mention is not None:
                parts.append(f"concept={rb.concept_mention:.2f}")
            if rb.contradiction is not None:
                parts.append(f"contra={rb.contradiction:.2f}")
            if rb.format_score is not None:
                parts.append(f"fmt={rb.format_score:.2f}")

            judge_str = ""
            if r.judge:
                j = r.judge
                judge_str = (
                    f" | judge: coh={j.coherence} read={j.readability} "
                    f"tone={j.coaching_tone}"
                )
            print(f"      {' '.join(parts)}{judge_str}")


def _build_representative_failures(results_with_context):
    """Package worst-scoring scenarios as dicts for the optimizer."""
    failures = []
    sorted_by_score = sorted(results_with_context, key=lambda x: x[2].composite)
    for s, o, r in sorted_by_score:
        gs = s.game_state
        so = s.solver_output
        failure = {
            "scenario_id": r.scenario_id,
            "composite": r.composite,
            "game_state": {
                "hand": gs.hand, "board": gs.board,
                "street": gs.street, "position": gs.position,
                "pot": gs.pot, "stack": gs.stack,
                "villain_action": gs.villain_action,
            },
            "solver_output": {
                "recommended_action": so.recommended_action,
                "equity": so.equity,
                "pot_odds": so.pot_odds,
                "ev": so.ev,
                "action_frequencies": so.action_frequencies,
            },
            "legal_actions": s.legal_actions,
            "coaching_action": o.recommended_action if o else "(generation failed)",
            "coaching_explanation": o.explanation if o else "(generation failed)",
            "gate_results": [
                {"name": g.name, "passed": g.passed, "detail": g.detail}
                for g in r.gates.gates
            ],
            "gate_failures": [
                g.name for g in r.gates.gates if not g.passed
            ],
            "rule_based_scores": (
                r.rule_based.model_dump(exclude_none=True)
                if r.rule_based else None
            ),
            "judge_scores": (
                {"coherence": r.judge.coherence, "readability": r.judge.readability,
                 "coaching_tone": r.judge.coaching_tone}
                if r.judge else None
            ),
        }
        # Only include annotation fields for annotated scenarios
        if isinstance(s, ScenarioAnnotated):
            failure["required_concepts"] = s.required_concepts
            failure["key_numbers"] = s.key_numbers
            failure["contradiction_rules"] = s.contradiction_rules

        failures.append(failure)
    return failures


def run_pipeline(train, holdout_annotated, initial_prompt, examples_pool=None):
    """Run the full optimization pipeline.

    Returns dict with: iterations, baseline_holdout, final_holdout, accepted,
    best_version, best_train_composite.
    """
    prompt_store.save_prompt("v0", initial_prompt)
    current_prompt = initial_prompt

    # Baseline holdout (full eval — annotation-based heuristics)
    print("\n--- Baseline holdout evaluation ---")
    baseline_holdout, _, baseline_details = _eval_set(
        holdout_annotated, current_prompt, full_eval=True)
    _print_eval_details(baseline_details)
    print(f"  ** Baseline: v0 (composite={baseline_holdout:.3f})")

    if all(o is None for _, o, _ in baseline_details):
        print("\n  ABORT: All LLM calls failed. Check API key / quota.")
        return {
            "iterations": [], "baseline_holdout": 0.0, "final_holdout": 0.0,
            "accepted": False, "best_version": "v0", "best_train_composite": 0.0,
            "baseline_details": baseline_details, "final_details": [],
        }

    best_prompt = current_prompt
    best_version = "v0"
    best_train_composite = 0.0

    iterations = []
    for i in range(config.NUM_ITERATIONS):
        print(f"\n--- Iteration {i + 1}/{config.NUM_ITERATIONS} ---")

        # Train eval (cheap — solver-derived heuristics only)
        avg_composite, gate_pass_rate, results_with_context = _eval_set(
            train, current_prompt, full_eval=False)

        print(f"  Gate pass rate: {gate_pass_rate:.1%}")
        print(f"  Avg composite:  {avg_composite:.3f}")
        _print_eval_details(results_with_context)

        if gate_pass_rate == 0.0 and all(o is None for _, o, _ in results_with_context):
            print("  SKIP: All coaching calls failed — skipping optimizer.")
            iterations.append({
                "version": f"v{i + 1}", "gate_pass_rate": 0.0,
                "avg_composite": 0.0, "edits": [],
            })
            continue

        # Collect failure diagnostics for the optimizer
        gate_failure_breakdown = {}
        for _, _, r in results_with_context:
            if not r.gates.all_passed:
                for g in r.gates.gates:
                    if not g.passed:
                        gate_failure_breakdown[g.name] = (
                            gate_failure_breakdown.get(g.name, 0) + 1
                        )

        if gate_failure_breakdown:
            print(f"  Gate failures: {gate_failure_breakdown}")

        representative_failures = _build_representative_failures(results_with_context)

        # 10% chance: include a gold-standard example for the optimizer
        example = None
        if examples_pool and random.random() < 0.10:
            example = random.choice(examples_pool)

        optimizer_output = propose_edits(
            current_prompt=current_prompt,
            metrics={"gate_pass_rate": gate_pass_rate, "avg_composite": avg_composite},
            gate_failure_breakdown=gate_failure_breakdown,
            representative_failures=representative_failures[:5],
            example=example,
        )
        new_prompt = apply_edits(current_prompt, optimizer_output.edits)

        for edit in optimizer_output.edits:
            print(f"  Edit: {edit.change[:80]}...")

        version = f"v{i + 1}"
        prompt_store.save_prompt(version, new_prompt)
        print(f"  Saved: {version} ({len(new_prompt)} chars)")

        if avg_composite > best_train_composite:
            best_train_composite = avg_composite
            best_prompt = new_prompt
            best_version = version
            print(f"  ** New best: {best_version} (composite={best_train_composite:.3f})")

        iterations.append({
            "version": version,
            "gate_pass_rate": gate_pass_rate,
            "avg_composite": avg_composite,
            "edits": [e.change for e in optimizer_output.edits],
        })

        current_prompt = new_prompt

    # Final holdout — use BEST prompt, not last
    print(f"\n--- Final holdout evaluation (using {best_version}) ---")
    final_holdout, _, final_details = _eval_set(
        holdout_annotated, best_prompt, full_eval=True)
    _print_eval_details(final_details)

    accepted = final_holdout > baseline_holdout

    return {
        "iterations": iterations,
        "baseline_holdout": baseline_holdout,
        "final_holdout": final_holdout,
        "accepted": accepted,
        "best_version": best_version,
        "best_train_composite": best_train_composite,
        "baseline_details": baseline_details,
        "final_details": final_details,
    }
