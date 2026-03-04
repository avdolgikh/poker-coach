"""Pipeline orchestrator: baseline → engine.optimize() → final holdout."""

from src import prompt_store
from src.coaching_agent import run_batch
from src.engine_protocol import OptimizationEngine
from src.evaluator import evaluate
from src.judge_agent import judge_coaching
from src.models import EvalResult, GateResult, GateResults


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


def _eval_set(scenarios, prompt, full_eval=False, no_judge=False):
    """Run coaching + judge + evaluate on a set of scenarios (text prompt).

    Returns (avg_composite, gate_pass_rate, results_with_context).
    """
    outputs = run_batch(scenarios, prompt)
    results_with_context = []
    for scenario, output in zip(scenarios, outputs):
        if output is None:
            results_with_context.append((scenario, None, _failed_eval(scenario.id)))
            continue
        result = evaluate(scenario, output, full_eval=full_eval)
        if result.gates.all_passed and not no_judge:
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


def _eval_set_fn(scenarios, coach_fn, full_eval=False, no_judge=False):
    """Run coach_fn + judge + evaluate on a set of scenarios (engine's coach_fn).

    Returns (avg_composite, gate_pass_rate, results_with_context).
    """
    results_with_context = []
    for scenario in scenarios:
        try:
            output = coach_fn(scenario)
        except Exception:
            output = None
        if output is None:
            results_with_context.append((scenario, None, _failed_eval(scenario.id)))
            continue
        result = evaluate(scenario, output, full_eval=full_eval)
        if result.gates.all_passed and not no_judge:
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


def run_pipeline(engine: OptimizationEngine, train, holdout_annotated, initial_prompt, **kwargs):
    """Run the full optimization pipeline.

    Returns dict with: iterations, baseline_holdout, final_holdout, accepted,
    best_version, best_train_composite.
    """
    no_judge = kwargs.get("no_judge", False)
    prompt_store.save_prompt("v0", initial_prompt)

    # Baseline holdout (text prompt, full_eval=True)
    print("\n--- Baseline holdout evaluation ---")
    baseline_holdout, _, baseline_details = _eval_set(
        holdout_annotated, initial_prompt, full_eval=True, no_judge=no_judge)
    _print_eval_details(baseline_details)
    print(f"  ** Baseline: v0 (composite={baseline_holdout:.3f})")

    if all(o is None for _, o, _ in baseline_details):
        print("\n  ABORT: All LLM calls failed. Check API key / quota.")
        return {
            "iterations": [], "baseline_holdout": 0.0, "final_holdout": 0.0,
            "accepted": False, "best_version": "v0", "best_train_composite": 0.0,
            "baseline_details": baseline_details, "final_details": [],
        }

    # Engine optimization
    result = engine.optimize(train, initial_prompt, **kwargs)

    # Final holdout (coach_fn, full_eval=True)
    print(f"\n--- Final holdout evaluation (using {result.best_label}) ---")
    final_holdout, _, final_details = _eval_set_fn(
        holdout_annotated, result.coach_fn, full_eval=True, no_judge=no_judge)
    _print_eval_details(final_details)

    accepted = final_holdout > baseline_holdout

    return {
        "iterations": result.iterations,
        "baseline_holdout": baseline_holdout,
        "final_holdout": final_holdout,
        "accepted": accepted,
        "best_version": result.best_label,
        "best_train_composite": result.best_train_composite,
        "baseline_details": baseline_details,
        "final_details": final_details,
    }
