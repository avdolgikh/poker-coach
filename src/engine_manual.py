"""Manual optimization engine: train/eval/optimize loop with LLM-based optimizer."""

import random

from src import config
from src import prompt_store
from src.coaching_agent import run_batch, generate_explanation
from src.evaluator import evaluate
from src.judge_agent import judge_coaching
from src.models import EvalResult, GateResult, GateResults, ScenarioAnnotated
from src.optimizer_agent import propose_edits, apply_edits
from src.engine_protocol import EngineResult


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
    """Run coaching + judge + evaluate on a set of scenarios.

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
        if isinstance(s, ScenarioAnnotated):
            failure["required_concepts"] = s.required_concepts
            failure["key_numbers"] = s.key_numbers
            failure["contradiction_rules"] = s.contradiction_rules

        failures.append(failure)
    return failures


class ManualEngine:
    """Original optimization loop: eval → optimizer proposes edits → repeat."""

    def optimize(self, train, initial_prompt, *, examples_pool=None, no_judge=False):
        current_prompt = initial_prompt
        best_prompt = current_prompt
        best_version = "v0"
        best_train_composite = 0.0

        iterations = []
        for i in range(config.NUM_ITERATIONS):
            print(f"\n--- Iteration {i + 1}/{config.NUM_ITERATIONS} ---")

            avg_composite, gate_pass_rate, results_with_context = _eval_set(
                train, current_prompt, full_eval=False, no_judge=no_judge)

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

        return EngineResult(
            coach_fn=lambda s: generate_explanation(s, best_prompt),
            iterations=iterations,
            best_train_composite=best_train_composite,
            best_label=best_version,
            metadata={"best_prompt": best_prompt},
        )
