import argparse
import json
import random
import sys

from src import config
from src.models import Scenario, ScenarioAnnotated
from src.pipeline import run_pipeline
from src.prompt_store import diff_prompts


def main():
    parser = argparse.ArgumentParser(description="Run poker coaching optimization pipeline")
    parser.add_argument("--train", type=int, default=None,
                        help="Max training scenarios to use (default: all)")
    parser.add_argument("--holdout", type=int, default=None,
                        help="Max holdout scenarios to use (default: all)")
    args = parser.parse_args()

    train_path = config.DATA_DIR / "scenarios.json"
    if not train_path.exists():
        print(f"ERROR: Training dataset not found at {train_path}")
        print("Run: python -m src.generate_data 10 --solver-only")
        sys.exit(1)

    train_data = json.loads(train_path.read_text(encoding="utf-8"))
    train = [Scenario.model_validate(d) for d in train_data]
    if args.train and args.train < len(train):
        train = random.sample(train, args.train)
    random.shuffle(train)
    print(f"Training scenarios: {len(train)} (solver-only)")

    annotated_path = config.DATA_DIR / "scenarios-annotated.json"
    if not annotated_path.exists():
        print(f"ERROR: Annotated dataset not found at {annotated_path}")
        print("Run: python -m src.generate_data 5")
        sys.exit(1)

    annotated_data = json.loads(annotated_path.read_text(encoding="utf-8"))
    annotated = [ScenarioAnnotated.model_validate(d) for d in annotated_data]
    random.shuffle(annotated)

    examples_pool = annotated[:5]   # TODO: get 10%
    holdout = annotated[5:]
    if args.holdout and args.holdout < len(holdout):
        holdout = random.sample(holdout, args.holdout)
    print(f"Annotated scenarios: {len(annotated)} total")
    print(f"  Examples pool: {len(examples_pool)}")
    print(f"  Holdout: {len(holdout)}")

    v0_path = config.PROMPTS_DIR / "v0.txt"
    initial_prompt = v0_path.read_text(encoding="utf-8")
    print(f"\nInitial prompt ({len(initial_prompt)} chars):")
    print(f"  {initial_prompt[:100]}...")

    print(f"\n{'='*60}")
    print("Running optimization pipeline...")
    print(f"{'='*60}")

    result = run_pipeline(train, holdout, initial_prompt,
                          examples_pool=examples_pool)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    for i, iteration in enumerate(result["iterations"], 1):
        marker = " <-- best" if iteration["version"] == result["best_version"] else ""
        print(f"  {iteration['version']}: gates={iteration['gate_pass_rate']:.0%}  composite={iteration['avg_composite']:.3f}{marker}")

    print()
    print(f"  Holdout baseline (v0):                  {result['baseline_holdout']:.3f}")
    print(f"  Holdout final    ({result['best_version']}): {' ' * (20 - len(result['best_version']))}{result['final_holdout']:.3f}")
    delta = result["final_holdout"] - result["baseline_holdout"]
    print(f"  Delta:                                  {delta:+.3f}")
    print()

    if result["accepted"]:
        print(f"  ACCEPTED — {result['best_version']} improves over v0 on holdout set.")
    else:
        print(f"  REJECTED — best prompt ({result['best_version']}) did NOT improve over v0 on holdout.")

    # Prompt diff
    best_v = result["best_version"]
    print(f"\n{'='*60}")
    print(f"PROMPT DIFF (v0 -> {best_v})")
    print(f"{'='*60}\n")
    try:
        diff = diff_prompts("v0", best_v)
        print(diff if diff else "(No differences)")
    except Exception as e:
        print(f"(Could not compute diff: {e})")

    # Before/after coaching samples
    print(f"\n{'='*60}")
    print("COACHING SAMPLES (v0 vs best)")
    print(f"{'='*60}")
    baseline_details = result.get("baseline_details", [])
    final_details = result.get("final_details", [])
    final_by_id = {s.id: (s, o, r) for s, o, r in final_details}
    shown = 0
    for s_base, o_base, r_base in baseline_details:
        if shown >= 2:
            break
        final_entry = final_by_id.get(s_base.id)
        if not final_entry or not o_base:
            continue
        s_fin, o_fin, r_fin = final_entry
        if not o_fin:
            continue
        gs = s_base.game_state
        print(f"\n  --- {s_base.id}: {gs.hand} on {gs.board or '(preflop)'} | "
              f"{gs.street} | {gs.position} | villain: {gs.villain_action} ---")
        print(f"  Solver: {s_base.solver_output.recommended_action} | "
              f"equity {s_base.solver_output.equity:.0%} | "
              f"pot odds {s_base.solver_output.pot_odds}")
        print(f"\n  [v0] (composite={r_base.composite:.3f}):")
        print(f"    Action: {o_base.recommended_action}")
        for line in o_base.explanation.split("\n"):
            print(f"    {line}")
        print(f"\n  [{best_v}] (composite={r_fin.composite:.3f}):")
        print(f"    Action: {o_fin.recommended_action}")
        for line in o_fin.explanation.split("\n"):
            print(f"    {line}")
        shown += 1


if __name__ == "__main__":
    main()
