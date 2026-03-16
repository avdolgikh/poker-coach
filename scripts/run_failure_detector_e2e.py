"""End-to-end test for the failure detector pipeline with real LLM calls.

Loads real scenarios, generates coaching output, then runs the full
failure detection pipeline: LLM claim extraction -> deterministic routing
-> hybrid verification (deterministic + LLM semantic) -> verdict synthesis.

Usage:
    uv run python scripts/run_failure_detector_e2e.py [--count N] [--scenario-id ID]

Cost: ~2 LLM calls per scenario (coaching PROD_MODEL + claim extraction TUNER_MODEL,
plus semantic verification calls if conceptual claims are found).
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path (scripts/ lives one level below)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Scenario, CoachingOutput
from src.coaching_agent import generate_explanation
from src.failure_detector import (
    detect_failures,
    extract_claims,
    route_claim,
    scenario_to_trace,
    verify_claim,
)
from src.prompt_store import load_prompt


def _sep(char: str = "=", width: int = 70) -> str:
    return char * width


def _print_scenario_header(scenario: Scenario) -> None:
    gs = scenario.game_state
    so = scenario.solver_output
    print(f"\n{_sep()}")
    print(f"  Scenario: {scenario.id}")
    print(f"  {gs.hand} | {gs.board or '(preflop)'} | {gs.street} | {gs.position}")
    print(f"  Pot: {gs.pot}  Stack: {gs.stack}  Villain: {gs.villain_action}")
    print(f"  Solver: {so.recommended_action} | equity={so.equity:.1%} | pot_odds={so.pot_odds} | EV={so.ev}")
    freqs = ", ".join(f"{k}: {v:.0%}" for k, v in so.action_frequencies.items())
    print(f"  Frequencies: {freqs}")
    print(_sep())


def _print_coaching_output(output: CoachingOutput) -> None:
    print(f"\n  [Coaching Output]")
    print(f"  Action: {output.recommended_action}")
    print(f"  Explanation:")
    for line in output.explanation.split("\n"):
        print(f"    {line}")


def _run_stepwise(output: CoachingOutput, scenario: Scenario) -> None:
    """Run the failure detector step-by-step with verbose output."""
    trace = scenario_to_trace(scenario)

    # Step 1: Extract claims (LLM)
    print(f"\n  [Step 1: Claim Extraction (LLM)]")
    claims = extract_claims(output)
    if not claims:
        print("    WARNING: No claims extracted (LLM failure or empty output)")
    for i, c in enumerate(claims, 1):
        print(f"    {i}. [{c.claim_type}] topic={c.topic}")
        print(f"       \"{c.text}\"")

    # Step 2: Route claims to tools (deterministic)
    print(f"\n  [Step 2: Routing (deterministic)]")
    routed = []
    for c in claims:
        tool = route_claim(c, trace)
        tool_name = tool.tool_name if tool else "NONE"
        print(f"    {c.topic:25s} -> {tool_name}")
        routed.append((c, tool))

    # Step 3: Verify claims (hybrid: deterministic + LLM semantic)
    print(f"\n  [Step 3: Verification (hybrid)]")
    verifications = []
    for c, tool in routed:
        v = verify_claim(c, tool)
        marker = "OK" if v.verdict == "supported" else "!!" if v.verdict == "contradicted" else "??"
        print(f"    [{marker}] {c.topic:25s} | {v.verdict:12s} | {v.severity:10s}")
        print(f"         {v.reasoning}")
        verifications.append(v)

    # Step 4: Full pipeline verdict (for comparison)
    print(f"\n  [Step 4: Full Pipeline Verdict]")
    verdict = detect_failures(output, trace)
    print(f"    Claims: {len(verdict.claims)}  |  Verifications: {len(verdict.verifications)}")
    print(f"    Skipped tools: {verdict.skipped_tools or 'none'}")
    print(f"    Failures: {len(verdict.failures)}")
    for f in verdict.failures:
        sev = "HARD" if f.failure_type in ("contradiction", "skipped_tool") else "soft"
        print(f"      [{sev}] {f.failure_type}: {f.tool_name} — {f.detail}")
    print(f"    Overall pass: {verdict.overall_pass}")
    print(f"    Summary: {verdict.summary}")

    return verdict


def main():
    parser = argparse.ArgumentParser(description="E2E failure detector test")
    parser.add_argument("--count", type=int, default=3, help="Number of scenarios to test (default: 3)")
    parser.add_argument("--scenario-id", type=str, default=None, help="Run a specific scenario by ID")
    parser.add_argument("--prompt", type=str, default="v0", help="Prompt version to use (default: v0)")
    args = parser.parse_args()

    # Load scenarios
    with open("data/scenarios.json", encoding="utf-8") as f:
        all_scenarios = [Scenario(**s) for s in json.load(f)]

    if args.scenario_id:
        scenarios = [s for s in all_scenarios if s.id == args.scenario_id]
        if not scenarios:
            print(f"ERROR: Scenario '{args.scenario_id}' not found.")
            print(f"Available IDs: {', '.join(s.id for s in all_scenarios[:10])}...")
            sys.exit(1)
    else:
        scenarios = all_scenarios[: args.count]

    # Load prompt
    pv = load_prompt(args.prompt)
    print(f"Prompt: {args.prompt} ({len(pv.prompt_text)} chars)")
    print(f"Scenarios: {len(scenarios)}")

    # Run E2E
    stats = {"total": 0, "passed": 0, "failed": 0, "claims_total": 0, "contradictions": 0}

    for scenario in scenarios:
        _print_scenario_header(scenario)

        print("\n  Generating coaching output...")
        try:
            output = generate_explanation(scenario, pv.prompt_text)
        except Exception as e:
            print(f"  ERROR generating coaching: {e}")
            stats["total"] += 1
            stats["failed"] += 1
            continue

        _print_coaching_output(output)
        verdict = _run_stepwise(output, scenario)

        stats["total"] += 1
        stats["claims_total"] += len(verdict.claims)
        stats["contradictions"] += len(verdict.failures)
        if verdict.overall_pass:
            stats["passed"] += 1
        else:
            stats["failed"] += 1

    # Summary
    print(f"\n{_sep()}")
    print(f"  SUMMARY")
    print(_sep())
    print(f"  Scenarios tested:  {stats['total']}")
    print(f"  Passed:            {stats['passed']}")
    print(f"  Failed:            {stats['failed']}")
    print(f"  Total claims:      {stats['claims_total']}")
    print(f"  Contradictions:    {stats['contradictions']}")
    avg_claims = stats["claims_total"] / stats["total"] if stats["total"] else 0
    print(f"  Avg claims/scenario: {avg_claims:.1f}")
    print(_sep())


if __name__ == "__main__":
    main()
