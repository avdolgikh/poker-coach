"""Manual smoke test for the failure detector pipeline.

Runs deterministic parts only (no LLM calls) to verify routing,
verification, skipped-tool detection, verdict synthesis, and metrics.
"""

from src.models import Claim, CoachingOutput, FailureLabel, FailureVerdict, ToolCall, ToolTrace
from src.failure_detector import (
    check_required_tools,
    compute_metrics,
    route_claim,
    scenario_to_trace,
    synthesize_verdict,
    verify_claim,
)
from tests.conftest import make_solver_scenario


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _trace_with_all_required(**overrides) -> ToolTrace:
    return ToolTrace(calls=[
        ToolCall(tool_name="equity_calculator", input={}, output={"equity": overrides.get("equity", 0.34)}),
        ToolCall(tool_name="pot_odds_calculator", input={}, output={"pot_odds": overrides.get("pot_odds", "3:1")}),
        ToolCall(tool_name="action_frequency_solver", input={}, output={
            "recommended_action": overrides.get("action", "call"),
            "frequencies": {"call": 0.6, "fold": 0.4},
        }),
    ])


def test_correct_output():
    _section("1. Correct output — all claims match")
    trace = _trace_with_all_required()
    claims = [
        Claim(text="Your equity is 34%", claim_type="numeric", topic="equity"),
        Claim(text="Pot odds are 3:1", claim_type="numeric", topic="pot_odds"),
        Claim(text="You should call", claim_type="action", topic="recommended_action"),
    ]
    verifications = []
    for c in claims:
        tool = route_claim(c, trace)
        v = verify_claim(c, tool)
        print(f"  {c.topic:25s} -> {v.verdict:12s} ({v.severity}) | {v.reasoning}")
        verifications.append(v)

    skipped = check_required_tools(trace)
    verdict = synthesize_verdict(verifications, skipped)
    print(f"\n  Overall pass: {verdict.overall_pass}  |  Failures: {len(verdict.failures)}")


def test_wrong_equity():
    _section("2. Wrong equity (55% vs 34%)")
    trace = _trace_with_all_required()
    claims = [
        Claim(text="Your equity is 55%", claim_type="numeric", topic="equity"),
        Claim(text="Pot odds are 3:1", claim_type="numeric", topic="pot_odds"),
        Claim(text="You should call", claim_type="action", topic="recommended_action"),
    ]
    verifications = []
    for c in claims:
        tool = route_claim(c, trace)
        v = verify_claim(c, tool)
        print(f"  {c.topic:25s} -> {v.verdict:12s} ({v.severity}) | {v.reasoning}")
        verifications.append(v)

    skipped = check_required_tools(trace)
    verdict = synthesize_verdict(verifications, skipped)
    print(f"\n  Overall pass: {verdict.overall_pass}  |  Hard failure: {verdict.has_hard_failure}")
    for f in verdict.failures:
        print(f"  FAILURE: {f.failure_type} @ {f.tool_name}")


def test_missing_tool():
    _section("3. Missing required tool (equity_calculator)")
    partial_trace = ToolTrace(calls=[
        ToolCall(tool_name="pot_odds_calculator", input={}, output={"pot_odds": "3:1"}),
        ToolCall(tool_name="action_frequency_solver", input={}, output={"recommended_action": "call"}),
    ])
    skipped = check_required_tools(partial_trace)
    print(f"  Skipped required tools: {skipped}")


def test_scenario_to_trace():
    _section("4. scenario_to_trace (from conftest defaults)")
    s = make_solver_scenario()
    t = scenario_to_trace(s)
    for call in t.calls:
        print(f"  {call.tool_name:30s} -> {call.output}")


def test_metrics():
    _section("5. compute_metrics")
    label = FailureLabel(failure_type="contradiction", tool_name="equity_calculator", detail="test")
    verdict = FailureVerdict(
        claims=[], verifications=[], skipped_tools=[],
        failures=[label], has_hard_failure=True, overall_pass=False, summary="test",
    )
    m = compute_metrics([verdict], [[label]])
    print(f"  Perfect match:  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}")

    extra = FailureLabel(failure_type="contradiction", tool_name="ev_calculator", detail="spurious")
    verdict2 = FailureVerdict(
        claims=[], verifications=[], skipped_tools=[],
        failures=[label, extra], has_hard_failure=True, overall_pass=False, summary="test",
    )
    m2 = compute_metrics([verdict2], [[label]])
    print(f"  +1 false pos:   P={m2['precision']:.2f}  R={m2['recall']:.2f}  F1={m2['f1']:.2f}")


def test_edge_cases():
    _section("6. Edge cases")
    trace = _trace_with_all_required()

    # Negation in text: "Do not call" still contains "call" substring
    c1 = Claim(text="Do not call, you should fold here", claim_type="action", topic="recommended_action")
    v1 = verify_claim(c1, route_claim(c1, trace))
    print(f"  Negation trick:  verdict={v1.verdict}")
    print(f"    claim='Do not call, you should fold'  tool=call")
    print(f"    BUG: categorical substring match finds 'call' -> {v1.verdict} (should be contradicted)")

    # Edge numeric: 3% vs 34%
    c2 = Claim(text="Your equity is only 3% here", claim_type="numeric", topic="equity")
    v2 = verify_claim(c2, route_claim(c2, trace))
    print(f"\n  Edge numeric:    verdict={v2.verdict} | {v2.reasoning}")

    # Unknown topic
    c3 = Claim(text="Implied odds are great", claim_type="conceptual", topic="implied_odds")
    v3 = verify_claim(c3, route_claim(c3, trace))
    print(f"\n  Unknown topic:   verdict={v3.verdict} ({v3.severity})")

    # Soft-fail: position contradiction (not required)
    pos_trace = ToolTrace(calls=[
        *trace.calls,
        ToolCall(tool_name="position_analyzer", input={}, output={"position": "BB"}),
    ])
    c4 = Claim(text="You are on the button", claim_type="categorical", topic="position")
    v4 = verify_claim(c4, route_claim(c4, pos_trace))
    print(f"\n  Position mismatch (soft): verdict={v4.verdict} ({v4.severity})")
    print(f"    claim='on the button'  tool=BB")


if __name__ == "__main__":
    test_correct_output()
    test_wrong_equity()
    test_missing_tool()
    test_scenario_to_trace()
    test_metrics()
    test_edge_cases()
    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}")
