"""Tests for the Agentic Orchestration Failure Detector.

Covers: claim extraction, tool routing, evidence verification,
skipped tool check, verdict synthesis, end-to-end detection, metrics,
and scenario-to-trace conversion.
"""

from unittest.mock import patch

import pytest

from src.models import (
    Claim,
    ClaimVerification,
    CoachingOutput,
    FailureLabel,
    FailureVerdict,
    ToolCall,
    ToolTrace,
)
from src.failure_detector import (
    ClaimList,
    SemanticVerificationResult,
    check_required_tools,
    compute_metrics,
    detect_failures,
    extract_claims,
    route_claim,
    scenario_to_trace,
    synthesize_verdict,
    verify_claim,
)
from src import config
from tests.conftest import make_solver_scenario


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _equity_tool(equity: float = 0.65) -> ToolCall:
    return ToolCall(
        tool_name="equity_calculator",
        input={"hand": "Ah Kd", "board": "Jd 7d 2c"},
        output={"equity": equity},
    )


def _pot_odds_tool(pot_odds: str = "3:1") -> ToolCall:
    return ToolCall(
        tool_name="pot_odds_calculator",
        input={"pot": 50, "bet": 25},
        output={"pot_odds": pot_odds},
    )


def _action_solver_tool(
    action: str = "call", frequencies: dict | None = None
) -> ToolCall:
    return ToolCall(
        tool_name="action_frequency_solver",
        input={},
        output={
            "recommended_action": action,
            "frequencies": frequencies or {"call": 0.6, "fold": 0.4},
        },
    )


def _ev_tool(ev: float = 36.0) -> ToolCall:
    return ToolCall(
        tool_name="ev_calculator",
        input={},
        output={"ev": ev},
    )


def _position_tool(position: str = "BTN") -> ToolCall:
    return ToolCall(
        tool_name="position_analyzer",
        input={},
        output={"position": position},
    )


def _texture_tool(texture: str = "wet") -> ToolCall:
    return ToolCall(
        tool_name="board_texture_analyzer",
        input={},
        output={"texture": texture, "draws": ["flush draw"]},
    )


def _full_trace(**overrides) -> ToolTrace:
    """Trace with all required tools + extras."""
    return ToolTrace(
        calls=[
            _equity_tool(overrides.get("equity", 0.65)),
            _pot_odds_tool(overrides.get("pot_odds", "3:1")),
            _action_solver_tool(overrides.get("action", "call")),
            _ev_tool(overrides.get("ev", 36.0)),
            _position_tool(overrides.get("position", "BTN")),
        ]
    )


def _claim(text: str, claim_type: str, topic: str) -> Claim:
    return Claim(text=text, claim_type=claim_type, topic=topic)


def _verification(
    claim: Claim,
    matched_tool: str | None,
    verdict: str,
    reasoning: str = "",
    severity: str = "info",
) -> ClaimVerification:
    return ClaimVerification(
        claim=claim,
        matched_tool=matched_tool,
        verdict=verdict,
        reasoning=reasoning,
        severity=severity,
    )


def _failure(
    failure_type: str,
    detail: str = "test",
    tool_name: str | None = None,
    claim_text: str | None = None,
) -> FailureLabel:
    return FailureLabel(
        failure_type=failure_type,
        tool_name=tool_name,
        claim_text=claim_text,
        detail=detail,
    )


# ===================================================================
# Claim Extraction
# ===================================================================


class TestClaimExtraction:
    """BDD Feature: Claim Extraction."""

    @patch("src.failure_detector.generate_structured")
    def test_extracts_multiple_claim_types(self, mock_gen):
        """Given coaching text with equity, pot odds, and action claims,
        extraction returns all three with correct types."""
        mock_gen.return_value = ClaimList(
            claims=[
                _claim("Your equity is 65%", "numeric", "equity"),
                _claim("Pot odds are 3:1", "numeric", "pot_odds"),
                _claim("You should call", "action", "recommended_action"),
            ]
        )
        output = CoachingOutput(
            recommended_action="call",
            explanation="Your equity is 65% and pot odds are 3:1. You should call.",
        )
        claims = extract_claims(output)

        assert len(claims) == 3
        assert any(c.topic == "equity" and c.claim_type == "numeric" for c in claims)
        assert any(c.topic == "pot_odds" for c in claims)
        assert any(c.topic == "recommended_action" for c in claims)
        mock_gen.assert_called_once()
        assert mock_gen.call_args.kwargs["model"] == config.TUNER_MODEL

    @patch("src.failure_detector.generate_structured")
    def test_empty_explanation_yields_no_claims(self, mock_gen):
        mock_gen.return_value = ClaimList(claims=[])
        output = CoachingOutput(recommended_action="call", explanation="")
        claims = extract_claims(output)

        assert len(claims) == 0

    @patch("src.failure_detector.generate_structured")
    def test_llm_failure_returns_empty_list(self, mock_gen):
        """Graceful degradation: LLM crash -> empty list, no exception."""
        mock_gen.side_effect = Exception("LLM API error")
        output = CoachingOutput(
            recommended_action="call", explanation="Some coaching text here."
        )
        claims = extract_claims(output)

        assert claims == []

    @patch("src.failure_detector.generate_structured")
    def test_conceptual_claims_extracted(self, mock_gen):
        mock_gen.return_value = ClaimList(
            claims=[
                _claim("The board is very wet", "conceptual", "board_texture"),
            ]
        )
        output = CoachingOutput(
            recommended_action="call",
            explanation="The board is very wet with multiple draws.",
        )
        claims = extract_claims(output)

        assert len(claims) == 1
        assert claims[0].claim_type == "conceptual"
        assert claims[0].topic == "board_texture"


# ===================================================================
# Tool Routing
# ===================================================================


class TestToolRouting:
    """BDD Feature: Tool Routing (deterministic, no mocking)."""

    def test_routes_equity_to_equity_calculator(self):
        claim = _claim("equity is 65%", "numeric", "equity")
        trace = ToolTrace(calls=[_equity_tool()])
        result = route_claim(claim, trace)

        assert result is not None
        assert result.tool_name == "equity_calculator"

    def test_routes_action_to_frequency_solver(self):
        claim = _claim("You should call", "action", "recommended_action")
        trace = ToolTrace(calls=[_action_solver_tool()])
        result = route_claim(claim, trace)

        assert result is not None
        assert result.tool_name == "action_frequency_solver"

    def test_missing_tool_returns_none(self):
        claim = _claim("equity is 65%", "numeric", "equity")
        trace = ToolTrace(calls=[_pot_odds_tool()])  # no equity_calculator
        result = route_claim(claim, trace)

        assert result is None

    def test_unknown_topic_returns_none(self):
        claim = _claim("implied odds are good", "conceptual", "implied_odds")
        trace = ToolTrace(calls=[_equity_tool(), _pot_odds_tool()])
        result = route_claim(claim, trace)

        assert result is None


# ===================================================================
# Evidence Verification
# ===================================================================


class TestEvidenceVerification:
    """BDD Feature: Evidence Verification (hybrid: deterministic + LLM)."""

    def test_supported_numeric_equity(self):
        """65% matches tool equity 0.65 -> supported."""
        claim = _claim("Your equity is 65%", "numeric", "equity")
        tool = _equity_tool(0.65)
        result = verify_claim(claim, tool)

        assert result.verdict == "supported"

    def test_contradicted_numeric_equity(self):
        """55% vs tool equity 0.34 -> contradicted, hard_fail."""
        claim = _claim("Your equity is 55%", "numeric", "equity")
        tool = _equity_tool(0.34)
        result = verify_claim(claim, tool)

        assert result.verdict == "contradicted"
        assert result.severity == "hard_fail"

    def test_numeric_within_tolerance(self):
        """66% vs tool equity 0.65 -> within ±2% -> supported."""
        claim = _claim("Your equity is 66%", "numeric", "equity")
        tool = _equity_tool(0.65)
        result = verify_claim(claim, tool)

        assert result.verdict == "supported"

    def test_unsupported_no_tool(self):
        """No matching tool -> unsupported with severity info."""
        claim = _claim("equity is 65%", "numeric", "equity")
        result = verify_claim(claim, None)

        assert result.verdict == "unsupported"
        assert result.severity == "info"

    def test_categorical_action_match(self):
        """Recommended action matches solver -> supported."""
        claim = _claim("You should call", "action", "recommended_action")
        tool = _action_solver_tool("call")
        result = verify_claim(claim, tool)

        assert result.verdict == "supported"

    def test_categorical_action_mismatch(self):
        """Recommended fold but solver says call -> contradicted, hard_fail."""
        claim = _claim("You should fold", "action", "recommended_action")
        tool = _action_solver_tool("call")
        result = verify_claim(claim, tool)

        assert result.verdict == "contradicted"
        assert result.severity == "hard_fail"

    def test_exact_pot_odds_match(self):
        """Pot odds "3:1" appears verbatim in claim -> supported."""
        claim = _claim("Pot odds are 3:1 here", "numeric", "pot_odds")
        tool = _pot_odds_tool("3:1")
        result = verify_claim(claim, tool)

        assert result.verdict == "supported"

    def test_exact_pot_odds_mismatch(self):
        """Claim says "2:1" but tool says "3:1" -> contradicted."""
        claim = _claim("Pot odds are 2:1", "numeric", "pot_odds")
        tool = _pot_odds_tool("3:1")
        result = verify_claim(claim, tool)

        assert result.verdict == "contradicted"

    @patch("src.failure_detector.generate_structured")
    def test_semantic_verification_calls_llm(self, mock_gen):
        """Board texture verified via LLM semantic check."""
        mock_gen.return_value = SemanticVerificationResult(
            is_supported=True,
            reasoning="Board has flush draws, consistent with 'wet' assessment.",
        )
        claim = _claim("The board is very wet", "conceptual", "board_texture")
        tool = _texture_tool("wet")
        result = verify_claim(claim, tool)

        assert result.verdict == "supported"
        mock_gen.assert_called_once()

    def test_unverifiable_unknown_topic(self):
        """Claim topic not in evidence contract -> unverifiable."""
        claim = _claim("implied odds are great", "conceptual", "implied_odds")
        tool = ToolCall(tool_name="some_tool", input={}, output={})
        result = verify_claim(claim, tool)

        assert result.verdict == "unverifiable"
        assert result.severity == "info"


# ===================================================================
# Skipped Tool Check
# ===================================================================


class TestSkippedToolCheck:
    """BDD Feature: Skipped Tool Detection (deterministic)."""

    def test_all_required_present(self):
        trace = _full_trace()
        missing = check_required_tools(trace)

        assert missing == []

    def test_missing_required_tool(self):
        """Trace without equity_calculator -> flagged."""
        trace = ToolTrace(
            calls=[_pot_odds_tool(), _action_solver_tool()]
        )
        missing = check_required_tools(trace)

        assert "equity_calculator" in missing


# ===================================================================
# Verdict Synthesis
# ===================================================================


class TestVerdictSynthesis:
    """BDD Feature: Verdict Synthesis (deterministic aggregation)."""

    def test_no_failures_passes(self):
        v = _verification(
            _claim("equity is 65%", "numeric", "equity"),
            "equity_calculator",
            "supported",
        )
        result = synthesize_verdict([v], [])

        assert result.overall_pass is True
        assert result.has_hard_failure is False
        assert len(result.failures) == 0
        assert len(result.claims) == 1

    def test_contradiction_hard_fail(self):
        v = _verification(
            _claim("equity is 55%", "numeric", "equity"),
            "equity_calculator",
            "contradicted",
            reasoning="55% vs 34%",
            severity="hard_fail",
        )
        result = synthesize_verdict([v], [])

        assert result.overall_pass is False
        assert result.has_hard_failure is True
        assert len(result.failures) == 1
        assert result.failures[0].failure_type == "contradiction"

    def test_soft_failure_still_passes(self):
        """Soft-fail contradiction does NOT set overall_pass to False."""
        v = _verification(
            _claim("hero is in position", "categorical", "position"),
            "position_analyzer",
            "contradicted",
            reasoning="Hero is OOP, not IP",
            severity="soft_fail",
        )
        result = synthesize_verdict([v], [])

        assert result.overall_pass is True
        assert result.has_hard_failure is False
        assert len(result.failures) == 1  # still reported

    def test_skipped_tool_hard_fail(self):
        """Skipped required tool -> hard failure even with no claim contradictions."""
        v = _verification(
            _claim("You should call", "action", "recommended_action"),
            "action_frequency_solver",
            "supported",
        )
        result = synthesize_verdict([v], ["equity_calculator"])

        assert result.overall_pass is False
        assert result.has_hard_failure is True
        assert any(f.failure_type == "skipped_tool" for f in result.failures)

    def test_multiple_failures_aggregated(self):
        v1 = _verification(
            _claim("equity is 55%", "numeric", "equity"),
            "equity_calculator",
            "contradicted",
            severity="hard_fail",
        )
        v2 = _verification(
            _claim("You should fold", "action", "recommended_action"),
            "action_frequency_solver",
            "contradicted",
            severity="hard_fail",
        )
        result = synthesize_verdict([v1, v2], [])

        assert result.overall_pass is False
        assert len(result.failures) == 2


# ===================================================================
# End-to-End: detect_failures
# ===================================================================


class TestDetectFailures:
    """BDD Feature: End-to-End Failure Detection."""

    @patch("src.failure_detector.generate_structured")
    def test_correct_output_passes(self, mock_gen):
        """All claims supported, all required tools present -> pass."""
        mock_gen.return_value = ClaimList(
            claims=[
                _claim("Your equity is 65%", "numeric", "equity"),
                _claim("Pot odds are 3:1", "numeric", "pot_odds"),
                _claim("You should call", "action", "recommended_action"),
            ]
        )
        output = CoachingOutput(
            recommended_action="call",
            explanation="Your equity is 65%, pot odds are 3:1. Call.",
        )
        trace = _full_trace(equity=0.65, pot_odds="3:1", action="call")

        verdict = detect_failures(output, trace)

        assert verdict.overall_pass is True
        assert len(verdict.failures) == 0
        assert len(verdict.claims) == 3
        # Only one generate_structured call (extraction); verification is deterministic
        mock_gen.assert_called_once()

    @patch("src.failure_detector.generate_structured")
    def test_contradicted_equity_fails(self, mock_gen):
        """Equity claim contradicts tool -> hard failure."""
        mock_gen.return_value = ClaimList(
            claims=[_claim("Your equity is 55%", "numeric", "equity")]
        )
        output = CoachingOutput(
            recommended_action="call", explanation="Your equity is 55%."
        )
        trace = _full_trace(equity=0.34)

        verdict = detect_failures(output, trace)

        assert verdict.overall_pass is False
        assert verdict.has_hard_failure is True
        assert any(f.failure_type == "contradiction" for f in verdict.failures)

    @patch("src.failure_detector.generate_structured")
    def test_missing_tool_fails(self, mock_gen):
        """Missing required tool -> hard failure via skipped_tools."""
        mock_gen.return_value = ClaimList(
            claims=[_claim("You should call", "action", "recommended_action")]
        )
        output = CoachingOutput(
            recommended_action="call", explanation="You should call."
        )
        # Trace missing equity_calculator (required)
        trace = ToolTrace(calls=[_pot_odds_tool(), _action_solver_tool()])

        verdict = detect_failures(output, trace)

        assert verdict.overall_pass is False
        assert "equity_calculator" in verdict.skipped_tools

    @patch("src.failure_detector.generate_structured")
    def test_extraction_failure_failsafe(self, mock_gen):
        """LLM crash during extraction -> fail-safe verdict (pass=False, 0 claims)."""
        mock_gen.side_effect = Exception("LLM API error")
        output = CoachingOutput(
            recommended_action="call", explanation="Some text."
        )
        trace = _full_trace()

        verdict = detect_failures(output, trace)

        assert verdict.overall_pass is False
        assert len(verdict.claims) == 0


# ===================================================================
# Metrics
# ===================================================================


class TestComputeMetrics:
    """BDD Feature: Precision/Recall computation."""

    def test_perfect_detection(self):
        label = _failure("contradiction", tool_name="equity_calculator")
        verdict = FailureVerdict(
            claims=[],
            verifications=[],
            skipped_tools=[],
            failures=[label],
            has_hard_failure=True,
            overall_pass=False,
            summary="test",
        )
        result = compute_metrics([verdict], [[label]])

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_false_positives_reduce_precision(self):
        real = _failure("contradiction", tool_name="equity_calculator")
        extra = _failure("contradiction", tool_name="ev_calculator")
        verdict = FailureVerdict(
            claims=[],
            verifications=[],
            skipped_tools=[],
            failures=[real, extra],
            has_hard_failure=True,
            overall_pass=False,
            summary="test",
        )
        result = compute_metrics([verdict], [[real]])

        assert result["precision"] == pytest.approx(0.5)
        assert result["recall"] == 1.0

    def test_false_negatives_reduce_recall(self):
        found = _failure("contradiction", tool_name="equity_calculator")
        missed = _failure("skipped_tool", tool_name="pot_odds_calculator")
        verdict = FailureVerdict(
            claims=[],
            verifications=[],
            skipped_tools=[],
            failures=[found],
            has_hard_failure=True,
            overall_pass=False,
            summary="test",
        )
        result = compute_metrics([verdict], [[found, missed]])

        assert result["precision"] == 1.0
        assert result["recall"] == pytest.approx(0.5)

    def test_empty_inputs(self):
        result = compute_metrics([], [])

        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0


# ===================================================================
# Scenario-to-Trace Conversion
# ===================================================================


class TestScenarioToTrace:
    """Helper: convert existing Scenario to ToolTrace for testing."""

    def test_converts_solver_output(self):
        scenario = make_solver_scenario()
        trace = scenario_to_trace(scenario)
        tool_names = [c.tool_name for c in trace.calls]

        assert "equity_calculator" in tool_names
        assert "pot_odds_calculator" in tool_names
        assert "action_frequency_solver" in tool_names

        # Check equity value matches scenario
        eq_tool = next(c for c in trace.calls if c.tool_name == "equity_calculator")
        assert eq_tool.output["equity"] == 0.34  # conftest default

    def test_includes_position_analyzer(self):
        scenario = make_solver_scenario()
        trace = scenario_to_trace(scenario)
        tool_names = [c.tool_name for c in trace.calls]

        assert "position_analyzer" in tool_names
        pos_tool = next(c for c in trace.calls if c.tool_name == "position_analyzer")
        assert pos_tool.output["position"] == "BTN"  # conftest default


# ===================================================================
# Regression tests for review findings
# ===================================================================


class TestActionFrequencyVerification:
    """Review finding #1: action_frequency claims crash _verify_numeric
    because output_field 'frequencies' is a dict, not a scalar."""

    def test_action_frequency_does_not_crash(self):
        """action_frequency claim must not raise TypeError."""
        claim = _claim("Call 60% of the time", "numeric", "action_frequency")
        tool = _action_solver_tool("call", {"call": 0.6, "fold": 0.4})
        # Current code: float({"call": 0.6, ...}) -> TypeError
        result = verify_claim(claim, tool)
        assert result.verdict in ("supported", "contradicted", "unverifiable")

    def test_action_frequency_supported(self):
        """Claim 'Call 60%' with frequencies={'call': 0.6} -> supported."""
        claim = _claim("Call 60% of the time", "numeric", "action_frequency")
        tool = _action_solver_tool("call", {"call": 0.6, "fold": 0.4})
        result = verify_claim(claim, tool)
        assert result.verdict == "supported"

    def test_action_frequency_contradicted(self):
        """Claim 'Call 80%' with frequencies={'call': 0.6} -> contradicted."""
        claim = _claim("Call 80% of the time", "numeric", "action_frequency")
        tool = _action_solver_tool("call", {"call": 0.6, "fold": 0.4})
        result = verify_claim(claim, tool)
        assert result.verdict == "contradicted"

    def test_action_frequency_no_matching_action(self):
        """Claim about 'raise' but frequencies has no 'raise' key -> unverifiable."""
        claim = _claim("Raise 50% of the time", "numeric", "action_frequency")
        tool = _action_solver_tool("call", {"call": 0.6, "fold": 0.4})
        result = verify_claim(claim, tool)
        assert result.verdict == "unverifiable"


class TestSubstringFalsePositives:
    """Review finding #2: exact/categorical substring checks produce false supports."""

    def test_exact_pot_odds_substring_false_positive(self):
        """pot_odds='3:1' must NOT support 'Pot odds are 13:1 here'."""
        claim = _claim("Pot odds are 13:1 here", "numeric", "pot_odds")
        tool = _pot_odds_tool("3:1")
        result = verify_claim(claim, tool)
        assert result.verdict == "contradicted"

    def test_categorical_negation_false_positive(self):
        """recommended_action='call' must NOT support 'Do not call'."""
        claim = _claim(
            "Do not call, you should fold here", "action", "recommended_action"
        )
        tool = _action_solver_tool("call")
        result = verify_claim(claim, tool)
        assert result.verdict == "contradicted"

    def test_exact_pot_odds_correct_match_still_works(self):
        """pot_odds='3:1' should still support 'Pot odds are 3:1 here'."""
        claim = _claim("Pot odds are 3:1 here", "numeric", "pot_odds")
        tool = _pot_odds_tool("3:1")
        result = verify_claim(claim, tool)
        assert result.verdict == "supported"

    def test_categorical_positive_match_still_works(self):
        """recommended_action='call' should still support 'You should call'."""
        claim = _claim("You should call", "action", "recommended_action")
        tool = _action_solver_tool("call")
        result = verify_claim(claim, tool)
        assert result.verdict == "supported"


class TestUnknownTopicVerdict:
    """Review finding #3: unknown topics get 'unsupported' instead of 'unverifiable'
    because verify_claim checks tool_call==None before checking evidence contract."""

    def test_unknown_topic_through_full_flow(self):
        """route_claim -> None for unknown topic, verify_claim should return
        'unverifiable', not 'unsupported'."""
        claim = _claim("Implied odds are great", "conceptual", "implied_odds")
        trace = _full_trace()
        tool = route_claim(claim, trace)
        assert tool is None  # expected: unknown topic not in contract
        result = verify_claim(claim, tool)
        assert result.verdict == "unverifiable"

    def test_missing_tool_still_unsupported(self):
        """Known topic but tool missing from trace -> 'unsupported' (not unverifiable)."""
        claim = _claim("Your equity is 65%", "numeric", "equity")
        trace = ToolTrace(calls=[_pot_odds_tool()])  # no equity_calculator
        tool = route_claim(claim, trace)
        assert tool is None
        result = verify_claim(claim, tool)
        assert result.verdict == "unsupported"


class TestEmptyClaimsSkippedTools:
    """Review finding #4: empty-claim path suppresses check_required_tools."""

    @patch("src.failure_detector.generate_structured")
    def test_extraction_failure_still_reports_skipped_tools(self, mock_gen):
        """When extraction fails, skipped_tools must still be populated."""
        mock_gen.side_effect = Exception("LLM API error")
        output = CoachingOutput(
            recommended_action="call", explanation="Some text."
        )
        # Trace missing equity_calculator (required)
        trace = ToolTrace(calls=[_pot_odds_tool(), _action_solver_tool()])

        verdict = detect_failures(output, trace)

        assert verdict.overall_pass is False
        assert "equity_calculator" in verdict.skipped_tools

    @patch("src.failure_detector.generate_structured")
    def test_empty_claims_still_reports_skipped_tools(self, mock_gen):
        """When extraction yields 0 claims, skipped_tools must still be populated."""
        mock_gen.return_value = ClaimList(claims=[])
        output = CoachingOutput(
            recommended_action="call", explanation=""
        )
        # Trace missing all required tools
        trace = ToolTrace(calls=[])

        verdict = detect_failures(output, trace)

        assert verdict.overall_pass is False
        assert len(verdict.skipped_tools) > 0
