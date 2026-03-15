"""Agentic Orchestration Failure Detector.

Multi-step pipeline: extract claims -> route to tools -> verify -> verdict.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from src import config
from src.llm import generate_structured
from src.models import (
    Claim,
    ClaimVerification,
    CoachingOutput,
    FailureLabel,
    FailureVerdict,
    Scenario,
    ToolCall,
    ToolTrace,
)


# --- Internal models for LLM structured output ---


class ClaimList(BaseModel):
    """Wrapper for structured claim extraction."""

    claims: list[Claim]


class SemanticVerificationResult(BaseModel):
    """LLM result for semantic claim verification."""

    is_supported: bool
    reasoning: str


# --- Prompts ---

_EXTRACT_CLAIMS_SYSTEM = """\
You are a claim extraction agent. Given a poker coaching explanation, extract every \
factual claim the coach makes.

For each claim, identify:
- text: the exact claim text from the coaching output
- claim_type: one of "numeric" (quantitative values like percentages, ratios), \
"categorical" (discrete values like position), "action" (recommended poker action), \
"conceptual" (qualitative assessments like board texture)
- topic: what the claim is about. Use one of: "equity", "pot_odds", "ev", \
"action_frequency", "recommended_action", "position", "board_texture". \
If the claim doesn't fit any topic, use the most descriptive short label.

Extract ALL factual assertions. Do not infer or add claims not present in the text."""

_SEMANTIC_VERIFY_SYSTEM = """\
You are an evidence verification agent. Given a claim from a poker coaching explanation \
and the output of a tool that should support it, determine whether the claim is \
consistent with the tool output.

Respond with:
- is_supported: true if the claim is consistent with the tool output, false otherwise
- reasoning: brief explanation of your judgment"""


# --- Public API ---


def extract_claims(coaching_output: CoachingOutput) -> list[Claim]:
    """LLM: Extract factual claims from coaching text.

    Returns empty list on LLM failure (graceful degradation).
    """
    try:
        result = generate_structured(
            _EXTRACT_CLAIMS_SYSTEM,
            coaching_output.explanation,
            ClaimList,
            model=config.TUNER_MODEL,
        )
        return result.claims
    except Exception:
        return []


def route_claim(claim: Claim, trace: ToolTrace) -> ToolCall | None:
    """Deterministic: Map claim to backing tool via evidence contract.

    Returns None if claim topic is unknown or tool is missing from trace.
    """
    entry = config.EVIDENCE_CONTRACT.get(claim.topic)
    if entry is None:
        return None
    required_tool = entry["required_tool"]
    for call in trace.calls:
        if call.tool_name == required_tool:
            return call
    return None


def verify_claim(claim: Claim, tool_call: ToolCall | None) -> ClaimVerification:
    """Verify claim against tool output.

    Dispatches by verification type from evidence contract:
    - numeric: deterministic (parse number, compare with tolerance)
    - exact: deterministic (verbatim string match)
    - categorical: deterministic (normalized string comparison)
    - semantic: LLM (semantic alignment check)

    Returns "unsupported" if tool_call is None.
    Returns "unverifiable" if claim topic not in evidence contract.
    """
    entry = config.EVIDENCE_CONTRACT.get(claim.topic)
    if entry is None:
        return ClaimVerification(
            claim=claim,
            matched_tool=tool_call.tool_name if tool_call else None,
            verdict="unverifiable",
            reasoning=f"Topic '{claim.topic}' not in evidence contract",
            severity="info",
        )

    if tool_call is None:
        return ClaimVerification(
            claim=claim,
            matched_tool=None,
            verdict="unsupported",
            reasoning="No matching tool in trace",
            severity="info",
        )

    vtype = entry["verification_type"]
    if vtype == "numeric":
        return _verify_numeric(claim, tool_call, entry)
    elif vtype == "exact":
        return _verify_exact(claim, tool_call, entry)
    elif vtype == "categorical":
        return _verify_categorical(claim, tool_call, entry)
    elif vtype == "semantic":
        return _verify_semantic(claim, tool_call, entry)
    elif vtype == "action_frequency":
        return _verify_action_frequency(claim, tool_call, entry)

    return ClaimVerification(
        claim=claim,
        matched_tool=tool_call.tool_name,
        verdict="unverifiable",
        reasoning=f"Unknown verification type '{vtype}'",
        severity="info",
    )


def check_required_tools(trace: ToolTrace) -> list[str]:
    """Deterministic: Return names of required tools missing from trace."""
    required = set()
    for entry in config.EVIDENCE_CONTRACT.values():
        if entry.get("required", False):
            required.add(entry["required_tool"])
    present = {c.tool_name for c in trace.calls}
    return sorted(required - present)


def synthesize_verdict(
    verifications: list[ClaimVerification],
    skipped_tools: list[str],
) -> FailureVerdict:
    """Deterministic: Aggregate verifications + skipped tools into final verdict."""
    claims = [v.claim for v in verifications]
    failures: list[FailureLabel] = []

    for v in verifications:
        if v.verdict == "contradicted":
            ftype = (
                "concept_misapplication"
                if v.claim.claim_type == "conceptual"
                else "contradiction"
            )
            failures.append(
                FailureLabel(
                    failure_type=ftype,
                    claim_text=v.claim.text,
                    tool_name=v.matched_tool,
                    detail=v.reasoning,
                )
            )

    for tool_name in skipped_tools:
        failures.append(
            FailureLabel(
                failure_type="skipped_tool",
                tool_name=tool_name,
                detail=f"Required tool '{tool_name}' missing from trace",
            )
        )

    has_hard = (
        any(v.severity == "hard_fail" for v in verifications)
        or len(skipped_tools) > 0
    )

    parts = []
    if not failures:
        parts.append("All claims verified successfully.")
    else:
        parts.append(f"{len(failures)} failure(s) detected:")
        for f in failures:
            parts.append(f"  - {f.failure_type}: {f.detail}")

    return FailureVerdict(
        claims=claims,
        verifications=verifications,
        skipped_tools=skipped_tools,
        failures=failures,
        has_hard_failure=has_hard,
        overall_pass=not has_hard,
        summary="\n".join(parts),
    )


def detect_failures(
    coaching_output: CoachingOutput, trace: ToolTrace
) -> FailureVerdict:
    """Agentic pipeline: extract claims -> route -> verify -> synthesize verdict.

    On extraction failure: returns fail-safe verdict (overall_pass=False, 0 claims).
    """
    claims = extract_claims(coaching_output)
    skipped = check_required_tools(trace)

    if not claims:
        failures = [
            FailureLabel(
                failure_type="skipped_tool",
                tool_name=t,
                detail=f"Required tool '{t}' missing from trace",
            )
            for t in skipped
        ]
        return FailureVerdict(
            claims=[],
            verifications=[],
            skipped_tools=skipped,
            failures=failures,
            has_hard_failure=True,
            overall_pass=False,
            summary="No claims extracted — cannot verify coaching output.",
        )

    verifications = []
    for claim in claims:
        tool_call = route_claim(claim, trace)
        verification = verify_claim(claim, tool_call)
        verifications.append(verification)

    return synthesize_verdict(verifications, skipped)


def scenario_to_trace(scenario: Scenario) -> ToolTrace:
    """Convert existing Scenario into a ToolTrace for testing.

    Decomposes solver_output into individual tool calls:
    - equity_calculator from solver_output.equity
    - pot_odds_calculator from solver_output.pot_odds
    - ev_calculator from solver_output.ev
    - action_frequency_solver from solver_output.action_frequencies + recommended_action
    - position_analyzer from game_state.position
    """
    so = scenario.solver_output
    gs = scenario.game_state
    return ToolTrace(
        calls=[
            ToolCall(
                tool_name="equity_calculator",
                input={"hand": gs.hand, "board": gs.board},
                output={"equity": so.equity},
            ),
            ToolCall(
                tool_name="pot_odds_calculator",
                input={"pot": gs.pot, "villain_action": gs.villain_action},
                output={"pot_odds": so.pot_odds},
            ),
            ToolCall(
                tool_name="ev_calculator",
                input={},
                output={"ev": so.ev},
            ),
            ToolCall(
                tool_name="action_frequency_solver",
                input={},
                output={
                    "recommended_action": so.recommended_action,
                    "frequencies": so.action_frequencies,
                },
            ),
            ToolCall(
                tool_name="position_analyzer",
                input={},
                output={"position": gs.position},
            ),
        ]
    )


def compute_metrics(
    verdicts: list[FailureVerdict],
    label_sets: list[list[FailureLabel]],
) -> dict:
    """Compute precision, recall, F1 on failure detection.

    Matches predictions to ground truth by (failure_type, tool_name).
    Returns {"precision": float, "recall": float, "f1": float}.
    """
    tp = fp = fn = 0
    for verdict, labels in zip(verdicts, label_sets):
        predicted = {(f.failure_type, f.tool_name) for f in verdict.failures}
        actual = {(l.failure_type, l.tool_name) for l in labels}
        tp += len(predicted & actual)
        fp += len(predicted - actual)
        fn += len(actual - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# --- Private verification helpers ---


def _verify_numeric(
    claim: Claim, tool_call: ToolCall, contract: dict
) -> ClaimVerification:
    """Parse number from claim, compare to tool output with tolerance."""
    output_field = contract["output_field"]
    tolerance = contract["tolerance"]
    expected = tool_call.output.get(output_field)

    if expected is None:
        return ClaimVerification(
            claim=claim,
            matched_tool=tool_call.tool_name,
            verdict="unverifiable",
            reasoning=f"Tool output missing field '{output_field}'",
            severity="info",
        )

    # Try percentage first (e.g., "65%"), then raw number (e.g., "+36")
    pct_matches = re.findall(r"([-+]?\d+(?:\.\d+)?)\s*%", claim.text)
    if pct_matches:
        claimed = float(pct_matches[0]) / 100.0
    else:
        num_matches = re.findall(r"([-+]?\d+(?:\.\d+)?)", claim.text)
        if not num_matches:
            return ClaimVerification(
                claim=claim,
                matched_tool=tool_call.tool_name,
                verdict="unverifiable",
                reasoning="Could not parse numeric value from claim",
                severity="info",
            )
        claimed = float(num_matches[0])

    diff = abs(claimed - float(expected))
    if diff <= tolerance:
        verdict = "supported"
        reasoning = (
            f"Claimed {claimed} matches expected {expected} "
            f"(diff={diff:.4f}, tolerance={tolerance})"
        )
    else:
        verdict = "contradicted"
        reasoning = (
            f"Claimed {claimed} differs from expected {expected} "
            f"(diff={diff:.4f}, tolerance={tolerance})"
        )

    return ClaimVerification(
        claim=claim,
        matched_tool=tool_call.tool_name,
        verdict=verdict,
        reasoning=reasoning,
        severity=_severity(contract, verdict),
    )


def _verify_exact(
    claim: Claim, tool_call: ToolCall, contract: dict
) -> ClaimVerification:
    """Check if tool output value appears as whole token in claim text."""
    output_field = contract["output_field"]
    expected = str(tool_call.output.get(output_field, ""))

    pattern = r"(?<!\w)" + re.escape(expected) + r"(?!\w)"
    if re.search(pattern, claim.text):
        verdict = "supported"
        reasoning = f"'{expected}' found verbatim in claim"
    else:
        verdict = "contradicted"
        reasoning = f"'{expected}' not found in claim text"

    return ClaimVerification(
        claim=claim,
        matched_tool=tool_call.tool_name,
        verdict=verdict,
        reasoning=reasoning,
        severity=_severity(contract, verdict),
    )


_NEGATION_WORDS = {"not", "don't", "dont", "never", "shouldn't", "shouldnt", "no"}


def _verify_categorical(
    claim: Claim, tool_call: ToolCall, contract: dict
) -> ClaimVerification:
    """Normalize and compare categorical values with negation awareness."""
    output_field = contract["output_field"]
    expected = str(tool_call.output.get(output_field, "")).lower().strip()
    claim_lower = claim.text.lower()

    pattern = r"(?<!\w)" + re.escape(expected) + r"(?!\w)"
    match = re.search(pattern, claim_lower)
    if match:
        # Check for negation in the 4 words preceding the match
        prefix = claim_lower[: match.start()].split()
        window = prefix[-4:] if len(prefix) >= 4 else prefix
        if any(w.strip(".,;:!?") in _NEGATION_WORDS for w in window):
            verdict = "contradicted"
            reasoning = f"'{expected}' found but negated in claim"
        else:
            verdict = "supported"
            reasoning = f"'{expected}' found in claim"
    else:
        verdict = "contradicted"
        reasoning = f"Expected '{expected}' not found in claim text"

    return ClaimVerification(
        claim=claim,
        matched_tool=tool_call.tool_name,
        verdict=verdict,
        reasoning=reasoning,
        severity=_severity(contract, verdict),
    )


def _verify_action_frequency(
    claim: Claim, tool_call: ToolCall, contract: dict
) -> ClaimVerification:
    """Parse action + frequency from claim, compare to frequencies dict."""
    output_field = contract["output_field"]
    tolerance = contract["tolerance"]
    frequencies = tool_call.output.get(output_field)

    if not isinstance(frequencies, dict):
        return ClaimVerification(
            claim=claim,
            matched_tool=tool_call.tool_name,
            verdict="unverifiable",
            reasoning=f"Tool output '{output_field}' is not a frequency map",
            severity="info",
        )

    # Find which action the claim references
    claim_lower = claim.text.lower()
    matched_action = None
    for action in frequencies:
        pattern = r"(?<!\w)" + re.escape(action.lower()) + r"(?!\w)"
        if re.search(pattern, claim_lower):
            matched_action = action
            break

    if matched_action is None:
        return ClaimVerification(
            claim=claim,
            matched_tool=tool_call.tool_name,
            verdict="unverifiable",
            reasoning=f"No recognized action found in claim for frequency map",
            severity="info",
        )

    expected_freq = frequencies[matched_action]

    # Parse percentage from claim
    pct_matches = re.findall(r"([-+]?\d+(?:\.\d+)?)\s*%", claim.text)
    if pct_matches:
        claimed = float(pct_matches[0]) / 100.0
    else:
        return ClaimVerification(
            claim=claim,
            matched_tool=tool_call.tool_name,
            verdict="unverifiable",
            reasoning="Could not parse frequency percentage from claim",
            severity="info",
        )

    diff = abs(claimed - float(expected_freq))
    if diff <= tolerance:
        verdict = "supported"
        reasoning = (
            f"Claimed {matched_action} {claimed:.0%} matches expected "
            f"{expected_freq} (diff={diff:.4f}, tolerance={tolerance})"
        )
    else:
        verdict = "contradicted"
        reasoning = (
            f"Claimed {matched_action} {claimed:.0%} differs from expected "
            f"{expected_freq} (diff={diff:.4f}, tolerance={tolerance})"
        )

    return ClaimVerification(
        claim=claim,
        matched_tool=tool_call.tool_name,
        verdict=verdict,
        reasoning=reasoning,
        severity=_severity(contract, verdict),
    )


def _verify_semantic(
    claim: Claim, tool_call: ToolCall, contract: dict
) -> ClaimVerification:
    """LLM semantic comparison for conceptual claims."""
    user_msg = (
        f'Claim: "{claim.text}"\n\n'
        f"Tool output ({tool_call.tool_name}):\n{tool_call.output}"
    )
    try:
        result = generate_structured(
            _SEMANTIC_VERIFY_SYSTEM,
            user_msg,
            SemanticVerificationResult,
            model=config.TUNER_MODEL,
        )
        verdict = "supported" if result.is_supported else "contradicted"
        reasoning = result.reasoning
    except Exception:
        verdict = "unverifiable"
        reasoning = "Semantic verification failed (LLM error)"

    return ClaimVerification(
        claim=claim,
        matched_tool=tool_call.tool_name,
        verdict=verdict,
        reasoning=reasoning,
        severity=_severity(contract, verdict),
    )


def _severity(contract: dict, verdict: str) -> str:
    """Determine severity based on contract required flag and verdict."""
    if verdict != "contradicted":
        return "info"
    return "hard_fail" if contract.get("required", False) else "soft_fail"
