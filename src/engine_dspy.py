"""DSPy optimization engines: GEPA and MIPROv2."""

import dspy

from src import config
from src.coaching_agent import _format_context
from src.evaluator import evaluate_soft
from src.models import CoachingOutput, Scenario, ScenarioAnnotated
from src.engine_protocol import EngineResult


# --- DSPy Signature and Module ---

class PokerCoaching(dspy.Signature):
    """Given a poker scenario with solver analysis, produce a coaching recommendation."""
    scenario: str = dspy.InputField(desc="Poker scenario with game state and solver output")
    recommended_action: str = dspy.OutputField(desc="The recommended poker action")
    explanation: str = dspy.OutputField(desc="Coaching explanation for the recommendation")


class PokerCoach(dspy.Module):
    def __init__(self):
        self.coach = dspy.Predict(PokerCoaching)

    def forward(self, scenario):
        return self.coach(scenario=scenario)


# --- Conversion helpers ---

def _to_dspy_example(scenario):
    """Convert a Scenario to a dspy.Example with input/output fields."""
    return dspy.Example(
        scenario=_format_context(scenario),
        scenario_id=scenario.id,
        recommended_action=scenario.solver_output.recommended_action,
    ).with_inputs("scenario")


def _make_coach_fn(compiled_module):
    """Wrap a compiled DSPy module as a coach_fn(Scenario) → CoachingOutput."""
    def coach_fn(scenario):
        context = _format_context(scenario)
        pred = compiled_module(scenario=context)
        return CoachingOutput(
            recommended_action=pred.recommended_action,
            explanation=pred.explanation,
        )
    return coach_fn


# --- Metric ---

def _build_metric(scenario_lookup, no_judge=False):
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        scenario = scenario_lookup[gold.scenario_id]
        output = CoachingOutput(
            recommended_action=pred.recommended_action,
            explanation=pred.explanation,
        )
        result = evaluate_soft(
            scenario, output,
            full_eval=isinstance(scenario, ScenarioAnnotated),
        )

        score = result.composite

        # GEPA's feedback_fn_creator passes pred_name; dspy.Evaluate does not.
        # dspy.Evaluate's parallelizer calls sum() on results — must be numeric.
        if pred_name is None:
            return score

        feedback_parts = []
        for g in result.gates.gates:
            if not g.passed:
                feedback_parts.append(f"GATE FAIL: {g.name} — {g.detail}")
        if result.rule_based:
            for field in ["concept_mention", "format_score", "ev_awareness", "position_awareness"]:
                val = getattr(result.rule_based, field, None)
                if val is not None and val < 0.5:
                    feedback_parts.append(f"LOW {field}={val:.2f}")

        return {"score": score, "feedback": "; ".join(feedback_parts) or "All checks passed."}
    return metric


def _build_mipro_metric(scenario_lookup, no_judge=False):
    base_metric = _build_metric(scenario_lookup, no_judge)

    def metric(example, prediction, trace=None):
        score = base_metric(example, prediction)  # returns float (pred_name=None)
        if trace is not None:
            return score >= 0.5
        return score
    return metric


# --- GEPA Engine ---

class GEPAEngine:
    def optimize(self, train, initial_prompt, *, examples_pool=None, no_judge=False):
        task_lm = dspy.LM(f"{config.LLM_PROVIDER}/{config.PROD_MODEL}")
        reflection_lm = dspy.LM(f"{config.LLM_PROVIDER}/{config.TUNER_MODEL}", temperature=1.0)
        dspy.configure(lm=task_lm)

        scenario_lookup = {s.id: s for s in train}
        trainset = [_to_dspy_example(s) for s in train]

        module = PokerCoach()
        module.coach.signature = module.coach.signature.with_instructions(initial_prompt)

        optimizer = dspy.GEPA(
            metric=_build_metric(scenario_lookup, no_judge),
            reflection_lm=reflection_lm,
            auto="light",
            num_threads=1,
        )
        compiled = optimizer.compile(module, trainset=trainset)

        config.DSPY_PROGRAMS_DIR.mkdir(exist_ok=True)
        compiled.save(str(config.DSPY_PROGRAMS_DIR / "gepa_optimized.json"))

        return EngineResult(
            coach_fn=_make_coach_fn(compiled),
            iterations=[{"optimizer": "gepa"}],
            best_train_composite=0.0,
            best_label="gepa_best",
            metadata={"optimized_instruction": compiled.coach.signature.instructions},
        )


# --- MIPROv2 Engine ---

class MIPROEngine:
    def optimize(self, train, initial_prompt, *, examples_pool=None, no_judge=False):
        task_lm = dspy.LM(f"{config.LLM_PROVIDER}/{config.PROD_MODEL}")
        prompt_lm = dspy.LM(f"{config.LLM_PROVIDER}/{config.TUNER_MODEL}")
        dspy.configure(lm=task_lm)

        scenario_lookup = {s.id: s for s in train}
        trainset = [_to_dspy_example(s) for s in train]

        module = PokerCoach()
        module.coach.signature = module.coach.signature.with_instructions(initial_prompt)

        optimizer = dspy.MIPROv2(
            metric=_build_mipro_metric(scenario_lookup, no_judge),
            prompt_model=prompt_lm,
            task_model=task_lm,
            auto="light",
            num_threads=1,
        )
        compiled = optimizer.compile(
            module, trainset=trainset,
            max_bootstrapped_demos=3, max_labeled_demos=3,
        )

        config.DSPY_PROGRAMS_DIR.mkdir(exist_ok=True)
        compiled.save(str(config.DSPY_PROGRAMS_DIR / "mipro_optimized.json"))

        return EngineResult(
            coach_fn=_make_coach_fn(compiled),
            iterations=[{"optimizer": "mipro"}],
            best_train_composite=0.0,
            best_label="mipro_best",
            metadata={
                "optimized_instruction": compiled.coach.signature.instructions,
                "num_demos": len(compiled.coach.demos),
            },
        )
