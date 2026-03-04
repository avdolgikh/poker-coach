"""Protocol and result type for optimization engines."""

from dataclasses import dataclass, field
from typing import Callable, Protocol

from src.models import CoachingOutput, Scenario, ScenarioAnnotated


@dataclass
class EngineResult:
    coach_fn: Callable[[Scenario], CoachingOutput | None]
    iterations: list[dict]
    best_train_composite: float
    best_label: str  # "v2", "gepa_best", "mipro_best"
    metadata: dict = field(default_factory=dict)


class OptimizationEngine(Protocol):
    def optimize(
        self,
        train: list[Scenario],
        initial_prompt: str,
        *,
        examples_pool: list[ScenarioAnnotated] | None = None,
        no_judge: bool = False,
    ) -> EngineResult: ...
