"""Shared provider adapter interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class ProviderExecution:
    """Normalized provider execution details."""

    provider: str
    role: str
    tier: str
    model: str
    output: str


class Provider(Protocol):
    """Provider contract for pipeline role execution."""

    name: str

    def run_role(
        self,
        *,
        role: str,
        prompt: str,
        repo_root: Path,
        schema: dict[str, Any] | None = None,
    ) -> ProviderExecution:
        """Run the requested role and return the final output plus resolved config."""
