"""Codex CLI provider adapter."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from src.coding_pipeline.core import EXIT_PROVIDER_EXEC_FAILED, PipelineError
from src.coding_pipeline.providers.base import ProviderExecution


@dataclass(frozen=True)
class RoleConfig:
    tier: str
    model: str


class CodexProvider:
    """Runs pipeline roles through the Codex CLI."""

    name = "codex"
    _sandbox_for_role = {
        "test-writer": "danger-full-access",
        "implementer": "danger-full-access",
        "reviewer": "danger-full-access",
    }

    def __init__(self) -> None:
        self.role_configs = {
            "test-writer": RoleConfig(
                tier="economy",
                model=os.getenv("CODEX_MODEL_TEST_WRITER", "gpt-5.1-codex-mini"),
            ),
            "implementer": RoleConfig(
                tier="economy",
                model=os.getenv("CODEX_MODEL_IMPLEMENTER", "gpt-5.1-codex"),
            ),
            "reviewer": RoleConfig(
                tier="premium",
                model=os.getenv("CODEX_MODEL_REVIEWER", "gpt-5.2-codex"),
            ),
        }
        self.executable = Path(os.getenv("APPDATA", "")) / "npm" / "codex.cmd"

    def _scratch_dir(self, repo_root: Path, role: str) -> Path:
        path = repo_root / ".pipeline-state" / "codex" / role
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _codex_home(self, repo_root: Path) -> Path:
        path = repo_root / ".pipeline-state" / "codex-home"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _seed_codex_home(self, repo_root: Path) -> Path:
        codex_home = self._codex_home(repo_root)
        source_home = Path.home() / ".codex"
        for filename in ("auth.json", "config.toml"):
            source = source_home / filename
            if source.exists():
                shutil.copy2(source, codex_home / filename)
        return codex_home

    def _provider_env(self, repo_root: Path) -> dict[str, str]:
        env = os.environ.copy()
        codex_home = self._seed_codex_home(repo_root)
        env["CODEX_HOME"] = str(codex_home)
        env["HOME"] = str(codex_home)
        return env

    def _tail(self, text: str, limit: int = 2000) -> str:
        stripped = text.strip()
        if len(stripped) <= limit:
            return stripped
        return "[truncated]\n" + stripped[-limit:]

    def _command(
        self,
        *,
        role: str,
        repo_root: Path,
        output_path: Path,
        schema_path: Path | None,
    ) -> list[str]:
        config = self.role_configs[role]
        command = [
            str(self.executable),
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--color",
            "never",
            "-C",
            str(repo_root),
            "--output-last-message",
            str(output_path),
            "--sandbox",
            self._sandbox_for_role[role],
            "--model",
            config.model,
        ]
        if schema_path is not None:
            command.extend(["--output-schema", str(schema_path)])
        command.append("-")
        return command

    def run_role(
        self,
        *,
        role: str,
        prompt: str,
        repo_root: Path,
        schema: dict[str, Any] | None = None,
    ) -> ProviderExecution:
        if not self.executable.exists():
            raise PipelineError(
                "FAIL: Codex CLI was not found at the expected Windows npm shim path.",
                EXIT_PROVIDER_EXEC_FAILED,
            )
        config = self.role_configs[role]
        scratch_dir = self._scratch_dir(repo_root, role)
        output_path = scratch_dir / "last_message.txt"
        if output_path.exists():
            output_path.unlink()
        schema_path: Path | None = None
        if schema is not None:
            schema_path = scratch_dir / "schema.json"
            schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        result = subprocess.run(
            self._command(
                role=role,
                repo_root=repo_root,
                output_path=output_path,
                schema_path=schema_path,
            ),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            input=prompt,
            env=self._provider_env(repo_root),
        )
        final_message = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        output = final_message.strip()
        if result.returncode != 0:
            details: list[str] = []
            if result.stdout.strip():
                details.append("stdout:\n" + self._tail(result.stdout))
            if result.stderr.strip():
                details.append("stderr:\n" + self._tail(result.stderr))
            if final_message.strip():
                details.append("last_message:\n" + self._tail(final_message))
            suffix = ("\n" + "\n\n".join(details)) if details else ""
            raise PipelineError(
                f"FAIL: Codex provider execution failed for role {role} (exit {result.returncode}).{suffix}",
                EXIT_PROVIDER_EXEC_FAILED,
            )
        return ProviderExecution(
            provider=self.name,
            role=role,
            tier=config.tier,
            model=config.model,
            output=output,
        )
