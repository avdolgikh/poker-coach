"""Provider-agnostic orchestration for the agentic TDD pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from src.coding_pipeline.providers.base import Provider, ProviderExecution


REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["approve", "revise"]},
        "summary": {"type": "string"},
        "blocking": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["decision", "summary", "blocking"],
    "additionalProperties": False,
}

STAGE_ORDER = {
    "SPEC_APPROVED": 0,
    "TESTS_GENERATED": 1,
    "TESTS_FROZEN": 2,
    "CODE_IMPLEMENTED": 3,
    "CODE_VALIDATED": 4,
    "DONE": 5,
}

REPO_HASH_TARGETS = [
    ".claude",
    "agents.md",
    "pyproject.toml",
    "scripts",
    "specs",
    "src",
    "tests",
]

TRANSIENT_PATH_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
TRANSIENT_SUFFIXES = {".pyc", ".pyo"}

EXIT_SUCCESS = 0
EXIT_SPEC_NOT_FOUND = 1
EXIT_TEST_REVISION_CAP = 2
EXIT_CODE_REVISION_CAP = 3
EXIT_FROZEN_TESTS_MODIFIED = 4
EXIT_TESTS_BROKE_AFTER_REVISION = 5
EXIT_STATE_PROVIDER_MISMATCH = 6
EXIT_REVIEWER_MODIFIED_FILES = 7
EXIT_INVALID_REVIEW_OUTPUT = 8
EXIT_PROVIDER_EXEC_FAILED = 9
EXIT_STAGE_NO_EFFECT = 10


class PipelineError(RuntimeError):
    """Pipeline failure with an explicit exit code."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass
class ReviewDecision:
    """Normalized reviewer decision."""

    decision: str
    summary: str
    blocking: list[str] = field(default_factory=list)
    fallback_used: bool = False


def _review_requests_missing_inputs(decision: ReviewDecision) -> bool:
    haystack = " ".join([decision.summary, *decision.blocking]).lower()
    phrases = [
        "missing review input",
        "missing review inputs",
        "review packet",
        "provide the spec",
        "provide the review packet",
        "what should i review",
        "where are the artifacts",
        "i don't have",
        "i do not have",
        "task id",
    ]
    return any(phrase in haystack for phrase in phrases)


def _stage_requested_more_input(raw: str) -> bool:
    haystack = raw.lower()
    phrases = [
        "role acknowledged",
        "please point me to",
        "please provide",
        "clarify any particular",
        "what task",
        "what should i review",
        "review packet",
        "share that too",
        "i'm set in the",
        "i?m set in the",
        "i'm ready to act",
        "i?m ready to act",
        "please include",
    ]
    return any(phrase in haystack for phrase in phrases)


@dataclass
class PipelineState:
    """Persisted pipeline state."""

    task: str
    provider: str
    stage: str
    iteration: int = 0
    frozen_tests_hash: str | None = None


class PipelineLogger:
    """Writes pipeline output to stdout and the transcript log."""

    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _write_stdout(self, message: str) -> None:
        stream = sys.stdout
        line = f"{message}\n"
        buffer = getattr(stream, "buffer", None)
        if buffer is not None:
            encoding = stream.encoding or "utf-8"
            buffer.write(line.encode(encoding, errors="replace"))
            buffer.flush()
            return
        stream.write(line)

    def log(self, message: str = "") -> None:
        self._write_stdout(message)
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(message)
            handle.write("\n")


class PromptBuilder:
    """Renders shared role prompts with stage-specific context."""

    def __init__(self, repo_root: Path, *, rules_file: str = "agents.md") -> None:
        self.repo_root = repo_root
        self.rules_file = rules_file
        self.prompts_dir = repo_root / "src" / "coding_pipeline" / "prompts"

    def role_prompt(self, role: str) -> str:
        filename = role.replace("-", "_") + ".md"
        prompt_path = self.prompts_dir / filename
        return prompt_path.read_text(encoding="utf-8-sig").strip()

    def render(
        self,
        *,
        role: str,
        task: str,
        spec_path: Path,
        stage_name: str,
        stage_instruction: str,
        iteration: int = 0,
        reviewer_feedback: list[str] | None = None,
    ) -> str:
        feedback = reviewer_feedback or []
        spec_text = spec_path.read_text(encoding="utf-8-sig").strip()
        sections = [
            self.role_prompt(role),
            "",
            "## Pipeline Context",
            f"- Task: {task}",
            f"- Stage: {stage_name}",
            f"- Iteration: {iteration}",
            f"- Spec path: {spec_path.as_posix()}",
            f"- Repo rules path: {self.rules_file}",
            "- This pipeline is non-interactive. Do not ask the human for more input; use the embedded spec/context and inspect the repository directly.",
            "- Tests must be run with `uv run pytest` or `uv run python -m pytest`.",
        ]
        if feedback:
            sections.extend(["- Reviewer blocking feedback to address:"])
            sections.extend([f"  - {item}" for item in feedback])
        sections.extend(
            [
                "",
                "## Approved Spec",
                spec_text,
                "",
                "## Stage Instructions",
                stage_instruction.strip(),
            ]
        )
        if role == "reviewer":
            sections.extend(
                [
                    "",
                    "## Required Final Response",
                    "Return exactly one raw JSON object matching this shape:",
                    '{"decision":"approve|revise","summary":"string","blocking":["string"]}',
                    "Do not ask questions. Do not add markdown fences. Do not add any prose before or after the JSON object.",
                ]
            )
        return "\n".join(sections).strip() + "\n"


def _json_candidates(raw: str) -> list[str]:
    candidates: list[str] = []
    stripped = raw.strip()
    if stripped:
        candidates.append(stripped)
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    candidates.extend(fenced)
    for start in (match.start() for match in re.finditer(r"\{", raw)):
        depth = 0
        for idx in range(start, len(raw)):
            char = raw[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(raw[start : idx + 1])
                    break
    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def normalize_review_output(raw: str) -> ReviewDecision:
    """Validate and normalize provider review output."""

    fallback_used = False
    last_error: Exception | None = None
    for candidate in _json_candidates(raw):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            fallback_used = True
            continue

        source = payload.get("structured_output", payload) if isinstance(payload, dict) else payload
        if not isinstance(source, dict):
            fallback_used = True
            continue
        decision = source.get("decision")
        summary = source.get("summary")
        blocking = source.get("blocking", [])
        if decision not in {"approve", "revise"}:
            fallback_used = True
            continue
        if not isinstance(summary, str) or not summary.strip():
            fallback_used = True
            continue
        if blocking is None:
            blocking = []
        if not isinstance(blocking, list) or not all(isinstance(item, str) for item in blocking):
            fallback_used = True
            continue
        return ReviewDecision(
            decision=decision,
            summary=summary,
            blocking=blocking,
            fallback_used=fallback_used,
        )

    detail = f" Unable to normalize review output: {last_error}" if last_error else ""
    raise PipelineError(
        f"FAIL: reviewer output did not match the canonical schema.{detail}",
        EXIT_INVALID_REVIEW_OUTPUT,
    )


def _is_hashable_file(repo_root: Path, path: Path) -> bool:
    relative = path.relative_to(repo_root)
    if any(part in TRANSIENT_PATH_PARTS for part in relative.parts):
        return False
    if path.suffix in TRANSIENT_SUFFIXES:
        return False
    return True


def _iter_hashable_files(repo_root: Path, relative_targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for relative in relative_targets:
        target = repo_root / relative
        if not target.exists():
            continue
        if target.is_file():
            if _is_hashable_file(repo_root, target):
                files.append(target)
            continue
        for child in sorted(path for path in target.rglob("*") if path.is_file()):
            if _is_hashable_file(repo_root, child):
                files.append(child)
    return sorted(files)


def hash_paths(repo_root: Path, relative_targets: list[str]) -> str:
    """Return a deterministic SHA-256 over file paths and content."""

    digest = hashlib.sha256()
    for path in _iter_hashable_files(repo_root, relative_targets):
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class PipelineRunner:
    """Runs the provider-generalized autonomous pipeline."""

    def __init__(
        self,
        *,
        repo_root: Path,
        task: str,
        provider: Provider,
        max_revisions: int = 4,
        rules_file: str = "agents.md",
    ) -> None:
        self.repo_root = repo_root
        self.task = task
        self.provider = provider
        self.max_revisions = max_revisions
        self.rules_file = rules_file
        self.spec_path = repo_root / "specs" / f"{task}-spec.md"
        self.state_dir = repo_root / ".pipeline-state"
        self.state_file = self.state_dir / f"{task}.json"
        self.log_file = self.state_dir / f"{task}.log"
        self.logger = PipelineLogger(self.log_file)
        self.prompts = PromptBuilder(repo_root, rules_file=rules_file)

    def _log_stage_header(self, label: str) -> None:
        self.logger.log("")
        self.logger.log("=" * len(label))
        self.logger.log(label)
        self.logger.log("=" * len(label))

    def _load_state(self) -> PipelineState:
        if not self.state_file.exists():
            return PipelineState(task=self.task, provider=self.provider.name, stage="SPEC_APPROVED")

        payload = json.loads(self.state_file.read_text(encoding="utf-8"))
        state = PipelineState(
            task=payload["task"],
            provider=payload["provider"],
            stage=payload["stage"],
            iteration=int(payload.get("iteration", 0)),
            frozen_tests_hash=payload.get("frozen_tests_hash"),
        )
        if state.provider != self.provider.name:
            raise PipelineError(
                (
                    "FAIL: resume provider mismatch. "
                    f"State recorded provider={state.provider}, requested provider={self.provider.name}."
                ),
                EXIT_STATE_PROVIDER_MISMATCH,
            )
        self.logger.log(
            f"[resume] Continuing from stage={state.stage} iteration={state.iteration} provider={state.provider}"
        )
        return state

    def _save_state(
        self, stage: str, *, iteration: int = 0, frozen_tests_hash: str | None = None
    ) -> None:
        current_hash = frozen_tests_hash
        if current_hash is None and self.state_file.exists():
            previous = json.loads(self.state_file.read_text(encoding="utf-8"))
            current_hash = previous.get("frozen_tests_hash")
        state = PipelineState(
            task=self.task,
            provider=self.provider.name,
            stage=stage,
            iteration=iteration,
            frozen_tests_hash=current_hash,
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
        details = f" provider={self.provider.name}"
        if state.frozen_tests_hash:
            details += f" frozen_tests_hash={state.frozen_tests_hash}"
        self.logger.log(f"[state] {stage} (iteration={iteration}{details})")

    def _past(self, resume_stage: str, target_stage: str) -> bool:
        return STAGE_ORDER[resume_stage] >= STAGE_ORDER[target_stage]

    def _start_iteration(self, state: PipelineState, loop_entry_stage: str) -> int:
        return state.iteration if state.stage == loop_entry_stage else 0

    def _repo_hash(self) -> str:
        return hash_paths(self.repo_root, REPO_HASH_TARGETS)

    def _tests_hash(self) -> str:
        return hash_paths(self.repo_root, ["tests"])

    def _enforce_reviewer_immutability(self, before_hash: str, stage_label: str) -> None:
        after_hash = self._repo_hash()
        if before_hash != after_hash:
            raise PipelineError(
                f"FAIL: reviewer stage modified repository files during {stage_label}.",
                EXIT_REVIEWER_MODIFIED_FILES,
            )

    def _enforce_test_freeze(self, frozen_tests_hash: str | None) -> None:
        if not frozen_tests_hash:
            raise PipelineError(
                "FAIL: frozen tests hash missing from pipeline state.",
                EXIT_FROZEN_TESTS_MODIFIED,
            )
        current_hash = self._tests_hash()
        if current_hash != frozen_tests_hash:
            raise PipelineError(
                "FAIL: frozen test files were modified after the test-freeze boundary.",
                EXIT_FROZEN_TESTS_MODIFIED,
            )

    def _spec_priority_terms(self) -> list[str]:
        terms = {self.task.lower(), self.task.lower().replace("-", "_")}
        try:
            spec_text = self.spec_path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            return sorted(term for term in terms if term)
        for raw in re.findall(r"`([^`\n]+)`", spec_text):
            normalized = raw.replace("\\", "/").strip().lower()
            if "/" not in normalized and not normalized.endswith(".py"):
                continue
            terms.add(normalized)
            terms.add(Path(normalized).name.lower())
            terms.add(Path(normalized).stem.lower())
            if normalized.endswith(".py"):
                terms.add(f"test_{Path(normalized).stem.lower()}.py")
        return sorted(term for term in terms if term)

    def _artifact_snapshot(self, relative_targets: list[str]) -> str:
        max_total_chars = 3000
        max_file_chars = 900
        candidate_files = [
            path
            for path in _iter_hashable_files(self.repo_root, relative_targets)
            if path.suffix
            in {
                ".py",
                ".md",
                ".toml",
                ".txt",
                ".json",
                ".yaml",
                ".yml",
                ".sh",
                ".ps1",
            }
        ]
        if not candidate_files:
            return ""

        priority_terms = self._spec_priority_terms()

        def sort_key(path: Path) -> tuple[int, int, int, str]:
            relative = path.relative_to(self.repo_root).as_posix().lower()
            hits = sum(1 for term in priority_terms if term in relative)
            exact = 0 if relative in priority_terms else 1
            return (exact, -hits, len(relative), relative)

        candidate_files = sorted(candidate_files, key=sort_key)

        sections: list[str] = []
        listed_files = [path.relative_to(self.repo_root).as_posix() for path in candidate_files]
        manifest_lines = listed_files[:20]
        if len(listed_files) > 20:
            manifest_lines.append(f"... [{len(listed_files) - 20} more files]")
        manifest = "### Workspace Files\n```text\n" + "\n".join(manifest_lines) + "\n```"
        sections.append(manifest)
        total_chars = len(manifest)
        truncated = False
        for path in candidate_files:
            relative = path.relative_to(self.repo_root).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if len(text) > max_file_chars:
                text = text[: max_file_chars - 15].rstrip() + "\n... [truncated]"
            section = f"### {relative}\n```text\n{text}\n```"
            if total_chars + len(section) > max_total_chars:
                truncated = True
                break
            sections.append(section)
            total_chars += len(section)
        if truncated:
            sections.append(
                "... [artifact snapshot truncated to stay within Windows command limits]"
            )
        return "\n\n".join(sections)

    def _ensure_tests_stage_effect(
        self,
        *,
        before_hash: str,
        execution: ProviderExecution,
        stage_label: str,
        allow_existing: bool = False,
    ) -> None:
        after_hash = self._tests_hash()
        if after_hash != before_hash:
            return
        if allow_existing and any(_iter_hashable_files(self.repo_root, ["tests"])):
            return
        base_message = f"FAIL: {stage_label} did not modify tests/."
        if stage_label == "Stage 1: Test Generation":
            raise PipelineError(base_message, EXIT_STAGE_NO_EFFECT)
        if "Revision" in stage_label:
            raise PipelineError(
                base_message,
                EXIT_STAGE_NO_EFFECT,
            )
        if _stage_requested_more_input(execution.output):
            raise PipelineError(
                f"FAIL: {stage_label} did not modify tests/ and instead requested more input.",
                EXIT_STAGE_NO_EFFECT,
            )

    def _run_pytest_gate(self, label: str) -> None:
        self.logger.log("")
        self.logger.log(label)
        result = subprocess.run(
            ["uv", "run", "python", "-m", "pytest"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            for line in result.stdout.rstrip().splitlines():
                self.logger.log(line)
        if result.stderr:
            for line in result.stderr.rstrip().splitlines():
                self.logger.log(line)
        if result.returncode != 0:
            raise PipelineError(
                "FAIL: tests broke after code revision.", EXIT_TESTS_BROKE_AFTER_REVISION
            )

    def _run_role(
        self,
        *,
        role: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        stage_label: str,
    ) -> ProviderExecution:
        self.logger.log(
            f"[provider] launching provider={self.provider.name} role={role} stage={stage_label}"
        )
        execution = self.provider.run_role(
            role=role,
            prompt=prompt,
            repo_root=self.repo_root,
            schema=schema,
        )
        self.logger.log(
            f"[provider] provider={execution.provider} role={execution.role} "
            f"tier={execution.tier} model={execution.model}"
        )
        if execution.output.strip():
            for line in execution.output.rstrip().splitlines():
                self.logger.log(line)
        else:
            self.logger.log(f"[provider] {stage_label} produced no terminal output.")
        return execution

    def _run_review_role(
        self,
        *,
        prompt: str,
        stage_label: str,
        before_hash: str,
    ) -> ReviewDecision:
        review_targets = ["tests"] if "Test Review" in stage_label else ["src", "tests", "scripts"]
        artifact_snapshot = self._artifact_snapshot(review_targets)
        review_prompt = (
            prompt
            + "\n\n## Artifact Snapshot\n"
            + (artifact_snapshot if artifact_snapshot else "(No matching text artifacts found.)")
            + "\n"
        )
        execution = self._run_role(
            role="reviewer",
            prompt=review_prompt,
            schema=REVIEW_SCHEMA,
            stage_label=stage_label,
        )
        self._enforce_reviewer_immutability(before_hash, stage_label)
        try:
            decision = normalize_review_output(execution.output)
            if _review_requests_missing_inputs(decision):
                raise PipelineError(
                    "FAIL: reviewer requested missing inputs even though the pipeline supplied context.",
                    EXIT_INVALID_REVIEW_OUTPUT,
                )
            return decision
        except PipelineError as exc:
            if exc.exit_code != EXIT_INVALID_REVIEW_OUTPUT:
                raise
            self.logger.log("[review] invalid output; attempting one repair pass")
            repair_prompt = (
                review_prompt
                + "\n\n## Repair Attempt\n"
                + "Your previous response was invalid because it was not exactly one usable review decision. "
                + "This pipeline is non-interactive: do not ask questions or request more input. "
                + "The approved spec and the current artifacts to review are already embedded above. "
                + "Return exactly one raw JSON object with keys `decision`, `summary`, and `blocking`.\n\n"
                + "Previous invalid response:\n"
                + execution.output.strip()
                + "\n"
            )
            repair_execution = self._run_role(
                role="reviewer",
                prompt=repair_prompt,
                schema=REVIEW_SCHEMA,
                stage_label=f"{stage_label} Output Repair",
            )
            self._enforce_reviewer_immutability(before_hash, f"{stage_label} Output Repair")
            decision = normalize_review_output(repair_execution.output)
            if _review_requests_missing_inputs(decision):
                raise PipelineError(
                    "FAIL: reviewer requested missing inputs even after a self-contained repair prompt.",
                    EXIT_INVALID_REVIEW_OUTPUT,
                )
            return decision

    def run(self) -> int:
        if not self.spec_path.exists():
            raise PipelineError(
                f"ERROR: Spec not found: {self.spec_path.as_posix()}", EXIT_SPEC_NOT_FOUND
            )

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logger.log("")
        self.logger.log(f"=== Pipeline: {self.task} ===")
        self.logger.log(f"=== Spec: {self.spec_path.as_posix()} ===")
        self.logger.log(f"=== Provider: {self.provider.name} ===")

        state = self._load_state()

        if not self._past(state.stage, "TESTS_GENERATED"):
            self._log_stage_header("Stage 1: Test Generation")
            before_tests_hash = self._tests_hash()
            prompt = self.prompts.render(
                role="test-writer",
                task=self.task,
                spec_path=self.spec_path,
                stage_name="Stage 1: Test Generation",
                stage_instruction=(
                    "Read the approved spec and the repo rules file. Write tests for this task covering all "
                    "acceptance criteria in tests/. Confirm the tests are red with `uv run python -m pytest`. "
                    "Do not write production code."
                ),
            )
            execution = self._run_role(
                role="test-writer",
                prompt=prompt,
                stage_label="Stage 1: Test Generation",
            )
            self._ensure_tests_stage_effect(
                before_hash=before_tests_hash,
                execution=execution,
                stage_label="Stage 1: Test Generation",
                allow_existing=True,
            )
            self._save_state("TESTS_GENERATED")
            state = self._load_state()

        if not self._past(state.stage, "TESTS_FROZEN"):
            start = self._start_iteration(state, "TESTS_GENERATED")
            for iteration in range(start, self.max_revisions + 1):
                self._log_stage_header(f"Stage 2: Test Review (iter {iteration})")
                before_hash = self._repo_hash()
                prompt = self.prompts.render(
                    role="reviewer",
                    task=self.task,
                    spec_path=self.spec_path,
                    stage_name="Stage 2: Test Review",
                    stage_instruction=(
                        "Review only the test files relevant to the approved spec for this task. Ignore unrelated workspace changes outside the spec scope. "
                        "Return only the canonical review decision JSON."
                    ),
                    iteration=iteration,
                )
                decision = self._run_review_role(
                    prompt=prompt,
                    stage_label="Stage 2: Test Review",
                    before_hash=before_hash,
                )
                if decision.fallback_used:
                    self.logger.log("[review] fallback normalization path used")
                self.logger.log(f"Decision: {decision.decision}")
                if decision.decision == "approve":
                    frozen_tests_hash = self._tests_hash()
                    self._save_state("TESTS_FROZEN", frozen_tests_hash=frozen_tests_hash)
                    break
                if iteration == self.max_revisions:
                    raise PipelineError(
                        f"FAIL: test revision cap reached after {self.max_revisions} iterations.",
                        EXIT_TEST_REVISION_CAP,
                    )
                self._log_stage_header(f"Stage 2b: Test Revision (iter {iteration})")
                before_tests_hash = self._tests_hash()
                revise_prompt = self.prompts.render(
                    role="test-writer",
                    task=self.task,
                    spec_path=self.spec_path,
                    stage_name="Stage 2b: Test Revision",
                    stage_instruction=(
                        "Revise the test suite to address the reviewer blocking feedback. "
                        "Do not touch production code. Re-run `uv run python -m pytest` after revisions."
                    ),
                    iteration=iteration,
                    reviewer_feedback=decision.blocking,
                )
                execution = self._run_role(
                    role="test-writer",
                    prompt=revise_prompt,
                    stage_label="Stage 2b: Test Revision",
                )
                self._ensure_tests_stage_effect(
                    before_hash=before_tests_hash,
                    execution=execution,
                    stage_label="Stage 2b: Test Revision",
                )
                self._save_state("TESTS_GENERATED", iteration=iteration + 1)
            self.logger.log("")
            self.logger.log(">>> Tests frozen <<<")
            state = self._load_state()

        frozen_tests_hash = state.frozen_tests_hash

        if not self._past(state.stage, "CODE_IMPLEMENTED"):
            self._log_stage_header("Stage 3: Implementation")
            prompt = self.prompts.render(
                role="implementer",
                task=self.task,
                spec_path=self.spec_path,
                stage_name="Stage 3: Implementation",
                stage_instruction=(
                    "Read the approved spec and frozen tests. Implement the minimal production code "
                    "needed to make the tests pass. Do not modify frozen tests. Run "
                    "`uv run python -m pytest` to verify."
                ),
            )
            self._run_role(
                role="implementer",
                prompt=prompt,
                stage_label="Stage 3: Implementation",
            )
            self._enforce_test_freeze(frozen_tests_hash)
            self._save_state("CODE_IMPLEMENTED", frozen_tests_hash=frozen_tests_hash)
            state = self._load_state()

        if not self._past(state.stage, "CODE_VALIDATED"):
            self._log_stage_header("Stage 4: Validation")
            prompt = self.prompts.render(
                role="implementer",
                task=self.task,
                spec_path=self.spec_path,
                stage_name="Stage 4: Validation",
                stage_instruction=(
                    "Validate the implementation end-to-end. If the spec defines runnable scripts or "
                    "CLI commands, run them with minimal arguments. If it does not, exercise the main "
                    "code via `uv run python -c ...`. Fix issues you find. Do not modify frozen tests. "
                    "Run `uv run python -m pytest` at the end."
                ),
            )
            self._run_role(
                role="implementer",
                prompt=prompt,
                stage_label="Stage 4: Validation",
            )
            self._enforce_test_freeze(frozen_tests_hash)
            self._save_state("CODE_VALIDATED", frozen_tests_hash=frozen_tests_hash)
            state = self._load_state()

        if not self._past(state.stage, "DONE"):
            start = self._start_iteration(state, "CODE_VALIDATED")
            for iteration in range(start, self.max_revisions + 1):
                self._log_stage_header(f"Stage 5: Code Review (iter {iteration})")
                before_hash = self._repo_hash()
                prompt = self.prompts.render(
                    role="reviewer",
                    task=self.task,
                    spec_path=self.spec_path,
                    stage_name="Stage 5: Code Review",
                    stage_instruction=(
                        "Review only the implementation and frozen tests relevant to the approved spec for this task. Ignore unrelated workspace changes outside the spec scope. "
                        "Observe test status with `uv run python -m pytest` and return only the "
                        "canonical review decision JSON."
                    ),
                    iteration=iteration,
                )
                decision = self._run_review_role(
                    prompt=prompt,
                    stage_label="Stage 5: Code Review",
                    before_hash=before_hash,
                )
                if decision.fallback_used:
                    self.logger.log("[review] fallback normalization path used")
                self.logger.log(f"Decision: {decision.decision}")
                if decision.decision == "approve":
                    self._save_state("DONE", frozen_tests_hash=frozen_tests_hash)
                    break
                if iteration == self.max_revisions:
                    raise PipelineError(
                        f"FAIL: code revision cap reached after {self.max_revisions} iterations.",
                        EXIT_CODE_REVISION_CAP,
                    )
                self._log_stage_header(f"Stage 5b: Code Revision (iter {iteration})")
                revise_prompt = self.prompts.render(
                    role="implementer",
                    task=self.task,
                    spec_path=self.spec_path,
                    stage_name="Stage 5b: Code Revision",
                    stage_instruction=(
                        "Revise the implementation to address the reviewer blocking feedback. "
                        "Do not modify frozen tests. Re-run `uv run python -m pytest` after revisions."
                    ),
                    iteration=iteration,
                    reviewer_feedback=decision.blocking,
                )
                self._run_role(
                    role="implementer",
                    prompt=revise_prompt,
                    stage_label="Stage 5b: Code Revision",
                )
                self._enforce_test_freeze(frozen_tests_hash)
                self._run_pytest_gate("Gate: pytest after code revision")
                self._save_state(
                    "CODE_VALIDATED",
                    iteration=iteration + 1,
                    frozen_tests_hash=frozen_tests_hash,
                )

        self.logger.log("")
        self.logger.log(f"=== Pipeline COMPLETE: {self.task} ===")
        return EXIT_SUCCESS


def run_from_cli(
    task: str,
    provider: Provider,
    repo_root: Path,
    max_revisions: int = 4,
    rules_file: str = "agents.md",
) -> int:
    """CLI entry point helper."""

    runner = PipelineRunner(
        repo_root=repo_root,
        task=task,
        provider=provider,
        max_revisions=max_revisions,
        rules_file=rules_file,
    )
    try:
        return runner.run()
    except PipelineError as exc:
        runner.logger.log(str(exc))
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - defensive fallback
        runner.logger.log(f"FAIL: unexpected pipeline error: {exc}")
        return EXIT_PROVIDER_EXEC_FAILED


def main(argv: list[str] | None = None) -> int:
    """Support module execution for debugging."""

    raise SystemExit("Use scripts/run_pipeline.py instead of module execution.")


if __name__ == "__main__":
    sys.exit(main())
