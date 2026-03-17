"""Run the provider-generalized autonomous pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repo root is on sys.path so `src.*` imports work.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.coding_pipeline.core import run_from_cli  # noqa: E402
from src.coding_pipeline.providers.claude import ClaudeProvider  # noqa: E402
from src.coding_pipeline.providers.codex import CodexProvider  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the autonomous TDD pipeline.")
    parser.add_argument("task", help="Task id. Spec is resolved as specs/<task>-spec.md.")
    parser.add_argument(
        "--provider",
        choices=["claude", "codex"],
        default="claude",
        help="Provider runtime (default: claude).",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=4,
        help="Maximum revision attempts per review loop.",
    )
    parser.add_argument(
        "--rules-file",
        default="agents.md",
        help="Repo rules file the agent reads (default: agents.md). Use a task-specific file to hide internal strategy.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    providers = {
        "claude": ClaudeProvider,
        "codex": CodexProvider,
    }
    provider = providers[args.provider]()
    return run_from_cli(
        task=args.task,
        provider=provider,
        repo_root=repo_root,
        max_revisions=args.max_revisions,
        rules_file=args.rules_file,
    )


if __name__ == "__main__":
    sys.exit(main())
