# Pipeline Rules

Conventions for the autonomous coding pipeline agents.

## Project

- **Python 3.11.9**, **uv** for package management
- `pyproject.toml` at repo root, `pythonpath = ["."]` for pytest
- Tests in `tests/`, source in `src/`

## Code Style

- Type hints on all public functions
- Pydantic for data models
- `from __future__ import annotations` in every file
- Keep modules small and focused

## Testing

- pytest with `uv run python -m pytest`
- Tests must be deterministic — no real LLM calls in unit tests (mock them)
- Use fixtures for shared test data

## LLM Usage

- Use OpenAI SDK (`openai` package, already in deps)
- All LLM calls go through a single wrapper module
- Structured output via Pydantic models

## Dataset

- Hand packages are in `data/eval-validation-package/dataset/`
- Contracts (evidence contract, schemas) are in `data/eval-validation-package/dataset/contracts/`
- Annotated examples are in `data/eval-validation-package/dataset/annotated_examples/`
- Validation hands are in `data/eval-validation-package/dataset/validation/`
- The flawed trace is in `data/eval-validation-package/dataset/flawed_trace/`
