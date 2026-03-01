import difflib
import json
from pathlib import Path

from src.config import PROMPTS_DIR
from src.models import PromptVersion


def save_prompt(version: str, prompt_text: str, metadata: dict | None = None) -> Path:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    prompt_path = PROMPTS_DIR / f"{version}.txt"
    meta_path = PROMPTS_DIR / f"{version}_meta.json"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    meta_path.write_text(json.dumps(metadata or {}, indent=2), encoding="utf-8")
    return prompt_path


def load_prompt(version: str) -> PromptVersion:
    prompt_path = PROMPTS_DIR / f"{version}.txt"
    meta_path = PROMPTS_DIR / f"{version}_meta.json"
    prompt_text = prompt_path.read_text(encoding="utf-8")
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return PromptVersion(version=version, prompt_text=prompt_text, metadata=metadata)


def diff_prompts(v1: str, v2: str) -> str:
    p1 = load_prompt(v1).prompt_text
    p2 = load_prompt(v2).prompt_text
    diff = difflib.unified_diff(
        p1.splitlines(keepends=True),
        p2.splitlines(keepends=True),
        fromfile=v1,
        tofile=v2,
    )
    return "".join(diff)
