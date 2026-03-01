"""LLM wrapper: supports OpenAI and Anthropic, with structured output and JSON fallback."""

import json
from typing import TypeVar

from pydantic import BaseModel

from src import config

T = TypeVar("T", bound=BaseModel)

_openai_client = None
_anthropic_client = None


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        _anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _anthropic_client


def generate(system: str, user: str, *, model: str) -> str:
    """Free-text LLM call."""
    if config.LLM_PROVIDER == "openai":
        client = _get_openai()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content
    else:
        client = _get_anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text


def _fallback_structured(
    system: str, user: str, response_model: type[T], *, model: str,
) -> T:
    """Fallback: ask for JSON in plain text, parse it, retry once."""
    schema_hint = json.dumps(response_model.model_json_schema(), indent=2)
    patched_system = f"{system}\n\nReturn ONLY valid JSON matching this schema:\n{schema_hint}"
    for attempt in range(2):
        raw = generate(patched_system, user, model=model)
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        try:
            return response_model.model_validate(json.loads(text))
        except Exception as e:
            if attempt == 0:
                print(f"  [WARN] JSON fallback parse failed (retrying): {e}")
            else:
                raise


def generate_structured(
    system: str, user: str, response_model: type[T], *, model: str,
) -> T:
    """Structured LLM call. Uses native structured output, falls back to JSON parsing."""
    if config.LLM_PROVIDER == "openai":
        client = _get_openai()
        try:
            resp = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format=response_model,
            )
            return resp.choices[0].message.parsed
        except Exception:
            return _fallback_structured(system, user, response_model, model=model)
    else:
        # Anthropic: use tool_use with forced tool_choice for structured output
        client = _get_anthropic()
        schema = response_model.model_json_schema()
        tool_name = response_model.__name__
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[{
                    "name": tool_name,
                    "description": f"Return structured {tool_name} data",
                    "input_schema": schema,
                }],
                tool_choice={"type": "tool", "name": tool_name},
            )
            tool_block = next(b for b in resp.content if b.type == "tool_use")
            return response_model.model_validate(tool_block.input)
        except Exception:
            return _fallback_structured(system, user, response_model, model=model)
