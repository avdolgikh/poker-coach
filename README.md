# Poker Coach — Self-Optimizing Prompt POC

A system that automatically improves its own coaching prompts. It generates poker coaching explanations using an LLM, evaluates them against solver ground truth, and iteratively rewrites the prompt to fix failures.

## How it works

```
                    ┌─────────────┐
                    │  Coaching   │
  Scenario ──────► │   Agent     │ ──────► CoachingOutput
  + Prompt         │  (LLM)      │         (action + explanation)
                    └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  Evaluator  │ ──────► Composite Score
                    │  (gates +   │         (0.0 if gates fail)
                    │  heuristics)│
                    └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  Judge      │ ──────► Style Scores
                    │  (LLM)      │         (coherence, readability, tone)
                    └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  Optimizer  │ ──────► Prompt Edits
                    │  (LLM)      │         (diagnosis + fix)
                    └─────────────┘
                          │
                          ▼
                    New prompt version
                    (repeat for N iterations)
```

The pipeline runs 3 iterations. Each iteration: coach all training scenarios, evaluate, collect failures, optimizer proposes prompt edits. After all iterations, the best prompt is validated on a held-out set to confirm it actually improved.

**Two-tier evaluation:**
- Training uses cheap, solver-derived heuristics (EV awareness, frequency awareness, position awareness, format)
- Holdout uses annotation-based heuristics (concept mention, contradiction detection) for rigorous validation

## Example run

The system starts with a generic one-line prompt and progressively sharpens it:

```
Iteration   Composite   What changed
─────────   ─────────   ────────────
v0 (base)   0.581       "You are a poker coach. Explain why the recommended action is correct."
v1          0.691       + "Always state equity as X% and include pot odds verbatim"
v2          0.734       + "Mention position and mixed strategy when relevant"
v3          0.691       + (over-constrained — optimizer rolled back)

Best: v2 (composite=0.734)
Holdout: 0.581 → 0.712 (+0.131) — ACCEPTED
```

The coaching output evolves from generic filler to precise, number-grounded explanations:

**v0:** "This is a good spot to call because you have decent equity and the pot odds support it."

**v2:** "Call. Your 34% equity against a pot offering 3:1 odds makes this clearly profitable. With the flush draw on a wet board, you have good implied odds from the BTN if you hit."

## Setup

```bash
uv sync
cp .env.example .env
# Add your API key to .env
```

## Run

```bash
# Full pipeline
uv run python run.py

# Smaller run for quick testing
uv run python run.py --train 5 --holdout 3

# Generate more training data
uv run python -m src.generate_data 50 --solver-only
```

## Tests

```bash
uv run pytest tests/ -v
```

## Design decisions

**No agentic frameworks.** This POC intentionally avoids any agentic frameworks. The optimization loop is simple enough that the overhead of a framework would obscure the core logic. Every LLM call, every evaluation step, every prompt edit is explicit and traceable. In production, a framework could reduce boilerplate — but for a POC, full visibility matters more.

## Project structure

```
run.py                     Entry point — loads data, runs pipeline, prints results
src/
  config.py                All configuration (models, weights, gates, paths)
  models.py                Pydantic data models (Scenario, CoachingOutput, EvalResult, etc.)
  llm.py                   LLM wrapper (OpenAI + Anthropic, structured output)
  coaching_agent.py        Generates coaching explanations for poker scenarios
  judge_agent.py           LLM-as-a-judge for communication quality (not strategy)
  evaluator.py             Hard gates + heuristic scoring + composite score
  optimizer_agent.py       Analyzes failures, proposes prompt edits
  pipeline.py              Main optimization loop (train → evaluate → optimize → repeat)
  generate_data.py         LLM-powered scenario generation with validation
  prompt_store.py          Save/load/diff prompt versions
data/
  scenarios.json           Solver-only training scenarios
  scenarios-annotated.json Annotated scenarios (holdout + examples)
prompts/
  v0.txt                   Initial coaching prompt
tests/                     Tests covering gates, heuristics, pipeline, validation
```

## Future directions

**Prompt improvements (low effort, high impact):**
- Add coaching tone and style examples to the prompt — show the LLM *how* a good coach speaks, not just what to include. Right now tone is judged but not guided.
- Include a compressed poker rules reference in the prompt. The system currently relies on the LLM's built-in poker knowledge, which can hallucinate odds or misread board textures.

**Optimization framework:**
- Try [DSPy](https://github.com/stanfordnlp/dspy) as the optimization backbone. The current hand-rolled optimizer loop works, but DSPy is purpose-built for prompt tuning and could replace the optimizer agent with more principled search.

**Data quality:**
- Replace LLM-generated solver data with real solver outputs. The current synthetic data is plausible but not ground truth.
- Use a small set of real human annotations for holdout validation — even 10-20 expert-reviewed scenarios would sharpen the signal.

**Production path:**
- Adopt an agentic framework (LangGraph, Google ADK, CrewAI) to reduce boilerplate once the core logic stabilizes.
- Fine-tune the coaching LLM if prompt optimization plateaus. The evaluation framework already produces scored examples that can serve as training data.
- Collect implicit user feedback in production (e.g., which coaching explanations users engage with) to continuously improve the training set without manual annotation.
