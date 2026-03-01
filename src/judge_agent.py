"""Judge agent: LLM-as-a-judge for coaching quality (style, not strategy)."""

from src import config
from src.llm import generate_structured
from src.models import GameState, SolverOutput, CoachingOutput, JudgeScores

SYSTEM_PROMPT = """\
You are an expert evaluator of poker coaching explanations. Your job is to score \
the QUALITY of the explanation — how well it communicates, not whether the strategy is correct.

DO NOT evaluate whether the recommended action or numbers are correct. \
Strategy correctness is checked elsewhere. You only judge communication quality.

A good coaching explanation is SHORT (3-5 sentences), direct, and sounds like a coach \
talking to a player at the table — not an essay or textbook analysis. Penalize verbosity heavily.

Score each dimension from 1 to 5:

## Coherence
How logically structured and internally consistent is the explanation?
1 = Contradicts itself, jumps between unrelated points
2 = Some logical gaps or redundant paragraphs
3 = Logical but includes unnecessary tangents or repetition
4 = Clean logical flow, no wasted sentences, each point builds on the previous
5 = Tight argument where every sentence earns its place

## Readability
How easy is it to understand quickly? Brevity is essential.
1 = Wall of text, dense jargon, or incoherent
2 = Too long or requires expert knowledge to parse
3 = Understandable but wordy — could say the same in half the words
4 = Concise and clear, 3-5 sentences, good use of terminology with context
5 = Immediately clear in one quick read, perfectly paced, nothing to cut

## Coaching Tone
Does it sound like a coach talking to a player, or like a textbook?
1 = Robotic data dump — just restates numbers with no insight
2 = Explains what to do but not why it matters to the player
3 = Some coaching elements but reads like an essay or analysis report
4 = Sounds like a coach: direct, practical, explains the "why" concisely
5 = Excellent coach — builds intuition in few words, player walks away understanding the concept

Return scores as integers from 1 to 5 for each dimension.\
"""


def _format_context(
    game_state: GameState, solver_output: SolverOutput,
    legal_actions: list[str], coaching_output: CoachingOutput,
) -> str:
    gs = game_state
    so = solver_output
    return (
        f"Game Context:\n"
        f"- Hand: {gs.hand}\n"
        f"- Board: {gs.board or '(preflop)'}\n"
        f"- Street: {gs.street}, Position: {gs.position}\n"
        f"- Pot: {gs.pot}, Stack: {gs.stack}\n"
        f"- Villain: {gs.villain_action}\n"
        f"- Legal actions: {', '.join(legal_actions)}\n"
        f"- Solver recommends: {so.recommended_action}\n\n"
        f"Coaching Output to Evaluate:\n"
        f"- Recommended action: {coaching_output.recommended_action}\n"
        f"- Explanation: {coaching_output.explanation}"
    )


def judge_coaching(
    game_state: GameState, solver_output: SolverOutput,
    legal_actions: list[str], coaching_output: CoachingOutput,
) -> JudgeScores:
    user_msg = _format_context(game_state, solver_output, legal_actions, coaching_output)
    return generate_structured(
        SYSTEM_PROMPT, user_msg, JudgeScores,
        model=config.TUNER_MODEL,
    )
