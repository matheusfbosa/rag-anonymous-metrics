"""Context Precision@K scoring (Ragas methodology).

For each retrieved chunk at rank k, the judge LLM decides if it is relevant (1)
or irrelevant (0) to producing the answer. Then::

    Context Precision@K = sum(precision@k * v_k) / total_relevant_in_top_K
        where precision@k = relevant_up_to_k / k
              v_k          = 1 if chunk k is relevant, else 0

Reference: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_precision/
"""

import logging

from rag_anonymous_metrics.metrics._judge import judge_yes_no

logger = logging.getLogger(__name__)

CHUNK_TRUNCATION = 300

CHUNK_RELEVANCE_PROMPT = """Given the question and the answer, determine whether EACH context chunk below is relevant to producing that answer.

Question:
{question}

Answer:
{answer}

Context chunks:
{numbered_chunks}

For each chunk, answer with exactly "Yes" or "No" on the same line after the pipe character.
Output ONLY the numbered lines with the verdict, nothing else. Example format:
1 | Yes
2 | No"""


def score_context_precision(
    chain,
    question: str,
    answer: str,
    chunks: list[str],
) -> float | None:
    """Compute context precision; returns float in [0, 1] or None on failure."""
    if not chunks:
        return None

    numbered = "\n".join(
        f"{i+1}. {chunk[:CHUNK_TRUNCATION]}" for i, chunk in enumerate(chunks)
    )
    verdicts = judge_yes_no(
        chain,
        CHUNK_RELEVANCE_PROMPT,
        {"question": question, "answer": answer, "numbered_chunks": numbered},
        expected_count=len(chunks),
    )
    if not verdicts:
        return None

    flags = _normalize_flags(verdicts, len(chunks))
    score = _context_precision_at_k(flags)
    logger.debug(
        "Context Precision: %d/%d chunks relevant, CP@K=%.2f",
        sum(flags), len(flags), score,
    )
    return score


def _normalize_flags(verdicts: list[bool], expected: int) -> list[bool]:
    """Pad with False or trim so the result matches the chunk count."""
    flags = verdicts[:expected]
    flags.extend([False] * (expected - len(flags)))
    return flags


def _context_precision_at_k(relevance_flags: list[bool]) -> float:
    """Aggregate per-rank relevance flags into the Ragas CP@K score."""
    total_relevant = sum(relevance_flags)
    if total_relevant == 0:
        return 0.0

    cumulative_relevant = 0
    weighted_sum = 0.0
    for k, is_relevant in enumerate(relevance_flags, start=1):
        if is_relevant:
            cumulative_relevant += 1
            weighted_sum += cumulative_relevant / k
    return weighted_sum / total_relevant
