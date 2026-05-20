import logging

from rag_anonymous_metrics.metrics._judge import invoke, judge_yes_no

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

CLAIMS_EXTRACTION_PROMPT = """Given the answer below, break it down into individual, atomic factual claims.
Each claim should be a single, self-contained statement.

Answer:
{answer}

List each claim on its own line, prefixed with a dash (-). Output ONLY the list, nothing else."""

CLAIMS_VERIFICATION_PROMPT = """Given the context and a list of claims, determine whether EACH claim can be inferred from the context.

Context:
{context}

For each claim below, answer with exactly "Yes" or "No" on the same line after the pipe character.

{numbered_claims}

Output ONLY the numbered lines with the verdict, nothing else. Example format:
1. Some claim | Yes
2. Another claim | No"""


def score_context_precision(
    chain,
    question: str,
    answer: str,
    chunks: list[str],
) -> float | None:
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


def score_faithfulness(chain, answer: str, contexts: list[str]) -> float | None:
    claims = _extract_claims(chain, answer)
    if not claims:
        logger.warning("No claims extracted from answer")
        return None

    context_text = "\n\n---\n\n".join(contexts)
    numbered = "\n".join(f"{i+1}. {claim}" for i, claim in enumerate(claims))
    verdicts = judge_yes_no(
        chain,
        CLAIMS_VERIFICATION_PROMPT,
        {"context": context_text, "numbered_claims": numbered},
        expected_count=len(claims),
    )
    if not verdicts:
        return None

    supported = sum(verdicts)
    total = len(verdicts)
    score = supported / total
    logger.debug("Faithfulness: %d/%d claims supported = %.2f", supported, total, score)
    return score


def _context_precision_at_k(relevance_flags: list[bool]) -> float:
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


def _extract_claims(chain, answer: str) -> list[str]:
    raw = invoke(chain, CLAIMS_EXTRACTION_PROMPT, {"answer": answer})
    claims = [line.lstrip("-").strip() for line in raw.strip().splitlines() if line.strip()]
    claims = [c for c in claims if c]
    logger.debug("Extracted %d claims", len(claims))
    return claims


def _normalize_flags(verdicts: list[bool], expected: int) -> list[bool]:
    flags = verdicts[:expected]
    flags.extend([False] * (expected - len(flags)))
    return flags
