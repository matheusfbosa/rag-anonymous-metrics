"""Faithfulness scoring (Ragas methodology).

Step 1: extract atomic claims from the answer.
Step 2: for each claim, ask whether it can be inferred from the context.
Score = supported_claims / total_claims, in [0, 1].

Reference: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/
"""

import logging

from rag_anonymous_metrics.metrics._judge import invoke, judge_yes_no

logger = logging.getLogger(__name__)

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


def score_faithfulness(chain, answer: str, contexts: list[str]) -> float | None:
    """Compute faithfulness following the Ragas formula.

    Returns a float in [0, 1] or None when no claims can be extracted or no
    verdicts parsed.
    """
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


def _extract_claims(chain, answer: str) -> list[str]:
    """Decompose an answer into atomic claims via the judge LLM."""
    raw = invoke(chain, CLAIMS_EXTRACTION_PROMPT, {"answer": answer})
    claims = [line.lstrip("-").strip() for line in raw.strip().splitlines() if line.strip()]
    claims = [c for c in claims if c]
    logger.debug("Extracted %d claims", len(claims))
    return claims
