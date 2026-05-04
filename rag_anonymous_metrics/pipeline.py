"""RAG pipeline construction for evaluation.

Wraps the user-facing :mod:`rag_anonymous` API into a single immutable
:class:`Pipeline` object so that :mod:`evaluator` can stay focused on the
per-question scoring loop.
"""

from dataclasses import dataclass
from typing import Any

from rag_anonymous import Anonymizer, build_llm, create_chain, load_vectordb


@dataclass(frozen=True)
class Pipeline:
    """Immutable bundle of the components needed to evaluate one strategy."""

    strategy: str
    gen_model: str
    chain: Any
    vectordb: Any
    anonymizer: Anonymizer | None


def build_pipeline(strategy: str, gen_model: str) -> Pipeline:
    """Construct retrieval chain, vector store, and (optional) anonymizer.

    Args:
        strategy: ``"offline"`` or ``"ondemand"``.
        gen_model: Ollama model tag for generation.

    Returns:
        A :class:`Pipeline` ready to be used by ``evaluate_question``.
    """
    vectordb = load_vectordb(collection_name=strategy)
    chain = create_chain(build_llm(gen_model))
    anonymizer = Anonymizer() if strategy == "ondemand" else None
    return Pipeline(
        strategy=strategy,
        gen_model=gen_model,
        chain=chain,
        vectordb=vectordb,
        anonymizer=anonymizer,
    )


def response_text(raw):
    """Extract a plain string from a chain response (LCEL may return message or str)."""
    if hasattr(raw, "content"):
        return raw.content
    return raw if isinstance(raw, str) else str(raw)
