from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv()
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    eval_strategy: str
    corpus_split: str
    retrieval_k_docs: int
    llm_model: str
    judge_model: str
    results_dir: Path
    max_questions: int | None

    @classmethod
    def load(cls) -> Settings:
        llm = os.getenv("RAG_ANON_LLM_MODEL", "qwen3:0.6b")
        judge = os.getenv("RAG_METRICS_JUDGE_MODEL") or llm
        results_rel = os.getenv("RAG_METRICS_RESULTS_DIR", "./data/output")
        return cls(
            eval_strategy=os.getenv("RAG_METRICS_EVAL_STRATEGY", "all"),
            corpus_split=os.getenv("RAG_METRICS_CORPUS_SPLIT", "dev"),
            retrieval_k_docs=int(os.getenv("RAG_METRICS_RETRIEVAL_K_DOCS", "5")),
            llm_model=llm,
            judge_model=judge,
            results_dir=PROJECT_ROOT / results_rel,
            max_questions=_parse_max_questions(os.getenv("RAG_METRICS_MAX_QUESTIONS")),
        )


__all__ = ["PROJECT_ROOT", "Settings"]


def _parse_max_questions(raw: str | None) -> int | None:
    if raw is None or not raw.strip():
        return None
    trimmed = raw.strip()
    try:
        n = int(trimmed)
    except ValueError as exc:
        raise ValueError(
            f"RAG_METRICS_MAX_QUESTIONS must be an integer, got {raw!r}"
        ) from exc
    if n <= 0:
        raise ValueError(f"RAG_METRICS_MAX_QUESTIONS must be positive, got {n}")
    return n
