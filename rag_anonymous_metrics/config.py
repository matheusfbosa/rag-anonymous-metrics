from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from rag_anonymous_metrics.questions_filter import FilterSpec, parse_questions_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv()
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    eval_strategy: str
    dataset: str
    retrieval_k_docs: int
    llm_model: str
    judge_model: str
    results_dir: Path
    questions_filter: FilterSpec

    @classmethod
    def load(cls) -> Settings:
        llm = os.getenv("RAG_ANON_LLM_MODEL", "qwen3:8b")
        judge = os.getenv("RAG_METRICS_JUDGE_MODEL") or llm
        results_rel = os.getenv("RAG_METRICS_RESULTS_DIR", "./data/output")
        return cls(
            eval_strategy=os.getenv("RAG_METRICS_EVAL_STRATEGY", "all"),
            dataset=os.getenv("RAG_METRICS_CORPUS_DATASET", "dev"),
            retrieval_k_docs=int(os.getenv("RAG_METRICS_RETRIEVAL_K_DOCS", "5")),
            llm_model=llm,
            judge_model=judge,
            results_dir=PROJECT_ROOT / results_rel,
            questions_filter=parse_questions_filter(
                os.getenv("RAG_METRICS_QUESTIONS_FILTER")
            ),
        )


__all__ = ["PROJECT_ROOT", "Settings"]
