from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from dotenv import load_dotenv

from rag_anonymous_metrics.questions_filter import (
    CategorySpec,
    RangeSpec,
    parse_questions_category,
    parse_questions_range,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv()
load_dotenv(PROJECT_ROOT / ".env")

StrategySpec = Literal["offline", "ondemand", "all"]


@dataclass(frozen=True)
class Settings:
    strategy: StrategySpec
    dataset: str
    retrieval_k_docs: int
    llm_model: str
    judge_model: str
    results_dir: Path
    questions_category: CategorySpec
    questions_range: RangeSpec

    @classmethod
    def load(cls) -> Settings:
        llm = os.getenv("RAG_ANON_LLM_MODEL", "qwen3:8b")
        judge = os.getenv("RAG_METRICS_JUDGE_MODEL") or llm
        results_rel = os.getenv("RAG_METRICS_RESULTS_DIR", "./data/output")

        strategy_raw = os.getenv("RAG_METRICS_STRATEGY")
        if strategy_raw is None or not strategy_raw.strip():
            raise ValueError(
                "RAG_METRICS_STRATEGY is required (offline|ondemand|all)"
            )
        strategy = strategy_raw.strip().lower()
        if strategy not in {"offline", "ondemand", "all"}:
            raise ValueError(
                "RAG_METRICS_STRATEGY must be 'offline', 'ondemand', or 'all', "
                f"got {strategy_raw!r}"
            )

        dataset = os.getenv("RAG_METRICS_CORPUS_DATASET")
        if dataset is None or not dataset.strip():
            raise ValueError(
                "RAG_METRICS_CORPUS_DATASET is required (dev|train|test)"
            )

        return cls(
            strategy=cast(StrategySpec, strategy),
            dataset=dataset.strip(),
            retrieval_k_docs=int(os.getenv("RAG_METRICS_RETRIEVAL_K_DOCS", "5")),
            llm_model=llm,
            judge_model=judge,
            results_dir=PROJECT_ROOT / results_rel,
            questions_category=parse_questions_category(
                os.getenv("RAG_METRICS_QUESTIONS_CATEGORY")
            ),
            questions_range=parse_questions_range(
                os.getenv("RAG_METRICS_QUESTIONS_RANGE")
            ),
        )


__all__ = ["PROJECT_ROOT", "Settings"]
