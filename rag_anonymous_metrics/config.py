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

StrategySpec = Literal["offline", "ondemand"]


@dataclass(frozen=True)
class Settings:
    strategy: StrategySpec
    dataset: str
    retrieval_k_docs: int
    llm_model: str
    judge_model: str
    judge_num_predict: int
    judge_timeout_sec: float
    results_dir: Path
    questions_category: CategorySpec
    questions_range: RangeSpec
    experiment_dir: Path | None
    judge_source_dir: Path | None
    overwrite: bool
    skip_judge: bool

    @classmethod
    def load(cls) -> Settings:
        llm = os.getenv("RAG_ANON_LLM_MODEL", "qwen3:8b")
        judge = os.getenv("RAG_METRICS_JUDGE_MODEL") or llm
        results_rel = os.getenv("RAG_METRICS_RESULTS_DIR", "./data/output")

        strategy_raw = os.getenv("RAG_METRICS_STRATEGY")
        if strategy_raw is None or not strategy_raw.strip():
            raise ValueError("RAG_METRICS_STRATEGY is required (offline|ondemand)")
        strategy = strategy_raw.strip().lower()
        if strategy not in {"offline", "ondemand"}:
            raise ValueError(
                "RAG_METRICS_STRATEGY must be 'offline' or 'ondemand', "
                f"got {strategy_raw!r}"
            )

        dataset = os.getenv("RAG_METRICS_CORPUS_DATASET")
        if dataset is None or not dataset.strip():
            raise ValueError("RAG_METRICS_CORPUS_DATASET is required (dev|train|test)")

        results_dir = PROJECT_ROOT / results_rel
        experiment_dir = _resolve_experiment_dir(
            os.getenv("RAG_METRICS_EXPERIMENT_DIR"),
            results_dir,
        )

        return cls(
            strategy=cast(StrategySpec, strategy),
            dataset=dataset.strip(),
            retrieval_k_docs=int(os.getenv("RAG_METRICS_RETRIEVAL_K_DOCS", "3")),
            llm_model=llm,
            judge_model=judge,
            judge_num_predict=int(os.getenv("RAG_METRICS_JUDGE_NUM_PREDICT", "1536")),
            judge_timeout_sec=float(os.getenv("RAG_METRICS_JUDGE_TIMEOUT_SEC", "1800")),
            results_dir=results_dir,
            questions_category=parse_questions_category(
                os.getenv("RAG_METRICS_QUESTIONS_CATEGORY")
            ),
            questions_range=parse_questions_range(
                os.getenv("RAG_METRICS_QUESTIONS_RANGE")
            ),
            experiment_dir=experiment_dir,
            judge_source_dir=_resolve_experiment_dir(
                os.getenv("RAG_METRICS_JUDGE_SOURCE_DIR"),
                results_dir,
            ),
            overwrite=_env_flag("RAG_METRICS_OVERWRITE"),
            skip_judge=_env_flag("RAG_METRICS_SKIP_JUDGE"),
        )


def _env_flag(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_experiment_dir(raw: str | None, results_dir: Path) -> Path | None:
    if raw is None or not raw.strip():
        return None
    path = Path(raw.strip())
    if path.is_absolute():
        return path
    if len(path.parts) == 1:
        return results_dir / path
    return PROJECT_ROOT / path


__all__ = ["PROJECT_ROOT", "Settings"]
