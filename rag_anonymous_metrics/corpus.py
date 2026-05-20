import json
import logging
from pathlib import Path

from rag_anonymous import load_corpus

from rag_anonymous_metrics.config import PROJECT_ROOT, Settings

__all__ = ["load_corpus", "load_questions"]

logger = logging.getLogger(__name__)


def load_questions(path: Path | None = None, split: str | None = None) -> list[dict]:
    if split:
        target = _questions_path_for_split(split)
    elif path is not None:
        target = path
    else:
        default_split = Settings.load().corpus_split
        target = _questions_path_for_split(default_split)

    if not target.exists():
        raise FileNotFoundError(f"Benchmark questions not found: {target}")

    with open(target, "r", encoding="utf-8") as f:
        questions = json.load(f)

    logger.info("Loaded %d benchmark questions from %s", len(questions), target)
    return questions


def _questions_path_for_split(split: str) -> Path:
    return PROJECT_ROOT / "data" / "input" / f"benchmark_questions_{split}.json"
