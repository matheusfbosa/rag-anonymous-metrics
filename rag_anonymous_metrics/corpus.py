import json
import logging
from pathlib import Path

from rag_anonymous import load_corpus
from rag_anonymous_metrics.config import PROJECT_ROOT, Settings

logger = logging.getLogger(__name__)

__all__ = ["load_adversarial_attacks", "load_corpus", "load_questions"]


def load_questions(path: Path | None = None, dataset: str | None = None) -> list[dict]:
    if dataset:
        target = _benchmark_path_for_dataset(dataset)
    elif path is not None:
        target = path
    else:
        default_dataset = Settings.load().dataset
        target = _benchmark_path_for_dataset(default_dataset)

    if not target.exists():
        raise FileNotFoundError(f"Benchmark questions not found: {target}")

    with open(target, encoding="utf-8") as f:
        questions = json.load(f)

    logger.info("Loaded benchmark questions: n=%d path=%s", len(questions), target)
    return questions


def load_adversarial_attacks(
    path: Path | None = None,
    dataset: str | None = None,
) -> list[dict]:
    if dataset:
        target = _adversarial_path_for_dataset(dataset)
    elif path is not None:
        target = path
    else:
        default_dataset = Settings.load().dataset
        target = _adversarial_path_for_dataset(default_dataset)

    if not target.exists():
        raise FileNotFoundError(f"Adversarial attacks not found: {target}")

    with open(target, encoding="utf-8") as f:
        attacks = json.load(f)

    logger.info("Loaded adversarial attacks: n=%d path=%s", len(attacks), target)
    return attacks


def _benchmark_path_for_dataset(dataset: str) -> Path:
    return PROJECT_ROOT / "data" / "input" / f"benchmark_questions_{dataset}.json"


def _adversarial_path_for_dataset(dataset: str) -> Path:
    return PROJECT_ROOT / "data" / "input" / f"adversarial_attacks_{dataset}.json"
