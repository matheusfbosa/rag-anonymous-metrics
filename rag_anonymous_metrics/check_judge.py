from __future__ import annotations

import asyncio
import logging
import math
import statistics

from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness

from rag_anonymous import build_llm
from rag_anonymous_metrics.config import Settings
from rag_anonymous_metrics.corpus import load_questions
from rag_anonymous_metrics.log_config import (
    log_item_progress,
    log_section,
    log_suite_header,
)
from rag_anonymous_metrics.questions_filter import (
    apply_category_filter,
    apply_id_range,
    describe_category,
    describe_range,
)

logger = logging.getLogger(__name__)

THRESHOLD = 0.9
SEPARATION = 0.3


def check_judge() -> int:
    cfg = Settings.load()
    if cfg.questions_category == "adversarial":
        logger.error(
            "Invalid category: suite=check-judge category=adversarial "
            "allowed=benchmark|factual|identity"
        )
        return 2
    return asyncio.run(_run(cfg))


async def _run(cfg: Settings) -> int:
    dataset = cfg.dataset
    category = cfg.questions_category
    id_range = cfg.questions_range
    judge = cfg.judge_model
    all_questions = load_questions(dataset=dataset)
    if category == "benchmark":
        questions = apply_id_range(all_questions, id_range)
    else:
        questions = apply_id_range(
            apply_category_filter(all_questions, category),
            id_range,
        )
    if not questions:
        logger.error(
            "No questions match: dataset=%s category=%s range=%s",
            dataset,
            describe_category(category),
            describe_range(id_range),
        )
        return 2

    log_suite_header("check-judge", category, id_range)
    logger.info(
        "Benchmark selection: category=%s range=%s selected=%d total=%d",
        describe_category(category),
        describe_range(id_range),
        len(questions),
        len(all_questions),
    )
    logger.info("Running judge check: judge=%s n=%d", judge, len(questions))

    metric = FactualCorrectness(
        llm=LangchainLLMWrapper(
            build_llm(
                judge,
                num_predict=cfg.judge_num_predict,
                timeout=cfg.judge_timeout_sec,
            )
        )
    )

    positives = []
    negatives = []
    width = len(str(len(questions)))
    for index, question in enumerate(questions, 1):
        reference = question["ground_truth"].strip()
        other = questions[index % len(questions)]["ground_truth"].strip()

        log_item_progress(
            logger,
            index,
            len(questions),
            str(question.get("id", "-")),
            question["question"],
        )
        positive = await _factual_correctness(
            metric, question["question"], reference, reference
        )
        negative = await _factual_correctness(
            metric, question["question"], other, reference
        )
        positives.append(positive)
        negatives.append(negative)
        logger.info(
            "  [%*d/%d] Scores: pos_fc=%s neg_fc=%s",
            width,
            index,
            len(questions),
            _fmt(positive),
            _fmt(negative),
        )

    return _report(judge, positives, negatives)


def _report(
    judge: str, positives: list[float | None], negatives: list[float | None]
) -> int:
    scored_positives = [s for s in positives if s is not None]
    scored_negatives = [s for s in negatives if s is not None]
    positive_mean = statistics.fmean(scored_positives) if scored_positives else 0.0
    negative_mean = statistics.fmean(scored_negatives) if scored_negatives else 0.0
    positive_failures = positives.count(None)
    above_threshold = sum(1 for s in scored_positives if s >= THRESHOLD)
    separation = positive_mean - negative_mean

    log_section(f"Summary: suite=check-judge judge={judge}")
    logger.info(
        "Positive scores: mean_fc=%.3f threshold=%.2f above=%d/%d failures=%d/%d",
        positive_mean,
        THRESHOLD,
        above_threshold,
        len(positives),
        positive_failures,
        len(positives),
    )
    logger.info("Negative scores: mean_fc=%.3f", negative_mean)
    logger.info(
        "Score separation: delta=%.3f required=%.3f",
        separation,
        SEPARATION,
    )

    reliable = (
        bool(scored_positives)
        and positive_mean >= THRESHOLD
        and positive_failures == 0
        and separation >= SEPARATION
    )
    if reliable:
        logger.info("Verdict: status=RELIABLE")
        return 0
    logger.info("Verdict: status=UNRELIABLE")
    return 1


async def _factual_correctness(
    metric: object, question: str, response: str, reference: str
) -> float | None:
    sample = SingleTurnSample(
        user_input=question, response=response, reference=reference
    )
    try:
        score = await metric.single_turn_ascore(sample)
    except Exception as exc:
        logger.warning("Judge failed: error=%s", exc.__class__.__name__)
        return None
    if score is None or math.isnan(score):
        return None
    return float(score)


def _fmt(score: float | None) -> str:
    return "FAIL" if score is None else f"{score:.2f}"
