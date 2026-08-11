from __future__ import annotations

import asyncio
import math
import statistics

from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness

from rag_anonymous import build_llm

from rag_anonymous_metrics.config import Settings
from rag_anonymous_metrics.corpus import load_questions
from rag_anonymous_metrics.questions_filter import apply_category_filter, apply_id_range


THRESHOLD = 0.9
SEPARATION = 0.3


def check_judge() -> int:
    cfg = Settings.load()
    if cfg.questions_category == "adversarial":
        print(
            "check-judge requires a benchmark category "
            "(all|factual|identity), not adversarial."
        )
        return 2
    return asyncio.run(_run(cfg))


async def _run(cfg: Settings):
    dataset = cfg.dataset
    category = cfg.questions_category
    id_range = cfg.questions_range
    judge = cfg.judge_model
    all_questions = load_questions(dataset=dataset)
    if category == "all":
        questions = apply_id_range(all_questions, id_range)
    else:
        questions = apply_id_range(
            apply_category_filter(all_questions, category),
            id_range,
        )
    if not questions:
        print(f"No questions match category={category} range={id_range} in {dataset}.")
        return 2

    print(f"Judge: {judge} | dataset: {dataset} | n: {len(questions)}\n")
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
    for index, question in enumerate(questions):
        reference = question["ground_truth"].strip()
        other = questions[(index + 1) % len(questions)]["ground_truth"].strip()

        positive = await _factual_correctness(metric, question["question"], reference, reference)
        negative = await _factual_correctness(metric, question["question"], other, reference)
        positives.append(positive)
        negatives.append(negative)
        print(f"[{index + 1:>2}/{len(questions)}] {question['id']}: POS fc={_fmt(positive)} | NEG fc={_fmt(negative)}")

    return _report(judge, positives, negatives)


def _report(judge, positives, negatives):
    scored_positives = [s for s in positives if s is not None]
    scored_negatives = [s for s in negatives if s is not None]
    positive_mean = statistics.fmean(scored_positives) if scored_positives else 0.0
    negative_mean = statistics.fmean(scored_negatives) if scored_negatives else 0.0
    positive_failures = positives.count(None)
    above_threshold = sum(1 for s in scored_positives if s >= THRESHOLD)

    print("\n" + "=" * 60)
    print(f"Summary — judge {judge}")
    print("=" * 60)
    print(f"POSITIVE (response = ground truth): mean fc = {positive_mean:.3f}")
    print(f"  fc >= {THRESHOLD:.2f}: {above_threshold}/{len(positives)} | failures: {positive_failures}/{len(positives)}")
    print(f"NEGATIVE (response = other ground truth): mean fc = {negative_mean:.3f}")

    reliable = (
        bool(scored_positives)
        and positive_mean >= THRESHOLD
        and positive_failures == 0
        and (positive_mean - negative_mean) >= SEPARATION
    )
    print("\nVerdict:")
    if reliable:
        print("  RELIABLE — judge scores ~1.0 on ground truth and separates positive from negative.")
        return 0
    print("  UNRELIABLE — metrics are contaminated; replace the judge.")
    return 1


async def _factual_correctness(metric, question, response, reference):
    sample = SingleTurnSample(user_input=question, response=response, reference=reference)
    try:
        score = await metric.single_turn_ascore(sample)
    except Exception as exc:
        print(f"    judge failed: {exc.__class__.__name__}")
        return None
    if score is None or math.isnan(score):
        return None
    return float(score)


def _fmt(score):
    return "FAIL" if score is None else f"{score:.2f}"
