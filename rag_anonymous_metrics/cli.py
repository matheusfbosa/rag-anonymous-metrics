import logging
import statistics
import sys
from typing import Iterable

from rag_anonymous import configure_logging

from rag_anonymous_metrics.config import Settings
from rag_anonymous_metrics.corpus import load_questions
from rag_anonymous_metrics.run import STRATEGIES, run_experiment

logger = logging.getLogger(__name__)

USAGE = "Usage: rag-metrics run"


def cmd_run_experiment() -> None:
    try:
        cfg = Settings.load()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    strategy_env = cfg.eval_strategy
    dataset = cfg.dataset
    k_docs = cfg.retrieval_k_docs
    llm_model = cfg.llm_model
    judge_model = cfg.judge_model

    strategies = STRATEGIES if strategy_env == "all" else [strategy_env]
    questions = load_questions(split=dataset)

    max_questions_limit = cfg.max_questions
    if max_questions_limit is not None and max_questions_limit < len(questions):
        logger.info(
            "Limiting benchmark to first %d of %d questions",
            max_questions_limit,
            len(questions),
        )
        questions = questions[:max_questions_limit]

    logger.info("Generator LLM: %s | Judge LLM: %s", llm_model, judge_model)

    results, experiment_dir = run_experiment(
        strategies=strategies,
        questions=questions,
        dataset=dataset,
        k_docs=k_docs,
        llm_model=llm_model,
        judge_model=judge_model,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Per-strategy summary")
    logger.info("=" * 60)
    for strategy, result in results.items():
        records = result.records
        avg_faith = _safe_mean(r.faithfulness for r in records)
        avg_ctx = _safe_mean(r.context_precision for r in records)
        avg_recall = _safe_mean(r.context_recall for r in records)
        avg_factual = _safe_mean(r.factual_correctness for r in records)
        avg_leak = _safe_mean(r.leakage_rate for r in records)
        n_ref = sum(1 for r in records if r.answer_is_refusal)
        ref_rate = n_ref / len(records) if records else 0.0

        logger.info(
            "[%-8s] faith=%.3f ctx=%.3f recall=%.3f factual*=%.3f leak=%.3f refusal=%.2f (%d/%d)",
            strategy, avg_faith, avg_ctx, avg_recall, avg_factual, avg_leak, ref_rate, n_ref, len(records),
        )
        logger.info("           -> %s", result.output_path)

    logger.info(
        "(faith/ctx/recall are conditional on answering; "
        "factual* is end-to-end, counting refusals as 0)"
    )

    logger.info("")
    logger.info("Done. Experiment directory: %s", experiment_dir)


COMMANDS = {
    "run": cmd_run_experiment,
    "evaluate": cmd_run_experiment,
}


def main() -> None:
    configure_logging()

    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(USAGE)

    COMMANDS[sys.argv[1]]()


def _safe_mean(values: Iterable[float | None]) -> float:
    nums = [v for v in values if v is not None]
    return statistics.fmean(nums) if nums else 0.0


if __name__ == "__main__":
    main()
