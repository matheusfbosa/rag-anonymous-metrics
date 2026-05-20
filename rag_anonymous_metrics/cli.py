import logging
import statistics
import sys
from typing import Iterable

from rag_anonymous import configure_logging

from rag_anonymous_metrics.config import Settings
from rag_anonymous_metrics.corpus import load_questions
from rag_anonymous_metrics.evaluator import STRATEGIES, run_evaluation
from rag_anonymous_metrics.export import export_csv

logger = logging.getLogger(__name__)

USAGE = "Usage: rag-metrics {evaluate|export}  (configure via env / .env)"


def cmd_evaluate() -> None:
    try:
        cfg = Settings.load()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    strategy_env = cfg.eval_strategy
    corpus_split = cfg.corpus_split
    k_docs = cfg.retrieval_k_docs
    llm_model = cfg.llm_model
    judge_model = cfg.judge_model

    strategies = STRATEGIES if strategy_env == "all" else [strategy_env]
    questions = load_questions(split=corpus_split)

    max_questions_limit = cfg.max_questions
    if max_questions_limit is not None and max_questions_limit < len(questions):
        logger.info(
            "Limiting benchmark to first %d of %d questions",
            max_questions_limit,
            len(questions),
        )
        questions = questions[:max_questions_limit]

    logger.info("Generator LLM: %s | Judge LLM: %s", llm_model, judge_model)

    results = run_evaluation(
        strategies=strategies,
        questions=questions,
        corpus_split=corpus_split,
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
        avg_leak = _safe_mean(r.leakage_rate for r in records)
        n_ref = sum(1 for r in records if r.answer_is_refusal)
        ref_rate = n_ref / len(records) if records else 0.0

        logger.info(
            "[%-8s] faith=%.3f ctx=%.3f leak=%.3f refusal=%.2f (%d/%d)",
            strategy, avg_faith, avg_ctx, avg_leak, ref_rate, n_ref, len(records),
        )
        logger.info("           -> %s", result.output_path)

    logger.info("")
    logger.info("Exporting metrics...")
    csv_path = export_csv()
    logger.info("Done. CSV at %s", csv_path)


def cmd_export() -> None:
    csv_path = export_csv()
    logger.info("Exported CSV to %s", csv_path)


def _safe_mean(values: Iterable[float | None]) -> float:
    nums = [v for v in values if v is not None]
    return statistics.fmean(nums) if nums else 0.0


COMMANDS = {
    "evaluate": cmd_evaluate,
    "export": cmd_export,
}


def main() -> None:
    configure_logging()

    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(USAGE)

    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()
