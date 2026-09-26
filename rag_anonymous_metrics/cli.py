import logging
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path

from rag_anonymous_metrics.check_judge import check_judge
from rag_anonymous_metrics.config import Settings
from rag_anonymous_metrics.corpus import (
    load_adversarial_attacks,
    load_corpus,
    load_questions,
)
from rag_anonymous_metrics.log_config import (
    configure_logging,
    init_command_output,
    log_section,
    log_suite_header,
)
from rag_anonymous_metrics.metrics.adversarial import (
    asr_by_type,
    asr_person_by_type,
    attack_success_rate,
    attack_success_rate_person,
    refusal_rate,
)
from rag_anonymous_metrics.metrics.privacy import (
    build_person_pii_index,
    build_pii_index,
)
from rag_anonymous_metrics.questions_filter import (
    apply_category_filter,
    apply_id_range,
    describe_category,
    describe_range,
)
from rag_anonymous_metrics.records import StrategyResult
from rag_anonymous_metrics.run import rejudge_experiment, run_experiment

logger = logging.getLogger(__name__)

USAGE = "Usage: rag-metrics {run|rejudge|check-judge}"


def cmd_run_experiment(experiment_dir: Path) -> None:
    try:
        cfg = Settings.load()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    strategy = cfg.strategy
    dataset = cfg.dataset
    k_docs = cfg.retrieval_k_docs
    llm_model = cfg.llm_model
    judge_model = cfg.judge_model
    category = cfg.questions_category
    id_range = cfg.questions_range

    strategies = [strategy]

    run_benchmark = category in {"benchmark", "factual", "identity"}
    run_adversarial = category == "adversarial"

    logger.info("Experiment output directory: dir=%s", experiment_dir)

    logger.info("Loading corpus for PII annotations: dataset=%s", dataset)
    corpus = load_corpus(dataset)
    pii_index = build_pii_index(corpus)
    person_pii_index = build_person_pii_index(corpus)
    logger.info(
        "PII index built: tab_docs=%d person_docs=%d",
        len(pii_index),
        len(person_pii_index),
    )

    if run_benchmark:
        log_suite_header("benchmark", category, id_range)
        all_questions = load_questions(dataset=dataset)
        if category == "benchmark":
            questions = apply_id_range(all_questions, id_range)
        else:
            questions = apply_id_range(
                apply_category_filter(all_questions, category),
                id_range,
            )
        logger.info(
            "Benchmark selection: category=%s range=%s selected=%d total=%d",
            describe_category(category),
            describe_range(id_range),
            len(questions),
            len(all_questions),
        )
        if not questions:
            raise SystemExit(
                "No benchmark questions match RAG_METRICS_QUESTIONS_CATEGORY/"
                "RAG_METRICS_QUESTIONS_RANGE"
            )

        results, _ = run_experiment(
            strategies=strategies,
            questions=questions,
            dataset=dataset,
            k_docs=k_docs,
            llm_model=llm_model,
            judge_model=judge_model,
            compute_utility=True,
            output_suffix="_benchmark",
            experiment_dir=experiment_dir,
            pii_index=pii_index,
            person_pii_index=person_pii_index,
        )
        _log_benchmark_summary(results)

    if run_adversarial:
        log_suite_header("adversarial", category, id_range)
        all_attacks = load_adversarial_attacks(dataset=dataset)
        attacks = apply_id_range(all_attacks, id_range)
        logger.info(
            "Adversarial selection: category=%s range=%s selected=%d total=%d",
            describe_category(category),
            describe_range(id_range),
            len(attacks),
            len(all_attacks),
        )
        if not attacks:
            raise SystemExit(
                "No adversarial attacks match RAG_METRICS_QUESTIONS_CATEGORY/"
                "RAG_METRICS_QUESTIONS_RANGE"
            )

        adv_results, _ = run_experiment(
            strategies=strategies,
            questions=attacks,
            dataset=dataset,
            k_docs=k_docs,
            llm_model=llm_model,
            judge_model=judge_model,
            compute_utility=False,
            output_suffix="_adversarial",
            experiment_dir=experiment_dir,
            pii_index=pii_index,
            person_pii_index=person_pii_index,
        )
        _log_adversarial_summary(adv_results)

    logger.info("Done: experiment_dir=%s", experiment_dir)


def _log_benchmark_summary(results: dict[str, StrategyResult]) -> None:
    log_section("Per-strategy summary: suite=benchmark")
    for strategy, result in results.items():
        records = result.records
        avg_faith = _safe_mean(r.faithfulness for r in records)
        avg_ctx = _safe_mean(r.context_precision for r in records)
        avg_recall = _safe_mean(r.context_recall for r in records)
        avg_factual = _safe_mean(r.factual_correctness for r in records)
        macro_avg_leak = _safe_mean(r.pii_leakage_rate_final for r in records)
        macro_avg_person_leak = _safe_mean(
            r.pii_person_leakage_rate_final for r in records
        )
        sum_leaked = sum(r.pii_leaked_final for r in records)
        sum_total = sum(r.pii_total_final for r in records)
        sum_person_leaked = sum(r.pii_person_leaked_final for r in records)
        sum_person_total = sum(r.pii_person_total_final for r in records)
        micro_avg_leak = sum_leaked / sum_total if sum_total else 0.0
        micro_avg_person_leak = (
            sum_person_leaked / sum_person_total if sum_person_total else 0.0
        )
        n_ref = sum(1 for r in records if r.answer_is_refusal)
        ref_rate = n_ref / len(records) if records else 0.0

        avg_query = _safe_mean(r.query_sec for r in records)
        avg_retrieval = _safe_mean(r.retrieval_sec for r in records)
        avg_generation = _safe_mean(r.generation_sec for r in records)
        avg_anonymize = _safe_mean(r.anonymize_sec for r in records)

        logger.info(
            "           strategy=%-8s faith=%.3f ctx=%.3f recall=%.3f factual*=%.3f "
            "leak_macro=%.3f leak_micro=%.3f (%d/%d) "
            "person_macro=%.3f person_micro=%.3f (%d/%d) refusal=%.2f (%d/%d)",
            strategy,
            avg_faith,
            avg_ctx,
            avg_recall,
            avg_factual,
            macro_avg_leak,
            micro_avg_leak,
            sum_leaked,
            sum_total,
            macro_avg_person_leak,
            micro_avg_person_leak,
            sum_person_leaked,
            sum_person_total,
            ref_rate,
            n_ref,
            len(records),
        )
        logger.info(
            "           strategy=%-8s query=%.1fs retrieval=%.2fs generation=%.1fs "
            "anonymize=%.3fs judge=%.1fs",
            strategy,
            avg_query,
            avg_retrieval,
            avg_generation,
            avg_anonymize,
            _safe_mean(r.judge_sec for r in records),
        )
        logger.info("           output_path=%s", result.output_path)

    logger.info(
        "(faith/ctx/recall are conditional on answering; "
        "factual* is end-to-end, counting refusals as 0; "
        "leak_macro=macro-avg of per-question rates, "
        "leak_micro=micro-avg (total leaked / total across all questions))"
    )


def _log_adversarial_summary(results: dict[str, StrategyResult]) -> None:
    log_section("Per-strategy summary: suite=adversarial metric=ASR")
    for strategy, result in results.items():
        records = result.records
        overall = attack_success_rate(records)
        overall_person = attack_success_rate_person(records)
        by_type = asr_by_type(records)
        by_type_person = asr_person_by_type(records)
        ref = refusal_rate(records)
        logger.info(
            "           strategy=%-8s ASR_tab=%.3f (%d/%d) ASR_person=%.3f (%d/%d) refusal=%.2f",
            strategy,
            overall["asr"],
            overall["successes"],
            overall["n"],
            overall_person["asr"],
            overall_person["successes"],
            overall_person["n"],
            ref,
        )
        logger.info(
            "           strategy=%-8s query=%.1fs retrieval=%.2fs generation=%.1fs "
            "anonymize=%.3fs",
            strategy,
            _safe_mean(r.query_sec for r in records),
            _safe_mean(r.retrieval_sec for r in records),
            _safe_mean(r.generation_sec for r in records),
            _safe_mean(r.anonymize_sec for r in records),
        )
        for type_name, stats in by_type.items():
            person_stats = by_type_person.get(
                type_name, {"asr": 0.0, "successes": 0, "n": 0}
            )
            logger.info(
                "           type=%-20s ASR_tab=%.3f (%d/%d) ASR_person=%.3f (%d/%d)",
                type_name,
                stats["asr"],
                stats["successes"],
                stats["n"],
                person_stats["asr"],
                person_stats["successes"],
                person_stats["n"],
            )
        gen_success = sum(1 for r in records if r.pii_any_leak_generated)
        fin_success = sum(1 for r in records if r.pii_any_leak_final)
        gen_person_success = sum(1 for r in records if r.pii_person_any_leak_generated)
        fin_person_success = sum(1 for r in records if r.pii_person_any_leak_final)
        if strategy == "ondemand" or gen_success != fin_success:
            logger.info(
                "           leak_any_tab: generated=%d final=%d n=%d",
                gen_success,
                fin_success,
                len(records),
            )
        if strategy == "ondemand" or gen_person_success != fin_person_success:
            logger.info(
                "           leak_any_person: generated=%d final=%d n=%d",
                gen_person_success,
                fin_person_success,
                len(records),
            )
        logger.info("           output_path=%s", result.output_path)


def cmd_check_judge() -> None:
    raise SystemExit(check_judge())


def cmd_rejudge(experiment_dir: Path) -> None:
    cfg = Settings.load()
    if cfg.judge_source_dir is None:
        raise SystemExit("RAG_METRICS_JUDGE_SOURCE_DIR is required for rejudge")
    logger.info(
        "Rejudge: source=%s output=%s judge=%s",
        cfg.judge_source_dir,
        experiment_dir,
        cfg.judge_model,
    )
    try:
        results = rejudge_experiment(
            cfg.judge_source_dir, experiment_dir, cfg.judge_model
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _log_benchmark_summary(results)
    logger.info("Done: experiment_dir=%s", experiment_dir)


def main() -> None:
    commands = ("run", "evaluate", "rejudge", "check-judge")
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit(USAGE)

    command = sys.argv[1]
    try:
        cfg = Settings.load()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    log_name = "check_judge.log" if command == "check-judge" else "run.log"
    reuse_dir = cfg.experiment_dir
    experiment_dir = init_command_output(
        cfg.results_dir,
        log_name,
        experiment_dir=reuse_dir,
        append_log=reuse_dir is not None and reuse_dir.exists(),
    )
    configure_logging()

    if command in ("run", "evaluate"):
        cmd_run_experiment(experiment_dir)
    elif command == "rejudge":
        cmd_rejudge(experiment_dir)
    else:
        cmd_check_judge()


def _safe_mean(values: Iterable[float | None]) -> float:
    nums = [v for v in values if v is not None]
    return statistics.fmean(nums) if nums else 0.0


if __name__ == "__main__":
    main()
