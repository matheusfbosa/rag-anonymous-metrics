import logging
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path

from rag_anonymous import Anonymizer
from rag_anonymous_metrics.check_judge import check_judge
from rag_anonymous_metrics.config import Settings
from rag_anonymous_metrics.corpus import (
    load_adversarial_attacks,
    load_corpus,
    load_questions,
)
from rag_anonymous_metrics.environment import log_environment
from rag_anonymous_metrics.judge_prompts import dump_judge_prompts
from rag_anonymous_metrics.log_config import (
    configure_logging,
    init_command_output,
    log_section,
    log_suite_header,
)
from rag_anonymous_metrics.metrics.adversarial import (
    PERSON_LEAK,
    TARGET_LEAK,
    asr_by_type,
    attack_success_rate,
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
from rag_anonymous_metrics.run import (
    reanonymize_experiment,
    rejudge_experiment,
    run_experiment,
)

logger = logging.getLogger(__name__)

USAGE = "Usage: rag-metrics {run|rejudge|reanonymize|judge-prompts|check-judge}"


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
        avg_factual = _safe_mean(r.factual_correctness for r in records)
        avg_accuracy = _safe_mean(r.answer_accuracy for r in records)
        avg_similarity = _safe_mean(r.semantic_similarity for r in records)
        avg_hit = _safe_mean(r.retrieval_hit for r in records)
        macro_avg_leak = _safe_mean(r.pii_leakage_rate_final for r in records)
        sum_leaked = sum(r.pii_leaked_final for r in records)
        sum_total = sum(r.pii_total_final for r in records)
        micro_avg_leak = sum_leaked / sum_total if sum_total else 0.0
        n_person_leak = sum(1 for r in records if r.pii_person_any_leak_final)
        n_ref = sum(1 for r in records if r.answer_is_refusal)
        ref_rate = n_ref / len(records) if records else 0.0

        avg_query = _safe_mean(r.query_sec for r in records)
        avg_retrieval = _safe_mean(r.retrieval_sec for r in records)
        avg_generation = _safe_mean(r.generation_sec for r in records)
        avg_anonymize = _safe_mean(r.anonymize_sec for r in records)

        logger.info(
            "           strategy=%-8s coverage=%.3f accuracy=%.3f similarity=%.3f "
            "faith=%.3f hit=%.3f "
            "person_any=%.3f (%d/%d) leak_macro=%.3f leak_micro=%.3f (%d/%d) "
            "refusal=%.2f (%d/%d)",
            strategy,
            avg_factual,
            avg_accuracy,
            avg_similarity,
            avg_faith,
            avg_hit,
            n_person_leak / len(records) if records else 0.0,
            n_person_leak,
            len(records),
            macro_avg_leak,
            micro_avg_leak,
            sum_leaked,
            sum_total,
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
        "(coverage/accuracy/similarity judge every answer; faith is null on refusal; "
        "context recall is cached once per judge in data/output/context_recall; "
        "person_any=share of answers with a person name; "
        "leak_macro/leak_micro=full TAB inventory, per-question / pooled)"
    )


def _log_adversarial_summary(results: dict[str, StrategyResult]) -> None:
    log_section("Per-strategy summary: suite=adversarial metric=ASR")
    for strategy, result in results.items():
        records = result.records
        target = attack_success_rate(records, TARGET_LEAK)
        person = attack_success_rate(records, PERSON_LEAK)
        target_by_type = asr_by_type(records, TARGET_LEAK)
        person_by_type = asr_by_type(records, PERSON_LEAK)
        logger.info(
            "           strategy=%-8s ASR_target=%.3f (%d/%d) ASR_person=%.3f (%d/%d) "
            "refusal=%.2f",
            strategy,
            target["asr"],
            target["successes"],
            target["n"],
            person["asr"],
            person["successes"],
            person["n"],
            refusal_rate(records),
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
        for type_name, stats in target_by_type.items():
            person_stats = person_by_type[type_name]
            logger.info(
                "           type=%-20s ASR_target=%.3f (%d/%d) ASR_person=%.3f (%d/%d)",
                type_name,
                stats["asr"],
                stats["successes"],
                stats["n"],
                person_stats["asr"],
                person_stats["successes"],
                person_stats["n"],
            )
        if strategy == "ondemand":
            for name, prefix in (("target", "pii_target"), ("person", "pii_person")):
                logger.info(
                    "           leak_any_%s: generated=%d final=%d n=%d",
                    name,
                    sum(
                        1 for r in records if getattr(r, f"{prefix}_any_leak_generated")
                    ),
                    sum(1 for r in records if getattr(r, f"{prefix}_any_leak_final")),
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


def cmd_reanonymize(experiment_dir: Path) -> None:
    cfg = Settings.load()
    if cfg.judge_source_dir is None:
        raise SystemExit("RAG_METRICS_JUDGE_SOURCE_DIR is required for reanonymize")
    logger.info(
        "Reanonymize: source=%s output=%s judge=%s",
        cfg.judge_source_dir,
        experiment_dir,
        cfg.judge_model,
    )
    corpus = load_corpus(cfg.dataset)
    try:
        results = reanonymize_experiment(
            cfg.judge_source_dir,
            experiment_dir,
            cfg.judge_model,
            Anonymizer(),
            build_pii_index(corpus),
            build_person_pii_index(corpus),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _log_benchmark_summary(results)
    logger.info("Done: experiment_dir=%s", experiment_dir)


def cmd_judge_prompts(experiment_dir: Path) -> None:
    cfg = Settings.load()
    if cfg.judge_source_dir is None:
        raise SystemExit("RAG_METRICS_JUDGE_SOURCE_DIR is required for judge-prompts")
    try:
        written = dump_judge_prompts(
            cfg.judge_source_dir,
            experiment_dir,
            cfg.judge_model,
            cfg.questions_range,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    logger.info("Done: prompts=%d experiment_dir=%s", len(written), experiment_dir)


def main() -> None:
    commands = (
        "run",
        "evaluate",
        "rejudge",
        "reanonymize",
        "judge-prompts",
        "check-judge",
    )
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
    log_environment(logger)

    if command in ("run", "evaluate"):
        cmd_run_experiment(experiment_dir)
    elif command == "rejudge":
        cmd_rejudge(experiment_dir)
    elif command == "reanonymize":
        cmd_reanonymize(experiment_dir)
    elif command == "judge-prompts":
        cmd_judge_prompts(experiment_dir)
    else:
        cmd_check_judge()


def _safe_mean(values: Iterable[float | None]) -> float:
    nums = [v for v in values if v is not None]
    return statistics.fmean(nums) if nums else 0.0


if __name__ == "__main__":
    main()
