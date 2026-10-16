import json
import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from rag_anonymous import (
    Anonymizer,
    build_llm,
    create_chain,
    load_retriever,
    query_rag,
)
from rag_anonymous_metrics.config import Settings
from rag_anonymous_metrics.context_recall import (
    attach_context_recall,
    context_recall_path,
)
from rag_anonymous_metrics.log_config import log_item_progress, log_section
from rag_anonymous_metrics.metrics.privacy import (
    build_person_pii_index,
    build_pii_index,
    compute_privacy_metrics,
    is_refusal_answer,
)
from rag_anonymous_metrics.records import QueryRecord, StrategyResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pipeline:
    strategy: str
    gen_model: str
    chain: Any
    retrieval_store: Any
    anonymizer: Anonymizer | None


def build_pipeline(strategy: str, gen_model: str, dataset: str) -> Pipeline:
    retrieval_store = load_retriever(collection_name=strategy, dataset=dataset)
    chain = create_chain(build_llm(gen_model))
    anonymizer = Anonymizer() if strategy == "ondemand" else None
    return Pipeline(
        strategy=strategy,
        gen_model=gen_model,
        chain=chain,
        retrieval_store=retrieval_store,
        anonymizer=anonymizer,
    )


def evaluate_question(
    pipeline: Pipeline,
    question_obj: dict,
    pii_index: dict,
    person_pii_index: dict,
    judge_model: str,
    run_id: str,
    dataset: str,
    k_docs: int,
) -> QueryRecord:
    question = question_obj["question"]
    result = query_rag(
        pipeline.chain, pipeline.retrieval_store, question, k_docs=k_docs
    )

    generated_answer = response_text(result["response"])
    anonymize_sec = 0.0
    if pipeline.anonymizer:
        started = time.perf_counter()
        final_answer = pipeline.anonymizer.anonymize(generated_answer)
        anonymize_sec = time.perf_counter() - started
    else:
        final_answer = generated_answer

    retrieval_sec = float(result["retrieval_sec"])
    generation_sec = float(result["generation_sec"])
    query_sec = retrieval_sec + generation_sec + anonymize_sec
    logger.info(
        "  query=%.2fs retrieval=%.2fs generation=%.2fs anonymize=%.3fs",
        query_sec,
        retrieval_sec,
        generation_sec,
        anonymize_sec,
    )

    contexts = [c["content"] for c in result["retrieved_chunks"]]
    doc_ids = [c["doc_id"] or "" for c in result["retrieved_chunks"]]
    chunk_ids = [c["chunk_id"] or "" for c in result["retrieved_chunks"]]

    privacy = compute_privacy_metrics(
        generated_answer,
        final_answer,
        doc_ids,
        pii_index,
        person_pii_index,
    )
    refusal = is_refusal_answer(final_answer)
    source_doc_id = question_obj["source_doc_id"]

    return QueryRecord(
        run_id=run_id,
        strategy=pipeline.strategy,
        question_id=question_obj.get("id", ""),
        question=question,
        ground_truth=question_obj.get("ground_truth", ""),
        llm_model=pipeline.gen_model,
        judge_model=judge_model,
        generated_answer=generated_answer,
        final_answer=final_answer,
        answer_is_refusal=refusal,
        contexts=contexts,
        doc_ids=doc_ids,
        dataset=dataset,
        item_type=question_obj.get("type"),
        category=question_obj.get("category"),
        retrieval_sec=retrieval_sec,
        generation_sec=generation_sec,
        anonymize_sec=anonymize_sec,
        query_sec=query_sec,
        chunk_ids=chunk_ids,
        source_doc_id=source_doc_id,
        retrieval_hit=int(source_doc_id in doc_ids),
        **cast(dict[str, Any], privacy),
    )


def evaluate_strategy(
    strategy: str,
    questions: list[dict],
    pii_index: dict,
    person_pii_index: dict,
    experiment_dir: Path,
    dataset: str,
    k_docs: int,
    llm_model: str | None = None,
    judge_model: str | None = None,
    *,
    compute_utility: bool = True,
    output_basename: str | None = None,
) -> StrategyResult:
    suite = "benchmark" if compute_utility else "adversarial"
    log_section(f"Evaluating: strategy={strategy} suite={suite}")

    cfg = Settings.load()
    gen_model = llm_model or cfg.llm_model
    judge_m = judge_model or cfg.judge_model

    basename = output_basename or f"results_{strategy}.jsonl"
    output_path = experiment_dir / basename

    existing_by_id: dict[str, QueryRecord] = {}
    if cfg.overwrite:
        if output_path.exists():
            logger.info("Overwrite enabled: replacing path=%s", output_path)
            output_path.unlink()
    else:
        for rec in _load_jsonl(output_path):
            qid = rec.question_id.strip()
            if qid:
                existing_by_id[qid] = rec
        if existing_by_id:
            logger.info(
                "Resuming from checkpoint: path=%s saved=%d",
                output_path,
                len(existing_by_id),
            )

    missing = [q for q in questions if _question_id(q) not in existing_by_id]
    pipeline = None
    run_id = str(uuid.uuid4())[:8]
    if missing:
        pipeline = build_pipeline(strategy, gen_model, dataset)
        if compute_utility:
            logger.info(
                "Running benchmark: llm=%s n=%d skipped=%d",
                gen_model,
                len(missing),
                len(questions) - len(missing),
            )
        else:
            logger.info(
                "Running adversarial attacks: llm=%s n=%d skipped=%d",
                gen_model,
                len(missing),
                len(questions) - len(missing),
            )
    else:
        logger.info(
            "Skipping RAG: all %d questions already saved path=%s",
            len(questions),
            output_path,
        )

    records: list[QueryRecord] = []
    for idx, q in enumerate(questions, 1):
        qid = _question_id(q)
        if qid in existing_by_id:
            records.append(existing_by_id[qid])
            continue
        log_item_progress(
            logger,
            idx,
            len(questions),
            qid or "-",
            q["question"],
        )
        if pipeline is None:
            pipeline = build_pipeline(strategy, gen_model, dataset)
        try:
            record = evaluate_question(
                pipeline,
                q,
                pii_index,
                person_pii_index,
                judge_m,
                run_id,
                dataset=dataset,
                k_docs=k_docs,
            )
        except Exception:
            logger.exception(
                "Skipping RAG failure: question_id=%s — retry on resume",
                qid or "-",
            )
            continue
        if not str(record.generated_answer).strip():
            logger.warning(
                "Skipping empty generation: question_id=%s — retry on resume",
                qid or "-",
            )
            continue
        records.append(record)
        if qid:
            existing_by_id[qid] = record
        _append_jsonl(record, output_path)

    if records and not any(r.pii_total_final for r in records):
        logger.error(
            "PII index mismatch: strategy=%s dataset=%s n=%d",
            strategy,
            dataset,
            len(records),
        )

    if compute_utility and not cfg.skip_judge:
        _attach_utility(records, judge_m, output_path)
    else:
        if compute_utility:
            logger.info("Skipping judge: RAG_METRICS_SKIP_JUDGE set")
        _upsert_jsonl(output_path, records)

    return StrategyResult(
        strategy=strategy, records=records, output_path=str(output_path)
    )


def response_text(raw: object) -> str:
    content = getattr(raw, "content", None)
    if isinstance(content, str):
        return content
    return raw if isinstance(raw, str) else str(raw)


def run_experiment(
    strategies: Sequence[str],
    questions: list[dict],
    dataset: str,
    k_docs: int,
    llm_model: str | None = None,
    judge_model: str | None = None,
    *,
    compute_utility: bool = True,
    output_suffix: str = "",
    experiment_dir: Path | None = None,
    pii_index: dict | None = None,
    person_pii_index: dict | None = None,
) -> tuple[dict[str, StrategyResult], Path]:
    from rag_anonymous_metrics.corpus import load_corpus

    cfg = Settings.load()
    if experiment_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = cfg.results_dir / stamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Experiment output directory: dir=%s", experiment_dir)

    if pii_index is None or person_pii_index is None:
        logger.info("Loading corpus for PII annotations: dataset=%s", dataset)
        corpus = load_corpus(dataset)
        if pii_index is None:
            pii_index = build_pii_index(corpus)
        if person_pii_index is None:
            person_pii_index = build_person_pii_index(corpus)
        logger.info(
            "PII index built: tab_docs=%d person_docs=%d",
            len(pii_index),
            len(person_pii_index),
        )

    results = {}
    for strategy in strategies:
        basename = f"results_{strategy}{output_suffix}.jsonl"
        results[strategy] = evaluate_strategy(
            strategy,
            questions,
            pii_index,
            person_pii_index,
            experiment_dir,
            dataset=dataset,
            k_docs=k_docs,
            llm_model=llm_model,
            judge_model=judge_model,
            compute_utility=compute_utility,
            output_basename=basename,
        )
    return results, experiment_dir


def rejudge_experiment(
    source_dir: Path,
    output_dir: Path,
    judge_model: str,
) -> dict[str, StrategyResult]:
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError(f"Rejudge output must differ from source: dir={source_dir}")
    sources = sorted(source_dir.glob("results_*_benchmark.jsonl"))
    if not sources:
        raise ValueError(f"No results_*_benchmark.jsonl in {source_dir}")

    for adversarial in sorted(source_dir.glob("results_*_adversarial.jsonl")):
        target = output_dir / adversarial.name
        if not target.exists():
            _save_jsonl(_load_jsonl(adversarial), target)

    results = {}
    for source in sources:
        output_path = output_dir / source.name
        if output_path.exists():
            records = _load_jsonl(output_path)
            logger.info("Resuming rejudge: path=%s n=%d", output_path, len(records))
        else:
            records = [_reset_utility(r) for r in _load_jsonl(source)]
            _save_jsonl(records, output_path)
        log_section(f"Rejudging: source={source} judge={judge_model}")
        _attach_utility(records, judge_model, output_path)
        strategy = records[0].strategy if records else source.stem
        results[strategy] = StrategyResult(
            strategy=strategy, records=records, output_path=str(output_path)
        )
    return results


def _reset_utility(record: QueryRecord) -> QueryRecord:
    return replace(
        record,
        faithfulness=None,
        factual_correctness=None,
        answer_accuracy=None,
        semantic_similarity=None,
        fc_n_resp=None,
        fc_n_resp_ok=None,
        fc_n_ref=None,
        fc_n_ref_ok=None,
        judge_sec=None,
        judge_metric_sec=None,
    )


def _attach_utility(
    records: list[QueryRecord],
    judge_model: str,
    output_path: Path,
) -> None:
    _attach_answer_utility(records, judge_model, output_path)
    attach_context_recall(
        records,
        judge_model,
        context_recall_path(Settings.load().results_dir, judge_model, output_path.name),
    )


def _attach_answer_utility(
    records: list[QueryRecord],
    judge_model: str,
    output_path: Path,
) -> None:
    from rag_anonymous_metrics.metrics.utility import COVERAGE_KEYS, UtilityScorer

    pending = [r for r in records if _needs_utility(r)]
    if not pending:
        logger.info("Utility metrics already complete: n=%d", len(records))
        _upsert_jsonl(output_path, records)
        return

    scorer = UtilityScorer(judge_model)
    logger.info(
        "Computing utility metrics: judge=%s pending=%d n=%d",
        judge_model,
        len(pending),
        len(records),
    )
    for idx, record in enumerate(records, 1):
        if not _needs_utility(record):
            continue
        log_item_progress(
            logger,
            idx,
            len(records),
            record.question_id or "-",
            record.question,
        )

        refusal = _is_refusal(record)
        started = time.perf_counter()
        try:
            scores = scorer.score(
                record.question,
                record.final_answer,
                record.contexts,
                record.ground_truth,
                refusal=refusal,
            )
        except Exception:
            logger.exception(
                "Skipping judge failure: question_id=%s — retry on resume",
                record.question_id or "-",
            )
            continue
        record.judge_model = judge_model
        record.judge_sec = time.perf_counter() - started
        record.judge_metric_sec = scores["judge_metric_sec"]
        if record.fc_n_ref is None:
            record.factual_correctness = scores["factual_correctness"]
            for key in COVERAGE_KEYS:
                setattr(record, key, scores[key])
        if record.answer_accuracy is None:
            record.answer_accuracy = scores["answer_accuracy"]
        if record.semantic_similarity is None:
            record.semantic_similarity = scores["semantic_similarity"]
        if record.faithfulness is None:
            record.faithfulness = scores["faithfulness"]
        logger.info("  judge=%.1fs refusal=%s", record.judge_sec, refusal)
        _upsert_jsonl(output_path, records)


def _is_refusal(record: QueryRecord) -> bool:
    return record.answer_is_refusal or is_refusal_answer(record.final_answer)


def _needs_utility(record: QueryRecord) -> bool:
    return (
        record.fc_n_ref is None
        or record.answer_accuracy is None
        or record.semantic_similarity is None
        or (
            record.faithfulness is None
            and bool(record.contexts)
            and not _is_refusal(record)
        )
    )


def _question_id(question_obj: dict) -> str:
    return str(question_obj.get("id", "")).strip()


def _record_from_dict(data: dict[str, Any]) -> QueryRecord:
    names = {f.name for f in fields(QueryRecord)}
    return QueryRecord(**{k: v for k, v in data.items() if k in names})


def _load_jsonl(path: Path) -> list[QueryRecord]:
    if not path.exists():
        return []
    records: list[QueryRecord] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping corrupt JSONL line: path=%s line=%d",
                    path,
                    line_no,
                )
                continue
            records.append(_record_from_dict(data))
    return records


def _append_jsonl(record: QueryRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        f.flush()
    logger.debug("Appended record: question_id=%s path=%s", record.question_id, path)


def _save_jsonl(records: list[QueryRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        f.flush()
    logger.info("Saved records: n=%d path=%s", len(records), path)


def _upsert_jsonl(path: Path, records: list[QueryRecord]) -> None:
    existing = _load_jsonl(path)
    by_id = {r.question_id: r for r in existing if r.question_id}
    incoming_ids: list[str] = []
    for rec in records:
        if rec.question_id and rec.question_id not in by_id:
            incoming_ids.append(rec.question_id)
        if rec.question_id:
            by_id[rec.question_id] = rec
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for rec in existing:
        qid = rec.question_id
        if qid and qid not in seen and qid in by_id:
            ordered_ids.append(qid)
            seen.add(qid)
    for qid in incoming_ids:
        if qid not in seen:
            ordered_ids.append(qid)
            seen.add(qid)
    merged = [by_id[qid] for qid in ordered_ids]
    extras = [r for r in records if not r.question_id]
    _save_jsonl(merged + extras, path)
