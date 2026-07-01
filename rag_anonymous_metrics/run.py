import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_anonymous import (
    Anonymizer,
    build_llm,
    create_chain,
    load_vectordb,
    query_rag,
)

from rag_anonymous_metrics.config import Settings
from rag_anonymous_metrics.metrics.privacy import build_pii_index, compute_pii_leakage, is_refusal_answer
from rag_anonymous_metrics.records import QueryRecord, StrategyResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pipeline:
    strategy: str
    gen_model: str
    chain: Any
    vectordb: Any
    anonymizer: Anonymizer | None


STRATEGIES = ["offline", "ondemand"]


def build_pipeline(strategy: str, gen_model: str, dataset: str) -> Pipeline:
    vectordb = load_vectordb(collection_name=strategy)
    _check_dataset(vectordb, strategy, dataset)
    chain = create_chain(build_llm(gen_model))
    anonymizer = Anonymizer() if strategy == "ondemand" else None
    return Pipeline(
        strategy=strategy,
        gen_model=gen_model,
        chain=chain,
        vectordb=vectordb,
        anonymizer=anonymizer,
    )


def evaluate_question(
    pipeline: Pipeline,
    question_obj: dict,
    pii_index: dict,
    judge_model: str,
    run_id: str,
    dataset: str,
    k_docs: int,
) -> QueryRecord:
    question = question_obj["question"]
    result = query_rag(pipeline.chain, pipeline.vectordb, question, k_docs=k_docs)

    raw_answer = response_text(result["response"])
    final_answer = (
        pipeline.anonymizer.anonymize(raw_answer) if pipeline.anonymizer else raw_answer
    )

    contexts = [c["content"] for c in result["retrieved_chunks"]]
    doc_ids = [c["doc_id"] for c in result["retrieved_chunks"]]

    privacy = compute_pii_leakage(final_answer, doc_ids, pii_index)
    refusal = is_refusal_answer(final_answer)

    return QueryRecord(
        run_id=run_id,
        strategy=pipeline.strategy,
        question_id=question_obj.get("id", ""),
        question=question,
        ground_truth=question_obj.get("ground_truth", ""),
        llm_model=pipeline.gen_model,
        judge_model=judge_model,
        raw_answer=raw_answer,
        final_answer=final_answer,
        answer_is_refusal=refusal,
        contexts=contexts,
        doc_ids=doc_ids,
        dataset=dataset,
        total_pii=privacy["total_pii"],
        leaked_pii=privacy["leaked_pii"],
        leakage_rate=privacy["leakage_rate"],
    )


def evaluate_strategy(
    strategy: str,
    questions: list[dict],
    pii_index: dict,
    experiment_dir: Path,
    dataset: str,
    k_docs: int,
    llm_model: str | None = None,
    judge_model: str | None = None,
) -> StrategyResult:
    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluating: strategy=%s", strategy)

    cfg = Settings.load()
    gen_model = llm_model or cfg.llm_model
    judge_m = judge_model or cfg.judge_model

    pipeline = build_pipeline(strategy, gen_model, dataset)
    run_id = str(uuid.uuid4())[:8]
    records: list[QueryRecord] = []
    width = len(str(len(questions)))

    logger.info(
        "Generating answers (LLM=%s) and computing privacy metrics for %d records...",
        gen_model, len(questions),
    )
    for idx, q in enumerate(questions, 1):
        logger.info(
            "  [%*d/%d] %s: %.80s",
            width, idx, len(questions), q.get("id", "-"), q["question"],
        )
        records.append(
            evaluate_question(pipeline, q, pii_index, judge_m, run_id, dataset=dataset, k_docs=k_docs)
        )

    if records and not any(r.total_pii for r in records):
        logger.error(
            "total_pii=0 for all %d records in strategy='%s'; "
            "PII index (dataset='%s') likely does not match the ChromaDB corpus.",
            len(records), strategy, dataset,
        )

    _attach_utility(records, judge_m)

    output_path = experiment_dir / f"results_{strategy}.jsonl"
    _save_jsonl(records, output_path)

    return StrategyResult(strategy=strategy, records=records, output_path=str(output_path))


def response_text(raw):
    if hasattr(raw, "content"):
        return raw.content
    return raw if isinstance(raw, str) else str(raw)


def run_experiment(
    strategies: list[str],
    questions: list[dict],
    dataset: str,
    k_docs: int,
    llm_model: str | None = None,
    judge_model: str | None = None,
) -> tuple[dict[str, StrategyResult], Path]:
    from rag_anonymous_metrics.corpus import load_corpus

    cfg = Settings.load()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = cfg.results_dir / stamp
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Experiment output directory: %s", experiment_dir)

    logger.info("Loading corpus (dataset=%s) for PII annotations...", dataset)
    corpus = load_corpus(dataset)
    pii_index = build_pii_index(corpus)
    logger.info("PII index built: %d documents with sensitive spans", len(pii_index))

    results = {
        strategy: evaluate_strategy(
            strategy,
            questions,
            pii_index,
            experiment_dir,
            dataset=dataset,
            k_docs=k_docs,
            llm_model=llm_model,
            judge_model=judge_model,
        )
        for strategy in strategies
    }
    return results, experiment_dir


def _check_dataset(vectordb, collection_name: str, expected: str) -> None:
    try:
        result = vectordb._collection.get(limit=1, include=["metadatas"])
        detected = ((result.get("metadatas") or [{}])[0] or {}).get("dataset")
    except Exception:
        detected = None
    if detected is None:
        logger.warning(
            "Collection '%s' has no 'dataset' metadata; skipping consistency check. "
            "Re-ingest to enable this guard.",
            collection_name,
        )
        return
    if detected != expected:
        raise SystemExit(
            f"Dataset mismatch: collection '{collection_name}' was ingested with "
            f"dataset='{detected}', but RAG_METRICS_CORPUS_DATASET='{expected}'. "
            f"Re-ingest with RAG_ANON_CORPUS_DATASET={expected}."
        )


def _attach_utility(records: list[QueryRecord], judge_model: str) -> None:
    from rag_anonymous_metrics.metrics.utility import UtilityScorer

    logger.info(
        "Computing utility metrics (judge=%s) for %d records...",
        judge_model, len(records),
    )
    scorer = UtilityScorer(judge_model)
    width = len(str(len(records)))

    for idx, record in enumerate(records, 1):
        logger.info(
            "  [%*d/%d] %s: %.80s",
            width, idx, len(records), record.question_id or "-", record.question,
        )

        if record.answer_is_refusal or is_refusal_answer(record.final_answer):
            logger.info(
                "  [%*d/%d] skipping judge: refusal / non-answer",
                width, idx, len(records),
            )
            record.factual_correctness = 0.0
            continue

        scores = scorer.score(
            record.question, record.final_answer, record.contexts, record.ground_truth,
        )
        record.faithfulness = scores["faithfulness"]
        record.context_precision = scores["context_precision"]
        record.context_recall = scores["context_recall"]
        record.factual_correctness = scores["factual_correctness"]


def _save_jsonl(records: list[QueryRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    logger.info("Saved %d records to %s", len(records), path)
