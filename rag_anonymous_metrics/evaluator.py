"""Per-question evaluation loop and JSONL persistence."""

import json
import logging
import os
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from rag_anonymous import query_rag

from rag_anonymous_metrics.metrics.pii_leakage import build_pii_index, compute_pii_leakage
from rag_anonymous_metrics.metrics.refusal import is_refusal_answer
from rag_anonymous_metrics.pipeline import Pipeline, build_pipeline, response_text
from rag_anonymous_metrics.records import QueryRecord, StrategyResult

logger = logging.getLogger(__name__)

STRATEGIES = ["offline", "ondemand"]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_evaluation(
    strategies: list[str],
    questions: list[dict],
    corpus_split: str,
    k_docs: int | None = None,
    llm_model: str | None = None,
    judge_model: str | None = None,
) -> dict[str, StrategyResult]:
    """Run the full evaluation pipeline for one or more strategies."""
    from rag_anonymous_metrics.corpus import load_corpus

    logger.info("Loading TAB corpus (split=%s) for PII annotations...", corpus_split)
    corpus = load_corpus(corpus_split)
    pii_index = build_pii_index(corpus)
    logger.info("PII index built: %d documents with sensitive spans", len(pii_index))

    return {
        strategy: evaluate_strategy(
            strategy,
            questions,
            pii_index,
            k_docs=k_docs,
            llm_model=llm_model,
            judge_model=judge_model,
        )
        for strategy in strategies
    }


def evaluate_strategy(
    strategy: str,
    questions: list[dict],
    pii_index: dict,
    k_docs: int | None = None,
    llm_model: str | None = None,
    judge_model: str | None = None,
) -> StrategyResult:
    """Evaluate one strategy across all benchmark questions and persist JSONL."""
    logger.info("")
    logger.info("=== Evaluating strategy: %s ===", strategy)

    gen_model = llm_model or _llm_model()
    judge_m = judge_model or _judge_model()
    if k_docs is None:
        k_docs = _k_docs()

    pipeline = build_pipeline(strategy, gen_model)
    run_id = str(uuid.uuid4())[:8]
    records: list[QueryRecord] = []
    width = len(str(len(questions)))

    for idx, q in enumerate(questions, 1):
        logger.info(
            "[%*d/%d] %s: %.80s",
            width, idx, len(questions), q.get("id", "-"), q["question"],
        )
        records.append(
            evaluate_question(pipeline, q, pii_index, judge_m, run_id, k_docs=k_docs)
        )

    _attach_utility(records, judge_m)

    output_path = _results_dir() / _jsonl_filename(strategy, gen_model)
    _save_jsonl(records, output_path)

    return StrategyResult(strategy=strategy, records=records, output_path=str(output_path))


def evaluate_question(
    pipeline: Pipeline,
    question_obj: dict,
    pii_index: dict,
    judge_model: str,
    run_id: str,
    k_docs: int | None = None,
) -> QueryRecord:
    """Run RAG, anonymize (if applicable), score privacy + refusal for one question."""
    if k_docs is None:
        k_docs = _k_docs()
    question = question_obj["question"]
    result = query_rag(pipeline.chain, pipeline.vectordb, question, k_docs=k_docs)

    raw_answer = response_text(result["response"])
    final_answer = (
        pipeline.anonymizer.anonymize(raw_answer) if pipeline.anonymizer else raw_answer
    )

    contexts = [c["content"] for c in result["retrieved_chunks"]]
    doc_ids = list({c["doc_id"] for c in result["retrieved_chunks"]})

    privacy = compute_pii_leakage(final_answer, doc_ids, pii_index)
    refusal = is_refusal_answer(final_answer)

    return QueryRecord(
        run_id=run_id,
        strategy=pipeline.strategy,
        question_id=question_obj.get("id", ""),
        question=question,
        llm_model=pipeline.gen_model,
        judge_model=judge_model,
        raw_answer=raw_answer,
        final_answer=final_answer,
        answer_is_refusal=refusal,
        contexts=contexts,
        doc_ids=doc_ids,
        total_pii=privacy["total_pii"],
        leaked_pii=privacy["leaked_pii"],
        leakage_rate=privacy["leakage_rate"],
    )


def _attach_utility(records: list[QueryRecord], judge_model: str) -> None:
    """Compute and attach faithfulness / context_precision in place."""
    from rag_anonymous_metrics.metrics import faithfulness, context_precision
    from rag_anonymous_metrics.metrics._judge import build_judge_chain

    logger.info(
        "Computing utility metrics for %d records (judge=%s)...",
        len(records), judge_model,
    )
    chain = build_judge_chain(judge_model)
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
            continue

        record.faithfulness = faithfulness.score_faithfulness(
            chain, record.final_answer, record.contexts,
        )
        record.context_precision = context_precision.score_context_precision(
            chain, record.question, record.final_answer, record.contexts,
        )


def _save_jsonl(records: list[QueryRecord], path: Path) -> None:
    """Write a batch of records as one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    logger.info("Saved %d records to %s", len(records), path)


def _jsonl_filename(strategy: str, llm_model: str) -> str:
    """Build output filename per the project naming convention."""
    model_slug = llm_model.replace(":", "_").replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"results_{model_slug}_{strategy}_{timestamp}.jsonl"


def _results_dir() -> Path:
    return _PROJECT_ROOT / os.getenv("RAG_METRICS_RESULTS_DIR", "./data/output")


def _llm_model() -> str:
    return os.getenv("RAG_ANON_LLM_MODEL", "qwen3:0.6b")


def _judge_model() -> str:
    return os.getenv("RAG_METRICS_JUDGE_MODEL") or _llm_model()


def _k_docs() -> int:
    return int(os.getenv("RAG_METRICS_RETRIEVAL_K_DOCS", "5"))
