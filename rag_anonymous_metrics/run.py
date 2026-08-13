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
    load_retriever,
    query_rag,
)

from rag_anonymous_metrics.config import Settings
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
    result = query_rag(pipeline.chain, pipeline.retrieval_store, question, k_docs=k_docs)

    generated_answer = response_text(result["response"])
    final_answer = (
        pipeline.anonymizer.anonymize(generated_answer)
        if pipeline.anonymizer
        else generated_answer
    )

    contexts = [c["content"] for c in result["retrieved_chunks"]]
    doc_ids = [c["doc_id"] for c in result["retrieved_chunks"]]

    privacy = compute_privacy_metrics(
        generated_answer,
        final_answer,
        doc_ids,
        pii_index,
        person_pii_index,
    )
    refusal = is_refusal_answer(final_answer)

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
        **privacy,
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

    pipeline = build_pipeline(strategy, gen_model, dataset)
    run_id = str(uuid.uuid4())[:8]
    records: list[QueryRecord] = []

    if compute_utility:
        logger.info(
            "Running benchmark: llm=%s n=%d",
            gen_model,
            len(questions),
        )
    else:
        logger.info(
            "Running adversarial attacks: llm=%s n=%d",
            gen_model,
            len(questions),
        )
    for idx, q in enumerate(questions, 1):
        log_item_progress(
            logger, idx, len(questions), str(q.get("id", "-")), q["question"],
        )
        records.append(
            evaluate_question(
                pipeline,
                q,
                pii_index,
                person_pii_index,
                judge_m,
                run_id,
                dataset=dataset,
                k_docs=k_docs,
            )
        )

    if records and not any(r.pii_total_final for r in records):
        logger.error(
            "PII index mismatch: strategy=%s dataset=%s n=%d",
            strategy,
            dataset,
            len(records),
        )

    if compute_utility:
        _attach_utility(records, judge_m)

    basename = output_basename or f"results_{strategy}.jsonl"
    output_path = experiment_dir / basename
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

def _attach_utility(records: list[QueryRecord], judge_model: str) -> None:
    from rag_anonymous_metrics.metrics.utility import UtilityScorer

    logger.info(
        "Computing utility metrics: judge=%s n=%d",
        judge_model,
        len(records),
    )
    scorer = UtilityScorer(judge_model)
    width = len(str(len(records)))

    for idx, record in enumerate(records, 1):
        log_item_progress(
            logger,
            idx,
            len(records),
            record.question_id or "-",
            record.question,
        )

        if record.answer_is_refusal or is_refusal_answer(record.final_answer):
            logger.info(
                "  [%*d/%d] Skipping judge: reason=refusal",
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
    logger.info("Saved records: n=%d path=%s", len(records), path)
