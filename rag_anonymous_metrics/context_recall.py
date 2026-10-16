import json
import logging
import time
from pathlib import Path

from rag_anonymous_metrics.log_config import log_item_progress
from rag_anonymous_metrics.records import QueryRecord

logger = logging.getLogger(__name__)


def context_recall_path(results_dir: Path, judge_model: str, basename: str) -> Path:
    return results_dir / "context_recall" / judge_model.replace(":", "_") / basename


def attach_context_recall(
    records: list[QueryRecord], judge_model: str, path: Path
) -> None:
    from rag_anonymous_metrics.metrics.utility import ContextRecallScorer

    rows = _load(path)
    pending = [r for r in records if not _cached(rows.get(r.question_id), r)]
    if not pending:
        logger.info("Context recall cached: n=%d path=%s", len(records), path)
        return

    logger.info(
        "Computing context recall: judge=%s pending=%d n=%d path=%s",
        judge_model,
        len(pending),
        len(records),
        path,
    )
    scorer = ContextRecallScorer(judge_model)
    for idx, record in enumerate(pending, 1):
        log_item_progress(
            logger, idx, len(pending), record.question_id, record.question
        )
        if record.question_id in rows:
            logger.warning(
                "Context recall cache mismatch: question_id=%s — recomputing",
                record.question_id,
            )
        started = time.perf_counter()
        score = scorer.score(record.question, record.contexts, record.ground_truth)
        judge_sec = time.perf_counter() - started
        if score is None:
            logger.warning(
                "Skipping judge failure: question_id=%s — retry on resume",
                record.question_id,
            )
            continue
        rows[record.question_id] = {
            "question_id": record.question_id,
            "strategy": record.strategy,
            "doc_ids": record.doc_ids,
            "chunk_ids": record.chunk_ids,
            "context_recall": score,
            "judge_model": judge_model,
            "judge_sec": judge_sec,
        }
        _save(path, rows)
        logger.info("  context_recall=%.2f judge=%.1fs", score, judge_sec)


def _cached(row: dict | None, record: QueryRecord) -> bool:
    return (
        row is not None
        and row["doc_ids"] == record.doc_ids
        and row["chunk_ids"] == record.chunk_ids
    )


def _load(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {str(row["question_id"]): row for row in rows}


def _save(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
