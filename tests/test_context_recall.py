import json
from pathlib import Path

import pytest
from conftest import FakeRecallScorer
from test_run_checkpoint import _record

from rag_anonymous_metrics.context_recall import (
    attach_context_recall,
    context_recall_path,
)

RECORDS = [
    _record(question_id="1", question="first", doc_ids=["d1"], chunk_ids=["c1"]),
    _record(question_id="2", question="fails", doc_ids=["d2"], chunk_ids=["c2"]),
]


@pytest.fixture
def scorers(monkeypatch) -> list[FakeRecallScorer]:
    created: list[FakeRecallScorer] = []

    class TrackingScorer(FakeRecallScorer):
        def __init__(self, judge_model: str):
            super().__init__(judge_model)
            created.append(self)

        def score(self, question, contexts, reference):
            super().score(question, contexts, reference)
            return None if question == "fails" else 0.5

    monkeypatch.setattr(
        "rag_anonymous_metrics.metrics.utility.ContextRecallScorer",
        TrackingScorer,
    )
    return created


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_attach_writes_scored_rows(tmp_path: Path, scorers):
    path = tmp_path / "recall.jsonl"

    attach_context_recall(RECORDS, "gemma4:26b", path)

    rows = _rows(path)
    assert [r["question_id"] for r in rows] == ["1"]
    assert rows[0]["context_recall"] == 0.5
    assert rows[0]["doc_ids"] == ["d1"]
    assert rows[0]["judge_model"] == "gemma4:26b"


def test_attach_reuses_cache_and_retries_failures(tmp_path: Path, scorers):
    path = tmp_path / "recall.jsonl"
    attach_context_recall(RECORDS, "judge", path)

    attach_context_recall(RECORDS, "judge", path)

    assert scorers[1].scored == ["fails"]


def test_attach_skips_scorer_when_cached(tmp_path: Path, scorers):
    path = tmp_path / "recall.jsonl"
    attach_context_recall(RECORDS[:1], "judge", path)

    attach_context_recall(RECORDS[:1], "judge", path)

    assert len(scorers) == 1


def test_attach_recomputes_on_retrieval_mismatch(tmp_path: Path, scorers, caplog):
    path = tmp_path / "recall.jsonl"
    attach_context_recall(RECORDS[:1], "judge", path)
    moved = _record(question_id="1", question="first", doc_ids=["d9"], chunk_ids=["c9"])

    with caplog.at_level("WARNING"):
        attach_context_recall([moved], "judge", path)

    assert "cache mismatch" in caplog.text
    assert _rows(path)[0]["doc_ids"] == ["d9"]


def test_context_recall_path_sanitizes_judge(tmp_path: Path):
    assert context_recall_path(tmp_path, "gemma4:26b", "r.jsonl") == (
        tmp_path / "context_recall" / "gemma4_26b" / "r.jsonl"
    )
