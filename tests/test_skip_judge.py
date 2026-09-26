from pathlib import Path

import pytest
from test_run_checkpoint import FakeScorer, _record, metrics_env  # noqa: F401

from rag_anonymous_metrics.run import _load_jsonl, evaluate_strategy

pytestmark = pytest.mark.usefixtures("metrics_env")

QUESTIONS = [
    {"id": "1", "question": "first", "ground_truth": "gt"},
    {"id": "2", "question": "second", "ground_truth": "gt"},
]


def _patch(monkeypatch, rag_ids: list[str], scorers: list[FakeScorer]) -> None:
    def fake_evaluate(pipeline, question_obj, *args, **kwargs):
        qid = str(question_obj["id"])
        rag_ids.append(qid)
        return _record(question_id=qid, question=question_obj["question"])

    class TrackingScorer(FakeScorer):
        def __init__(self, judge_model: str):
            super().__init__(judge_model)
            scorers.append(self)

    monkeypatch.setattr(
        "rag_anonymous_metrics.run.build_pipeline",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr("rag_anonymous_metrics.run.evaluate_question", fake_evaluate)
    monkeypatch.setattr(
        "rag_anonymous_metrics.metrics.utility.UtilityScorer",
        TrackingScorer,
    )


def _evaluate(tmp_path: Path):
    return evaluate_strategy(
        "offline",
        QUESTIONS,
        pii_index={"d1": set()},
        person_pii_index={},
        experiment_dir=tmp_path,
        dataset="dev",
        k_docs=3,
        compute_utility=True,
        output_basename="results_offline_benchmark.jsonl",
    )


def test_skip_judge_saves_answers_without_utility(tmp_path: Path, monkeypatch):
    rag_ids: list[str] = []
    scorers: list[FakeScorer] = []
    _patch(monkeypatch, rag_ids, scorers)
    monkeypatch.setenv("RAG_METRICS_SKIP_JUDGE", "1")

    _evaluate(tmp_path)

    records = _load_jsonl(tmp_path / "results_offline_benchmark.jsonl")
    assert [r.question_id for r in records] == ["1", "2"]
    assert all(r.faithfulness is None for r in records)
    assert all(r.factual_correctness is None for r in records)
    assert scorers == []


def test_judge_after_skip_reuses_answers(tmp_path: Path, monkeypatch):
    rag_ids: list[str] = []
    scorers: list[FakeScorer] = []
    _patch(monkeypatch, rag_ids, scorers)
    monkeypatch.setenv("RAG_METRICS_SKIP_JUDGE", "1")
    _evaluate(tmp_path)

    monkeypatch.delenv("RAG_METRICS_SKIP_JUDGE")
    monkeypatch.setenv("RAG_METRICS_JUDGE_MODEL", "llama")
    _evaluate(tmp_path)

    records = _load_jsonl(tmp_path / "results_offline_benchmark.jsonl")
    assert rag_ids == ["1", "2"]
    assert len(scorers) == 1
    assert scorers[0].scored_questions == ["first", "second"]
    assert all(r.faithfulness == 0.9 for r in records)
    assert all(r.judge_model == "llama" for r in records)
