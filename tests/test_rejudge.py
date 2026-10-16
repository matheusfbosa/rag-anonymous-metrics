from pathlib import Path

import pytest
from test_run_checkpoint import FakeScorer, _record

from rag_anonymous_metrics.run import _load_jsonl, _save_jsonl, rejudge_experiment


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    src = tmp_path / "source"
    scored = dict(
        judge_model="gemma",
        faithfulness=0.1,
        factual_correctness=0.1,
        judge_sec=5.0,
    )
    _save_jsonl(
        [
            _record(question_id="1", question="first", **scored),
            _record(question_id="2", question="second", **scored),
        ],
        src / "results_offline_benchmark.jsonl",
    )
    _save_jsonl(
        [_record(question_id="a1", question="attack")],
        src / "results_offline_adversarial.jsonl",
    )
    return src


@pytest.fixture
def fake_scorer(monkeypatch):
    monkeypatch.setattr(
        "rag_anonymous_metrics.metrics.utility.UtilityScorer",
        FakeScorer,
    )


def test_rejudge_rescores_into_new_dir(tmp_path: Path, source_dir: Path, fake_scorer):
    out = tmp_path / "out"

    rejudge_experiment(source_dir, out, "llama")

    records = _load_jsonl(out / "results_offline_benchmark.jsonl")
    assert [r.question_id for r in records] == ["1", "2"]
    assert all(r.judge_model == "llama" for r in records)
    assert all(r.faithfulness == 0.9 for r in records)
    assert all(r.factual_correctness == 0.6 for r in records)
    assert records[0].final_answer == "the applicant"


def test_rejudge_keeps_source_unchanged(tmp_path: Path, source_dir: Path, fake_scorer):
    rejudge_experiment(source_dir, tmp_path / "out", "llama")

    source = _load_jsonl(source_dir / "results_offline_benchmark.jsonl")
    assert all(r.judge_model == "gemma" for r in source)
    assert all(r.faithfulness == 0.1 for r in source)


def test_rejudge_copies_adversarial(tmp_path: Path, source_dir: Path, fake_scorer):
    out = tmp_path / "out"

    rejudge_experiment(source_dir, out, "llama")

    adversarial = _load_jsonl(out / "results_offline_adversarial.jsonl")
    assert [r.question_id for r in adversarial] == ["a1"]


def test_rejudge_resumes_without_rescoring(
    tmp_path: Path, source_dir: Path, monkeypatch
):
    out = tmp_path / "out"
    scorers: list[FakeScorer] = []

    class TrackingScorer(FakeScorer):
        def __init__(self, judge_model: str):
            super().__init__(judge_model)
            scorers.append(self)

    monkeypatch.setattr(
        "rag_anonymous_metrics.metrics.utility.UtilityScorer",
        TrackingScorer,
    )
    rejudge_experiment(source_dir, out, "llama")
    rejudge_experiment(source_dir, out, "llama")

    assert len(scorers) == 1
    assert scorers[0].scored_questions == ["first", "second"]


def test_rejudge_judges_refusal(tmp_path: Path, fake_scorer):
    src = tmp_path / "source"
    _save_jsonl(
        [_record(question_id="1", answer_is_refusal=True, factual_correctness=0.0)],
        src / "results_offline_benchmark.jsonl",
    )

    rejudge_experiment(src, tmp_path / "out", "llama")

    (record,) = _load_jsonl(tmp_path / "out" / "results_offline_benchmark.jsonl")
    assert record.factual_correctness == 0.6
    assert record.fc_n_ref == 5
    assert record.faithfulness is None


def test_rejudge_rejects_same_dir(source_dir: Path):
    with pytest.raises(ValueError, match="must differ"):
        rejudge_experiment(source_dir, source_dir, "llama")


def test_rejudge_requires_benchmark_files(tmp_path: Path):
    with pytest.raises(ValueError, match="No results_"):
        rejudge_experiment(tmp_path, tmp_path / "out", "llama")


def test_rejudge_caches_context_recall_per_judge(
    tmp_path: Path, source_dir: Path, fake_scorer
):
    rejudge_experiment(source_dir, tmp_path / "out", "gemma4:26b")

    cache = (
        tmp_path
        / "output"
        / "context_recall"
        / "dev"
        / "gemma4_26b"
        / "results_offline_benchmark.jsonl"
    )
    assert [line for line in cache.read_text().splitlines() if line]
