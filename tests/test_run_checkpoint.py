import json
from pathlib import Path

import pytest

from rag_anonymous_metrics.records import QueryRecord
from rag_anonymous_metrics.run import (
    _append_jsonl,
    _is_refusal,
    _load_jsonl,
    _needs_utility,
    _save_jsonl,
    evaluate_strategy,
)

_REQUIRED_ENV = {
    "RAG_METRICS_STRATEGY": "offline",
    "RAG_METRICS_CORPUS_DATASET": "dev",
    "RAG_METRICS_QUESTIONS_CATEGORY": "benchmark",
    "RAG_METRICS_QUESTIONS_RANGE": "all",
}


def _record(**overrides) -> QueryRecord:
    data = dict(
        run_id="abc12345",
        strategy="offline",
        question_id="1",
        question="Who applied?",
        ground_truth="the applicant",
        llm_model="qwen",
        judge_model="gemma",
        generated_answer="the applicant",
        final_answer="the applicant",
        answer_is_refusal=False,
        contexts=["ctx"],
        doc_ids=["d1"],
        dataset="dev",
        pii_total_generated=1,
        pii_leaked_generated=0,
        pii_leakage_rate_generated=0.0,
        pii_any_leak_generated=0,
        pii_total_final=1,
        pii_leaked_final=0,
        pii_leakage_rate_final=0.0,
        pii_any_leak_final=0,
        pii_person_total_generated=0,
        pii_person_leaked_generated=0,
        pii_person_leakage_rate_generated=0.0,
        pii_person_any_leak_generated=0,
        pii_person_total_final=0,
        pii_person_leaked_final=0,
        pii_person_leakage_rate_final=0.0,
        pii_person_any_leak_final=0,
    )
    data.update(overrides)
    return QueryRecord(**data)


@pytest.fixture
def metrics_env(monkeypatch):
    for key, value in _REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("RAG_METRICS_OVERWRITE", raising=False)


class FakeScorer:
    def __init__(self, judge_model: str):
        self.judge_model = judge_model
        self.scored_questions: list[str] = []

    def score(self, question, answer, contexts, reference):
        self.scored_questions.append(question)
        return {
            "faithfulness": 0.9,
            "context_precision": 0.8,
            "context_recall": 0.7,
            "factual_correctness": 0.6,
        }


def test_load_jsonl_skips_corrupt_line(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    rec = _record(question_id="1")
    path.write_text(
        '{"not": "json"\n' + json.dumps(asdict_safe(rec), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    loaded = _load_jsonl(path)
    assert [r.question_id for r in loaded] == ["1"]


def test_load_jsonl_missing_timing_fields_default_to_none(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    payload = asdict_safe(_record(question_id="1"))
    for key in (
        "retrieval_sec",
        "generation_sec",
        "anonymize_sec",
        "query_sec",
        "judge_sec",
        "chunk_ids",
    ):
        payload.pop(key, None)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    loaded = _load_jsonl(path)[0]

    assert loaded.question_id == "1"
    assert loaded.retrieval_sec is None
    assert loaded.generation_sec is None
    assert loaded.anonymize_sec is None
    assert loaded.query_sec is None
    assert loaded.judge_sec is None
    assert loaded.chunk_ids is None


def asdict_safe(record: QueryRecord) -> dict:
    from dataclasses import asdict

    return asdict(record)


def test_append_jsonl_keeps_distinct_ids(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    _append_jsonl(_record(question_id="1"), path)
    _append_jsonl(_record(question_id="2"), path)
    loaded = _load_jsonl(path)
    assert [r.question_id for r in loaded] == ["1", "2"]


def test_needs_utility_false_when_scored():
    rec = _record(
        faithfulness=0.5,
        context_precision=0.4,
        context_recall=0.3,
        factual_correctness=0.2,
    )
    assert _needs_utility(rec) is False


def test_needs_utility_true_when_unscored():
    rec = _record()
    assert _needs_utility(rec) is True


def test_needs_utility_true_when_partially_scored():
    rec = _record(faithfulness=0.5, context_precision=0.4)
    assert _needs_utility(rec) is True


def test_refusal_does_not_need_judge_once_factual_is_set():
    rec = _record(answer_is_refusal=True, factual_correctness=0.0)
    assert _is_refusal(rec) is True
    assert _needs_utility(rec) is False


def test_evaluate_strategy_skips_saved_question_ids(
    tmp_path: Path, monkeypatch, metrics_env
):
    path = tmp_path / "results_offline_benchmark.jsonl"
    _save_jsonl([_record(question_id="1", question="first")], path)

    rag_ids: list[str] = []

    def fake_evaluate(pipeline, question_obj, *args, **kwargs):
        qid = str(question_obj["id"])
        rag_ids.append(qid)
        return _record(question_id=qid, question=question_obj["question"])

    monkeypatch.setattr(
        "rag_anonymous_metrics.run.build_pipeline",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr("rag_anonymous_metrics.run.evaluate_question", fake_evaluate)
    monkeypatch.setattr(
        "rag_anonymous_metrics.metrics.utility.UtilityScorer",
        FakeScorer,
    )

    questions = [
        {"id": "1", "question": "first", "ground_truth": "gt"},
        {"id": "2", "question": "second", "ground_truth": "gt"},
    ]
    result = evaluate_strategy(
        "offline",
        questions,
        pii_index={"d1": set()},
        person_pii_index={},
        experiment_dir=tmp_path,
        dataset="dev",
        k_docs=3,
        compute_utility=True,
        output_basename="results_offline_benchmark.jsonl",
    )

    assert rag_ids == ["2"]
    assert [r.question_id for r in result.records] == ["1", "2"]
    reloaded = _load_jsonl(path)
    assert [r.question_id for r in reloaded] == ["1", "2"]
    assert len(reloaded) == 2


def test_evaluate_strategy_scores_only_unscored_records(
    tmp_path: Path, monkeypatch, metrics_env
):
    scored = _record(
        question_id="1",
        question="first",
        faithfulness=0.1,
        context_precision=0.2,
        context_recall=0.3,
        factual_correctness=0.4,
    )
    pending = _record(question_id="2", question="second")
    _save_jsonl([scored, pending], tmp_path / "results_offline_benchmark.jsonl")

    rag_ids: list[str] = []
    scorer_holder: dict[str, FakeScorer] = {}

    def fake_evaluate(pipeline, question_obj, *args, **kwargs):
        rag_ids.append(str(question_obj["id"]))
        return _record(question_id=str(question_obj["id"]))

    class TrackingScorer(FakeScorer):
        def __init__(self, judge_model: str):
            super().__init__(judge_model)
            scorer_holder["scorer"] = self

    monkeypatch.setattr(
        "rag_anonymous_metrics.run.build_pipeline",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("RAG should be skipped")),
    )
    monkeypatch.setattr("rag_anonymous_metrics.run.evaluate_question", fake_evaluate)
    monkeypatch.setattr(
        "rag_anonymous_metrics.metrics.utility.UtilityScorer",
        TrackingScorer,
    )

    questions = [
        {"id": "1", "question": "first", "ground_truth": "gt"},
        {"id": "2", "question": "second", "ground_truth": "gt"},
    ]
    result = evaluate_strategy(
        "offline",
        questions,
        pii_index={"d1": set()},
        person_pii_index={},
        experiment_dir=tmp_path,
        dataset="dev",
        k_docs=3,
        compute_utility=True,
        output_basename="results_offline_benchmark.jsonl",
    )

    assert rag_ids == []
    assert scorer_holder["scorer"].scored_questions == ["second"]
    by_id = {r.question_id: r for r in result.records}
    assert by_id["1"].faithfulness == 0.1
    assert by_id["2"].faithfulness == 0.9
    assert by_id["1"].judge_sec is None
    assert by_id["2"].judge_sec is not None
    assert by_id["2"].judge_sec >= 0.0


def test_evaluate_strategy_skips_judge_for_refusal(
    tmp_path: Path, monkeypatch, metrics_env
):
    refusal = _record(
        question_id="1",
        question="secret?",
        answer_is_refusal=True,
        final_answer="I cannot help with that",
    )
    _save_jsonl([refusal], tmp_path / "results_offline_benchmark.jsonl")

    class BoomScorer:
        def __init__(self, judge_model: str):
            raise AssertionError("judge should not be constructed")

    monkeypatch.setattr(
        "rag_anonymous_metrics.run.build_pipeline",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr(
        "rag_anonymous_metrics.metrics.utility.UtilityScorer",
        BoomScorer,
    )

    result = evaluate_strategy(
        "offline",
        [{"id": "1", "question": "secret?", "ground_truth": "gt"}],
        pii_index={"d1": set()},
        person_pii_index={},
        experiment_dir=tmp_path,
        dataset="dev",
        k_docs=3,
        compute_utility=True,
        output_basename="results_offline_benchmark.jsonl",
    )

    assert result.records[0].factual_correctness == 0.0
    assert result.records[0].judge_sec == 0.0
    loaded = _load_jsonl(tmp_path / "results_offline_benchmark.jsonl")
    assert loaded[0].factual_correctness == 0.0
    assert loaded[0].judge_sec == 0.0


def test_evaluate_strategy_overwrite_reruns_everything(
    tmp_path: Path, monkeypatch, metrics_env
):
    monkeypatch.setenv("RAG_METRICS_OVERWRITE", "1")
    saved = _record(question_id="1", question="old", generated_answer="old-answer")
    _save_jsonl([saved], tmp_path / "results_offline_benchmark.jsonl")

    rag_ids: list[str] = []

    def fake_evaluate(pipeline, question_obj, *args, **kwargs):
        qid = str(question_obj["id"])
        rag_ids.append(qid)
        return _record(
            question_id=qid,
            question=question_obj["question"],
            generated_answer="new",
        )

    monkeypatch.setattr(
        "rag_anonymous_metrics.run.build_pipeline",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr("rag_anonymous_metrics.run.evaluate_question", fake_evaluate)
    monkeypatch.setattr(
        "rag_anonymous_metrics.metrics.utility.UtilityScorer",
        FakeScorer,
    )

    result = evaluate_strategy(
        "offline",
        [{"id": "1", "question": "first", "ground_truth": "gt"}],
        pii_index={"d1": set()},
        person_pii_index={},
        experiment_dir=tmp_path,
        dataset="dev",
        k_docs=3,
        compute_utility=True,
        output_basename="results_offline_benchmark.jsonl",
    )

    assert rag_ids == ["1"]
    assert result.records[0].generated_answer == "new"
    assert result.records[0].judge_sec is not None


def test_evaluate_strategy_records_judge_wall_clock(
    tmp_path: Path, monkeypatch, metrics_env
):
    pending = _record(question_id="1", question="first")
    _save_jsonl([pending], tmp_path / "results_offline_benchmark.jsonl")

    ticks = iter([10.0, 22.5])
    monkeypatch.setattr(
        "rag_anonymous_metrics.run.time.perf_counter", lambda: next(ticks)
    )
    monkeypatch.setattr(
        "rag_anonymous_metrics.run.build_pipeline",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("RAG should be skipped")),
    )
    monkeypatch.setattr(
        "rag_anonymous_metrics.metrics.utility.UtilityScorer",
        FakeScorer,
    )

    result = evaluate_strategy(
        "offline",
        [{"id": "1", "question": "first", "ground_truth": "gt"}],
        pii_index={"d1": set()},
        person_pii_index={},
        experiment_dir=tmp_path,
        dataset="dev",
        k_docs=3,
        compute_utility=True,
        output_basename="results_offline_benchmark.jsonl",
    )

    assert result.records[0].judge_sec == pytest.approx(12.5)
    loaded = _load_jsonl(tmp_path / "results_offline_benchmark.jsonl")
    assert loaded[0].judge_sec == pytest.approx(12.5)


def test_evaluate_strategy_rescores_partial_utility(
    tmp_path: Path, monkeypatch, metrics_env
):
    partial = _record(
        question_id="1",
        question="first",
        faithfulness=0.1,
        context_precision=0.2,
    )
    _save_jsonl([partial], tmp_path / "results_offline_benchmark.jsonl")

    monkeypatch.setattr(
        "rag_anonymous_metrics.run.build_pipeline",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("RAG should be skipped")),
    )
    monkeypatch.setattr(
        "rag_anonymous_metrics.metrics.utility.UtilityScorer",
        FakeScorer,
    )

    result = evaluate_strategy(
        "offline",
        [{"id": "1", "question": "first", "ground_truth": "gt"}],
        pii_index={"d1": set()},
        person_pii_index={},
        experiment_dir=tmp_path,
        dataset="dev",
        k_docs=3,
        compute_utility=True,
        output_basename="results_offline_benchmark.jsonl",
    )

    rec = result.records[0]
    assert rec.faithfulness == 0.1
    assert rec.context_precision == 0.2
    assert rec.context_recall == 0.7
    assert rec.factual_correctness == 0.6


def test_evaluate_strategy_skips_empty_generation(
    tmp_path: Path, monkeypatch, metrics_env
):
    def fake_evaluate(pipeline, question_obj, *args, **kwargs):
        return _record(
            question_id=str(question_obj["id"]),
            generated_answer="   ",
            final_answer="",
        )

    monkeypatch.setattr(
        "rag_anonymous_metrics.run.build_pipeline",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr("rag_anonymous_metrics.run.evaluate_question", fake_evaluate)
    monkeypatch.setattr(
        "rag_anonymous_metrics.metrics.utility.UtilityScorer",
        FakeScorer,
    )

    result = evaluate_strategy(
        "offline",
        [{"id": "1", "question": "first", "ground_truth": "gt"}],
        pii_index={"d1": set()},
        person_pii_index={},
        experiment_dir=tmp_path,
        dataset="dev",
        k_docs=3,
        compute_utility=True,
        output_basename="results_offline_benchmark.jsonl",
    )

    assert result.records == []
    path = tmp_path / "results_offline_benchmark.jsonl"
    assert _load_jsonl(path) == []


def test_evaluate_strategy_skips_rag_exception(
    tmp_path: Path, monkeypatch, metrics_env
):
    def fake_evaluate(pipeline, question_obj, *args, **kwargs):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(
        "rag_anonymous_metrics.run.build_pipeline",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr("rag_anonymous_metrics.run.evaluate_question", fake_evaluate)
    monkeypatch.setattr(
        "rag_anonymous_metrics.metrics.utility.UtilityScorer",
        FakeScorer,
    )

    result = evaluate_strategy(
        "offline",
        [{"id": "1", "question": "first", "ground_truth": "gt"}],
        pii_index={"d1": set()},
        person_pii_index={},
        experiment_dir=tmp_path,
        dataset="dev",
        k_docs=3,
        compute_utility=True,
        output_basename="results_offline_benchmark.jsonl",
    )

    assert result.records == []
    path = tmp_path / "results_offline_benchmark.jsonl"
    assert _load_jsonl(path) == []
