import pytest

from rag_anonymous_metrics.run import Pipeline, evaluate_question


def _rag_result(**overrides) -> dict:
    data = {
        "response": "the applicant",
        "retrieved_chunks": [
            {"content": "ctx", "doc_id": "d1", "chunk_id": "c1"},
        ],
        "retrieval_sec": 0.11,
        "generation_sec": 1.25,
    }
    data.update(overrides)
    return data


def test_evaluate_question_records_offline_timings(monkeypatch):
    monkeypatch.setattr(
        "rag_anonymous_metrics.run.query_rag",
        lambda *a, **k: _rag_result(),
    )
    pipeline = Pipeline(
        strategy="offline",
        gen_model="qwen",
        chain=object(),
        retrieval_store=object(),
        anonymizer=None,
    )

    record = evaluate_question(
        pipeline,
        {"id": "1", "question": "Who applied?", "ground_truth": "gt"},
        pii_index={"d1": set()},
        person_pii_index={},
        judge_model="gemma",
        run_id="abc",
        dataset="dev",
        k_docs=3,
    )

    assert record.retrieval_sec == pytest.approx(0.11)
    assert record.generation_sec == pytest.approx(1.25)
    assert record.anonymize_sec == 0.0
    assert record.query_sec == pytest.approx(1.36)
    assert record.final_answer == "the applicant"


def test_evaluate_question_times_ondemand_anonymize(monkeypatch):
    class FakeAnonymizer:
        def anonymize(self, text: str) -> str:
            return f"anon:{text}"

    ticks = iter([50.0, 50.04])
    monkeypatch.setattr("rag_anonymous_metrics.run.time.perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        "rag_anonymous_metrics.run.query_rag",
        lambda *a, **k: _rag_result(),
    )
    pipeline = Pipeline(
        strategy="ondemand",
        gen_model="qwen",
        chain=object(),
        retrieval_store=object(),
        anonymizer=FakeAnonymizer(),
    )

    record = evaluate_question(
        pipeline,
        {"id": "1", "question": "Who applied?", "ground_truth": "gt"},
        pii_index={"d1": set()},
        person_pii_index={},
        judge_model="gemma",
        run_id="abc",
        dataset="dev",
        k_docs=3,
    )

    assert record.final_answer == "anon:the applicant"
    assert record.anonymize_sec == pytest.approx(0.04)
    assert record.query_sec == pytest.approx(1.4)
