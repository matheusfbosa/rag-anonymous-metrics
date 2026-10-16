import pytest


class FakeRecallScorer:
    def __init__(self, judge_model: str):
        self.scored: list[str] = []

    def score(self, question, contexts, reference):
        self.scored.append(question)
        return 0.5


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_METRICS_STRATEGY", "offline")
    monkeypatch.setenv("RAG_METRICS_CORPUS_DATASET", "dev")
    monkeypatch.setenv("RAG_METRICS_QUESTIONS_CATEGORY", "benchmark")
    monkeypatch.setenv("RAG_METRICS_QUESTIONS_RANGE", "all")
    monkeypatch.setenv("RAG_METRICS_RESULTS_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(
        "rag_anonymous_metrics.metrics.utility.ContextRecallScorer",
        FakeRecallScorer,
    )
