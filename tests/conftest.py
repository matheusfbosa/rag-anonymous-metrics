import pytest


class FakeRecallScorer:
    def __init__(self, judge_model: str):
        self.scored: list[str] = []

    def score(self, question, contexts, reference):
        self.scored.append(question)
        return 0.5


@pytest.fixture(autouse=True)
def context_recall_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_METRICS_RESULTS_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(
        "rag_anonymous_metrics.metrics.utility.ContextRecallScorer",
        FakeRecallScorer,
    )
