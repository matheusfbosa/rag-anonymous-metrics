from rag_anonymous_metrics.config import Settings


class TestJudgeLimitsSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.setenv("RAG_METRICS_STRATEGY", "offline")
        monkeypatch.setenv("RAG_METRICS_CORPUS_DATASET", "dev")
        monkeypatch.setenv("RAG_METRICS_QUESTIONS_CATEGORY", "all")
        monkeypatch.setenv("RAG_METRICS_QUESTIONS_RANGE", "all")
        monkeypatch.delenv("RAG_METRICS_JUDGE_NUM_PREDICT", raising=False)
        monkeypatch.delenv("RAG_METRICS_JUDGE_TIMEOUT_SEC", raising=False)

        s = Settings.load()

        assert s.judge_num_predict == 1536
        assert s.judge_timeout_sec == 1800.0

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("RAG_METRICS_STRATEGY", "offline")
        monkeypatch.setenv("RAG_METRICS_CORPUS_DATASET", "dev")
        monkeypatch.setenv("RAG_METRICS_QUESTIONS_CATEGORY", "all")
        monkeypatch.setenv("RAG_METRICS_QUESTIONS_RANGE", "all")
        monkeypatch.setenv("RAG_METRICS_JUDGE_NUM_PREDICT", "2048")
        monkeypatch.setenv("RAG_METRICS_JUDGE_TIMEOUT_SEC", "900")

        s = Settings.load()

        assert s.judge_num_predict == 2048
        assert s.judge_timeout_sec == 900.0
