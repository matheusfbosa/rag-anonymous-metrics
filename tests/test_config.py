import pytest

from rag_anonymous_metrics.config import Settings


class TestJudgeLimitsSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.setenv("RAG_METRICS_STRATEGY", "offline")
        monkeypatch.setenv("RAG_METRICS_CORPUS_DATASET", "dev")
        monkeypatch.setenv("RAG_METRICS_QUESTIONS_CATEGORY", "benchmark")
        monkeypatch.setenv("RAG_METRICS_QUESTIONS_RANGE", "all")
        monkeypatch.delenv("RAG_METRICS_JUDGE_NUM_PREDICT", raising=False)
        monkeypatch.delenv("RAG_METRICS_JUDGE_TIMEOUT_SEC", raising=False)
        monkeypatch.delenv("RAG_METRICS_OVERWRITE", raising=False)

        s = Settings.load()

        assert s.judge_num_predict == 1536
        assert s.judge_timeout_sec == 1800.0
        assert s.overwrite is False

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("RAG_METRICS_STRATEGY", "offline")
        monkeypatch.setenv("RAG_METRICS_CORPUS_DATASET", "dev")
        monkeypatch.setenv("RAG_METRICS_QUESTIONS_CATEGORY", "benchmark")
        monkeypatch.setenv("RAG_METRICS_QUESTIONS_RANGE", "all")
        monkeypatch.setenv("RAG_METRICS_JUDGE_NUM_PREDICT", "2048")
        monkeypatch.setenv("RAG_METRICS_JUDGE_TIMEOUT_SEC", "900")

        s = Settings.load()

        assert s.judge_num_predict == 2048
        assert s.judge_timeout_sec == 900.0
        assert s.overwrite is False

    def test_overwrite_flag(self, monkeypatch):
        monkeypatch.setenv("RAG_METRICS_STRATEGY", "offline")
        monkeypatch.setenv("RAG_METRICS_CORPUS_DATASET", "dev")
        monkeypatch.setenv("RAG_METRICS_QUESTIONS_CATEGORY", "benchmark")
        monkeypatch.setenv("RAG_METRICS_QUESTIONS_RANGE", "all")
        monkeypatch.setenv("RAG_METRICS_OVERWRITE", "1")

        s = Settings.load()

        assert s.overwrite is True

    def test_strategy_all_rejected(self, monkeypatch):
        monkeypatch.setenv("RAG_METRICS_STRATEGY", "all")
        monkeypatch.setenv("RAG_METRICS_CORPUS_DATASET", "dev")
        monkeypatch.setenv("RAG_METRICS_QUESTIONS_CATEGORY", "benchmark")
        monkeypatch.setenv("RAG_METRICS_QUESTIONS_RANGE", "all")

        with pytest.raises(ValueError, match="offline"):
            Settings.load()
