import asyncio

import pytest
from test_run_checkpoint import metrics_env  # noqa: F401

from rag_anonymous_metrics.metrics.utility import UtilityScorer

pytestmark = pytest.mark.usefixtures("metrics_env")

CLAIMS = {
    "answer": ["a1", "a2", "a3"],
    "reference": ["r1", "r2", "r3", "r4"],
}
VERDICTS = {
    "reference": [True, False, True],
    "answer": [True, True, False, False],
}


class FakeFactualCorrectness:
    async def decompose_claims(self, text, callbacks):
        return CLAIMS.get(text, [])

    async def verify_claims(self, premise, hypothesis_list, callbacks):
        return VERDICTS.get(premise, [False] * len(hypothesis_list))


class FakeMetric:
    def __init__(self, score):
        self.score = score
        self.calls = 0
        self.loops: list[asyncio.AbstractEventLoop] = []

    async def single_turn_ascore(self, sample):
        self.calls += 1
        self.loops.append(asyncio.get_running_loop())
        return self.score


@pytest.fixture
def scorer():
    scorer = UtilityScorer("judge")
    scorer._factual_correctness = FakeFactualCorrectness()
    scorer._answer_accuracy = FakeMetric(0.75)
    scorer._semantic_similarity = FakeMetric(0.8)
    scorer._faithfulness = FakeMetric(0.9)
    return scorer


def test_score_reports_coverage_counts(scorer):
    scores = scorer.score("q", "answer", ["ctx"], "reference")

    assert scores["fc_n_resp"] == 3
    assert scores["fc_n_resp_ok"] == 2
    assert scores["fc_n_ref"] == 4
    assert scores["fc_n_ref_ok"] == 2
    assert scores["factual_correctness"] == pytest.approx(0.5)
    assert scores["answer_accuracy"] == 0.75
    assert scores["semantic_similarity"] == 0.8
    assert scores["faithfulness"] == 0.9
    assert set(scores["judge_metric_sec"]) == {
        "factual_correctness",
        "answer_accuracy",
        "semantic_similarity",
        "faithfulness",
    }


def test_score_refusal_skips_faithfulness(scorer):
    scores = scorer.score("q", "answer", ["ctx"], "reference", refusal=True)

    assert scores["faithfulness"] is None
    assert scorer._faithfulness.calls == 0
    assert "faithfulness" not in scores["judge_metric_sec"]
    assert scores["factual_correctness"] == pytest.approx(0.5)


def test_score_coverage_none_without_reference_claims(scorer):
    scores = scorer.score("q", "answer", ["ctx"], "empty")

    assert scores["fc_n_ref"] == 0
    assert scores["factual_correctness"] is None


def test_score_nan_metric_becomes_none(scorer):
    scorer._answer_accuracy = FakeMetric(float("nan"))

    scores = scorer.score("q", "answer", ["ctx"], "reference")

    assert scores["answer_accuracy"] is None


def test_score_reuses_one_event_loop(scorer):
    scorer.score("q", "answer", ["ctx"], "reference")
    scorer.score("q", "answer", ["ctx"], "reference")

    first, second = scorer._answer_accuracy.loops
    assert first is second
    assert not first.is_closed()
