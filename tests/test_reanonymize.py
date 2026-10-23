from pathlib import Path

import pytest
from test_run_checkpoint import SCORED, FakeScorer, _record

from rag_anonymous import strip_markdown_emphasis
from rag_anonymous_metrics.run import (
    _load_jsonl,
    _save_jsonl,
    reanonymize_experiment,
    refresh_refusals,
)

PERSON_INDEX = {"d1": {"mr x"}}


class FakeAnonymizer:
    def anonymize_answer(self, text: str) -> str:
        return strip_markdown_emphasis(text).replace("Mr X", "<PERSON>")


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    src = tmp_path / "source"
    leaked = dict(
        strategy="ondemand",
        generated_answer="It was **Mr X**.",
        final_answer="It was **Mr X**.",
        pii_person_total_final=1,
        pii_person_leaked_final=1,
        pii_person_leakage_rate_final=1.0,
        pii_person_any_leak_final=1,
    )
    formatted = dict(
        strategy="ondemand",
        generated_answer="It was **Mr X** in **Rome**.",
        final_answer="It was **<PERSON>** in **Rome**.",
    )
    _save_jsonl(
        [
            _record(question_id="1", judge_model="gemma", **leaked, **SCORED),
            _record(question_id="2", judge_model="gemma", **formatted, **SCORED),
        ],
        src / "results_ondemand_benchmark.jsonl",
    )
    _save_jsonl(
        [_record(question_id="1", final_answer="**bold**", **SCORED)],
        src / "results_offline_benchmark.jsonl",
    )
    _save_jsonl(
        [_record(question_id="a1", **leaked)],
        src / "results_ondemand_adversarial.jsonl",
    )
    return src


@pytest.fixture
def scorers(monkeypatch) -> list[FakeScorer]:
    created: list[FakeScorer] = []

    class TrackingScorer(FakeScorer):
        def __init__(self, judge_model: str):
            super().__init__(judge_model)
            created.append(self)

    monkeypatch.setattr(
        "rag_anonymous_metrics.metrics.utility.UtilityScorer",
        TrackingScorer,
    )
    return created


def _run(source_dir: Path, out: Path) -> None:
    reanonymize_experiment(
        source_dir, out, "llama", FakeAnonymizer(), {"d1": set()}, PERSON_INDEX
    )


def test_changed_answer_is_reanonymized_and_rejudged(
    tmp_path: Path, source_dir: Path, scorers
):
    out = tmp_path / "out"

    _run(source_dir, out)

    first, _ = _load_jsonl(out / "results_ondemand_benchmark.jsonl")
    assert first.final_answer == "It was <PERSON>."
    assert first.pii_person_any_leak_generated == 1
    assert first.pii_person_any_leak_final == 0
    assert first.judge_model == "llama"
    assert first.factual_correctness == 0.6
    assert scorers[0].scored_questions == ["Who applied?"]


def test_markdown_only_change_keeps_scores(tmp_path: Path, source_dir: Path, scorers):
    out = tmp_path / "out"

    _run(source_dir, out)

    _, second = _load_jsonl(out / "results_ondemand_benchmark.jsonl")
    assert second.final_answer == "It was <PERSON> in Rome."
    assert second.judge_model == "gemma"
    assert second.factual_correctness == SCORED["factual_correctness"]


def test_offline_and_source_unchanged(tmp_path: Path, source_dir: Path, scorers):
    out = tmp_path / "out"

    _run(source_dir, out)

    (offline,) = _load_jsonl(out / "results_offline_benchmark.jsonl")
    assert offline.final_answer == "**bold**"
    assert offline.factual_correctness == SCORED["factual_correctness"]
    first, _ = _load_jsonl(source_dir / "results_ondemand_benchmark.jsonl")
    assert first.final_answer == "It was **Mr X**."


def test_adversarial_reanonymized_without_judge(
    tmp_path: Path, source_dir: Path, scorers
):
    out = tmp_path / "out"

    _run(source_dir, out)

    (attack,) = _load_jsonl(out / "results_ondemand_adversarial.jsonl")
    assert attack.final_answer == "It was <PERSON>."
    assert attack.pii_person_any_leak_final == 0
    assert attack.factual_correctness is None


def test_resume_does_not_rescore(tmp_path: Path, source_dir: Path, scorers):
    out = tmp_path / "out"

    _run(source_dir, out)
    _run(source_dir, out)

    assert len(scorers) == 1


def test_rejects_same_dir(source_dir: Path):
    with pytest.raises(ValueError, match="must differ"):
        _run(source_dir, source_dir)


def test_refresh_refusals_flags_missed_refusal(tmp_path: Path):
    path = tmp_path / "results_offline_benchmark.jsonl"
    _save_jsonl(
        [
            _record(question_id="1", final_answer="It cannot be answered.", **SCORED),
            _record(question_id="2", **SCORED),
        ],
        path,
    )

    assert refresh_refusals(tmp_path) == {"results_offline_benchmark.jsonl": 1}

    first, second = _load_jsonl(path)
    assert first.answer_is_refusal is True
    assert first.faithfulness is None
    assert first.factual_correctness == SCORED["factual_correctness"]
    assert second.answer_is_refusal is False
