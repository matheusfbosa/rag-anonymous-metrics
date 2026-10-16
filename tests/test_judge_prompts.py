from pathlib import Path

import pytest
from ragas.prompt.pydantic_prompt import PydanticPrompt
from test_run_checkpoint import _record, metrics_env  # noqa: F401

from rag_anonymous_metrics.judge_prompts import _capture_prompts, dump_judge_prompts
from rag_anonymous_metrics.metrics.utility import ContextRecallScorer
from rag_anonymous_metrics.run import _save_jsonl

pytestmark = pytest.mark.usefixtures("metrics_env")


@pytest.fixture(autouse=True)
def real_context_recall(monkeypatch):
    monkeypatch.setattr(
        "rag_anonymous_metrics.metrics.utility.ContextRecallScorer",
        ContextRecallScorer,
    )


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    src = tmp_path / "source"
    _save_jsonl(
        [
            _record(question_id="1", question="first", contexts=["c1", "c2", "c3"]),
            _record(question_id="2", question="second"),
            _record(question_id="3", answer_is_refusal=True),
        ],
        src / "results_offline_benchmark.jsonl",
    )
    return src


def test_dump_writes_prompts_for_selected_question(tmp_path: Path, source_dir: Path):
    written = dump_judge_prompts(source_dir, tmp_path / "out", "judge", (1, 1))

    folder = tmp_path / "out" / "judge_prompts" / "offline_q1"
    assert written and all(p.parent == folder for p in written)
    names = [p.name for p in written]
    assert sum("ClaimDecomposition" in n for n in names) == 2
    assert any("StatementGenerator" in n for n in names)
    assert any("ContextRecall" in n for n in names)
    assert names[-2:] == ["08_AnswerAccuracy1.txt", "09_AnswerAccuracy2.txt"]
    assert not any("ContextPrecision" in n for n in names)
    assert "first" in written[-2].read_text()


def test_dump_refusal_skips_faithfulness(tmp_path: Path, source_dir: Path):
    written = dump_judge_prompts(source_dir, tmp_path / "out", "judge", (3, 3))

    names = [p.name for p in written]
    assert any("ClaimDecomposition" in n for n in names)
    assert not any("StatementGenerator" in n for n in names)


def test_capture_restores_ragas(tmp_path: Path, source_dir: Path):
    original = PydanticPrompt.generate_multiple
    with _capture_prompts():
        assert PydanticPrompt.generate_multiple is not original
    assert PydanticPrompt.generate_multiple is original


def test_dump_requires_benchmark_files(tmp_path: Path):
    with pytest.raises(ValueError, match="No results_"):
        dump_judge_prompts(tmp_path, tmp_path / "out", "judge", "all")
