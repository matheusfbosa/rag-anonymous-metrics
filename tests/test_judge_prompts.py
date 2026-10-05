from pathlib import Path

import pytest
from ragas.prompt.pydantic_prompt import PydanticPrompt
from test_run_checkpoint import _record

from rag_anonymous_metrics.judge_prompts import _capture_prompts, dump_judge_prompts
from rag_anonymous_metrics.run import _save_jsonl

pytestmark = pytest.mark.usefixtures("metrics_env")


@pytest.fixture
def metrics_env(monkeypatch):
    monkeypatch.setenv("RAG_METRICS_STRATEGY", "offline")
    monkeypatch.setenv("RAG_METRICS_CORPUS_DATASET", "dev")


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
    assert any("ContextPrecision" in n for n in names)
    assert sum("ContextPrecision" in n for n in names) == 3
    assert "first" in written[0].read_text()


def test_dump_skips_refusals(tmp_path: Path, source_dir: Path):
    written = dump_judge_prompts(source_dir, tmp_path / "out", "judge", (3, 3))

    assert written == []


def test_capture_restores_ragas(tmp_path: Path, source_dir: Path):
    original = PydanticPrompt.generate_multiple
    with _capture_prompts():
        assert PydanticPrompt.generate_multiple is not original
    assert PydanticPrompt.generate_multiple is original


def test_dump_requires_benchmark_files(tmp_path: Path):
    with pytest.raises(ValueError, match="No results_"):
        dump_judge_prompts(tmp_path, tmp_path / "out", "judge", "all")
