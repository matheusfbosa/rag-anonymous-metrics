import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ragas.prompt.pydantic_prompt import PydanticPrompt

from rag_anonymous_metrics.questions_filter import RangeSpec, apply_id_range
from rag_anonymous_metrics.run import _is_refusal, _load_jsonl

logger = logging.getLogger(__name__)


def dump_judge_prompts(
    source_dir: Path,
    output_dir: Path,
    judge_model: str,
    range_spec: RangeSpec,
) -> list[Path]:
    from rag_anonymous_metrics.metrics.utility import UtilityScorer

    sources = sorted(source_dir.glob("results_*_benchmark.jsonl"))
    if not sources:
        raise ValueError(f"No results_*_benchmark.jsonl in {source_dir}")

    scorer = UtilityScorer(judge_model)
    written: list[Path] = []
    for source in sources:
        items = [{"id": r.question_id, "record": r} for r in _load_jsonl(source)]
        for item in apply_id_range(items, range_spec):
            record = item["record"]
            if _is_refusal(record):
                logger.info("Skipping refusal: question_id=%s", record.question_id)
                continue
            with _capture_prompts() as calls:
                scorer.score(
                    record.question,
                    record.final_answer,
                    record.contexts,
                    record.ground_truth,
                )
            folder = (
                output_dir
                / "judge_prompts"
                / f"{record.strategy}_q{record.question_id}"
            )
            folder.mkdir(parents=True, exist_ok=True)
            for idx, (name, text) in enumerate(calls, 1):
                path = folder / f"{idx:02d}_{name}.txt"
                path.write_text(text, encoding="utf-8")
                written.append(path)
            logger.info("Saved judge prompts: n=%d dir=%s", len(calls), folder)
    return written


@contextmanager
def _capture_prompts() -> Iterator[list[tuple[str, str]]]:
    # Later steps get the prompt's first few-shot example as the judge output.
    calls: list[tuple[str, str]] = []
    original = PydanticPrompt.generate_multiple

    async def record(self: Any, llm: Any, data: Any, n: int = 1, **_: Any) -> list:
        calls.append((type(self).__name__, self.to_string(data)))
        if self.examples:
            return [self.examples[0][1]] * n
        return [self.output_model.model_construct()] * n

    PydanticPrompt.generate_multiple = record  # type: ignore[method-assign,assignment]
    try:
        yield calls
    finally:
        PydanticPrompt.generate_multiple = original  # type: ignore[method-assign]
