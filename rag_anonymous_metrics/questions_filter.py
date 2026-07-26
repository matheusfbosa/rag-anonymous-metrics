from __future__ import annotations

import re
from typing import Literal, Sequence

CategoryFilter = Literal["factual", "identity"]
FilterSpec = None | tuple[int, int] | CategoryFilter

_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_CATEGORY_FILTERS: frozenset[str] = frozenset({"factual", "identity"})


def parse_questions_filter(raw: str | None) -> FilterSpec:
    """Parse RAG_METRICS_QUESTIONS_FILTER.

    Returns:
        None — all questions (unset, empty, or ``all``)
        ``factual`` / ``identity`` — filter by question ``category``
        ``(start, end)`` — 1-based inclusive index range in the loaded list
    """
    if raw is None or not raw.strip():
        return None
    trimmed = raw.strip().lower()
    if trimmed == "all":
        return None
    if trimmed in _CATEGORY_FILTERS:
        return trimmed
    match = _RANGE_RE.match(trimmed)
    if not match:
        raise ValueError(
            "RAG_METRICS_QUESTIONS_FILTER must be 'all', 'factual', 'identity', "
            f"or a range like '1-60', got {raw!r}"
        )
    start = int(match.group(1))
    end = int(match.group(2))
    if start <= 0 or end <= 0:
        raise ValueError(
            f"RAG_METRICS_QUESTIONS_FILTER range must be positive, got {start}-{end}"
        )
    if start > end:
        raise ValueError(
            f"RAG_METRICS_QUESTIONS_FILTER start must be <= end, got {start}-{end}"
        )
    return (start, end)


def apply_questions_filter(
    questions: Sequence[dict],
    filter_spec: FilterSpec,
) -> list[dict]:
    if filter_spec is None:
        return list(questions)
    if filter_spec in _CATEGORY_FILTERS:
        return [q for q in questions if q.get("category") == filter_spec]
    start, end = filter_spec
    if start > len(questions):
        return []
    slice_end = min(end, len(questions))
    return list(questions[start - 1:slice_end])


def describe_questions_filter(filter_spec: FilterSpec) -> str:
    if filter_spec is None:
        return "all"
    if isinstance(filter_spec, str):
        return filter_spec
    start, end = filter_spec
    return f"{start}-{end}"
