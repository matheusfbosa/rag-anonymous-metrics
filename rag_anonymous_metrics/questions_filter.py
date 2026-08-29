from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, cast

CategorySpec = Literal["benchmark", "factual", "identity", "adversarial"]
RangeSpec = Literal["all"] | tuple[int, int] | frozenset[int]

_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_LIST_RE = re.compile(r"^\s*\d+(?:\s*,\s*\d+)*\s*$")
_CATEGORIES: frozenset[str] = frozenset(
    {"benchmark", "factual", "identity", "adversarial"}
)


def parse_questions_category(raw: str | None) -> CategorySpec:
    if raw is None or not raw.strip():
        raise ValueError(
            "RAG_METRICS_QUESTIONS_CATEGORY is required "
            "(benchmark|factual|identity|adversarial)"
        )
    trimmed = raw.strip().lower()
    if trimmed not in _CATEGORIES:
        raise ValueError(
            "RAG_METRICS_QUESTIONS_CATEGORY must be "
            "'benchmark', 'factual', 'identity', or 'adversarial', "
            f"got {raw!r}"
        )
    return cast(CategorySpec, trimmed)


def parse_questions_range(raw: str | None) -> RangeSpec:
    if raw is None or not raw.strip():
        raise ValueError(
            "RAG_METRICS_QUESTIONS_RANGE is required "
            "(all, START-END, or comma-separated ids like '52,57,59')"
        )
    trimmed = raw.strip().lower()
    if trimmed == "all":
        return "all"

    range_match = _RANGE_RE.match(trimmed)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        if start <= 0 or end <= 0:
            raise ValueError(
                "RAG_METRICS_QUESTIONS_RANGE bounds must be positive, "
                f"got {start}-{end}"
            )
        if start > end:
            raise ValueError(
                f"RAG_METRICS_QUESTIONS_RANGE start must be <= end, got {start}-{end}"
            )
        return (start, end)

    if _LIST_RE.match(trimmed):
        ids = frozenset(int(part.strip()) for part in trimmed.split(","))
        if any(item_id <= 0 for item_id in ids):
            raise ValueError(
                f"RAG_METRICS_QUESTIONS_RANGE ids must be positive, got {raw!r}"
            )
        return ids

    raise ValueError(
        "RAG_METRICS_QUESTIONS_RANGE must be 'all', a range like '1-60', "
        f"or a comma-separated list like '52,57,59', got {raw!r}"
    )


def apply_category_filter(
    items: Sequence[dict],
    category: CategorySpec,
) -> list[dict]:
    if category == "benchmark":
        return list(items)
    return [q for q in items if q.get("category") == category]


def apply_id_range(
    items: Sequence[dict],
    range_spec: RangeSpec,
) -> list[dict]:
    if range_spec == "all":
        return list(items)

    if isinstance(range_spec, frozenset):
        allowed = range_spec
        selected: list[dict] = []
        for item in items:
            try:
                item_id = int(str(item.get("id", "")).strip())
            except ValueError:
                continue
            if item_id in allowed:
                selected.append(item)
        return selected

    start, end = range_spec
    selected = []
    for item in items:
        try:
            item_id = int(str(item.get("id", "")).strip())
        except ValueError:
            continue
        if start <= item_id <= end:
            selected.append(item)
    return selected


def describe_category(category: CategorySpec) -> str:
    return category


def describe_range(range_spec: RangeSpec) -> str:
    if range_spec == "all":
        return "all"
    if isinstance(range_spec, frozenset):
        return ",".join(str(item_id) for item_id in sorted(range_spec))
    start, end = range_spec
    return f"{start}-{end}"
