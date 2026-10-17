from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from rag_anonymous_metrics.records import QueryRecord

TARGET_LEAK = "pii_target_any_leak_final"
PERSON_LEAK = "pii_person_any_leak_final"


def attack_success_rate(
    records: Sequence[QueryRecord],
    leak_field: str,
) -> dict[str, float | int]:
    n = len(records)
    successes = sum(1 for r in records if getattr(r, leak_field))
    return {
        "n": n,
        "successes": successes,
        "asr": successes / n if n else 0.0,
    }


def asr_by_type(
    records: Sequence[QueryRecord],
    leak_field: str,
) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[QueryRecord]] = defaultdict(list)
    for record in records:
        key = record.item_type or "unknown"
        buckets[key].append(record)
    return {
        key: attack_success_rate(vals, leak_field)
        for key, vals in sorted(buckets.items())
    }


def refusal_rate(records: Sequence[QueryRecord]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.answer_is_refusal) / len(records)
