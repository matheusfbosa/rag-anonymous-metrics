from test_run_checkpoint import _record

from rag_anonymous_metrics.metrics.adversarial import (
    PERSON_LEAK,
    TARGET_LEAK,
    asr_by_type,
    attack_success_rate,
)

RECORDS = [
    _record(
        item_type="direct", pii_target_any_leak_final=1, pii_person_any_leak_final=1
    ),
    _record(item_type="direct", pii_person_any_leak_final=1),
    _record(item_type="social"),
]


def test_attack_success_rate_uses_given_leak_field():
    assert attack_success_rate(RECORDS, TARGET_LEAK)["successes"] == 1
    assert attack_success_rate(RECORDS, PERSON_LEAK)["successes"] == 2
    assert attack_success_rate([], TARGET_LEAK)["asr"] == 0.0


def test_asr_by_type_splits_attack_types():
    by_type = asr_by_type(RECORDS, PERSON_LEAK)

    assert by_type["direct"] == {"n": 2, "successes": 2, "asr": 1.0}
    assert by_type["social"]["successes"] == 0
