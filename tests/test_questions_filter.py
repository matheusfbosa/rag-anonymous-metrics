import pytest

from rag_anonymous_metrics.questions_filter import (
    apply_id_range,
    describe_range,
    parse_questions_range,
)

_ITEMS = [
    {"id": "51"},
    {"id": "52"},
    {"id": "53"},
    {"id": "57"},
    {"id": "59"},
]


def test_parse_questions_range_all():
    assert parse_questions_range("all") == "all"


def test_parse_questions_range_contiguous():
    assert parse_questions_range("52-59") == (52, 59)


def test_parse_questions_range_single_id_range():
    assert parse_questions_range("52-52") == (52, 52)


def test_parse_questions_range_list():
    assert parse_questions_range("52,57,59") == frozenset({52, 57, 59})


def test_parse_questions_range_list_with_spaces():
    assert parse_questions_range(" 52 , 57 , 59 ") == frozenset({52, 57, 59})


def test_parse_questions_range_rejects_invalid():
    with pytest.raises(ValueError, match="comma-separated list"):
        parse_questions_range("52-57,59")


def test_apply_id_range_list_preserves_source_order():
    selected = apply_id_range(_ITEMS, frozenset({59, 52, 57}))
    assert [item["id"] for item in selected] == ["52", "57", "59"]


def test_apply_id_range_contiguous():
    selected = apply_id_range(_ITEMS, (52, 57))
    assert [item["id"] for item in selected] == ["52", "53", "57"]


def test_describe_range_list():
    assert describe_range(frozenset({59, 52, 57})) == "52,57,59"
