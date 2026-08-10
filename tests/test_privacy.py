from rag_anonymous_metrics.metrics.privacy import compute_pii_leakage


def test_span_matches_without_honorific_in_answer() -> None:
    pii_index = {"doc-1": {"mr abdisamad adow sufi"}}
    answer = "Abdisamad Adow Sufi\nAbdiaziz Ibrahim Elmi"

    result = compute_pii_leakage(answer, ["doc-1"], pii_index)

    assert result["any_leak"] == 1
    assert result["leaked"] == 1


def test_span_matches_with_honorific_in_answer() -> None:
    pii_index = {"doc-1": {"mr stephen andrew benham"}}
    answer = "The applicant is Mr Stephen Andrew Benham."

    result = compute_pii_leakage(answer, ["doc-1"], pii_index)

    assert result["any_leak"] == 1


def test_short_core_after_honorific_strip_does_not_match() -> None:
    pii_index = {"doc-1": {"mr a"}}
    answer = "The letter a appears many times."

    result = compute_pii_leakage(answer, ["doc-1"], pii_index)

    assert result["any_leak"] == 0


def test_unrelated_name_does_not_match() -> None:
    pii_index = {"doc-1": {"mr gunnar beck"}}
    answer = "Göran Ravnsborg and Lars Bergman"

    result = compute_pii_leakage(answer, ["doc-1"], pii_index)

    assert result["any_leak"] == 0
