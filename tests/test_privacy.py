from rag_anonymous_metrics.metrics.privacy import compute_pii_leakage, is_refusal_answer


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


def test_privacy_prompt_refusal_is_detected() -> None:
    answer = (
        "I cannot disclose personal identifiers. "
        "I can summarize the case without naming individuals."
    )
    assert is_refusal_answer(answer) is True


def test_factual_answer_is_not_a_refusal() -> None:
    assert is_refusal_answer("The application was lodged in 2002.") is False


def test_cannot_comply_is_a_refusal() -> None:
    assert is_refusal_answer("I cannot comply with this request.") is True


def test_not_provided_in_opening_sentence_is_a_refusal() -> None:
    answer = "The outcome is not provided in the context. Nothing else to add."

    assert is_refusal_answer(answer) is True


def test_not_provided_after_an_answer_is_not_a_refusal() -> None:
    answer = (
        "The applicant argued his appeal in person. "
        "However, the details of the outcome are not provided in the context."
    )

    assert is_refusal_answer(answer) is False


def test_name_list_with_trailing_note_is_not_a_refusal() -> None:
    answer = "<PERSON>\nFerranti plc\n\n(Names of individuals are not provided in the given context.)"

    assert is_refusal_answer(answer) is False
