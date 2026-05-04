"""Heuristic detection of safety-style refusals and non-answers.

Used to tag rows so faithfulness / context-precision judging can be skipped
when the generator declined to answer, avoiding ambiguous low scores.
"""

import re
import unicodedata

_APOSTROPHE_VARIANTS = (
    "\u2019",  # right single quotation
    "\u2018",  # left single quotation
    "\u02bc",  # modifier letter apostrophe
)

_REFUSAL_MARKERS = (
    "i can't help",
    "i cannot help",
    "can't assist",
    "cannot assist",
    "can't provide",
    "cannot provide",
    "unable to provide",
    "unable to help",
    "i'm not able to",
    "i am not able to",
    "i can't give",
    "cannot give you",
    "do not have access",
    "private citizen",
    "personal information",
    "personal data",
    "identify a private",
    "identify individuals",
    "anything else i can help",
    "is there anything else i can help",
    "can i help you with something else",
    "won't provide",
    "will not provide",
    "not comfortable providing",
    "illegal or harmful activities",
    "including defamation",
)

_NON_ANSWER_MAX_LEN = 22
_NON_ANSWER_PATTERN = re.compile(
    r"^(?:i\s+don'?t\s+know|not\s+enough\s+context|no\s+information)\.?$",
    re.IGNORECASE,
)


def is_refusal_answer(text: str | None) -> bool:
    """Return True if the answer is a safety-style refusal or empty non-answer."""
    if text is None or not str(text).strip():
        return True

    normalised = _normalise(str(text))
    if any(marker in normalised for marker in _REFUSAL_MARKERS):
        return True

    stripped = normalised.strip()
    if len(stripped) <= _NON_ANSWER_MAX_LEN and _NON_ANSWER_PATTERN.match(stripped):
        return True

    return False


def _normalise(text: str) -> str:
    """Lowercase, NFKC-normalise, and unify curly apostrophes for matching."""
    s = unicodedata.normalize("NFKC", text).lower()
    for ch in _APOSTROPHE_VARIANTS:
        s = s.replace(ch, "'")
    return s
