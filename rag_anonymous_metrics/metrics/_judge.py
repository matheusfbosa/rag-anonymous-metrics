import logging
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_anonymous import build_llm

from rag_anonymous_metrics.config import Settings

logger = logging.getLogger(__name__)

VERDICT_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def build_judge_chain(model_name: str | None = None):
    llm = build_llm(model_name or _default_judge_model())
    return llm | StrOutputParser()


def invoke(chain, template: str, variables: dict[str, Any]) -> str:
    prompt = ChatPromptTemplate.from_template(template)
    return (prompt | chain).invoke(variables)


def judge_yes_no(
    chain,
    template: str,
    variables: dict[str, Any],
    expected_count: int,
) -> list[bool]:
    raw = invoke(chain, template, variables)
    return parse_verdicts(raw, expected_count)


def parse_verdicts(text: str, expected_count: int) -> list[bool]:
    verdicts = [
        match.group(1).lower() == "yes"
        for line in text.strip().splitlines()
        if (match := VERDICT_PATTERN.search(line))
    ]
    if len(verdicts) < expected_count:
        logger.warning(
            "Expected %d verdicts but parsed %d — raw output:\n%s",
            expected_count, len(verdicts), text,
        )
    return verdicts


def _default_judge_model() -> str:
    return Settings.load().judge_model
