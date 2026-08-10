from __future__ import annotations

import logging

from rag_anonymous.log_config import configure_logging as _configure_base_logging

SECTION_WIDTH = 60
ITEM_PREVIEW_LEN = 80


def configure_logging() -> None:
    _configure_base_logging()


def log_section(title: str) -> None:
    log = logging.getLogger(__name__)
    log.info("")
    log.info("=" * SECTION_WIDTH)
    log.info(title)
    log.info("=" * SECTION_WIDTH)


def log_suite_header(suite: str, category: str, id_range: str) -> None:
    from rag_anonymous_metrics.questions_filter import describe_category, describe_range

    log_section(
        f"suite={suite} category={describe_category(category)} "
        f"range={describe_range(id_range)}"
    )


def log_item_progress(
    log: logging.Logger,
    idx: int,
    total: int,
    item_id: str,
    question: str,
) -> None:
    width = len(str(total))
    log.info(
        "  [%*d/%d] id=%s preview=%.*s",
        width,
        idx,
        total,
        item_id or "-",
        ITEM_PREVIEW_LEN,
        question,
    )
