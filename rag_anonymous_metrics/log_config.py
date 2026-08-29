from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Protocol

from rag_anonymous.log_config import configure_logging as _configure_base_logging
from rag_anonymous_metrics.questions_filter import (
    CategorySpec,
    RangeSpec,
    describe_category,
    describe_range,
)

SECTION_WIDTH = 60
ITEM_PREVIEW_LEN = 80


def init_command_output(
    results_dir: Path,
    log_name: str,
    *,
    experiment_dir: Path | None = None,
    append_log: bool = False,
) -> Path:
    if experiment_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = results_dir / stamp
    experiment_dir.mkdir(parents=True, exist_ok=True)
    log_path = experiment_dir / log_name
    mode = "a" if append_log else "w"
    log_file = log_path.open(mode, encoding="utf-8")
    stdout = sys.__stdout__
    stderr = sys.__stderr__
    if stdout is None or stderr is None:
        raise RuntimeError("standard streams are not available")
    sys.stdout = _Tee(stdout, log_file)
    sys.stderr = _Tee(stderr, log_file)
    return experiment_dir


def configure_logging() -> None:
    _configure_base_logging()


def log_section(title: str) -> None:
    log = logging.getLogger(__name__)
    log.info("")
    log.info("=" * SECTION_WIDTH)
    log.info(title)
    log.info("=" * SECTION_WIDTH)


def log_suite_header(suite: str, category: CategorySpec, id_range: RangeSpec) -> None:
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


class _Stream(Protocol):
    def write(self, data: str) -> object: ...
    def flush(self) -> object: ...


class _Tee:
    def __init__(self, *streams: _Stream) -> None:
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        primary = self._streams[0]
        isatty = getattr(primary, "isatty", None)
        return bool(isatty()) if isatty else False
