"""Aggregate per-run JSONL results into a metrics_summary.csv grouped by strategy."""

import json
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

METRIC_COLUMNS = (
    "faithfulness",
    "context_precision",
    "answer_is_refusal",
    "leakage_rate",
    "total_pii",
    "leaked_pii",
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def export_csv(results_dir: Path | None = None, output_path: Path | None = None) -> Path:
    """Aggregate JSONL results into ``metrics_summary.csv`` and return its path."""
    results_dir = Path(results_dir or _results_dir())
    output_path = Path(output_path or results_dir / "metrics_summary.csv")

    records = _load_all_jsonl(results_dir)
    df = pd.DataFrame(records)

    if "answer_is_refusal" not in df.columns:
        df["answer_is_refusal"] = False
    df["answer_is_refusal"] = df["answer_is_refusal"].astype(float)

    summary = df.groupby("strategy")[list(METRIC_COLUMNS)].agg(["mean", "std", "count"])
    summary.columns = [
        f"{col}_num_questions" if stat == "count" else f"{col}_{stat}"
        for col, stat in summary.columns
    ]
    summary = summary.reset_index()

    summary.to_csv(output_path, index=False)
    logger.info("Exported metrics summary to %s", output_path)
    logger.info("\n%s", _format_summary(summary))
    return output_path


def _format_summary(summary: pd.DataFrame) -> str:
    """Render a compact, terminal-friendly view of the per-strategy summary.

    Strategies become columns and metrics become rows, so the line width
    grows with the (small) number of strategies rather than the (large)
    number of metric/stat combinations. The full ``mean``/``std``/``count``
    grid still lives in the CSV; this view is for the log only.
    """
    means = (
        summary.set_index("strategy")
        .filter(regex=r"_mean$")
        .rename(columns=lambda c: c.removesuffix("_mean"))
        .T.round(4)
    )
    counts = (
        summary.set_index("strategy")
        .filter(regex=r"_num_questions$")
        .rename(columns=lambda c: c.removesuffix("_num_questions"))
        .astype(int)
        .T
    )

    lines = [means.to_string(), ""]
    if (counts.nunique(axis=0) == 1).all():
        ns = ", ".join(f"{strategy}={n}" for strategy, n in counts.iloc[0].items())
        lines.append(f"n: {ns}")
    else:
        lines.append("Defined records per metric:")
        lines.append(counts.to_string())
    return "\n".join(lines)


def _load_all_jsonl(results_dir: Path) -> list[dict]:
    """Read every ``results_*.jsonl`` file in ``results_dir`` into one list."""
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    jsonl_files = sorted(results_dir.glob("results_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL result files found in {results_dir}")

    records: list[dict] = []
    for path in jsonl_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        logger.info("Loaded %s", path.name)

    logger.info("Total records loaded: %d", len(records))
    return records


def _results_dir() -> Path:
    return _PROJECT_ROOT / os.getenv("RAG_METRICS_RESULTS_DIR", "./data/output")
