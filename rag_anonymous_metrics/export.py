import json
import logging
from pathlib import Path

import pandas as pd

from rag_anonymous_metrics.config import Settings

logger = logging.getLogger(__name__)

METRIC_COLUMNS = (
    "faithfulness",
    "context_precision",
    "answer_is_refusal",
    "leakage_rate",
    "total_pii",
    "leaked_pii",
)

def export_csv(results_dir: Path | None = None, output_path: Path | None = None) -> Path:
    cfg = Settings.load()
    results_dir = cfg.results_dir if results_dir is None else Path(results_dir)
    output_path = results_dir / "metrics_summary.csv" if output_path is None else Path(output_path)

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
