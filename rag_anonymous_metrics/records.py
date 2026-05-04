"""Per-query evaluation record schema.

The :class:`QueryRecord` dataclass is the single source of truth for the
fields persisted to ``data/output/results_*.jsonl`` and re-aggregated into
``data/output/metrics_summary.csv``. Use ``dataclasses.asdict`` for serialisation.
"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class QueryRecord:
    """One evaluated benchmark question for a single strategy."""

    run_id: str
    strategy: str
    question_id: str
    question: str
    llm_model: str
    judge_model: str
    raw_answer: str
    final_answer: str
    answer_is_refusal: bool
    contexts: list[str]
    doc_ids: list[str]
    total_pii: int
    leaked_pii: int
    leakage_rate: float
    faithfulness: float | None = None
    context_precision: float | None = None


@dataclass(frozen=True, slots=True)
class StrategyResult:
    """Output of :func:`rag_anonymous_metrics.evaluator.evaluate_strategy`."""

    strategy: str
    records: list[QueryRecord] = field(default_factory=list)
    output_path: str = ""
