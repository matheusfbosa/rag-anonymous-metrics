from dataclasses import dataclass, field


@dataclass(slots=True)
class QueryRecord:
    run_id: str
    strategy: str
    question_id: str
    question: str
    ground_truth: str
    llm_model: str
    judge_model: str
    raw_answer: str
    final_answer: str
    answer_is_refusal: bool
    contexts: list[str]
    doc_ids: list[str]
    dataset: str
    total_generated: int
    leaked_generated: int
    leakage_rate_generated: float
    any_leak_generated: int
    total_anonymized: int
    leaked_anonymized: int
    leakage_rate_anonymized: float
    any_leak_anonymized: int
    pii_total_anonymized: int
    pii_leaked_anonymized: int
    pii_leakage_rate_anonymized: float
    faithfulness: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    factual_correctness: float | None = None


@dataclass(frozen=True, slots=True)
class StrategyResult:
    strategy: str
    records: list[QueryRecord] = field(default_factory=list)
    output_path: str = ""
