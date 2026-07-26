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
    generated_answer: str
    final_answer: str
    answer_is_refusal: bool
    contexts: list[str]
    doc_ids: list[str]
    dataset: str
    pii_total_generated: int
    pii_leaked_generated: int
    pii_leakage_rate_generated: float
    pii_any_leak_generated: int
    pii_total_final: int
    pii_leaked_final: int
    pii_leakage_rate_final: float
    pii_any_leak_final: int
    faithfulness: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    factual_correctness: float | None = None


@dataclass(frozen=True, slots=True)
class StrategyResult:
    strategy: str
    records: list[QueryRecord] = field(default_factory=list)
    output_path: str = ""
