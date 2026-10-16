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
    pii_person_total_generated: int
    pii_person_leaked_generated: int
    pii_person_leakage_rate_generated: float
    pii_person_any_leak_generated: int
    pii_person_total_final: int
    pii_person_leaked_final: int
    pii_person_leakage_rate_final: float
    pii_person_any_leak_final: int
    source_doc_id: str
    retrieval_hit: int
    privacy_prompt: bool
    item_type: str | None = None
    category: str | None = None
    faithfulness: float | None = None
    factual_correctness: float | None = None
    retrieval_sec: float | None = None
    generation_sec: float | None = None
    anonymize_sec: float | None = None
    query_sec: float | None = None
    judge_sec: float | None = None
    chunk_ids: list[str] | None = None
    answer_accuracy: float | None = None
    semantic_similarity: float | None = None
    fc_n_resp: int | None = None
    fc_n_resp_ok: int | None = None
    fc_n_ref: int | None = None
    fc_n_ref_ok: int | None = None
    judge_metric_sec: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class StrategyResult:
    strategy: str
    records: list[QueryRecord] = field(default_factory=list)
    output_path: str = ""
