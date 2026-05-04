"""rag-anonymous-metrics: Privacy / Utility evaluation for rag-anonymous.

Research-only consumer of the rag-anonymous pipeline. Provides:

- LLM-as-a-Judge faithfulness and context precision (Ragas methodology).
- Retrieval-conditioned PII leakage rate over TAB annotations.
- Heuristic refusal detection used to censor judge calls on non-answers.
- JSONL per-run results plus aggregated CSV summaries.
"""

from dotenv import load_dotenv

load_dotenv()
