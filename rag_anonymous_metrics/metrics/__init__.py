"""Metric scoring primitives for rag-anonymous-metrics.

Modules:

- ``_judge``: shared judge-LLM chain construction and Yes/No verdict parsing.
- ``faithfulness``: claim-level faithfulness scoring.
- ``context_precision``: retrieval ranking quality scoring.
- ``pii_leakage``: substring-based PII leakage rate.
- ``refusal``: heuristic refusal / non-answer detection.
"""
