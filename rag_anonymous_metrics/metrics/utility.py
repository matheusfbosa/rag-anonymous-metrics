import asyncio
import logging
import math

from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)

from rag_anonymous import build_llm

logger = logging.getLogger(__name__)


class UtilityScorer:
    def __init__(self, judge_model: str):
        llm = LangchainLLMWrapper(build_llm(judge_model))
        self._faithfulness = Faithfulness(llm=llm)
        self._context_precision = LLMContextPrecisionWithoutReference(llm=llm)
        self._context_recall = LLMContextRecall(llm=llm)
        self._factual_correctness = FactualCorrectness(llm=llm)

    def score(
        self, question: str, answer: str, contexts: list[str], reference: str
    ) -> dict[str, float | None]:
        if not contexts:
            return {
                "faithfulness": None,
                "context_precision": None,
                "context_recall": None,
                "factual_correctness": None,
            }
        return asyncio.run(self._ascore(question, answer, contexts, reference))

    async def _ascore(
        self, question: str, answer: str, contexts: list[str], reference: str
    ) -> dict[str, float | None]:
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference,
        )
        faithfulness, context_precision, context_recall, factual_correctness = (
            await asyncio.gather(
                self._safe(self._faithfulness, sample, "faithfulness"),
                self._safe(self._context_precision, sample, "context_precision"),
                self._safe(self._context_recall, sample, "context_recall"),
                self._safe(self._factual_correctness, sample, "factual_correctness"),
            )
        )
        return {
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "factual_correctness": factual_correctness,
        }

    @staticmethod
    async def _safe(metric, sample: SingleTurnSample, name: str) -> float | None:
        try:
            score = await metric.single_turn_ascore(sample)
        except Exception:
            logger.warning("Ragas %s failed", name, exc_info=True)
            return None
        if score is None or math.isnan(score):
            return None
        return float(score)
