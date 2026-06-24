import asyncio
import logging
import math

from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference

from rag_anonymous import build_llm

logger = logging.getLogger(__name__)


class UtilityScorer:
    def __init__(self, judge_model: str):
        llm = LangchainLLMWrapper(build_llm(judge_model))
        self._faithfulness = Faithfulness(llm=llm)
        self._context_precision = LLMContextPrecisionWithoutReference(llm=llm)

    def score(
        self, question: str, answer: str, contexts: list[str]
    ) -> tuple[float | None, float | None]:
        if not contexts:
            return None, None
        return asyncio.run(self._ascore(question, answer, contexts))

    async def _ascore(
        self, question: str, answer: str, contexts: list[str]
    ) -> tuple[float | None, float | None]:
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        return await asyncio.gather(
            self._safe(self._faithfulness, sample, "faithfulness"),
            self._safe(self._context_precision, sample, "context_precision"),
        )

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
