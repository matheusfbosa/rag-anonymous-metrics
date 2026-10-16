import asyncio
import logging
import math
import time
from collections.abc import Awaitable
from typing import Any

from langchain_ollama import OllamaEmbeddings
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    AnswerAccuracy,
    FactualCorrectness,
    Faithfulness,
    LLMContextRecall,
    SemanticSimilarity,
)

from rag_anonymous import Settings as AnonSettings
from rag_anonymous import build_llm
from rag_anonymous_metrics.config import Settings
from rag_anonymous_metrics.types import SingleTurnMetric

logger = logging.getLogger(__name__)

COVERAGE_KEYS = ("fc_n_resp", "fc_n_resp_ok", "fc_n_ref", "fc_n_ref_ok")


def judge_llm(judge_model: str) -> LangchainLLMWrapper:
    cfg = Settings.load()
    return LangchainLLMWrapper(
        build_llm(
            judge_model,
            num_predict=cfg.judge_num_predict,
            timeout=cfg.judge_timeout_sec,
        )
    )


def _embeddings() -> LangchainEmbeddingsWrapper:
    s = AnonSettings.load()
    return LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            model=s.embedding_model,
            base_url=s.ollama_base_url,
            num_ctx=s.embedding_num_ctx,
        )
    )


class UtilityScorer:
    def __init__(self, judge_model: str) -> None:
        llm = judge_llm(judge_model)
        self._factual_correctness = FactualCorrectness(llm=llm)
        self._answer_accuracy = AnswerAccuracy(llm=llm)
        self._faithfulness = Faithfulness(llm=llm)
        self._semantic_similarity = SemanticSimilarity(embeddings=_embeddings())

    def score(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        reference: str,
        *,
        refusal: bool = False,
    ) -> dict[str, Any]:
        return asyncio.run(
            self._ascore(question, answer, contexts, reference, refusal=refusal)
        )

    async def _ascore(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        reference: str,
        *,
        refusal: bool,
    ) -> dict[str, Any]:
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference,
        )
        timings: dict[str, float] = {}
        coverage = await _timed(
            timings, "factual_correctness", self._coverage(answer, reference)
        )
        accuracy = await _timed(
            timings,
            "answer_accuracy",
            _safe(self._answer_accuracy, sample, "answer_accuracy"),
        )
        similarity = await _timed(
            timings,
            "semantic_similarity",
            _safe(self._semantic_similarity, sample, "semantic_similarity"),
        )
        faithfulness = None
        if contexts and not refusal:
            faithfulness = await _timed(
                timings,
                "faithfulness",
                _safe(self._faithfulness, sample, "faithfulness"),
            )
        return {
            **coverage,
            "answer_accuracy": accuracy,
            "semantic_similarity": similarity,
            "faithfulness": faithfulness,
            "judge_metric_sec": timings,
        }

    def run_prompt_metrics(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        reference: str,
        *,
        refusal: bool = False,
    ) -> None:
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference,
        )

        async def run() -> None:
            await self._coverage(answer, reference)
            if contexts and not refusal:
                await _safe(self._faithfulness, sample, "faithfulness")

        asyncio.run(run())

    def answer_accuracy_prompts(
        self, question: str, answer: str, reference: str
    ) -> list[tuple[str, str]]:
        metric = self._answer_accuracy
        return [
            (
                "AnswerAccuracy1",
                metric.template_accuracy1.format(
                    query=question,
                    answer0="User Answer",
                    answer1="Reference Answer",
                    sentence_inference=answer,
                    sentence_true=reference,
                ),
            ),
            (
                "AnswerAccuracy2",
                metric.template_accuracy2.format(
                    query=question,
                    answer0="Reference Answer",
                    answer1="User Answer",
                    sentence_inference=reference,
                    sentence_true=answer,
                ),
            ),
        ]

    async def _coverage(self, answer: str, reference: str) -> dict[str, Any]:
        # Ragas' recall mode divides answer-claim TP by reference-claim FN,
        # so reference coverage is counted here from the same judge calls.
        try:
            resp_ok = await self._verified_claims(answer, premise=reference)
            ref_ok = await self._verified_claims(reference, premise=answer)
        except Exception as exc:
            logger.warning(
                "Ragas metric failed: metric=factual_correctness error=%s",
                exc.__class__.__name__,
            )
            logger.debug("Ragas factual_correctness traceback", exc_info=True)
            return {"factual_correctness": None, **dict.fromkeys(COVERAGE_KEYS)}
        n_ref = len(ref_ok)
        n_ref_ok = sum(ref_ok)
        return {
            "factual_correctness": n_ref_ok / n_ref if n_ref else None,
            "fc_n_resp": len(resp_ok),
            "fc_n_resp_ok": sum(resp_ok),
            "fc_n_ref": n_ref,
            "fc_n_ref_ok": n_ref_ok,
        }

    async def _verified_claims(self, text: str, *, premise: str) -> list[bool]:
        metric = self._factual_correctness
        claims = await metric.decompose_claims(text, None)
        if not claims:
            return []
        verdicts = await metric.verify_claims(
            premise=premise, hypothesis_list=claims, callbacks=None
        )
        return [bool(v) for v in verdicts]


class ContextRecallScorer:
    def __init__(self, judge_model: str) -> None:
        self._context_recall = LLMContextRecall(llm=judge_llm(judge_model))

    def score(self, question: str, contexts: list[str], reference: str) -> float | None:
        if not contexts:
            return None
        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts,
            reference=reference,
        )
        return asyncio.run(_safe(self._context_recall, sample, "context_recall"))


async def _timed[T](timings: dict[str, float], name: str, task: Awaitable[T]) -> T:
    started = time.perf_counter()
    try:
        return await task
    finally:
        timings[name] = time.perf_counter() - started


async def _safe(
    metric: SingleTurnMetric, sample: SingleTurnSample, name: str
) -> float | None:
    try:
        score = await metric.single_turn_ascore(sample)
    except Exception as exc:
        logger.warning(
            "Ragas metric failed: metric=%s error=%s",
            name,
            exc.__class__.__name__,
        )
        logger.debug("Ragas %s traceback", name, exc_info=True)
        return None
    if score is None or math.isnan(score):
        return None
    return float(score)
