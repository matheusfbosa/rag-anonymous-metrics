from typing import Protocol

from ragas.dataset_schema import SingleTurnSample


class SingleTurnMetric(Protocol):
    async def single_turn_ascore(self, sample: SingleTurnSample) -> float | None: ...
