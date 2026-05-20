from dataclasses import dataclass
from typing import Any

from rag_anonymous import Anonymizer, build_llm, create_chain, load_vectordb


@dataclass(frozen=True)
class Pipeline:
    strategy: str
    gen_model: str
    chain: Any
    vectordb: Any
    anonymizer: Anonymizer | None


def build_pipeline(strategy: str, gen_model: str) -> Pipeline:
    vectordb = load_vectordb(collection_name=strategy)
    chain = create_chain(build_llm(gen_model))
    anonymizer = Anonymizer() if strategy == "ondemand" else None
    return Pipeline(
        strategy=strategy,
        gen_model=gen_model,
        chain=chain,
        vectordb=vectordb,
        anonymizer=anonymizer,
    )


def response_text(raw):
    if hasattr(raw, "content"):
        return raw.content
    return raw if isinstance(raw, str) else str(raw)
