# rag-anonymous-metrics

Computes **Utility** and **Privacy** metrics to quantify the trade-off between information quality and PII protection across Offline/On-Demand anonymization.

## Overview

This package evaluates two anonymization strategies against a benchmark question set drawn from the *[Text Anonymization Benchmark* (TAB) corpus](https://github.com/NorskRegnesentral/text-anonymization-benchmark):

- **Offline:** the corpus is anonymized by Presidio before ingestion; the retriever and generator never see raw PII.
- **On-Demand:** the raw corpus is ingested as-is; Presidio anonymizes only the generated answer at query time.

For each question × strategy the evaluator:

1. Runs the full RAG pipeline (retrieval + generation + optional post-generation anonymization).
2. Computes **privacy metrics** (PII leakage rate, answer refusal) immediately from the final answer.
3. Computes **utility metrics** (faithfulness, context precision) via a separate judge LLM once all answers are collected — refusals are skipped and recorded as `null`.
4. Persists per-query results to JSONL.

```mermaid
flowchart TD
    corpus["TAB Corpus (split)"] --> pii_index["PII Index\n(per doc_id)"]
    questions["Benchmark Questions"] --> loop["Per-question loop\n(offline × on-demand)"]

    subgraph rag_pipeline [RAG Pipeline]
        loop --> retriever["Retriever\n(ChromaDB, k chunks)"]
        retriever --> generator["Generator LLM\n(RAG_ANON_LLM_MODEL)"]
        generator --> anon["Anonymizer\n(offline: no-op | on-demand: Presidio)"]
        anon --> final_answer["final_answer"]
    end

    final_answer --> pii_leakage["PII Leakage Rate\n(substring match vs PII index)"]
    final_answer --> refusal["Refusal Detection\n(heuristic)"]
    pii_index --> pii_leakage

    refusal -->|"is_refusal = false"| judge_block
    refusal -->|"is_refusal = true"| null_scores["faithfulness = null\ncontext_precision = null"]

    subgraph judge_block [Judge LLM - RAG_METRICS_JUDGE_MODEL]
        faithfulness["Faithfulness\n(claim-level)"]
        ctx_precision["Context Precision@K\n(no gold reference)"]
    end

    pii_leakage --> jsonl["JSONL per run"]
    null_scores --> jsonl
    faithfulness --> jsonl
    ctx_precision --> jsonl
```



## Project structure

```
rag-anonymous-metrics/
├── pyproject.toml          # Package metadata; rag-metrics entry point; depends on rag-anonymous
├── rag_anonymous_metrics/
│   ├── __init__.py
│   ├── config.py           # Load env vars
│   ├── cli.py              # rag-metrics run
│   ├── corpus.py           # Load benchmark questions
│   ├── records.py          # QueryRecord dataclass (JSONL schema)
│   ├── run.py              # Pipeline per-question loop
│   └── metrics/
│       ├── privacy.py      # PII leakage index/rate + refusal heuristics
│       └── utility.py      # Ragas faithfulness + context precision (judge LLM)
├── data/
│   ├── input/
│   │   ├── benchmark_questions_dev.json
│   │   ├── benchmark_questions_train.json
│   │   └── benchmark_questions_test.json
│   └── output/             # Results and metrics
├── notebooks/              # Jupyter Notebooks for analysis
├── Makefile
└── .env.example
```

## Metrics

Utility metrics (faithfulness and context precision) are computed by the [Ragas](https://docs.ragas.io) framework, driven by a dedicated **judge LLM** (`RAG_METRICS_JUDGE_MODEL`) that is intentionally different from the generator (`RAG_ANON_LLM_MODEL`) to avoid self-evaluation bias. Privacy metrics (PII leakage rate, answer refusal rate) are computed deterministically and do not involve Ragas or the judge.

### Faithfulness

The **Faithfulness** metric measures how factually consistent a **response** is with the **retrieved context**. It ranges from **0 to 1**, with higher scores indicating better consistency.

A response is considered **faithful** if all its claims can be supported by the retrieved context.

To calculate this:

1. Identify all the claims in the response.
2. Check each claim to see if it can be inferred from the retrieved context.
3. Compute the faithfulness score using the formula:

$$
\text{Faithfulness} = \frac{\text{Number of claims in the response supported by the retrieved context}}{\text{Total number of claims in the response}}.
$$

**How it’s calculated:** Computed by Ragas' [`Faithfulness`](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/) metric. The **response** is `final_answer` (after any strategy-specific anonymisation) and the **retrieved context** is the list of chunk texts returned by the retriever, passed as a `SingleTurnSample`. Ragas performs steps (1)–(2) — claim decomposition and per-claim verification — internally via the judge LLM. If no claims are extracted the score is **undefined** (Ragas returns `NaN`) and stored as `null`; otherwise the score is in $[0,1]$.

### Context precision

**Context precision** evaluates the retriever’s ability to place **relevant** chunks **higher** than irrelevant ones: it rewards relevant items that appear near the top of the ranked `retrieved_contexts` list.

Ragas defines **Context Precision@K** as:

$$
\text{Context Precision@}K = \frac{\sum_{k=1}^{K} \bigl(\text{Precision@}k \times v_k\bigr)}{\text{Total number of relevant items in the top } K \text{ results}}
$$

$$
\text{Precision@}k = \frac{\text{true positives@}k}{\text{true positives@}k + \text{false positives@}k}
$$

where $K$ is the number of chunks in `retrieved_contexts` and $v_k \in 0,1$ is the relevance indicator at rank $k$. With one binary label per rank, the denominator at cut $k$ is $k$, and $\text{true positives@}k$ is the count of relevant chunks among positions $1..k$, so $\text{Precision@}k$ matches the cumulative precision used here.

**How it’s calculated:** Computed by Ragas' [`LLMContextPrecisionWithoutReference`](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_precision/) metric. Relevance $v_k$ is judged by the **judge LLM** from the **user question** and the **model response** (whether each chunk is useful *for producing that answer*) — the without-reference variant, since we have no gold `reference` string. Ragas computes the weighted Precision@$k$ formula above internally over the full `retrieved_contexts` list. If $\sum_k v_k = 0$ the score is **0**; if the metric cannot be computed it returns `NaN` and is stored as `null`.

### Answer refusal rate

Let $R_{\mathrm{ref}} \in 0,1$ be an **abstention indicator** computed by deterministic string heuristics.

Over $N$ benchmark items, the **refusal rate** is

$$  
\rho = \frac{1}{N}\sum_{j=1}^{N} R_{\mathrm{ref}}^{(j)}.  
$$

**Censoring of utility:** When $R_{\mathrm{ref}}=1$, faithfulness and context precision are **omitted** (recorded as `null`) so the judge is not applied to non-answers.

### PII leakage rate

Let $\mathcal{D}_{\mathrm{ret}}$ be the set of **source document IDs** associated with chunks returned for a query. From TAB annotations, define per-document sensitive span sets $\mathcal{P}_d$ as lowercased `span_text` values with `identifier_type` $\in \mathrm{DIRECT},\mathrm{QUASI}$ and length $\geq 2$ after trimming. The **retrieval-conditioned PII inventory** is

$$
\mathcal{P} = \bigcup_{d \in \mathcal{D}_{\mathrm{ret}}} \mathcal{P}_d.
$$

Let $a_{\mathrm{low}} = \mathrm{lower}(a)$ for the final answer $a$. **Leakage** is operationalised as **substring disclosure**:

$$
\mathcal{L} =  p \in \mathcal{P} \mid p \text{ occurs in } a_{\mathrm{low}} .
$$

**Per-query quantities:** `total_pii` $= |\mathcal{P}|$, `leaked_pii` $= |\mathcal{L}|$, and

$$
L = \begin{cases}
|\mathcal{L}|/|\mathcal{P}| & |\mathcal{P}| > 0, 
0 & |\mathcal{P}| = 0.
\end{cases}
$$

Thus $L \in [0,1]$ is **conditional recall of annotated PII** over retrieved documents: it does **not** flag hallucinated identifiers absent from $\mathcal{P}$.

## Setup

### 1. Ollama

The pipeline talks to Ollama over its HTTP API at `RAG_ANON_OLLAMA_BASE_URL` (default `http://localhost:11434`). Two equivalent ways to bring it up — pick one:

**Option A — Native CLI / desktop app** (install from [https://ollama.com](https://ollama.com)):

```sh
ollama serve                      # or just launch the Ollama desktop app
make ollama-pull-models           # pull models via the host CLI
```

Both paths expose the same API on port `11434`, so the rest of the pipeline doesn't need to know which one is running. **Don't run both at once** — they'll race for the port and the second one will fail to bind.

**Option B — Docker** (uses `[docker-compose.yml](./docker-compose.yml)`):

```sh
make docker-up                    # start the ollama container
make docker-ollama-pull-models    # pull models inside the container
make docker-down                  # stop the container when done
```

### 2. Python package

```sh
python3.12 -m venv .venv
source .venv/bin/activate

pip install -e ../rag-anonymous
pip install -e .

cp .env.example .env
```

## Usage

### Prerequisites

1. Offline and On-Demand strategies ingested for the split you will evaluate (`dev`, `train`, or `test`):
  ```sh
   cd ../rag-anonymous
   RAG_ANON_ANONYMIZER_STRATEGY=offline RAG_ANON_CORPUS_SPLIT=dev make ingest
   RAG_ANON_ANONYMIZER_STRATEGY=ondemand RAG_ANON_CORPUS_SPLIT=dev make ingest
  ```

### Run full evaluation (Offline and On-Demand strategies)

```sh
make run
```

### Run a single strategy or corpus split

```sh
RAG_METRICS_EVAL_STRATEGY=offline make run
RAG_METRICS_EVAL_STRATEGY=ondemand make run
RAG_METRICS_CORPUS_DATASET=train make run
RAG_METRICS_CORPUS_DATASET=test make run
```

`RAG_METRICS_CORPUS_DATASET` selects both the TAB JSON used for the PII index and the matching `data/input/benchmark_questions_<split>.json` file.

## Output

- **Per experiment:** outputs go under `data/output/<YYYYMMDD_HHMMSS>/` (timestamp when the run starts). Each folder contains:
  - `results_offline.jsonl` and/or `results_ondemand.jsonl` (depending on `RAG_METRICS_EVAL_STRATEGY`) — includes `llm_model`, `judge_model`, `answer_is_refusal`, and per-query utility scores.

## Configuration


| Variable                        | Default                                                    | Description                                                                                                                                                                    |
| ------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `RAG_ANON_CHROMADB_PERSIST_DIR` | `../rag-anonymous/chromadb`                                | Shared ChromaDB path                                                                                                                                                           |
| `RAG_ANON_EMBEDDING_MODEL`      | `nomic-embed-text`                                         | Embeddings model                                                                                                                                                               |
| `RAG_ANON_LLM_MODEL`            | `qwen3:8b`                                                 | Generator LLM                                                                                                                                                                  |
| `RAG_ANON_LLM_REASONING`        | `false`                                                    | Forwards to `ChatOllama(reasoning=...)`; `false` disables thinking for both generator and judge (faster, cleaner Yes/No verdicts). Accepted truthy values: `1`, `true`, `yes`. |
| `RAG_ANON_LLM_TEMPERATURE`      | `0.0`                                                      | Generation temperature                                                                                                                                                         |
| `RAG_ANON_LOG_LEVEL`            | `INFO`                                                     | Application loggers / root level. Accepts `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL` or a numeric level. Inherited from the `rag-anonymous` package.                          |
| `RAG_ANON_LOG_LEVEL_HTTP`       | `WARNING`                                                  | Level for HTTP client loggers (`httpx`, `httpcore`, `urllib3`). Default hides per-request lines; raise to `INFO` to inspect every Ollama call.                                 |
| `RAG_ANON_LOG_LEVEL_PRESIDIO`   | `ERROR`                                                    | Level for `presidio-analyzer` / `presidio-anonymizer`. Default suppresses the recognizer startup banner; raise to `INFO` to debug recognizer issues.                           |
| `RAG_ANON_OLLAMA_BASE_URL`      | `http://localhost:11434`                                   | Ollama API endpoint                                                                                                                                                            |
| `RAG_METRICS_CORPUS_DATASET`    | `dev`                                                      | TAB split (train/dev/test)                                                                                                                                                     |
| `RAG_METRICS_EVAL_STRATEGY`     | `all`                                                      | `offline`, `ondemand`, or `all`                                                                                                                                                |
| `RAG_METRICS_JUDGE_MODEL`       | `gemma4:12b` (falls back to `RAG_ANON_LLM_MODEL` if unset) | Judge LLM for faithfulness / context precision only                                                                                                                            |
| `RAG_METRICS_MAX_QUESTIONS`     | *(unset)*                                                  | If set, evaluate only the first N benchmark questions                                                                                                                          |
| `RAG_METRICS_RESULTS_DIR`       | `./data/output`                                            | Base directory; each run creates a timestamped subfolder with JSONL outputs                                                                                                    |
| `RAG_METRICS_RETRIEVAL_K_DOCS`  | `5`                                                        | Retrieved chunks per query                                                                                                                                                     |


