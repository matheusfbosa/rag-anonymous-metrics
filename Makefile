CATEGORY ?= benchmark
DATASET  ?= dev
EXPERIMENT_DIR ?=
OLLAMA_DOCKER_CONTAINER ?= ollama
OLLAMA_VIA_DOCKER ?=
OVERWRITE ?=
RAG_ANON_DIR ?= ../rag-anonymous
RANGE    ?= all
RANGE_SMOKE ?= 1-5
STRATEGY ?= offline

# Lint / format

.PHONY: format lint test hooks
format:
	ruff format .
	ruff check --fix .

lint:
	ruff check .
	ruff format --check .
	mypy

hooks:
	pre-commit install

# Test

test:
	pytest -q

# Prerequisites

ingest-prereq:
	$(MAKE) -C $(RAG_ANON_DIR) ingest-$(DATASET)-strategies

ingest-dev-prereq:
	$(MAKE) ingest-prereq DATASET=dev

ingest-train-prereq:
	$(MAKE) ingest-prereq DATASET=train

ingest-test-prereq:
	$(MAKE) ingest-prereq DATASET=test

# Judge sanity check

check-judge:
	RAG_METRICS_STRATEGY=$(STRATEGY) \
	RAG_METRICS_CORPUS_DATASET=$(DATASET) \
	RAG_METRICS_QUESTIONS_CATEGORY=$(CATEGORY) \
	RAG_METRICS_QUESTIONS_RANGE=$(RANGE) \
	rag-metrics check-judge

check-judge-dev:
	$(MAKE) check-judge DATASET=dev

check-judge-train:
	$(MAKE) check-judge DATASET=train

check-judge-test:
	$(MAKE) check-judge DATASET=test

check-judge-smoke:
	$(MAKE) check-judge RANGE=$(RANGE_SMOKE)

# Run

run:
	RAG_METRICS_STRATEGY=$(STRATEGY) \
	RAG_METRICS_CORPUS_DATASET=$(DATASET) \
	RAG_METRICS_QUESTIONS_CATEGORY=$(CATEGORY) \
	RAG_METRICS_QUESTIONS_RANGE=$(RANGE) \
	RAG_METRICS_EXPERIMENT_DIR=$(EXPERIMENT_DIR) \
	RAG_METRICS_OVERWRITE=$(OVERWRITE) \
	rag-metrics run

run-offline:
	$(MAKE) run STRATEGY=offline

run-ondemand:
	$(MAKE) run STRATEGY=ondemand

run-dev:
	$(MAKE) run DATASET=dev

run-train:
	$(MAKE) run DATASET=train

run-test:
	$(MAKE) run DATASET=test

run-benchmark:
	$(MAKE) run CATEGORY=benchmark

run-factual:
	$(MAKE) run CATEGORY=factual

run-identity:
	$(MAKE) run CATEGORY=identity

run-adversarial:
	$(MAKE) run CATEGORY=adversarial

run-smoke:
	$(MAKE) run RANGE=$(RANGE_SMOKE)

# Unload Ollama models between strategy runs
# OLLAMA_VIA_DOCKER: 1 = force docker exec, 0 = force host CLI, unset = auto
ollama-unload:
	@set -a; [ -f .env ] && . ./.env; set +a; \
	OLLAMA_VIA_DOCKER="$(OLLAMA_VIA_DOCKER)"; \
	_use_docker() { \
	  case "$${OLLAMA_VIA_DOCKER}" in \
	    1|true|yes) return 0 ;; \
	    0|false|no) return 1 ;; \
	  esac; \
	  docker inspect -f '{{.State.Running}}' $(OLLAMA_DOCKER_CONTAINER) 2>/dev/null | grep -qx true; \
	}; \
	_ollama_stop() { \
	  model="$$1"; \
	  [ -n "$$model" ] || return 0; \
	  if _use_docker; then \
	    docker exec $(OLLAMA_DOCKER_CONTAINER) ollama stop "$$model" || true; \
	  elif command -v ollama >/dev/null 2>&1; then \
	    export OLLAMA_HOST="$${OLLAMA_HOST:-$${RAG_ANON_OLLAMA_BASE_URL:-http://127.0.0.1:11434}}"; \
	    ollama stop "$$model" || true; \
	  else \
	    echo "ollama-unload: skip $$model (no running container '$(OLLAMA_DOCKER_CONTAINER)' and no host ollama CLI)" >&2; \
	  fi; \
	}; \
	_ollama_stop "$$RAG_METRICS_JUDGE_MODEL"; \
	_ollama_stop "$$RAG_ANON_LLM_MODEL"; \
	_ollama_stop "$$RAG_ANON_EMBEDDING_MODEL"

run-strategy-suite:
	@if [ "$(CATEGORY)" = "benchmark" ]; then \
	  $(MAKE) run STRATEGY=$(STRATEGY) DATASET=$(DATASET) CATEGORY=benchmark RANGE=$(RANGE) EXPERIMENT_DIR=$(EXPERIMENT_DIR) OVERWRITE=$(OVERWRITE) && \
	  $(MAKE) ollama-unload && \
	  $(MAKE) run STRATEGY=$(STRATEGY) DATASET=$(DATASET) CATEGORY=adversarial RANGE=$(RANGE) EXPERIMENT_DIR=$(EXPERIMENT_DIR) OVERWRITE=$(OVERWRITE); \
	else \
	  $(MAKE) run STRATEGY=$(STRATEGY) DATASET=$(DATASET) CATEGORY=$(CATEGORY) RANGE=$(RANGE) EXPERIMENT_DIR=$(EXPERIMENT_DIR) OVERWRITE=$(OVERWRITE); \
	fi

run-strategies:
	@stamp="$(if $(EXPERIMENT_DIR),$(EXPERIMENT_DIR),$$(date +%Y%m%d_%H%M%S))"; \
	echo "Experiment dir: data/output/$$stamp"; \
	$(MAKE) run-strategy-suite STRATEGY=offline DATASET=$(DATASET) CATEGORY=$(CATEGORY) RANGE=$(RANGE) EXPERIMENT_DIR=$$stamp OVERWRITE=$(OVERWRITE) && \
	$(MAKE) ollama-unload && \
	$(MAKE) run-strategy-suite STRATEGY=ondemand DATASET=$(DATASET) CATEGORY=$(CATEGORY) RANGE=$(RANGE) EXPERIMENT_DIR=$$stamp OVERWRITE=$(OVERWRITE)

run-dev-strategies:
	$(MAKE) run-strategies DATASET=dev

run-train-strategies:
	$(MAKE) run-strategies DATASET=train

run-test-strategies:
	$(MAKE) run-strategies DATASET=test

run-smoke-strategies:
	$(MAKE) run-strategies RANGE=$(RANGE_SMOKE)

run-all:
	$(MAKE) run-dev-strategies
	$(MAKE) ollama-unload
	$(MAKE) run-train-strategies
	$(MAKE) ollama-unload
	$(MAKE) run-test-strategies
