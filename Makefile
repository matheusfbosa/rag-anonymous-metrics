CATEGORY ?= benchmark
DATASET  ?= $(or $(RAG_METRICS_CORPUS_DATASET),$(shell sed -n 's/^RAG_METRICS_CORPUS_DATASET=//p' .env 2>/dev/null),dev)
EXPERIMENT_DIR ?=
OLLAMA_DOCKER_CONTAINER ?= ollama
OLLAMA_VIA_DOCKER ?=
OVERWRITE ?=
RAG_ANON_DIR ?= ../rag-anonymous
RANGE    ?= all
RANGE_SMOKE ?= 1-5
SKIP_JUDGE ?=
SOURCE_DIR ?=
STRATEGY ?= offline

RUN_ENV = RAG_METRICS_STRATEGY=$(STRATEGY) \
	RAG_METRICS_CORPUS_DATASET=$(DATASET) \
	RAG_METRICS_QUESTIONS_CATEGORY=$(CATEGORY) \
	RAG_METRICS_QUESTIONS_RANGE=$(RANGE) \
	RAG_METRICS_EXPERIMENT_DIR=$(EXPERIMENT_DIR) \
	RAG_METRICS_OVERWRITE=$(OVERWRITE) \
	RAG_METRICS_SKIP_JUDGE=$(SKIP_JUDGE)

STAMP := $(or $(EXPERIMENT_DIR),$(shell date +%Y%m%d_%H%M%S))

require = @test -n "$($(1))" || { echo "$@: set $(1)=<run folder under data/output>" >&2; exit 1; }

.DEFAULT_GOAL := help
.PHONY: help format lint test hooks ingest check-judge check-judge-smoke \
	run-strategies generate-strategies judge-strategies run-smoke-strategies run-all \
	run generate judge rejudge judge-prompts ollama-unload

help: ## List targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  %-22s %s\n", $$1, $$2} /^## / {printf "\n%s\n", substr($$0, 4)}' $(MAKEFILE_LIST)
	@printf "\nVariables: DATASET=%s CATEGORY=%s RANGE=%s STRATEGY=%s EXPERIMENT_DIR=%s\n" "$(DATASET)" "$(CATEGORY)" "$(RANGE)" "$(STRATEGY)" "$(EXPERIMENT_DIR)"

## Setup

ingest: ## Build offline + ondemand indices in rag-anonymous for DATASET
	$(MAKE) -C $(RAG_ANON_DIR) ingest-$(DATASET)-strategies

check-judge: ## Sanity-check RAG_METRICS_JUDGE_MODEL (own vs foreign ground truth)
	RAG_METRICS_STRATEGY=$(STRATEGY) \
	RAG_METRICS_CORPUS_DATASET=$(DATASET) \
	RAG_METRICS_QUESTIONS_CATEGORY=$(CATEGORY) \
	RAG_METRICS_QUESTIONS_RANGE=$(RANGE) \
	rag-metrics check-judge

check-judge-smoke: ## check-judge on RANGE_SMOKE
	$(MAKE) check-judge RANGE=$(RANGE_SMOKE)

## Experiment: both strategies in one folder

run-strategies: ## End to end: generate both strategies, unload, then judge both
	$(MAKE) generate-strategies EXPERIMENT_DIR=$(STAMP)
	$(MAKE) ollama-unload
	$(MAKE) judge-strategies EXPERIMENT_DIR=$(STAMP)

generate-strategies: ## Phase 1: retrieval + generation + privacy, no judge
	@echo "Experiment dir: data/output/$(STAMP)"
	$(MAKE) generate STRATEGY=offline EXPERIMENT_DIR=$(STAMP)
	$(MAKE) generate STRATEGY=ondemand EXPERIMENT_DIR=$(STAMP)

judge-strategies: ## Phase 2: Ragas utility on a generated folder (EXPERIMENT_DIR)
	$(call require,EXPERIMENT_DIR)
	$(MAKE) judge STRATEGY=offline
	$(MAKE) judge STRATEGY=ondemand

run-smoke-strategies: ## run-strategies on RANGE_SMOKE
	$(MAKE) run-strategies RANGE=$(RANGE_SMOKE)

run-all: ## run-strategies for dev, train and test
	$(MAKE) run-strategies DATASET=dev
	$(MAKE) ollama-unload
	$(MAKE) run-strategies DATASET=train
	$(MAKE) ollama-unload
	$(MAKE) run-strategies DATASET=test

## Pieces: one strategy (STRATEGY=offline|ondemand)

run: ## One CATEGORY, generate + judge (resumes a folder given EXPERIMENT_DIR)
	$(RUN_ENV) rag-metrics run

generate: ## Generation only; CATEGORY=benchmark also runs the adversarial suite
	$(MAKE) run SKIP_JUDGE=1
	$(if $(filter benchmark,$(CATEGORY)),$(MAKE) run SKIP_JUDGE=1 CATEGORY=adversarial)

judge: ## Judge only, on answers already in EXPERIMENT_DIR
	$(call require,EXPERIMENT_DIR)
	$(MAKE) run SKIP_JUDGE= OVERWRITE=

## Rejudge: same answers, another judge, new folder

rejudge: ## Rescore SOURCE_DIR with RAG_METRICS_JUDGE_MODEL into a new folder
	$(call require,SOURCE_DIR)
	RAG_METRICS_STRATEGY=$(STRATEGY) \
	RAG_METRICS_CORPUS_DATASET=$(DATASET) \
	RAG_METRICS_JUDGE_SOURCE_DIR=$(SOURCE_DIR) \
	RAG_METRICS_EXPERIMENT_DIR=$(EXPERIMENT_DIR) \
	rag-metrics rejudge

judge-prompts: ## Save the exact Ragas judge prompts for RANGE of SOURCE_DIR (no LLM call)
	$(call require,SOURCE_DIR)
	RAG_METRICS_STRATEGY=$(STRATEGY) \
	RAG_METRICS_CORPUS_DATASET=$(DATASET) \
	RAG_METRICS_JUDGE_SOURCE_DIR=$(SOURCE_DIR) \
	RAG_METRICS_QUESTIONS_RANGE=$(RANGE) \
	RAG_METRICS_EXPERIMENT_DIR=$(EXPERIMENT_DIR) \
	rag-metrics judge-prompts

## Utilities

# OLLAMA_VIA_DOCKER: 1 = force docker exec, 0 = force host CLI, unset = auto
ollama-unload: ## Unload judge, generator and embedding models from Ollama
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

format: ## ruff format + fix
	ruff format .
	ruff check --fix .

lint: ## ruff + mypy
	ruff check .
	ruff format --check .
	mypy

test: ## pytest
	pytest -q

hooks: ## Install pre-commit hooks
	pre-commit install
