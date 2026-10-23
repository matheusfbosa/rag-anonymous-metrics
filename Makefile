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
SOURCE_RUN_ID ?=
STRATEGY ?= offline

comma := ,
env_value = $(shell sed -n 's/^$(1)=//p' .env 2>/dev/null)
env_list = $(strip $(subst $(comma), ,$(1)))

RAG_METRICS_GENERATORS ?= $(call env_value,RAG_METRICS_GENERATORS)
RAG_METRICS_JUDGES ?= $(call env_value,RAG_METRICS_JUDGES)
RAG_METRICS_PRIVACY_PROMPT_GENERATORS ?= $(call env_value,RAG_METRICS_PRIVACY_PROMPT_GENERATORS)
GENERATORS := $(call env_list,$(RAG_METRICS_GENERATORS))
JUDGES := $(call env_list,$(RAG_METRICS_JUDGES))
PRIVACY_PROMPT_GENERATORS := $(call env_list,$(RAG_METRICS_PRIVACY_PROMPT_GENERATORS))
RUN_ID := $(or $(RUN_ID),$(shell date +%Y%m%d_%H%M%S))
GEN ?=
JUDGE ?=
PRIVACY_PROMPT ?= 0

RUN_ENV = RAG_METRICS_STRATEGY=$(STRATEGY) \
	RAG_METRICS_CORPUS_DATASET=$(DATASET) \
	RAG_METRICS_QUESTIONS_CATEGORY=$(CATEGORY) \
	RAG_METRICS_QUESTIONS_RANGE=$(RANGE) \
	RAG_METRICS_EXPERIMENT_DIR=$(EXPERIMENT_DIR) \
	RAG_METRICS_OVERWRITE=$(OVERWRITE) \
	RAG_METRICS_SKIP_JUDGE=$(SKIP_JUDGE)

STAMP := $(or $(EXPERIMENT_DIR),$(shell date +%Y%m%d_%H%M%S))

require = @test -n "$($(1))" || { echo "$@: set $(1)=<run folder under data/output>" >&2; exit 1; }
slug = $(subst :,-,$(1))
run_dir = $(RUN_ID)_$(call slug,$(GEN))_pp$(PRIVACY_PROMPT)_$(call slug,$(1))
MODEL_ENV = RAG_ANON_LLM_MODEL=$(GEN) RAG_ANON_PRIVACY_PROMPT=$(PRIVACY_PROMPT)

.DEFAULT_GOAL := help
.PHONY: help format lint test hooks ingest check-judge check-judge-smoke \
	run-experiments run-model \
	run-strategies generate-strategies judge-strategies run-smoke-strategies run-all \
	run generate judge rejudge judge-prompts ollama-unload \
	reanonymize reanonymize-model reanonymize-experiments refresh-refusals

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

## Experiments: generators × judges × privacy prompt

run-experiments: ## All runs: RAG_METRICS_GENERATORS with prompt off, RAG_METRICS_PRIVACY_PROMPT_GENERATORS with prompt on, each judged by every RAG_METRICS_JUDGES model; resume with RUN_ID
	@test -n "$(GENERATORS)" || { echo "$@: set RAG_METRICS_GENERATORS in .env (comma-separated)" >&2; exit 1; }
	@test -n "$(JUDGES)" || { echo "$@: set RAG_METRICS_JUDGES in .env (comma-separated)" >&2; exit 1; }
	@echo "run-experiments: RUN_ID=$(RUN_ID) DATASET=$(DATASET)"
	@echo "  generators (prompt off): $(GENERATORS)"
	@echo "  generators (prompt on):  $(PRIVACY_PROMPT_GENERATORS)"
	@echo "  judges:                  $(JUDGES)"
	@failed=""; \
	for g in $(GENERATORS); do \
	  $(MAKE) run-model RUN_ID=$(RUN_ID) GEN=$$g PRIVACY_PROMPT=0 || failed="$$failed $$g:pp0"; \
	done; \
	for g in $(PRIVACY_PROMPT_GENERATORS); do \
	  $(MAKE) run-model RUN_ID=$(RUN_ID) GEN=$$g PRIVACY_PROMPT=1 || failed="$$failed $$g:pp1"; \
	done; \
	if [ -n "$$failed" ]; then \
	  echo "run-experiments: failed:$$failed — rerun with RUN_ID=$(RUN_ID) to resume" >&2; exit 1; \
	fi; \
	echo "run-experiments: done RUN_ID=$(RUN_ID)"

run-model: ## One generator (GEN, PRIVACY_PROMPT=0|1): generate once, judge with the first of RAG_METRICS_JUDGES, rejudge with the others
	@test -n "$(GEN)" || { echo "$@: set GEN=<generator tag>" >&2; exit 1; }
	@test -n "$(JUDGES)" || { echo "$@: set RAG_METRICS_JUDGES in .env (comma-separated)" >&2; exit 1; }
	$(MODEL_ENV) RAG_METRICS_JUDGE_MODEL=$(firstword $(JUDGES)) \
		$(MAKE) generate-strategies EXPERIMENT_DIR=$(call run_dir,$(firstword $(JUDGES)))
	$(MODEL_ENV) RAG_METRICS_JUDGE_MODEL=$(firstword $(JUDGES)) $(MAKE) ollama-unload
	$(MODEL_ENV) RAG_METRICS_JUDGE_MODEL=$(firstword $(JUDGES)) \
		$(MAKE) judge-strategies EXPERIMENT_DIR=$(call run_dir,$(firstword $(JUDGES)))
	$(foreach judge,$(wordlist 2,$(words $(JUDGES)),$(JUDGES)),\
		$(MODEL_ENV) RAG_METRICS_JUDGE_MODEL=$(firstword $(JUDGES)) $(MAKE) ollama-unload && \
		$(MODEL_ENV) RAG_METRICS_JUDGE_MODEL=$(judge) \
		$(MAKE) rejudge SOURCE_DIR=$(call run_dir,$(firstword $(JUDGES))) EXPERIMENT_DIR=$(call run_dir,$(judge)) &&) true
	$(MODEL_ENV) RAG_METRICS_JUDGE_MODEL=$(lastword $(JUDGES)) $(MAKE) ollama-unload

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

## Reanonymize: same generations, On-Demand answers anonymized again

reanonymize: ## Reanonymize the On-Demand answers of SOURCE_DIR into a new folder; judge (RAG_METRICS_JUDGE_MODEL) only answers whose text changed
	$(call require,SOURCE_DIR)
	RAG_METRICS_STRATEGY=$(STRATEGY) \
	RAG_METRICS_CORPUS_DATASET=$(DATASET) \
	RAG_METRICS_JUDGE_SOURCE_DIR=$(SOURCE_DIR) \
	RAG_METRICS_EXPERIMENT_DIR=$(EXPERIMENT_DIR) \
	rag-metrics reanonymize

reanonymize-model: ## One folder: reanonymize SOURCE_RUN_ID's (GEN, PRIVACY_PROMPT, JUDGE) into RUN_ID
	@test -n "$(SOURCE_RUN_ID)" || { echo "$@: set SOURCE_RUN_ID=<RUN_ID of the source folders>" >&2; exit 1; }
	@test -n "$(GEN)" || { echo "$@: set GEN=<generator tag>" >&2; exit 1; }
	@test -n "$(JUDGE)" || { echo "$@: set JUDGE=<judge tag>" >&2; exit 1; }
	RAG_METRICS_JUDGE_MODEL=$(JUDGE) $(MAKE) reanonymize \
		SOURCE_DIR=$(SOURCE_RUN_ID)_$(call slug,$(GEN))_pp$(PRIVACY_PROMPT)_$(call slug,$(JUDGE)) \
		EXPERIMENT_DIR=$(call run_dir,$(JUDGE))

reanonymize-experiments: ## Every folder of SOURCE_RUN_ID (same generators, prompts and judges as run-experiments) into RUN_ID; resume with RUN_ID
	@test -n "$(SOURCE_RUN_ID)" || { echo "$@: set SOURCE_RUN_ID=<RUN_ID of the source folders>" >&2; exit 1; }
	@test -n "$(GENERATORS)" || { echo "$@: set RAG_METRICS_GENERATORS in .env (comma-separated)" >&2; exit 1; }
	@test -n "$(JUDGES)" || { echo "$@: set RAG_METRICS_JUDGES in .env (comma-separated)" >&2; exit 1; }
	@echo "reanonymize-experiments: SOURCE_RUN_ID=$(SOURCE_RUN_ID) RUN_ID=$(RUN_ID) DATASET=$(DATASET)"
	@failed=""; \
	for j in $(JUDGES); do \
	  for g in $(GENERATORS); do \
	    $(MAKE) reanonymize-model RUN_ID=$(RUN_ID) GEN=$$g PRIVACY_PROMPT=0 JUDGE=$$j || failed="$$failed $$g:pp0:$$j"; \
	  done; \
	  for g in $(PRIVACY_PROMPT_GENERATORS); do \
	    $(MAKE) reanonymize-model RUN_ID=$(RUN_ID) GEN=$$g PRIVACY_PROMPT=1 JUDGE=$$j || failed="$$failed $$g:pp1:$$j"; \
	  done; \
	  RAG_METRICS_JUDGE_MODEL=$$j $(MAKE) ollama-unload; \
	done; \
	if [ -n "$$failed" ]; then \
	  echo "reanonymize-experiments: failed:$$failed — rerun with RUN_ID=$(RUN_ID) to resume" >&2; exit 1; \
	fi; \
	echo "reanonymize-experiments: done RUN_ID=$(RUN_ID)"

refresh-refusals: ## Reapply the refusal detector to every results_*.jsonl in EXPERIMENT_DIR (in place, no LLM call)
	$(call require,EXPERIMENT_DIR)
	RAG_METRICS_STRATEGY=$(STRATEGY) \
	RAG_METRICS_CORPUS_DATASET=$(DATASET) \
	RAG_METRICS_EXPERIMENT_DIR=$(EXPERIMENT_DIR) \
	rag-metrics refresh-refusals

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
	@_judge="$$RAG_METRICS_JUDGE_MODEL"; _llm="$$RAG_ANON_LLM_MODEL"; _embed="$$RAG_ANON_EMBEDDING_MODEL"; \
	set -a; [ -f .env ] && . ./.env; set +a; \
	RAG_METRICS_JUDGE_MODEL="$${_judge:-$$RAG_METRICS_JUDGE_MODEL}"; \
	RAG_ANON_LLM_MODEL="$${_llm:-$$RAG_ANON_LLM_MODEL}"; \
	RAG_ANON_EMBEDDING_MODEL="$${_embed:-$$RAG_ANON_EMBEDDING_MODEL}"; \
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
