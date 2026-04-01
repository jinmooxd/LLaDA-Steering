#!/usr/bin/env bash
# Run TruthfulQA (MC2), COPA (SuperGLUE), and StoryCloze on LLaDA via lm-evaluation-harness.
# Usage (from repo root):
#   bash benchmarks/run_truthfulqa_copa_storycloze.sh
#   MODEL_PATH=GSAI-ML/LLaDA-8B-Base bash benchmarks/run_truthfulqa_copa_storycloze.sh
#
# Requires: see benchmarks/README.md for pip packages and env vars.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-true}"

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Base}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MC_NUM="${MC_NUM:-128}"

# Shared model args (boolean-like flags must be JSON: False not false)
COMMON_ARGS="model_path='${MODEL_PATH}',is_check_greedy=False,mc_num=${MC_NUM}"

echo "==> TruthfulQA (MC2), cfg=2.0 (matches eval_llada_lm_eval.sh)"
accelerate launch eval_llada.py \
  --tasks truthfulqa_mc2 \
  --num_fewshot 0 \
  --model llada_dist \
  --batch_size "${BATCH_SIZE}" \
  --model_args "${COMMON_ARGS},cfg=2.0"

echo "==> COPA (SuperGLUE), cfg=0.5"
accelerate launch eval_llada.py \
  --tasks copa \
  --num_fewshot 0 \
  --model llada_dist \
  --batch_size "${BATCH_SIZE}" \
  --model_args "${COMMON_ARGS},cfg=0.5"

echo "==> StoryCloze 2016, cfg=0.5"
accelerate launch eval_llada.py \
  --tasks storycloze_2016 \
  --num_fewshot 0 \
  --model llada_dist \
  --batch_size "${BATCH_SIZE}" \
  --model_args "${COMMON_ARGS},cfg=0.5"

echo "Done. For StoryCloze 2018 instead, run with --tasks storycloze_2018"
