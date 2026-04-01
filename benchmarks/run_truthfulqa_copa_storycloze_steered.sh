#!/usr/bin/env bash
# Same three benchmarks as run_truthfulqa_copa_storycloze.sh, but with CAA steering
# applied on every forward pass during log-likelihood evaluation (see eval_llada.py).
#
# Required env:
#   STEERING_VECTORS_PATH — path to a .pt dict {layer_idx: tensor} from steering/extract_vectors.py
# Optional env:
#   STEERING_MULTIPLIER   (default 1.0; use negative to subtract the behavior vector)
#   STEERING_LAYERS       (default "all"; or e.g. "10,11,12")
#   STEERING_START_POSITION (default 0 = steer full sequence; matches CAA MC-style eval)
#   MODEL_PATH
#
# Usage (from repo root):
#   export STEERING_VECTORS_PATH=./vectors/sycophancy_vectors.pt
#   bash benchmarks/run_truthfulqa_copa_storycloze_steered.sh
#
# Steering is read from env inside LLaDAEvalHarness (avoids comma/quoting issues in model_args).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${STEERING_VECTORS_PATH:-}" ]]; then
  echo "ERROR: Set STEERING_VECTORS_PATH to your CAA vectors .pt file (e.g. ./vectors/sycophancy_vectors.pt)" >&2
  exit 1
fi

export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-true}"

export STEERING_MULTIPLIER="${STEERING_MULTIPLIER:-1.0}"
export STEERING_LAYERS="${STEERING_LAYERS:-all}"

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Base}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MC_NUM="${MC_NUM:-128}"

COMMON_ARGS="model_path='${MODEL_PATH}',is_check_greedy=False,mc_num=${MC_NUM}"

echo "==> TruthfulQA (MC2) WITH steering, cfg=2.0"
accelerate launch eval_llada.py \
  --tasks truthfulqa_mc2 \
  --num_fewshot 0 \
  --model llada_dist \
  --batch_size "${BATCH_SIZE}" \
  --model_args "${COMMON_ARGS},cfg=2.0"

echo "==> COPA WITH steering, cfg=0.5"
accelerate launch eval_llada.py \
  --tasks copa \
  --num_fewshot 0 \
  --model llada_dist \
  --batch_size "${BATCH_SIZE}" \
  --model_args "${COMMON_ARGS},cfg=0.5"

echo "==> StoryCloze 2016 WITH steering, cfg=0.5"
accelerate launch eval_llada.py \
  --tasks storycloze_2016 \
  --num_fewshot 0 \
  --model llada_dist \
  --batch_size "${BATCH_SIZE}" \
  --model_args "${COMMON_ARGS},cfg=0.5"

echo "Done (steered runs). Compare to baseline: bash benchmarks/run_truthfulqa_copa_storycloze.sh"
