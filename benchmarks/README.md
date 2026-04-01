# Standard benchmarks: TruthfulQA, COPA, StoryCloze

This folder documents how to run **TruthfulQA**, **COPA** (SuperGLUE), and **StoryCloze** on LLaDA using the same stack as the upstream repo: **EleutherAI lm-evaluation-harness** with the custom model `llada_dist` defined in [`eval_llada.py`](../eval_llada.py).

You do **not** need a custom contrastive dataset: these tasks ship with `lm_eval` and are loaded from Hugging Face Datasets.

## Requirements

Install the same versions as [`eval_llada_lm_eval.sh`](../eval_llada_lm_eval.sh) (from the repository root):

```bash
pip install transformers==4.49.0 lm_eval==0.4.8 accelerate==0.34.2
pip install antlr4-python3-runtime==4.11 math_verify sympy hf_xet
```

Use a **conda env** (or `python -m pip`) so packages install into that env, not system Python.

If your steering development env pins an older `transformers` (e.g. 4.40.x for Mac CPU), create a **separate conda env** for `lm_eval` benchmarks so versions match the upstream evaluation script and Hugging Face remote code for LLaDA loads cleanly.

```bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
```

## Task names (lm_eval 0.4.8)

| Benchmark   | `lm_eval` task name   | Notes |
|------------|------------------------|--------|
| TruthfulQA | `truthfulqa_mc2`       | Multiple-choice; matches the setting in `eval_llada_lm_eval.sh`. |
| COPA       | `copa`                 | SuperGLUE COPA (English), defined under `lm_eval/tasks/super_glue/copa/`. |
| StoryCloze | `storycloze_2016`    | LSD2016 split; use `storycloze_2018` if you prefer the 2018 corpus. |

To list or verify tasks after install:

```bash
python -m lm_eval --tasks list | grep -Ei 'truthful|copa|storycloze'
```

## How evaluation works (LLaDA-8B-Base)

Per [`EVAL.md`](../EVAL.md), **LLaDA-8B-Base** uses **conditional likelihood** (not pure generation) for these-style multiple-choice tasks. The harness calls `LLaDAEvalHarness.loglikelihood`, which uses masked forward passes and Monte Carlo estimation (`mc_num`) as in the paper.

**LLaDA-8B-Instruct** is often evaluated with generation-only tooling (e.g. OpenCompass). For apples-to-apples comparison with the paper’s Table 1, use **Base** for `lm_eval` MC benchmarks unless you add a separate gen-based pipeline.

## Run all three (recommended)

From the **repository root**:

```bash
bash benchmarks/run_truthfulqa_copa_storycloze.sh
```

Override model path:

```bash
MODEL_PATH=GSAI-ML/LLaDA-8B-Base bash benchmarks/run_truthfulqa_copa_storycloze.sh
```

## Run a single task

See the commands inside `run_truthfulqa_copa_storycloze.sh`, or run manually, e.g.:

```bash
cd /path/to/LLaDA-Steering
accelerate launch eval_llada.py \
  --tasks copa \
  --num_fewshot 0 \
  --model llada_dist \
  --batch_size 8 \
  --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128
```

## Classifier-free guidance (`cfg`)

Upstream uses different `cfg` values per task (see [`eval_llada_lm_eval.sh`](../eval_llada_lm_eval.sh)):

- **TruthfulQA**: `cfg=2.0`
- **COPA / StoryCloze** (and similar MC): `cfg=0.5` in line with ARC / Hellaswag-style runs

The bundled script runs **one job per benchmark** so each can keep its own `cfg`.

## CAA steering (same benchmarks)

`eval_llada.LLaDAEvalHarness` can apply **CAA residual steering** on every forward pass used for **log-likelihood / multiple-choice** evaluation (`get_logits` → MC likelihood). That matches how you would compare baseline vs steered on TruthfulQA, COPA, and StoryCloze without changing task code.

**1. Extract vectors** (contrastive pairs → `.pt` dict `{layer_idx: tensor}`), e.g.:

```bash
python -m steering.extract_vectors --behavior sycophancy --output vectors/sycophancy_vectors.pt
```

**2. Run the same three benchmarks with steering** (from repo root so `import steering` resolves):

```bash
export STEERING_VECTORS_PATH=./vectors/sycophancy_vectors.pt
export STEERING_MULTIPLIER=1.0    # optional; use negative to subtract the vector
export STEERING_LAYERS=all        # optional; or e.g. 10,11,12
bash benchmarks/run_truthfulqa_copa_storycloze_steered.sh
```

Optional overrides: `MODEL_PATH`, `BATCH_SIZE`, `MC_NUM`, `STEERING_START_POSITION` (default `0` steers the full sequence for MC likelihood).

**3. Compare** to baseline metrics from `bash benchmarks/run_truthfulqa_copa_storycloze.sh` (same `cfg` and task settings per script).

You can also pass steering via `model_args` on `llada_dist` (`steering_vectors_path=...`, `steering_multiplier=...`, `steering_layers=...`) if you avoid commas inside paths; otherwise prefer **environment variables** as above.

**Note:** Steering is applied on the **likelihood** path only. Generation-heavy benchmarks that use `generate_until` are not steered unless extended similarly.
