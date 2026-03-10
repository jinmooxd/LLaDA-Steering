## Project overview

- **Overall purpose**  
  - This fork turns the original LLaDA codebase into a **testbed for activation-space steering** (Contrastive Activation Addition, CAA) on a diffusion language model.  
  - Goal: **extract behavior directions** (e.g. sycophancy, corrigibility, refusal) from LLaDA’s residual stream, **inject** them during diffusion sampling, and **measure how well we can control behavior** without retraining the model.

---

## Progress so far

- **Environment & baseline generation**
  - Created a dedicated conda env and installed a compatible **torch + transformers + numpy** stack.
  - Fixed `generate.py` to:
    - Auto-select **CPU vs GPU** and choose appropriate dtypes.
    - Use **lighter sampling settings on CPU** so we can sanity-check generation without a GPU.

- **Repository “reorientation”**
  - Rewrote the top-level `README.md` to clearly state:
    - **Scope**: steering LLaDA via CAA.
    - **High-level pipeline**: dataset → vector extraction → steering → evaluation.
    - **How to get started** (env setup, smoke-test generation).

- **Core steering infrastructure (implemented)**
  - `steering/hooks.py`
    - **`ActivationStore`**: central container for capturing per-layer residual activations via forward hooks.
    - **`_get_transformer_layers`**: abstracts away the internal layout of the LLaDA HF model and returns a list of Transformer blocks.
    - **`register_residual_hooks`**: attaches forward hooks on arbitrary layer indices and saves outputs under keys like `layer_13`.
  - `steering/steering.py`
    - **`SteeringHook`**:
      - Implements the core intervention \(h_{b,t,:} \leftarrow h_{b,t,:} + \alpha \cdot v\) at a given layer.
      - Supports:
        - **Token range selection** (`start_position`, optional `end_position`) → lets us do **response-only steering** (after the prompt).
        - **Mask-aware steering** (optional restriction to `[MASK]` tokens) for future variants.
      - Exposes `register(model, layer_idx)` that returns a removable handle, so we can plug this into **every denoising step** without touching LLaDA’s sampler logic.
  - `steering/extract_vectors.py`
    - **`extract_steering_vectors_for_pairs`**:
      - Accepts a list of contrast pairs with `positive_text` / `negative_text` (prompt + answer letter).
      - For each pair:
        - Runs an **unmasked forward pass** on both texts.
        - Uses the hooks to get residual activations at the **answer token** (last token).
        - Accumulates per-layer differences (`pos - neg`).
      - Averages across pairs → **per-layer mean difference vectors** (the CAA steering vectors).
    - **`extract_and_save_behavior_vectors`**:
      - Loads a LLaDA model + tokenizer.
      - Reads `{behavior}_gen.json` from `./data/formatted/`.
      - Calls the extractor and saves `{behavior}_vectors.pt` as `{layer_idx: tensor}`.
    - Includes a **CLI** (`python -m steering.extract_vectors --behavior sycophancy`) to run extraction behavior-by-behavior.

---

## Why the current code achieves its part (and why it should work)

- **Unmasked extraction is the right analogue of CAA for LLaDA**
  - In the original CAA for Llama 2, steering vectors are computed from a **single forward pass** over full prompt+answer sequences, at a fixed layer, with **no extra noise process**.
  - For LLaDA, calling `model(input_ids)` at **t = 0** is the exact analogue: it runs the same Transformer backbone over **fully observed text**, so the residual activations at the answer token encode the model’s internal representation of “choosing A vs B” without diffusion noise.
  - The `extract_steering_vectors_for_pairs` function mirrors this exactly:
    - Same tokenizer pipeline.
    - Same “answer token = last token” assumption.
    - Same **mean difference** aggregation per layer.

- **Why the mean-difference vectors are meaningful**
  - The CAA and representation-engineering literature (including the CAA paper and related MD/MDLM work) shows that, for many behaviors:
    - The difference between **behavior+** and **behavior−** activations at a given position is **approximately linear** in a shared direction.
    - Averaging over many contrast pairs cancels out unrelated variation and isolates a **“concept direction”** in the residual stream.
  - Our implementation:
    - Uses **hundreds of pairs** per behavior (once datasets are wired in).
    - Takes a simple average across pairs → the same **mean-difference** construction used in CAA and related work.
    - Stores these as **per-layer steering vectors** so we can:
      - Apply them at their “native” layer.
      - Or transfer them across layers (all-layer steering or single-layer ablations).

- **Why residual-stream steering hooks are the right intervention point**
  - Both CAA and concurrent work on diffusion steering find that:
    - **Residual-stream interventions dominate attention/MLP-only edits** in terms of effect size and generality.
    - Adding vectors at **all layers** (or at mid-layers) is particularly effective.
  - `SteeringHook` plugs directly into the **residual stream output** of each Transformer block:
    - The hook sees the same hidden states that CAA manipulates in autoregressive models.
    - By registering the hook **before the diffusion loop** and not touching the sampling code, we ensure:
      - Every denoising step’s forward pass is steered.
      - We can easily vary which layers and which token positions are affected without rewriting the sampler.

- **Why this is compatible with diffusion generation in LLaDA**
  - LLaDA’s `generate(...)` / `diffusion_generate(...)` pattern:
    - Iteratively calls `model(x, attention_mask=...)` across discrete time steps.
  - Our design relies completely on HF’s **forward hook mechanism**:
    - We don’t need to modify the time loop; we only register hooks once.
    - At each step, when the model calls the Transformer, hooks:
      - Capture activations (for extraction) or
      - Modify activations (for steering).
  - This makes the implementation **robust and minimally invasive**:
    - If LLaDA’s sampler evolves, as long as it still goes through the same backbone, hooks continue to work.

---

## What’s left to finish the project

- **1. Dataset plumbing and formatting**
  - Implement / finish scripts to:
    - Download or load **Anthropic-style behavioral datasets** (or your own equivalents).
    - Convert them into the `{"positive_text", "negative_text"}` format expected by `extract_vectors.py`, matching LLaDA’s prompt style (chat vs plain).
  - Split into **generation vs test** sets (e.g. N for vector extraction, last 50 for evaluation) per behavior.

- **2. Wiring steering into the diffusion sampler**
  - Build a `steered_generation.py` helper that:
    - Computes `prompt_length` from tokenized input.
    - Registers `SteeringHook` instances on:
      - All layers (for “all-layer steering”).
      - Or specific layers (for single-layer ablations).
    - Calls LLaDA’s `generate(...)` / `diffusion_generate(...)` with no internal changes.
    - Cleans up hooks afterward and returns only the **response tokens**.
  - Initially implement **response-only steering** (positions ≥ prompt length), then add a **prompt+response** variant as an ablation.

- **3. Multiple-choice behavioral evaluation**
  - Implement an `eval_mc.py` that:
    - Loads `{behavior}_vectors.pt`.
    - For each behavior and multiplier α:
      - Registers appropriate `SteeringHook`s (all layers or a subset).
      - Uses LLaDA’s conditional likelihood (or a simplified logits-based method at the answer position) to compute:
        - P(behavior-aligned answer) across held-out test questions.
    - Produces plots / tables of **P(behavior)** vs α, comparing:
      - No steering
      - Positive steering
      - Negative steering

- **4. Open-ended evaluation**
  - Implement `eval_openended.py` that:
    - Loads a small set of **open-ended prompts** per behavior.
    - Calls your `steered_generation` helper with different α values.
    - Optionally uses a rater model (e.g. GPT-4) or a simpler heuristic to assign a **behavior score** to each answer.
  - This mirrors the CAA paper’s GPT-4 evaluation and checks that steering generalizes beyond multiple-choice.

- **5. Capability preservation checks**
  - Integrate with `lm-evaluation-harness` or a custom script to:
    - Run **MMLU** (or a subset) under α ∈ {−1, 0, +1}.
    - Confirm **minimal degradation** in accuracy / log-prob, as in the original CAA paper.

- **6. Ablations and analysis**
  - **Layer sweep**:
    - Use `extract_vectors.py` vectors to:
      - Steer with a single layer at a time.
      - Identify mid-layers where behavior control is strongest.
  - **All-layer vs single-layer**:
    - Compare effect sizes and stability.
  - **Prompt-only vs response-only vs prompt+response** steering:
    - Implement via different `start_position` / `end_position` configs in `SteeringHook`.
  - **Interpretability**:
    - Use `ActivationStore` to:
      - Collect per-token activations and do PCA plots (letter vs behavior clustering).
      - Compute cosine similarities between steering vectors and token activations across text.

Once these pieces are in place, you’ll have a full pipeline that:

1. **Extracts CAA vectors** from LLaDA.  
2. **Steers** its diffusion sampling at arbitrary layers and token ranges.  
3. **Quantitatively evaluates** behavioral control and capability preservation.  
4. **Qualitatively inspects** how LLaDA’s internal representations line up with the targeted behaviors.

