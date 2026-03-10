## Goal / Objective

- **High-level aim**: Turn this fork of the LLaDA repository into a **research testbed for Contrastive Activation Addition (CAA)** on a diffusion language model.
- **Concrete objectives**:
  - Extract **behavior steering vectors** (e.g. sycophancy, corrigibility, refusal) from LLaDA’s residual stream using unmasked contrastive examples.
  - Inject these vectors into LLaDA’s **diffusion sampling loop** at chosen layers and token positions to modulate behaviors at inference time.
  - Evaluate the steering’s effect on:
    - Multiple-choice behavioral benchmarks.
    - Open-ended generation.
    - General capabilities (e.g. MMLU) to check for degradation.

---

## Key decisions

- **Model & environment**
  - Use `GSAI-ML/LLaDA-8B-Instruct` (and optionally `LLaDA-8B-Base`) via Hugging Face with `trust_remote_code=True`.
  - Create a dedicated conda env (`llada-steering`) and pin a **compatible stack**:
    - `torch==2.2.2`, `transformers==4.40.2`, `numpy<2`.
  - Allow CPU-only operation for development (MacBook), with **lighter sampling settings**; use GPU for full experiments.

- **Device handling in `generate.py`**
  - Auto-select `device = 'cuda'` if available, else `'cpu'`.
  - Use `torch.bfloat16` on GPU, `torch.float32` on CPU.
  - On CPU, use reduced `steps` / `gen_length` to keep smoke tests tractable; on GPU, use full settings close to the paper’s configuration.

- **CAA extraction strategy**
  - Perform **unmasked extraction at t = 0** (standard forward pass on full prompt + answer letter):
    - This is taken as the diffusion-model analogue of CAA extraction in autoregressive models.
    - Do **not** attempt to extract at noisy diffusion times (e.g. `t ≈ 0.99`) for the main method.
  - Treat the **answer token as the last token** in `positive_text` / `negative_text` contrast pairs.
  - Compute **mean-difference vectors per layer**:
    - For each pair and layer: `diff = pos_acts - neg_acts`.
    - Average diffs across all pairs to get a steering vector per layer.

- **Steering (injection) design**
  - Intervene directly in the **residual stream** of the Transformer blocks:
    - Use PyTorch forward hooks on `model.model.layers[i]` (or equivalent) to add `α · v` to hidden states.
  - Apply steering at **every denoising step** by registering hooks **before** calling LLaDA’s diffusion sampler; no changes to the sampler’s control flow.
  - Start with **response-only steering**:
    - Use token position range `[prompt_length, end)` so only generated response tokens are steered, mirroring CAA’s “post-prompt” intervention.
  - Later ablations:
    - Prompt+response steering.
    - Single-layer vs all-layer steering.
    - Optional mask-aware steering (only at `[MASK]` positions).

- **Repository orientation**
  - Rewrote `README.md` to explicitly position this fork as a **CAA-on-LLaDA** project, not just a copy of the original LLaDA repo.
  - Added `STEERING_OVERVIEW.md` and `dev-notes/session-context.md` to capture high-level design and session context.

---

## Code written or modified

### New files

- `steering/hooks.py`
  - **Purpose**: Provide reusable building blocks to access LLaDA’s internal activations.
  - Key components:
    - `ActivationStore`:
      - Dataclass holding a `Dict[str, torch.Tensor]` of activations.
      - Methods:
        - `clear()` to reset stored activations between forward passes.
        - `save_hook(name)` returning a forward hook that:
          - Handles HF outputs (tensor or tuple).
          - Stores the hidden states under `activations[name]` as a detached clone.
    - `_get_transformer_layers(model)`:
      - Introspects the HF LLaDA model to find Transformer blocks via:
        - `model.model.layers` (preferred, LLaDA-style).
        - Or fallbacks: `model.layers`, `model.transformer.layers`.
    - `register_residual_hooks(model, layer_indices, store)`:
      - Registers forward hooks on the specified layer indices.
      - On each forward pass, writes block outputs to `store.activations[f"layer_{idx}"]`.
      - Returns a list of `RemovableHandle` objects for cleanup.

- `steering/steering.py`
  - **Purpose**: Implement Residual-stream steering for LLaDA via forward hooks.
  - Key components:
    - `SteeringHook`:
      - Constructor args:
        - `steering_vector: Tensor(hidden_dim,)` – the CAA direction for a layer.
        - `multiplier: float` – `α` controlling steering strength and sign.
        - `start_position: int`, `end_position: Optional[int]` – token index interval to steer.
        - `mask_token_id: Optional[int]`, `input_ids: Optional[Tensor]`, `mask_only: bool` – for optional `[MASK]`-only steering.
      - `hook_fn(module, inputs, output)`:
        - Unpacks HF outputs (tensor or first element of tuple).
        - Builds a boolean position mask for tokens in `[start, end)` and, if requested, additionally where `input_ids == mask_token_id`.
        - Adds `α · v` to `hidden[mask]` via broadcasting.
      - `register(model, layer_idx)`:
        - Uses `_get_transformer_layers(model)` to locate the `layer_idx` block.
        - Attaches `hook_fn` as a forward hook.
        - Returns a `RemovableHandle` for later `.remove()`.

- `steering/extract_vectors.py`
  - **Purpose**: Implement CAA-style **vector extraction** from LLaDA using unmasked forward passes.
  - Key components:
    - `extract_steering_vectors_for_pairs(model, tokenizer, contrast_pairs, layer_indices=None, device="cuda") -> Dict[int, Tensor]`:
      - Moves `model` to `device` and switches to eval mode.
      - If `layer_indices` is `None`, extracts from **all backbone layers**.
      - Registers residual hooks (`register_residual_hooks`).
      - For each contrast pair:
        - Tokenizes and forwards `positive_text`:
          - Stores residual activation at the **last token** (answer) for each layer.
        - Tokenizes and forwards `negative_text`, similarly.
        - Computes `pos_acts[layer_idx] - neg_acts[layer_idx]` and appends to per-layer lists.
      - Removes hooks.
      - Averages over pairs for each layer → `steering_vectors[layer_idx]`.
    - `extract_and_save_behavior_vectors(model_name_or_path, behavior, data_dir="./data/formatted", output_dir="./vectors", device=None, layer_indices=None) -> Path`:
      - Loads tokenizer + LLaDA model for `model_name_or_path`.
      - Reads `data/formatted/{behavior}_gen.json` (list of dicts with `positive_text`, `negative_text`).
      - Calls `extract_steering_vectors_for_pairs`.
      - Saves result as `./vectors/{behavior}_vectors.pt` (dict `{layer_idx: tensor}`) and returns the path.
    - CLI entry in `if __name__ == "__main__":`:
      - Usage example:
        ```bash
        python -m steering.extract_vectors --model GSAI-ML/LLaDA-8B-Instruct --behavior sycophancy
        ```

- `STEERING_OVERVIEW.md`
  - Summary document explaining:
    - What the steering project is trying to achieve.
    - Implemented hook + extraction infrastructure.
    - Theoretical justification for mean-difference steering vectors.
    - Remaining tasks and roadmap.

- `dev-notes/session-context.md` (this file)
  - Captures the current session’s design decisions, code changes, and next steps.

### Modified files

- `generate.py`
  - Added **device-aware model loading**:
    - `device = 'cuda' if torch.cuda.is_available() else 'cpu'`.
    - `dtype = bfloat16` on CUDA, `float32` on CPU.
  - Adjusted sampling parameters based on device:
    - On GPU: `steps = 128`, `gen_length = 128`.
    - On CPU: reduced to `steps = 16`, `gen_length = 32` (or smaller as needed for quick tests).
  - Preserved the original diffusion sampling logic (`generate(...)`), only wrapping it with more practical defaults for development.

- `README.md`
  - Rewritten from the upstream LLaDA marketing / FAQ into a **steering-focused** introduction:
    - Describes the goal of applying CAA to LLaDA.
    - Outlines the pipeline: env setup → generation → extraction → steering → evaluation.

---

## Current status / what’s incomplete

**Working / implemented:**

- Environment set up with a compatible `torch`/`transformers`/`numpy` combo.
- LLaDA-8B-Instruct can be **loaded and sampled** (slowly) on CPU, with automated CPU vs GPU configs.
- Core steering infrastructure:
  - Residual activations can be **captured** at arbitrary layers via `ActivationStore` and `register_residual_hooks`.
  - Steering vectors can be **injected** into the residual stream via `SteeringHook` during forward passes.
  - A generic CAA extraction routine (`extract_steering_vectors_for_pairs`) can build per-layer steering vectors from contrastive text pairs.
  - A behavior-level wrapper (`extract_and_save_behavior_vectors`) can load data and save `*_vectors.pt`.
- Documentation:
  - `README.md`, `STEERING_OVERVIEW.md`, and this `dev-notes/session-context.md` describe the purpose, design, and remaining work.

**Not yet implemented / incomplete:**

- **Dataset pipeline**:
  - Scripts to:
    - Download or load Anthropic-style behavioral datasets (or equivalents).
    - Reformat them into the `{"positive_text", "negative_text"}` schema under `./data/formatted/`.
    - Split into **generation** (for vector extraction) and **test** (for evaluation) subsets.

- **Steered diffusion generation helper**:
  - A `steered_generation.py` (or similar) that:
    - Uses `SteeringHook` to register steering on a chosen set of layers.
    - Computes `prompt_length` and sets `start_position` appropriately.
    - Calls LLaDA’s sampling API (`generate(...)` / `diffusion_generate(...)`) and returns decoded outputs.
  - This is conceptually straightforward but not coded yet.

- **Multiple-choice behavioral evaluation**:
  - `eval_mc.py` that:
    - Loads `{behavior}_vectors.pt`.
    - Iterates over α values and layer configurations (all layers or selected ones).
    - Computes P(behavior-matching answer) on held-out contrastive test questions using LLaDA’s conditional likelihood or logits at the answer token.
  - Needs to integrate with the existing LLaDA likelihood utilities, or implement a simple logit-based scoring at the answer position.

- **Open-ended evaluation**:
  - `eval_openended.py` that:
    - Uses a small curated set of prompts per behavior.
    - Calls the steered generation function for multiple α values.
    - Optionally calls an external rater (e.g. GPT-4) or uses heuristic scoring to rate the presence of the behavior.

- **Capability checks (MMLU, etc.)**:
  - Wiring up an MMLU evaluation under **no steering**, **positive steering**, and **negative steering** to quantify capability degradation.
  - Likely via `lm-evaluation-harness` with a custom LLaDA wrapper.

- **Ablation & analysis scripts**:
  - Layer sweep: single-layer steering vs all-layer steering.
  - Prompt-only vs response-only vs prompt+response steering.
  - PCA / cosine-similarity tools that use `ActivationStore` to visualize behavior representations and validate linear separability.

---

## Next steps

Short-term (to get a full end-to-end steering demo):

- [ ] Implement data preparation utilities:
  - A script (e.g. `prepare_datasets.py`) that:
    - Reads raw behavioral datasets.
    - Produces `./data/formatted/{behavior}_gen.json` and `{behavior}_test.json` with `positive_text` / `negative_text`.
- [ ] Build a `steered_generation.py`:
  - Function `generate_with_steering(model, tokenizer, prompt, vectors, layer_indices, multiplier, ...)` that:
    - Computes `prompt_length`.
    - Registers `SteeringHook` on the chosen layers with `start_position=prompt_length`.
    - Calls `generate(...)` / `diffusion_generate(...)`.
    - Removes hooks and returns the decoded response.
- [ ] Implement a simple `eval_mc.py`:
  - Load one behavior’s vectors and test set.
  - For a few α values and a simple layer config (e.g. all layers):
    - Score behavior-matching answer probabilities.
  - Verify steering has the expected directionality (α > 0 increases behavior, α < 0 decreases it).

Medium-term (toward a paper-level results replication):

- [ ] Generalize `eval_mc.py` across all behaviors and produce full plots.
- [ ] Implement `eval_openended.py` with a rater model (or approximate scoring).
- [ ] Integrate MMLU (and optionally TruthfulQA) under steering.
- [ ] Add ablation scripts:
  - Layer sweeps.
  - Prompt vs response vs prompt+response steering.
  - Mask-only vs full-position steering.
- [ ] Add interpretability utilities (PCA, cosine similarity, visualization of token-level behavior “heatmaps”).

Once these steps are complete, the repo will support:

1. **Extraction** of behavior steering vectors from LLaDA.  
2. **Intervention** on the residual stream at arbitrary layers and token windows during diffusion sampling.  
3. **Evaluation** of steering efficacy and safety trade-offs across multiple tasks and behaviors.  
4. **Analysis** of how LLaDA internally represents alignment-relevant behaviors in a diffusion-native setting.

