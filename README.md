## Steering LLaDA with Contrastive Activation Addition

This repository is a **research fork of LLaDA** (Large Language Diffusion Models) focused on a single goal:

**Implement and study activation-space steering (Contrastive Activation Addition, CAA) for diffusion-based language models.**

Concretely, we adapt the CAA method from *“Steering Llama 2 via Contrastive Activation Addition”* to the diffusion language model LLaDA‑8B. The aim is to:

- **Extract linear “behavior vectors”** (e.g. sycophancy, corrigibility, refusal) from LLaDA’s residual stream.
- **Inject those vectors during LLaDA’s diffusion sampling** to increase or decrease target behaviors.
- **Evaluate steering** on multiple-choice behavioral benchmarks, open-ended generations, and capability metrics (e.g. MMLU).

The key question this project explores is:

> *Do the same kind of linear control knobs (CAA steering vectors) that work for autoregressive LLMs also emerge in diffusion-based language models like LLaDA?*

---

### Overview of the base model (LLaDA)

LLaDA (**L**arge **La**nguage **D**iffusion with m**A**sking) is an 8B-parameter diffusion language model trained from scratch. It replaces left-to-right next-token prediction with a **masked diffusion process**:

- **Forward process**: randomly masks tokens in a sequence until fully masked.
- **Reverse process**: a Transformer “mask predictor” iteratively denoises the masked sequence back to text.

For details on LLaDA itself, see the original paper and models:

- **Paper**: `https://arxiv.org/abs/2502.09992`
- **Models**: `GSAI-ML/LLaDA-8B-Base`, `GSAI-ML/LLaDA-8B-Instruct` on Hugging Face

This fork keeps the original inference and evaluation code, but repurposes the repo around activation steering.

---

### What this project adds on top of LLaDA

- **Activation extraction hooks**  
  - Instrument LLaDA’s Transformer layers to capture **residual stream activations** at specified layers and token positions.
  - Support **unmasked extraction at \(t = 0\)** (full prompt + answer letter, no random masking), matching the CAA setup in autoregressive models.

- **Steering hooks for diffusion sampling**  
  - Inject steering vectors directly into the **residual stream** at chosen layers.
  - Apply vectors **at every denoising step**, restricted to **response tokens** (positions after the prompt) by default, with prompt+response steering as an ablation.

- **Contrastive behavior datasets & formatting**  
  - Use behaviorally-labeled multiple-choice datasets (e.g. Anthropic’s “Advanced AI Risk” evals) to build **contrast pairs**:
    - Same question, two answers: one exhibiting the target behavior, one its opposite.
  - Format prompts to match LLaDA’s SFT chat or plain-text style.

- **CAA-style steering vectors**  
  - For each behavior:
    - Run **positive vs negative** answer completions through LLaDA in an **unmasked forward pass**.
    - Extract residual activations at the **answer token position** across a set of contrast pairs.
    - Compute **mean difference vectors** per layer → the behavior “steering vectors”.

- **Evaluation suite (planned / in progress)**  
  - **Multiple-choice behavioral evals**: probability on behavior-matching answers with/without steering.
  - **Open-ended generations**: GPT-4-style rater prompts for sycophancy, hallucination, etc.
  - **Capability preservation**: MMLU under positive / negative steering to check that general performance is not heavily degraded.

---

### High-level steering pipeline

At a high level, the steering experiments follow this pipeline:

- **1. Load LLaDA and verify generation**  
  - Use `generate.py` to run diffusion sampling (`generate(...)`) on a few toy prompts.  
  - The script auto-selects GPU vs CPU and adjusts sampling settings accordingly.

- **2. Prepare contrastive datasets**  
  - Build JSONL/JSON files of multiple-choice questions with **A/B answers** and a label of which letter matches the target behavior.
  - Format into contrast pairs with full text prompts, e.g.:
    - question text  
    - `Choices: (A) ... (B) ...`  
    - `Answer: (A)` or `Answer: (B)`

- **3. Extract steering vectors (CAA)**  
  - Register activation hooks on a set of Transformer layers (e.g. all 32 in LLaDA‑8B).
  - For each contrast pair:
    - Run **unmasked forward passes** on the positive and negative completions.
    - Capture residual activations at the last token (the answer letter).
    - Accumulate `pos − neg` differences per layer and average over the dataset.

- **4. Steered diffusion generation**  
  - During LLaDA’s iterative sampling loop, add **α · v** to the residual stream at each chosen layer:
    - **Every denoising step**.
    - Restricted to **response positions** (tokens after the prompt) in the default configuration.
  - The scalar **multiplier α** controls steering strength; positive vs negative values increase vs decrease the behavior.

- **5. Evaluation & analysis**  
  - **Multiple-choice**: evaluate P(behavior-aligned answer) with α ∈ {−k, …, 0, …, +k}.  
  - **Open-ended**: generate answers to free-form questions under steering, score with a rater model.  
  - **Capability**: run MMLU (or a subset) under steering to quantify any degradation.  
  - **Interpretability**: PCA and cosine similarity analyses of activations vs steering vectors to check for linear representations and layer-wise structure.

---

### Environment and quickstart

- **Hardware**  
  - For full experiments, a GPU with ≥80 GB (e.g. A100) is recommended for LLaDA‑8B.  
  - On CPU (e.g. a MacBook), you can still run **small test generations** to validate hooks and extraction, but it will be slow.

- **Basic setup**

```bash
conda create -n llada-steering python=3.10 -y
conda activate llada-steering

python -m pip install "numpy<2"
python -m pip install torch==2.2.2 transformers==4.40.2 accelerate
python -m pip install datasets scipy scikit-learn matplotlib seaborn tqdm