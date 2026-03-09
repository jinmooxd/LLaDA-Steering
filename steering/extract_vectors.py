"""Extraction of CAA-style steering vectors from LLaDA.

This module implements the *extraction* side of Contrastive Activation Addition
for LLaDA. Given contrastive prompt pairs of the form:

- (question, behavior‑matching answer letter), and
- (question, behavior‑opposite answer letter),

we run LLaDA in a **single unmasked forward pass** for each completion, read
out residual stream activations at the answer token, and compute **mean
difference vectors** per layer:

.. math::

    v_L = \\mathbb{E}_{(p, c^+, c^-)}[a_L(p, c^+) - a_L(p, c^-)]

where:

- ``L`` is a Transformer layer index,
- ``a_L(p, c)`` is the residual activation at layer ``L`` at the position of
  the answer token when we feed prompt ``p`` with completion ``c``.

These per‑layer vectors are the diffusion‑model analogue of the steering
vectors used in the original CAA paper on Llama 2 and in subsequent
replications on other autoregressive models.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from .hooks import ActivationStore, register_residual_hooks, _get_transformer_layers


def extract_steering_vectors_for_pairs(
    model: torch.nn.Module,
    tokenizer,
    contrast_pairs: List[dict],
    layer_indices: Optional[List[int]] = None,
    device: str = "cuda",
) -> Dict[int, torch.Tensor]:
    """Extract CAA-style steering vectors from contrastive prompt pairs.

    Parameters
    ----------
    model:
        The instantiated LLaDA model (already loaded, but not necessarily on
        the right device yet). This function will call ``model.to(device)``
        and switch it to ``eval()`` mode.
    tokenizer:
        The tokenizer corresponding to the model (e.g. from
        :func:`AutoTokenizer.from_pretrained`).
    contrast_pairs:
        A list of dictionaries, each representing one contrastive example.
        Each entry **must** contain:

        - ``"positive_text"`` – full prompt + answer for the behavior‑matching
          completion.
        - ``"negative_text"`` – full prompt + answer for the behavior‑opposite
          completion.

        The answer token is assumed to be the **last token** in each of these
        texts (as in CAA’s multiple‑choice A/B formatting).
    layer_indices:
        Optional list of Transformer layer indices from which to extract
        vectors. If ``None``, we extract from **all** layers in the model.
    device:
        Device string (``"cuda"`` or ``"cpu"``) on which to run extraction.

    Returns
    -------
    dict
        A mapping ``{layer_index: steering_vector}`` where each steering vector
        is a 1‑D tensor of shape ``(hidden_dim,)`` representing the mean
        difference direction for that layer.

    Extraction procedure
    --------------------

    For each contrast pair:

    1. Run an unmasked forward pass on ``positive_text`` and capture
       residual activations at the answer token position.
    2. Run an unmasked forward pass on ``negative_text`` and do the same.
    3. Compute the per‑layer difference ``pos - neg`` and accumulate this
       vector into a list for each layer.

    After processing all pairs, we take the **mean** of the accumulated
    differences per layer, yielding one steering vector per layer.
    """

    model.eval()
    model.to(device)

    # Determine which layers to use if not specified.
    if layer_indices is None:
        layers = _get_transformer_layers(model)
        layer_indices = list(range(len(layers)))

    store = ActivationStore()
    hooks = register_residual_hooks(model, layer_indices, store)

    # Accumulate per-layer differences.
    diffs: Dict[int, List[torch.Tensor]] = {idx: [] for idx in layer_indices}

    with torch.no_grad():
        for pair in contrast_pairs:
            # Positive example
            pos_text = pair["positive_text"]
            pos_inputs = tokenizer(pos_text, return_tensors="pt", padding=False).to(device)
            pos_ids = pos_inputs["input_ids"]

            store.clear()
            _ = model(pos_ids)
            answer_pos = pos_ids.shape[1] - 1
            pos_acts = {
                idx: store.activations[f"layer_{idx}"][0, answer_pos, :].clone()
                for idx in layer_indices
            }

            # Negative example
            neg_text = pair["negative_text"]
            neg_inputs = tokenizer(neg_text, return_tensors="pt", padding=False).to(device)
            neg_ids = neg_inputs["input_ids"]

            store.clear()
            _ = model(neg_ids)
            # Answer token is again the last token.
            neg_answer_pos = neg_ids.shape[1] - 1
            neg_acts = {
                idx: store.activations[f"layer_{idx}"][0, neg_answer_pos, :].clone()
                for idx in layer_indices
            }

            # Accumulate (pos - neg) for this pair at each layer.
            for idx in layer_indices:
                diffs[idx].append(pos_acts[idx] - neg_acts[idx])

    # Clean up hooks.
    for h in hooks:
        h.remove()

    # Average over pairs to get mean-difference vectors.
    steering_vectors: Dict[int, torch.Tensor] = {}
    for idx in layer_indices:
        if not diffs[idx]:
            continue
        stacked = torch.stack(diffs[idx], dim=0)  # (num_pairs, hidden_dim)
        steering_vectors[idx] = stacked.mean(dim=0)  # (hidden_dim,)

    return steering_vectors


def extract_and_save_behavior_vectors(
    model_name_or_path: str,
    behavior: str,
    data_dir: str = "./data/formatted",
    output_dir: str = "./vectors",
    device: Optional[str] = None,
    layer_indices: Optional[List[int]] = None,
) -> Path:
    """Load a LLaDA model and extract steering vectors for a single behavior.

    This is a convenience wrapper around
    :func:`extract_steering_vectors_for_pairs` that handles model and tokenizer
    loading, reading contrast pairs from disk, and saving the resulting
    steering vectors to a ``.pt`` file for later reuse.

    Parameters
    ----------
    model_name_or_path:
        Hugging Face model identifier or a local path pointing to a LLaDA
        checkpoint (e.g. ``"GSAI-ML/LLaDA-8B-Instruct"``).
    behavior:
        Name of the behavior whose contrastive dataset should be used. This is
        used to construct the input file path
        ``{data_dir}/{behavior}_gen.json``.
    data_dir:
        Directory containing the formatted generation split for this behavior.
        The expected JSON file must contain a list of objects, each with at
        least ``"positive_text"`` and ``"negative_text"`` fields.
    output_dir:
        Directory where the steering vectors will be saved as
        ``{behavior}_vectors.pt``.
    device:
        Optional device override (``"cuda"`` or ``"cpu"``). If ``None``, the
        function picks ``"cuda"`` when available, otherwise ``"cpu"``.
    layer_indices:
        Optional list of layer indices to extract from. If ``None``, the code
        extracts steering vectors for **all** layers discovered by
        ``_get_transformer_layers``.

    Returns
    -------
    pathlib.Path
        The filesystem path to the saved ``.pt`` file, which contains a dict
        mapping ``layer_index`` to the corresponding steering vector tensor.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    # Load generation pairs for this behavior.
    data_path = Path(data_dir) / f"{behavior}_gen.json"
    with data_path.open() as f:
        contrast_pairs = json.load(f)

    vectors = extract_steering_vectors_for_pairs(
        model=model,
        tokenizer=tokenizer,
        contrast_pairs=contrast_pairs,
        layer_indices=layer_indices,
        device=device,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / f"{behavior}_vectors.pt"
    torch.save(vectors, save_path)

    return save_path


if __name__ == "__main__":
    # Example CLI usage:
    #
    #   python -m steering.extract_vectors \
    #       --model GSAI-ML/LLaDA-8B-Instruct \
    #       --behavior sycophancy
    #
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Extract Contrastive Activation Addition (CAA) steering vectors "
            "for a given behavior from a LLaDA model."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Hugging Face model name or local path.",
    )
    parser.add_argument(
        "--behavior",
        type=str,
        required=True,
        help="Behavior name (expects {behavior}_gen.json under data/formatted).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/formatted",
        help="Directory containing formatted contrast pairs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./vectors",
        help="Directory to save steering vectors (.pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu). Defaults to cuda if available.",
    )
    args = parser.parse_args()

    save_path = extract_and_save_behavior_vectors(
        model_name_or_path=args.model,
        behavior=args.behavior,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(f"Saved steering vectors for behavior '{args.behavior}' to: {save_path}")

