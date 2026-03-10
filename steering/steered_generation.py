"""Helpers for running LLaDA diffusion sampling with CAA steering vectors.

This module provides a thin wrapper around the base ``generate`` function
defined in ``generate.py``. It wires in :class:`SteeringHook` so that
pre‑computed behavior vectors are added to the residual stream during every
denoising step.

The intended usage mirrors the CAA paper:

- Extract per‑layer steering vectors with ``steering.extract_vectors``.
- Load those vectors from disk (``{behavior}_vectors.pt``).
- Call ``generate_with_steering`` with:
  - a loaded LLaDA model and tokenizer,
  - a list of prompts,
  - the steering vectors dict,
  - a choice of layers and multiplier ``alpha``.
"""

from typing import Dict, Iterable, List, Optional

import torch

from transformers import AutoModel, AutoTokenizer

from .steering import SteeringHook
from .hooks import _get_transformer_layers
from generate import generate as diffusion_generate


def generate_with_steering(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: Iterable[str],
    vectors: Dict[int, torch.Tensor],
    layer_indices: Optional[Iterable[int]] = None,
    multiplier: float = 1.0,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    mask_id: int = 126336,
) -> List[str]:
    """Run LLaDA generation with CAA steering applied at selected layers.

    Parameters
    ----------
    model:
        A LLaDA mask‑predictor model (e.g. ``AutoModel.from_pretrained(...)``)
        already moved to the appropriate device and set to ``eval()``.
    tokenizer:
        The corresponding tokenizer. For Instruct checkpoints we assume the
        standard chat template interface and will left‑pad inputs.
    prompts:
        Iterable of user prompts (plain text) to steer.
    vectors:
        Dict mapping ``layer_index`` to a 1‑D steering vector tensor of shape
        ``(hidden_dim,)``. Typically loaded from
        ``torch.load('./vectors/{behavior}_vectors.pt')``.
    layer_indices:
        Optional iterable of layer indices to steer on. If ``None``, we steer on
        all layers that appear in ``vectors`` (intersection with the model's
        layers).
    multiplier:
        Scalar ``α`` controlling direction and strength of steering. Positive
        values amplify the behavior; negative values suppress it.
    steps, gen_length, block_length, temperature, cfg_scale, mask_id:
        Passed through to the underlying diffusion ``generate`` function.

    Returns
    -------
    list of str
        Decoded model completions, one per input prompt.
    """

    device = next(model.parameters()).device

    # Ensure tokenizer is configured the way LLaDA sampling expects.
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    assert tokenizer.pad_token_id != mask_id, (
        "LLaDA's sampling loop assumes pad_token_id != mask_id; "
        "if this is not true, the generate() function must be adapted."
    )

    # Wrap plain prompts for Instruct models using the chat template.
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    chat_prompts = [
        tokenizer.apply_chat_template(
            [message], add_generation_prompt=True, tokenize=False
        )
        for message in messages
    ]

    encoded = tokenizer(
        chat_prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Decide which layers to steer on.
    all_layers = _get_transformer_layers(model)
    max_layer_idx = len(all_layers) - 1

    if layer_indices is None:
        # Use all layers for which we actually have vectors and which exist on
        # the current backbone.
        layer_indices = sorted(
            idx for idx in vectors.keys() if 0 <= idx <= max_layer_idx
        )
    else:
        layer_indices = [idx for idx in layer_indices if 0 <= idx <= max_layer_idx]

    # Register steering hooks on the chosen layers.
    handles = []
    prompt_length = input_ids.shape[1]

    for idx in layer_indices:
        vec = vectors[idx].to(device)
        hook = SteeringHook(
            steering_vector=vec,
            multiplier=multiplier,
            start_position=prompt_length,
            end_position=None,
            mask_token_id=None,
            input_ids=None,
            mask_only=False,
        )
        handle = hook.register(model, layer_idx=idx)
        handles.append(handle)

    try:
        sampled = diffusion_generate(
            model,
            input_ids,
            attention_mask=attention_mask,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking="low_confidence",
            mask_id=mask_id,
        )
    finally:
        for h in handles:
            h.remove()

    # Decode only the generated continuation (positions after the original prompt).
    completions = tokenizer.batch_decode(
        sampled[:, prompt_length:], skip_special_tokens=True
    )
    return completions


def load_model_and_tokenizer(
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct",
    device: Optional[str] = None,
) -> tuple[AutoModel, AutoTokenizer]:
    """Convenience helper to load a LLaDA model and tokenizer for steering.

    This mirrors the loading logic in ``generate.main`` but returns the objects
    instead of running a demo.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    return model, tokenizer

