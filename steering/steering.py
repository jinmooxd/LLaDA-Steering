"""Residual-stream steering utilities for LLaDA.

This module defines :class:`SteeringHook`, a small helper that can be attached
as a PyTorch forward hook to any Transformer block in LLaDA. It implements the
core intervention used throughout the project:

.. math::

    h_{b, t, :} \\leftarrow h_{b, t, :} + \\alpha \\cdot v

where:

- ``h`` is the residual stream at a given layer,
- ``b`` is the batch index,
- ``t`` is the token position,
- ``v`` is a pre‑computed steering vector, and
- ``α`` is a scalar multiplier controlling steering strength.

We use this hook in two distinct regimes:

1. **During diffusion sampling**:
   - At every denoising step, LLaDA runs a full Transformer forward pass.
   - By registering this hook on selected layers before sampling, we ensure
     that the residual stream is nudged along the steering direction on **every
     step**.
   - Typically we set ``start_position`` to the prompt length so that only
     **response tokens** are steered, mirroring the “post‑prompt” steering in
     the original CAA work.

2. **During likelihood evaluation / ablations**:
   - We can also enable steering for *all* positions (``start_position = 0``)
     to study how steering interacts with multiple‑choice scoring and general
     capabilities (e.g. MMLU).
"""

import torch
from typing import Optional, List

from .hooks import _get_transformer_layers


class SteeringHook:
    """Residual-stream steering hook for LLaDA.

    This class encapsulates all the logic needed to add a CAA steering vector
    to the residual stream at a specific Transformer block.

    It is designed to be used as a **forward hook**:

    .. code-block:: python

        hook = SteeringHook(steering_vector=v, multiplier=alpha, start_position=prompt_len)
        handle = hook.register(model, layer_idx=13)
        # run generation / evaluation ...
        handle.remove()

    Parameters
    ----------
    steering_vector:
        A 1‑D tensor of shape ``(hidden_dim,)`` representing the behavior
        direction for this layer (e.g. a mean‑difference vector from CAA).
    multiplier:
        Scalar ``α`` controlling how strongly to steer along the vector.
        Positive values add the behavior; negative values subtract it.
    start_position:
        First token index (inclusive) at which to apply the intervention.
        For response‑only steering, this should be set to the prompt length.
    end_position:
        Optional token index (exclusive) at which to stop applying the
        intervention. If ``None``, the hook steers up to the end of the
        sequence.
    mask_token_id:
        If provided together with ``mask_only=True`` and ``input_ids``, the
        hook will restrict steering to tokens equal to this ID (e.g. the
        LLaDA ``[MASK]`` token).
    input_ids:
        Optional tensor of shape ``(batch, seq_len)`` used to construct a
        **mask‑aware** positional filter when ``mask_only=True`` is set.
    mask_only:
        If ``True``, steering is applied **only** at positions where
        ``input_ids == mask_token_id`` and within the ``[start_position,
        end_position)`` window. If ``False``, all positions in that window
        are steered.
    """

    def __init__(
        self,
        steering_vector: torch.Tensor,  # shape: (hidden_dim,)
        multiplier: float = 1.0,
        start_position: int = 0,
        end_position: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        input_ids: Optional[torch.Tensor] = None,  # shape: (batch, seq_len)
        mask_only: bool = False,
    ) -> None:
        self.steering_vector = steering_vector
        self.multiplier = multiplier
        self.start_position = start_position
        self.end_position = end_position
        self.mask_token_id = mask_token_id
        self.input_ids = input_ids
        self.mask_only = mask_only

    def hook_fn(self, module, inputs, output):
        """Forward-hook implementation that applies the steering vector.

        This method matches the PyTorch forward‑hook signature and is called
        automatically by PyTorch every time the attached layer runs.

        It performs three steps:

        1. **Unpack the hidden states** from Hugging Face style outputs
           (either a tensor or a tuple whose first element is the tensor).
        2. **Construct a token‑level mask** based on ``start_position``,
           ``end_position``, and optionally ``input_ids`` / ``mask_token_id``.
        3. **Apply the intervention** by adding ``α · v`` to all masked
           positions in the residual stream.

        The returned object has the same structure as ``output`` so that the
        rest of the model remains unaware of the intervention.
        """
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        # hidden: (batch, seq_len, hidden_dim)
        bsz, seqlen, hdim = hidden.shape

        start = self.start_position
        end = self.end_position if self.end_position is not None else seqlen
        start = max(0, min(start, seqlen))
        end = max(start, min(end, seqlen))

        # Broadcast steering vector to hidden size and scale it.
        vec = self.steering_vector.to(hidden.device, hidden.dtype)  # (hidden_dim,)
        vec = self.multiplier * vec  # (hidden_dim,)

        # Base positional mask: all positions in [start, end) are candidates.
        pos_mask = torch.zeros((bsz, seqlen), dtype=torch.bool, device=hidden.device)
        if start < end:
            pos_mask[:, start:end] = True

        # Optionally restrict to [MASK] tokens only.
        if self.mask_only and self.input_ids is not None and self.mask_token_id is not None:
            mask_positions = self.input_ids.to(hidden.device) == self.mask_token_id
            pos_mask = pos_mask & mask_positions

        if pos_mask.any():
            pos_mask_3d = pos_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            hidden = hidden + pos_mask_3d * vec  # broadcast add

        if rest is not None:
            return (hidden,) + rest
        return hidden

    def register(self, model: torch.nn.Module, layer_idx: int):
        """Attach this hook to a specific Transformer block in LLaDA.

        Parameters
        ----------
        model:
            The instantiated LLaDA model (Hugging Face ``AutoModel`` with
            ``trust_remote_code=True``).
        layer_idx:
            Index of the Transformer block to steer (0‑based). For LLaDA‑8B
            there are usually 32 layers, indexed from 0 to 31.

        Returns
        -------
        torch.utils.hooks.RemovableHandle
            A handle that you should keep and later call ``handle.remove()``
            on once you are done steering. Forgetting to remove hooks can lead
            to accumulating interventions across experiments and to memory
            leaks.
        """
        layers: List[torch.nn.Module] = _get_transformer_layers(model)
        layer = layers[layer_idx]
        handle = layer.register_forward_hook(self.hook_fn)
        return handle

