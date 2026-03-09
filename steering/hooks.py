import torch
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ActivationStore:
    """
    Serves as a container for activations captured from forward hooks.

    The basic idea is that we register a forward hook on every Transformer
    block we care about. 
    
    Each hook points at ``ActivationStore.save_hook(name)``,
    where ``name`` is typically of the form ``"layer_{idx}"``. When the model
    runs, each block’s forward call deposits its output tensor into this store.

    Attributes:
    - activations:
        - A mapping from arbitrary string keys (e.g. ``"layer_13"``) to the
        corresponding activation tensor. For LLaDA, each tensor has shape
        ``(batch, seq_len, hidden_dim)`` and represents the *residual stream*
        at the output of that block.

    Typical Lifecycle:
    - Construct once at the start of an experiment.
    - Register hooks with ``register_residual_hooks``.
    - Inside an evaluation loop:
      - Call ``clear()`` before each forward pass to avoid stale entries,
      - After the forward pass, read activations from ``activations``.
    """

    activations: Dict[str, torch.Tensor] = field(default_factory=dict)

    def clear(self) -> None:
        """Remove all stored activations.

        This is important when reusing the same ``ActivationStore`` across
        multiple forward passes. Without clearing, you risk reading activations
        from a previous example or having a mixture of shapes from different
        sequence lengths under the same keys.
        """
        self.activations.clear()

    def save_hook(self, name: str):
        """
        Return a forward hook that saves the layer output under ``name``.

        The returned function conforms to the PyTorch forward‑hook API:

        .. code-block:: python

            def hook_fn(module, inputs, output): ...

        It inspects ``output`` and handles both common Hugging Face patterns:

        - If ``output`` is a tensor, we store it directly.
        - If ``output`` is a tuple, we assume the *first* element is the
          hidden states (as with ``BaseModelOutput``) and store that.

        All tensors are detached and cloned so that:

        - They do not hold onto the computational graph (no autograd overhead).
        - Subsequent in‑place operations on the model do not mutate stored
          activations.
        """

        def hook_fn(module, inputs, output):
            # Hugging Face models typically return either:
            # - a single tensor of shape (batch, seq, hidden_dim), or
            # - a tuple where the first element is that tensor.
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self.activations[name] = hidden.detach().clone()

        return hook_fn


def _get_transformer_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Return the list of Transformer layers for LLaDA.

    This helper encapsulates all the model‑structure assumptions so that the
    rest of the steering code can treat the backbone as a simple list of
    blocks. It tries a few common patterns in order:

    1. ``model.model.layers`` – current LLaDA implementation.
    2. ``model.layers`` – a direct attribute on the top‑level model.
    3. ``model.transformer.layers`` – more traditional “transformer” container.

    If none of these are found, we raise an ``AttributeError`` with a helpful
    message, so it is obvious that the internal structure of the HF model has
    changed and this function needs to be updated.
    """

    # Common LLaMA-style / LLaDA-style layout: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)

    # Fallback: layers directly on the model
    if hasattr(model, "layers"):
        return list(model.layers)

    # Fallback: nested transformer.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        return list(model.transformer.layers)

    raise AttributeError(
        "Could not find Transformer layers on the LLaDA model. "
        "Please update `_get_transformer_layers` to match the model structure."
    )


def register_residual_hooks(
    model: torch.nn.Module,
    layer_indices: List[int],
    store: ActivationStore,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Attach forward hooks to capture residual stream activations.

    Parameters
    ----------
    model:
        The instantiated LLaDA model (e.g. from ``AutoModel.from_pretrained``).
    layer_indices:
        Indices of the Transformer blocks on which to register hooks. For
        LLaDA‑8B, there are typically 32 layers indexed from 0 to 31.
    store:
        An :class:`ActivationStore` instance that will hold the captured
        activations. Each hook writes into
        ``store.activations[f"layer_{idx}"]``.

    Returns
    -------
    list of torch.utils.hooks.RemovableHandle
        A list of hook handles. You **must** call ``handle.remove()`` on each
        when you are done (e.g. at the end of vector extraction) to avoid
        memory leaks and unintended side‑effects on future forward passes.

    Notes
    -----

    The hooks are *passive*: they do not modify the forward computation in any
    way; they simply copy the hidden states out of the residual stream. This
    makes them safe to use both during:

    - **Unmasked extraction** – a single forward pass at ``t = 0`` to build
      CAA steering vectors.
    - **Diagnostic runs** – e.g. collecting activations for PCA plots or
      cosine‑similarity analyses.
    """

    layers = _get_transformer_layers(model)
    handles: List[torch.utils.hooks.RemovableHandle] = []

    for idx in layer_indices:
        layer = layers[idx]
        handle = layer.register_forward_hook(store.save_hook(f"layer_{idx}"))
        handles.append(handle)

    return handles

