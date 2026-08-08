"""Runtime low-rank branches for quantized MiniMax H3 weights.

Merging a LoRA into a w4a8 / int4 weight destroys it.  ComfyUI's stock path is
dequantize -> add delta -> ``requantize_from_float(scale="recalculate")``, and
that round trip is not idempotent: re-fitting the codebook and re-rounding to
int4 injects ~1.5% relative weight noise, while these deltas are 0.01-0.08% of
the weight.  The merge therefore replaces the adapter with noise -- measured
end to end it recovers under a third of the LoRA magnitude at cos 0.12-0.14
against the correct result.

Keeping the branch separate is exact, keeps the quantized kernel, and costs
~1.5% extra FLOPs at rank 64.

Stacking is fused: N LoRAs on one Linear become a single pair by concatenating
along the rank axis,

    sum_i s_i * B_i @ A_i @ x  ==  [s_1 B_1 | ... | s_N B_N] @ [A_1; ...; A_N] @ x

so a ten-LoRA stack still costs one extra matmul pair per layer rather than
ten, and only one set of factors has to be moved to the GPU.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management


class LoraBank(nn.Module):
    """Holds every runtime branch's factors as one movable, accountable module."""

    def __init__(self, pairs: "OrderedDict[str, tuple[torch.Tensor, torch.Tensor]]"):
        super().__init__()
        self.index: dict[str, int] = {}
        for i, (name, (up, down)) in enumerate(pairs.items()):
            self.register_parameter(f"up{i}", nn.Parameter(up, requires_grad=False))
            self.register_parameter(f"down{i}", nn.Parameter(down, requires_grad=False))
            self.index[name] = i

    def get(self, name: str):
        i = self.index[name]
        return getattr(self, f"up{i}"), getattr(self, f"down{i}")


class LoraBranch:
    """Object patch for a Linear's ``forward``: ``y = W_q(x) + up @ down @ x``.

    Patches ``.forward`` rather than the module itself -- replacing the module
    would shift the memory manager's ``named_modules`` keys.  Holds the
    *patcher* rather than the bare module so comfy's weakref bookkeeping does
    not report a leaked model.
    """

    def __init__(self, bank_patcher, name: str, original):
        self.bank_patcher = bank_patcher
        self.bank: LoraBank = bank_patcher.model
        self.name = name
        self.original = original

    def __call__(self, input: torch.Tensor, *args, **kwargs):
        out = self.original(input, *args, **kwargs)
        up, down = self.bank.get(self.name)
        h = F.linear(input, comfy.model_management.cast_to_device(down, input.device, input.dtype))
        delta = F.linear(h, comfy.model_management.cast_to_device(up, input.device, input.dtype))
        return out + delta.to(out.dtype)


def fuse(contributions, compute_dtype):
    """Concatenate per-LoRA ``(up, down, scale)`` triples into one pair.

    The strength/alpha scale is folded into ``up`` so the forward path stays a
    bare pair of matmuls.
    """
    ups, downs = [], []
    for up, down, scale in contributions:
        ups.append((up.to(torch.float32) * float(scale)).to(compute_dtype))
        downs.append(down.to(compute_dtype))
    if len(ups) == 1:
        return ups[0], downs[0]
    return torch.cat(ups, dim=1).contiguous(), torch.cat(downs, dim=0).contiguous()


def attach(model_patcher, per_module, compute_dtype, tag: str):
    """Build the bank, register it for VRAM accounting, patch every forward.

    ``per_module`` maps a module path to a list of ``(up, down, scale)``.
    Returns the number of branched layers.
    """
    if not per_module:
        return 0

    import comfy.model_patcher

    pairs: "OrderedDict[str, tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
    for module_path, contributions in per_module.items():
        pairs[module_path] = fuse(contributions, compute_dtype)

    bank = LoraBank(pairs)
    bank_patcher = comfy.model_patcher.ModelPatcher(
        bank,
        load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device(),
    )
    model_patcher.set_additional_models(tag, [bank_patcher])

    for module_path in pairs:
        forward_key = f"{module_path}.forward"
        original = model_patcher.get_model_object(forward_key)
        model_patcher.add_object_patch(forward_key, LoraBranch(bank_patcher, module_path, original))
    return len(pairs)


def bank_bytes(per_module, compute_dtype) -> int:
    """Approximate VRAM the bank will occupy, for reporting."""
    itemsize = torch.empty((), dtype=compute_dtype).element_size()
    total = 0
    for contributions in per_module.values():
        for up, down, _ in contributions:
            total += (up.numel() + down.numel()) * itemsize
    return total
