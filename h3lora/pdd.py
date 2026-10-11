"""Peel Acc/PDD output-head banks off H3 LoRAs and blend them at sample time.

alibaba-pai Acc LoRAs store a 32-interval bank as ``final_layer.video_out.set_weight``
of shape ``[N * 96, hidden]`` (and the audio twin). Stock Comfy ``copy_``s that
onto the native ``[96, hidden]`` head and crashes. Core PR 15908 teaches
``FinalLayer`` to blend those rows; this module does the same inside the Power
LoRA Stack so those files load without a core change.

The bank is kept beside the model (object patch + DIFFUSION_MODEL wrapper) and
is registered as an additional model so ComfyUI can move and account for it.
The native head is not resized, and PDD strength interpolates from the native
head to the blended bank.
"""

from __future__ import annotations

import logging
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.model_patcher
from comfy.ldm.minimax.model import _mod_row, time_shift_sigma
from comfy.patcher_extension import WrappersMP
from .schedule import progress_from

LOG = logging.getLogger("h3.powerlorastack")

_VIDEO_WEIGHT = (
    "diffusion_model.final_layer.video_out.set_weight",
    "final_layer.video_out.set_weight",
    "proj_out.weight",
)
_VIDEO_BIAS = (
    "diffusion_model.final_layer.video_out.set_bias",
    "final_layer.video_out.set_bias",
    "proj_out.bias",
)
_AUDIO_WEIGHT = (
    "diffusion_model.final_layer.audio_out.set_weight",
    "final_layer.audio_out.set_weight",
    "audio_proj_out.weight",
)
_AUDIO_BIAS = (
    "diffusion_model.final_layer.audio_out.set_bias",
    "final_layer.audio_out.set_bias",
    "audio_proj_out.bias",
)


def _pop_first(sd, names):
    for name in names:
        if name in sd:
            return sd.pop(name), name
    return None, None


def _as_bank(tensor, out_features):
    """``[N, O, I]`` or ``[N*O, I]`` / ``[N*O]`` -> ``(N, flat)`` or None."""
    if tensor is None or out_features <= 0:
        return None
    t = tensor.detach().float().contiguous()
    if t.ndim == 3 and t.shape[1] == out_features:
        n = t.shape[0]
        return n, t.reshape(n * out_features, *t.shape[2:])
    if t.ndim == 2 and t.shape[1] == out_features:
        n = t.shape[0]
        return n, t.reshape(n * out_features)
    if t.ndim == 2 and t.shape[0] % out_features == 0:
        n = t.shape[0] // out_features
        return n, t
    if t.ndim == 1 and t.shape[0] % out_features == 0:
        n = t.shape[0] // out_features
        return n, t
    return None


def peel(sd, video_dim, audio_dim, hidden):
    """Pull a PDD bank out of a LoRA state dict. Returns ``(bank, note)``.

    ``bank`` is None when the file has no Acc heads. Matching keys that are
    not a whole number of native heads are dropped so stock ``set`` cannot crash.
    """
    vw, vw_key = _pop_first(sd, _VIDEO_WEIGHT)
    vb, vb_key = _pop_first(sd, _VIDEO_BIAS)
    aw, aw_key = _pop_first(sd, _AUDIO_WEIGHT)
    ab, ab_key = _pop_first(sd, _AUDIO_BIAS)
    if vw is None and aw is None:
        if vb is not None:
            sd[vb_key] = vb
        if ab is not None:
            sd[ab_key] = ab
        return None, "PDD bias without head ignored" if vb is not None or ab is not None else ""

    def _restore():
        if vw is not None:
            sd[vw_key] = vw
        if vb is not None:
            sd[vb_key] = vb
        if aw is not None:
            sd[aw_key] = aw
        if ab is not None:
            sd[ab_key] = ab

    video = _as_bank(vw, video_dim) if vw is not None else None
    audio = _as_bank(aw, audio_dim) if aw is not None else None
    notes = []
    if video is None and vw is not None:
        LOG.warning("H3 PowerLoraStack: ignored %s shape %s (native video head is %d)",
                    vw_key, tuple(vw.shape), video_dim)
        notes.append("video weight dropped")
    if audio is None and aw is not None:
        LOG.warning("H3 PowerLoraStack: ignored %s shape %s (native audio head is %d)",
                    aw_key, tuple(aw.shape), audio_dim)
        notes.append("audio weight dropped")
    if video is None and audio is None:
        return None, "PDD heads dropped (shape mismatch)"

    n = video[0] if video is not None else audio[0]
    if video is not None and video[0] != n:
        return None, f"PDD video n={video[0]} != audio n={n}"
    if audio is not None and audio[0] != n:
        return None, f"PDD audio n={audio[0]} != video n={n}"
    if n <= 1:
        _restore()
        return None, ""

    video_w = video[1] if video is not None else None
    audio_w = audio[1] if audio is not None else None
    if video_w is not None and video_w.shape[-1] != hidden:
        return None, f"PDD video in_features {video_w.shape[-1]} != {hidden}"
    if audio_w is not None and audio_w.shape[-1] != hidden:
        return None, f"PDD audio in_features {audio_w.shape[-1]} != {hidden}"

    video_b = None
    if vb is not None:
        packed = _as_bank(vb, video_dim)
        video_b = (packed[1] if video is not None and packed is not None
                   and packed[0] == n else None)
        if video_b is None:
            notes.append("video bias dropped")
    audio_b = None
    if ab is not None:
        packed = _as_bank(ab, audio_dim)
        audio_b = (packed[1] if audio is not None and packed is not None
                   and packed[0] == n else None)
        if audio_b is None:
            notes.append("audio bias dropped")

    bank = {
        "n": n,
        "video_dim": video_dim,
        "audio_dim": audio_dim,
        "video_w": video_w,
        "video_b": video_b,
        "audio_w": audio_w,
        "audio_b": audio_b,
    }
    note = f"PDD bank n={n}"
    if notes:
        note += " (" + ", ".join(notes) + ")"
    return bank, note


def _blend(weight, bias, h, n, start, stop, flow_shift):
    grid = torch.linspace(1.0, 0.0, n + 1, dtype=torch.float64)
    dt = (1.0 - flow_shift * grid / (1.0 + (flow_shift - 1.0) * grid)).diff()[start:stop]
    w = (dt / dt.sum()).to(device=h.device, dtype=h.dtype)
    rows = weight.to(device=h.device, dtype=h.dtype).reshape(n, -1, weight.shape[-1])[start:stop]
    out_w = torch.einsum("n,noi->oi", w, rows)
    out_b = None
    if bias is not None:
        b_rows = bias.to(device=h.device, dtype=h.dtype).reshape(n, -1)[start:stop]
        out_b = torch.einsum("n,no->o", w, b_rows)
    return F.linear(h, out_w, out_b)


def interval(sigma, sigma_next, n, video_shift):
    start, stop = (round(float(1.0 - time_shift_sigma(s, video_shift, 1.0)) * n)
                   for s in (sigma, sigma_next))
    start = max(0, min(n - 1, start))
    stop = max(start + 1, min(n, stop))
    return start, stop


class PddFinalLayer(nn.Module):
    def __init__(self, inner, bank, ctx):
        super().__init__()
        # The bank belongs to the additional-model patcher, so keep that
        # reference out of this module.  Register the native head children at
        # their original paths so the object patch preserves the model state
        # dict and its main-patcher ownership.
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "bank", bank)
        object.__setattr__(self, "ctx", ctx)
        self.norm = inner.norm
        self.adaln_proj = inner.adaln_proj
        self.video_out = inner.video_out
        self.audio_out = inner.audio_out

    def forward(self, x, t_emb, video_seg, audio_seg,
                sigma=None, sample_sigmas=None, shifts=None):
        """Run PDD under both current and legacy MiniMax H3 final-layer ABIs.

        Current ComfyUI passes sigma, the sampler schedule, and stream shifts
        directly. Older cores expose only the first four arguments, so retain
        the outer wrapper's thread-local values as a compatibility fallback.
        """
        inner = self.inner
        bank = self.bank
        ctx = self.ctx
        n = bank.n

        if sigma is None:
            sigma = ctx.get("sigma")
        if sample_sigmas is None:
            sample_sigmas = ctx.get("sample_sigmas")
        if shifts is None:
            shifts = ctx.get("shifts")
        if shifts is None:
            shifts = (12.0, 3.0)
        if sample_sigmas is None or sigma is None:
            raise ValueError("MiniMax H3 PDD heads need the sampler's sigma schedule")

        sigma = torch.as_tensor(sigma)
        sample_sigmas = torch.as_tensor(
            sample_sigmas, device=sigma.device
        ).flatten()
        if sample_sigmas.numel() == 0:
            raise ValueError("MiniMax H3 PDD heads need a non-empty sigma schedule")

        shift, scale = inner.adaln_proj(t_emb)

        def mod(seg):
            a, b, row = seg
            return (inner.norm(x[a:b]) * (1.0 + _mod_row(scale, row, scale.dtype))
                    + _mod_row(shift, row, shift.dtype)).to(torch.float32)

        i = int((sample_sigmas - sigma).abs().argmin())
        sigma_next = sample_sigmas[min(i + 1, sample_sigmas.shape[0] - 1)]
        strength = ctx.get("strength")
        if strength is None:
            strength = ctx.default_strength
        strength = float(strength)
        start, stop = interval(sigma, sigma_next, n, shifts[0])
        vh = mod(video_seg)
        ah = mod(audio_seg)
        native_video = inner.video_out(vh)
        if bank.video_w is None:
            video = native_video
        else:
            pdd_video = _blend(bank.video_w, bank.video_b, vh, n, start, stop, shifts[0])
            video = native_video + strength * (pdd_video - native_video)
        native_audio = inner.audio_out(ah)
        if bank.audio_w is None:
            audio = native_audio
        else:
            pdd_audio = _blend(bank.audio_w, bank.audio_b, ah, n, start, stop, shifts[1])
            audio = native_audio + strength * (pdd_audio - native_audio)
        return video, audio


class PddBank(nn.Module):
    """Movable, accountable storage for a PDD head bank."""

    def __init__(self, bank):
        super().__init__()
        self.n = bank["n"]
        self.video_w = nn.Parameter(bank["video_w"], requires_grad=False) if bank["video_w"] is not None else None
        self.video_b = nn.Parameter(bank["video_b"], requires_grad=False) if bank["video_b"] is not None else None
        self.audio_w = nn.Parameter(bank["audio_w"], requires_grad=False) if bank["audio_w"] is not None else None
        self.audio_b = nn.Parameter(bank["audio_b"], requires_grad=False) if bank["audio_b"] is not None else None


class PddState:
    def __init__(self, strength, schedule):
        self.default_strength = float(strength)
        self.schedule = schedule
        self.local = threading.local()

    def set(self, sigma, sample_sigmas, shifts, strength):
        self.local.values = {
            "sigma": sigma,
            "sample_sigmas": sample_sigmas,
            "shifts": shifts,
            "strength": strength,
        }

    def get(self, key):
        return getattr(self.local, "values", {}).get(key)

    def clear(self):
        self.local.values = {}


def bank_bytes(bank):
    return sum(
        tensor.numel() * tensor.element_size()
        for key in ("video_w", "video_b", "audio_w", "audio_b")
        if (tensor := bank.get(key)) is not None
    )


def attach(patcher, bank, strength=1.0, schedule=None):
    fl = patcher.get_model_object("diffusion_model.final_layer")
    bank_model = PddBank(bank)
    bank_patcher = comfy.model_patcher.ModelPatcher(
        bank_model,
        load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device(),
    )
    n = 0
    while patcher.get_additional_models_with_key(f"h3_power_lora_pdd_bank_{n}"):
        n += 1
    patcher.set_additional_models(f"h3_power_lora_pdd_bank_{n}", [bank_patcher])

    ctx = PddState(strength, schedule)
    wrapped = PddFinalLayer(fl, bank_model, ctx)
    patcher.add_object_patch("diffusion_model.final_layer", wrapped)

    def wrapper(executor, x, timestep, context, transformer_options=None, **kwargs):
        transformer_options = transformer_options or {}
        sigma = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        sample_sigmas = transformer_options.get("sample_sigmas")
        if sample_sigmas is None:
            sample_sigmas = transformer_options.get("sigmas")
        effective_strength = ctx.default_strength
        if ctx.schedule is not None:
            schedule_options = transformer_options
            if "sample_sigmas" not in schedule_options and sample_sigmas is not None:
                schedule_options = dict(transformer_options)
                schedule_options["sample_sigmas"] = sample_sigmas
            progress = progress_from(timestep, schedule_options, ctx.schedule.domain)
            effective_strength *= ctx.schedule.evaluate(progress)
        ctx.set(sigma, sample_sigmas, (
            float(transformer_options.get("minimax_h3_sigma_shift_video", 12.0)),
            float(transformer_options.get("minimax_h3_sigma_shift_audio", 3.0)),
        ), effective_strength)
        try:
            return executor(x, timestep, context, transformer_options, **kwargs)
        finally:
            ctx.clear()

    n = 0
    while patcher.get_wrappers(WrappersMP.DIFFUSION_MODEL, f"h3_power_lora_pdd_{n}"):
        n += 1
    patcher.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, f"h3_power_lora_pdd_{n}", wrapper)
    return n
