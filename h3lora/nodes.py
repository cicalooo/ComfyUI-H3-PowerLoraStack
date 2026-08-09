"""Node definitions for the MiniMax H3 Power LoRA Stack."""

from __future__ import annotations

import logging

import folder_paths

from . import apply as apply_mod
from . import detect
from .schedule import Schedule

LOG = logging.getLogger("h3.powerlorastack")
CATEGORY = "MiniMax-H3/lora"


class AnyType(str):
    def __ne__(self, other):
        return False


ANY = AnyType("*")


class FlexibleOptionalInputType(dict):
    """Accepts the arbitrary ``lora_N`` inputs the frontend adds at runtime.

    ComfyUI validates a prompt against INPUT_TYPES; a stack whose row count is
    decided in the browser has no fixed schema, so this dict reports that it
    contains every key and hands back a permissive type for the ones it does
    not know about.
    """

    def __init__(self, type, data=None):
        super().__init__()
        self.type = type
        self.data = data or {}
        self.update(self.data)

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        return (self.type,)

    def __contains__(self, key):
        return True


def _resolve_lora(name: str):
    """Map a stored lora name onto a real file, tolerating separator drift."""
    if not name or name == "None":
        return None
    path = folder_paths.get_full_path("loras", name)
    if path:
        return path
    wanted = name.replace("\\", "/").lower()
    for candidate in folder_paths.get_filename_list("loras"):
        normalized = candidate.replace("\\", "/").lower()
        if normalized == wanted or normalized.endswith("/" + wanted.rsplit("/", 1)[-1]):
            return folder_paths.get_full_path("loras", candidate)
    return None


def _collect(kwargs):
    """Pull enabled rows out of the dynamic ``lora_N`` inputs, in UI order."""
    rows = []
    for key, value in kwargs.items():
        if not key.lower().startswith("lora_") or not isinstance(value, dict):
            continue
        if "lora" not in value:
            continue
        try:
            order = int(key.split("_", 1)[1])
        except (IndexError, ValueError):
            order = 1 << 30
        rows.append((order, value))
    rows.sort(key=lambda item: item[0])

    entries = []
    for order, value in rows:
        if not value.get("on", True):
            continue
        strength = float(value.get("strength", 1.0))
        if strength == 0.0:
            continue
        name = value.get("lora")
        if not name or name == "None":
            continue        # a row the user added but has not filled in yet
        path = _resolve_lora(name)
        if path is None:
            LOG.warning("H3 Power LoRA Stack: could not find lora %r, skipping", name)
            continue
        entries.append({"name": name, "path": path, "strength": strength, "row": order})
    return entries


class H3PowerLoraStack:
    """Stacked multi-LoRA loader for MiniMax H3 across every weight format.

    Handles the three things the stock loader and rgthree's Power Lora Loader
    get wrong on H3:

    * quantized bases -- w4a8/int4 weights are patched with an exact runtime
      branch instead of a lossy dequantize/requantize merge;
    * adaLN basis mismatch -- LoRAs trained against a dense checkpoint are
      rebased onto a pruned checkpoint's curve, instead of being skipped;
    * key conventions -- ai-toolkit, kohya, lycoris, peft and bare-prefix
      layouts all resolve against the model's own key set.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": FlexibleOptionalInputType(ANY, {
                "model": ("MODEL",),
                "quantized_layers": (["auto", "branch", "merge"], {
                    "default": "auto",
                    "tooltip": "auto: runtime branch for quantized layers, merge for the rest. "
                               "branch: never modify a weight. merge: stock behaviour "
                               "(destroys LoRAs on w4a8/int4).",
                }),
                "adaln_port": (["auto", "off"], {
                    "default": "auto",
                    "tooltip": "Rebase adaLN LoRA pairs between dense (2688) and curve (8) "
                               "checkpoints. Needs h3_silu_temb_grid.safetensors.",
                }),
                "adaln_modality": ("H3_MODALITY", {
                    "tooltip": "Optional. Wire a MiniMax H3 adaLN Modality node here to "
                               "scale every stacked LoRA's adaLN modulation per modality.",
                }),
                "schedule": ("H3_SCHEDULE", {
                    "tooltip": "Optional. Wire an H3 LoRA Schedule chain here to vary selected "
                               "row strengths over the denoising trajectory.",
                }),
            }),
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "report")
    OUTPUT_TOOLTIPS = ("Patched model", "Per-LoRA account of what was applied")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = __doc__

    def apply(self, model=None, quantized_layers="auto", adaln_port="auto",
              adaln_modality=None, schedule=None, **kwargs):
        if model is None:
            raise ValueError("H3 Power LoRA Stack: no model connected")

        entries = _collect(kwargs)
        if not entries:
            return (model, "no LoRAs enabled")

        patcher, report = apply_mod.apply_stack(
            model, entries, mode=quantized_layers, adaln_mode=adaln_port,
            modality=adaln_modality, schedule=schedule,
        )
        text = report.text()
        LOG.info("H3 Power LoRA Stack:\n%s", text)
        return (patcher, text)


class H3AdalnModality:
    """Scale a stacked LoRA's adaLN modulation per modality.

    H3 runs audio and video through the same 50 blocks, so there are no audio
    layers to target the way LTX 2.3 allows. The one pathway that does separate
    cleanly is adaLN: its projection emits three contiguous row blocks, one per
    modality, so scaling a slice steers that modality's modulation exactly.

    Wire the output into the stack's ``adaln_modality`` input. Leaving all three
    at 1.0 is a no-op; 0.0 removes that modality's share of the adapter.

    Only affects LoRAs that carry adaLN pairs -- the stack's report says, per
    LoRA, whether the control was actually live. It does not isolate a modality:
    attention is joint over the packed sequence, so this changes where a LoRA is
    applied, not everything it eventually reaches.
    """

    @classmethod
    def INPUT_TYPES(cls):
        slider = {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}
        return {"required": {
            "video": ("FLOAT", dict(slider, tooltip="Scale for the video modality (tag 0).")),
            "text": ("FLOAT", dict(slider, tooltip="Scale for the text/conditioning modality (tag 1).")),
            "audio": ("FLOAT", dict(slider, tooltip="Scale for the audio modality (tag 2).")),
        }}

    RETURN_TYPES = ("H3_MODALITY", "STRING")
    RETURN_NAMES = ("adaln_modality", "info")
    OUTPUT_TOOLTIPS = ("Wire into the stack's adaln_modality input",
                       "What this setting will do")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = __doc__

    def build(self, video=1.0, text=1.0, audio=1.0):
        values = {"video": float(video), "text": float(text), "audio": float(audio)}
        if all(v == 1.0 for v in values.values()):
            info = "no-op (all modalities at 1.0)"
        else:
            info = "adaLN " + ", ".join(f"{k} x{v:g}" for k, v in values.items())
        return (values, info)


class H3LoraSchedule:
    """Schedule selected Power LoRA Stack rows over sampler steps or sigma.

    Chain multiple nodes when different rows need different schedules; the
    later node wins where row selectors overlap. The stack reads ComfyUI's
    sampler timeline automatically, so no SIGMAS connection is required.
    """

    @classmethod
    def INPUT_TYPES(cls):
        strength = {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}
        percent = {"min": 0.0, "max": 100.0, "step": 1.0}
        return {
            "required": {
                "rows": ("STRING", {"default": "all", "tooltip": "all, 1,3, or 2-4"}),
                "start_strength": ("FLOAT", dict(strength, default=1.0)),
                "end_strength": ("FLOAT", dict(strength, default=0.1)),
                "curve": (["linear", "cosine", "smoothstep", "power", "step", "explicit"],),
                "curve_power": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 20.0, "step": 0.05}),
                "explicit_strengths": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Comma- or space-separated strengths, one per intended step.",
                }),
                "start_percent": ("FLOAT", dict(percent, default=0.0)),
                "end_percent": ("FLOAT", dict(percent, default=100.0)),
                "domain": (["steps", "sigma"], {
                    "tooltip": "steps follows model-call indices; sigma follows the scheduler's "
                               "actual normalized noise values.",
                }),
            },
            "optional": {
                "schedule": ("H3_SCHEDULE", {
                    "tooltip": "Optional earlier schedule link. This node wins on overlaps.",
                }),
            },
        }

    RETURN_TYPES = ("H3_SCHEDULE", "STRING")
    RETURN_NAMES = ("schedule", "info")
    OUTPUT_TOOLTIPS = ("Wire into the Power LoRA Stack's schedule input", "Schedule summary")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = __doc__

    def build(self, rows="all", start_strength=1.0, end_strength=0.1,
              curve="linear", curve_power=2.0, explicit_strengths="",
              start_percent=0.0, end_percent=100.0, domain="steps", schedule=None):
        item = Schedule(
            rows=str(rows), start_strength=float(start_strength),
            end_strength=float(end_strength), curve=curve,
            curve_power=float(curve_power), explicit_strengths=str(explicit_strengths),
            start_percent=float(start_percent), end_percent=float(end_percent), domain=domain,
        )
        previous = (schedule,) if isinstance(schedule, Schedule) else tuple(schedule or ())
        chain = previous + (item,)
        info = (f"rows {rows}: {start_strength:g} \u2192 {end_strength:g} {curve} "
                f"({domain} {start_percent:g}\u2013{end_percent:g}%)")
        if curve == "explicit":
            info += f", {len(item._explicit)} values"
        return (chain, info)


class H3LoraInspector:
    """Report the format, rank and adaLN basis of a LoRA without loading it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lora_name": (folder_paths.get_filename_list("loras"),),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "inspect"
    CATEGORY = CATEGORY
    DESCRIPTION = __doc__

    def inspect(self, lora_name):
        path = _resolve_lora(lora_name)
        if path is None:
            return (f"{lora_name}: not found",)
        info = detect.inspect(path)
        if info.get("error"):
            return (f"{info['name']}: {info['error']}",)
        lines = [
            f"file      : {info['name']}",
            f"size      : {info['size'] / (1024 ** 2):.0f} MB, {info['tensors']} tensors",
            f"format    : {info['convention']}"
            + (f" ({info['trainer']})" if info.get("trainer") else ""),
            f"dtype     : {', '.join(info['dtypes'])}",
            f"rank      : {'/'.join(str(r) for r in info['ranks']) or '?'}"
            + ("  (alpha present)" if info["alpha"] else "  (no alpha)"),
            f"adaLN     : {'/'.join(str(d) for d in info['adaln_dims']) or 'none'}",
            f"modules   : {', '.join(info['modules'])}",
        ]
        if info.get("base_model"):
            lines.append(f"base      : {info['base_model']}")
        if info.get("passenger"):
            lines.append(f"passenger : {', '.join(info['passenger'])}")
        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {
    "H3PowerLoraStack": H3PowerLoraStack,
    "H3AdalnModality": H3AdalnModality,
    "H3LoraSchedule": H3LoraSchedule,
    "H3LoraInspector": H3LoraInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PowerLoraStack": "MiniMax H3 Power LoRA Stack",
    "H3AdalnModality": "MiniMax H3 adaLN Modality",
    "H3LoraSchedule": "MiniMax H3 LoRA Schedule",
    "H3LoraInspector": "MiniMax H3 LoRA Inspector",
}
