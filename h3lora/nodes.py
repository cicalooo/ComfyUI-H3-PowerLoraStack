"""Node definitions for the MiniMax H3 Power LoRA Stack."""

from __future__ import annotations

import logging

import folder_paths

from . import apply as apply_mod
from . import detect

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
    for _, value in rows:
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
        entries.append({"name": name, "path": path, "strength": strength})
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
            }),
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "report")
    OUTPUT_TOOLTIPS = ("Patched model", "Per-LoRA account of what was applied")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = __doc__

    def apply(self, model=None, quantized_layers="auto", adaln_port="auto", **kwargs):
        if model is None:
            raise ValueError("H3 Power LoRA Stack: no model connected")

        entries = _collect(kwargs)
        if not entries:
            return (model, "no LoRAs enabled")

        patcher, report = apply_mod.apply_stack(
            model, entries, mode=quantized_layers, adaln_mode=adaln_port,
        )
        text = report.text()
        LOG.info("H3 Power LoRA Stack:\n%s", text)
        return (patcher, text)


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
    "H3LoraInspector": H3LoraInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PowerLoraStack": "MiniMax H3 Power LoRA Stack",
    "H3LoraInspector": "MiniMax H3 LoRA Inspector",
}
