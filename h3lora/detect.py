"""Cheap safetensors introspection for MiniMax H3 LoRA files.

Reads only the header, so scanning a 2.5 GB LoRA costs a few kilobytes of IO.
Used both for the node's diagnostic report and for deciding, before anything is
loaded, whether a file needs adaLN porting.
"""

from __future__ import annotations

import json
import os
import re
import struct

_cache: dict[tuple, dict] = {}

_CONVENTIONS = (
    ("kohya", re.compile(r"^lora_unet_")),
    ("lycoris", re.compile(r"^lycoris_")),
    ("peft", re.compile(r"^(base_model\.model\.|transformer\.)")),
    ("comfy", re.compile(r"^diffusion_model\.")),
)


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(length))


def _convention(keys) -> str:
    for name, pattern in _CONVENTIONS:
        if any(pattern.match(k) for k in keys):
            return name
    if any(k.startswith(("blocks.", "token_refiner.", "final_layer.")) for k in keys):
        return "bare"
    return "unknown"


def inspect(path: str) -> dict:
    """Summarise one LoRA file.  Cached on (path, mtime, size)."""
    try:
        st = os.stat(path)
    except OSError as exc:
        return {"error": str(exc), "name": os.path.basename(path)}
    key = (path, st.st_mtime_ns, st.st_size)
    hit = _cache.get(key)
    if hit is not None:
        return hit

    info: dict = {"name": os.path.basename(path), "path": path, "size": st.st_size}
    try:
        header = read_header(path)
    except Exception as exc:
        info["error"] = f"unreadable header: {exc}"
        _cache[key] = info
        return info

    metadata = header.pop("__metadata__", {}) or {}
    keys = list(header)
    info["tensors"] = len(keys)
    info["convention"] = _convention(keys)
    info["dtypes"] = sorted({v.get("dtype", "?") for v in header.values()})
    info["base_model"] = metadata.get("ss_base_model_version", "")
    info["trainer"] = ""
    try:
        info["trainer"] = json.loads(metadata.get("software", "{}")).get("name", "")
    except Exception:
        pass
    if not info["trainer"] and metadata.get("ss_network_module"):
        info["trainer"] = metadata["ss_network_module"]

    ranks: set[int] = set()
    adaln_dims: set[int] = set()
    modules: set[str] = set()
    has_alpha = False
    for k, v in header.items():
        if k.endswith(".alpha"):
            has_alpha = True
            continue
        shape = v.get("shape") or []
        is_down = k.endswith(("lora_A.weight", "lora_down.weight"))
        if is_down and len(shape) == 2:
            ranks.add(int(shape[0]))
            if "adaln" in k:
                adaln_dims.add(int(shape[1]))
        for tag in ("adaln_proj", "attn.qkv_proj", "attn_qkv_proj", "attn.out_proj",
                    "attn_out_proj", "mlp.fc1", "mlp_fc1", "mlp.fc2", "mlp_fc2",
                    "token_refiner", "final_layer"):
            if tag in k:
                modules.add(tag.replace("_proj", "").replace(".", "_").replace("attn_qkv", "qkv")
                            .replace("attn_out", "out"))

    info["ranks"] = sorted(ranks)
    info["alpha"] = has_alpha
    info["adaln_dims"] = sorted(adaln_dims)
    info["modules"] = sorted(modules)
    info["passenger"] = sorted(
        k for k in keys
        if not re.search(r"(lora_A|lora_B|lora_down|lora_up|alpha|diff|dora|hada|lokr|oft)", k)
    )[:8]
    _cache[key] = info
    return info


def summary_line(info: dict) -> str:
    if info.get("error"):
        return f"{info['name']}: {info['error']}"
    ranks = "/".join(str(r) for r in info.get("ranks", [])) or "?"
    adaln = info.get("adaln_dims")
    adaln_s = f", adaLN dim {'/'.join(str(d) for d in adaln)}" if adaln else ", no adaLN"
    return (f"{info['name']}: {info.get('tensors', 0)} tensors, {info.get('convention', '?')} "
            f"format, rank {ranks}{adaln_s}")
