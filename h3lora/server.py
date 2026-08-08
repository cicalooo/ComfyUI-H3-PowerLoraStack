"""HTTP route backing the node's auto-balance button.

The measurement has to happen server-side: it needs the LoRA files themselves,
which the browser has no access to.  Results are cached in :mod:`.gain` on
(path, mtime, size), so re-pressing the button, or toggling a row, is free after
the first read of each file.
"""

from __future__ import annotations

import logging

from . import gain

LOG = logging.getLogger("h3.powerlorastack")

ROUTE = "/h3_power_lora_stack/balance"


def measure_names(names):
    """``[lora_name]`` -> ``{lora_name: {rel, factor, layers, note}}``.

    Imported lazily by the route so the module stays importable without a
    running server (the calibration scripts use it directly).
    """
    from .nodes import _resolve_lora

    out = {}
    seen: dict[str, str] = {}
    for name in names:
        if not name or name == "None" or name in out:
            continue
        path = _resolve_lora(name)
        if path is None:
            out[name] = {"rel": None, "factor": 1.0, "layers": 0,
                         "note": "not found"}
            continue
        info = gain.measure_file(path)
        entry = {
            "rel": info.get("rel"),
            "factor": float(info.get("factor", 1.0)),
            "layers": info.get("layers", 0),
            "ranks": info.get("ranks", []),
            "note": "",
        }
        if info.get("error"):
            entry["note"] = "unreadable"
        elif not info.get("rel"):
            entry["note"] = "no measurable layers"
        else:
            fp = info.get("fingerprint")
            first = seen.get(fp) if fp else None
            if first is not None:
                entry["note"] = f"same adapter as {first}"
            elif fp:
                seen[fp] = name
        out[name] = entry
    return out


def register():
    """Attach the route to ComfyUI's aiohttp app, if there is one."""
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:                       # running outside ComfyUI
        return False
    instance = getattr(PromptServer, "instance", None)
    if instance is None or not hasattr(instance, "routes"):
        return False

    @instance.routes.post(ROUTE)
    async def _balance(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad request"}, status=400)
        names = body.get("loras")
        if not isinstance(names, list):
            return web.json_response({"error": "expected {'loras': [...]}"}, status=400)
        try:
            results = measure_names([n for n in names if isinstance(n, str)])
        except Exception as exc:
            LOG.exception("H3 Power LoRA Stack: balance failed")
            return web.json_response({"error": str(exc)}, status=500)
        return web.json_response({
            "results": results,
            "reference": gain.REFERENCE_REL,
        })

    return True
