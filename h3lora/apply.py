"""Stack orchestration: normalise, port, partition, apply."""

from __future__ import annotations

import logging
import math
from collections import OrderedDict

import torch
import comfy.lora
import comfy.patcher_extension
import comfy.utils
from comfy.quant_ops import QuantizedTensor
from comfy.weight_adapter import LoRAAdapter

from . import adaln as adaln_mod
from . import branch as branch_mod
from . import gain
from . import keymap
from . import modality as modality_mod
from . import pdd as pdd_mod
from . import schedule as schedule_mod
from . import sidecars as sidecar_mod

LOG = logging.getLogger("h3.powerlorastack")
sidecar_mod.ensure_load_hook()

# Layers whose forward is bypassed by a fused kernel, per weight layout.  A
# runtime branch on these would be silently dropped, so they must merge.
# Grouped W3A8-W6A8 use branch.SlicedFC2Branch at the MLP boundary.
_FORWARD_BYPASSED = {"TensorWiseINT8Layout": ("mlp.fc2",)}


class StackReport:
    """Human-readable account of what happened, surfaced on the node."""

    def __init__(self):
        self.lines: list[str] = []
        self.merged = 0
        self.branched = 0
        self.skipped = 0
        self.rejected = 0
        self.bank_bytes = 0

    def add(self, line: str):
        self.lines.append(line)

    def text(self) -> str:
        return "\n".join(self.lines)


def _module_weight(model_patcher, module_path: str):
    try:
        module = model_patcher.get_model_object(module_path)
    except Exception:
        return None
    return getattr(module, "weight", None)


def _is_plain_lora(adapter) -> bool:
    """True for a bare rank-decomposition with no locon/dora/reshape extras."""
    return (
        isinstance(adapter, LoRAAdapter)
        and adapter.weights[3] is None      # locon mid
        and adapter.weights[4] is None      # dora scale
        and adapter.weights[5] is None      # reshape / pad
        and adapter.weights[0].ndim == 2
    )


def _branchable(weight, module_path: str, mode: str) -> bool:
    if weight is None or mode == "merge":
        return False
    if mode == "auto" and not isinstance(weight, QuantizedTensor):
        return False        # unquantized merges are exact and free at runtime
    layout = getattr(weight, "_layout_cls", None)
    for suffix in _FORWARD_BYPASSED.get(layout, ()):
        if module_path.endswith(suffix):
            return False
    return True


def detect_quantization(model_patcher) -> str:
    """Report the DiT's dominant weight format for the node's status line."""
    layouts: dict[str, int] = {}
    plain = 0
    try:
        diffusion_model = model_patcher.get_model_object("diffusion_model")
    except Exception:
        return "unknown"
    for module in diffusion_model.modules():
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        if isinstance(weight, QuantizedTensor):
            layout = getattr(weight, "_layout_cls", "quantized")
            layouts[layout] = layouts.get(layout, 0) + 1
        else:
            plain += 1
    if not layouts:
        return "unquantized"
    parts = [f"{k.replace('TensorCore', '').replace('TensorWise', '').replace('Layout', '')}"
             f" x{v}" for k, v in sorted(layouts.items(), key=lambda kv: -kv[1])]
    if plain:
        parts.append(f"plain x{plain}")
    return ", ".join(parts)


def apply_stack(model, entries, mode="auto", adaln_mode="auto", grid_path="",
                report: StackReport | None = None, modality=None, schedule=None):
    """Apply a list of ``{'path', 'name', 'strength'}`` LoRAs to an H3 model.

    ``mode`` is ``auto`` (branch quantized layers, merge the rest), ``merge``
    (stock behaviour) or ``branch`` (never touch a weight).

    ``adaln_mode`` is ``auto`` (rebase this stack and any adaLN adapters already
    on ``model``), ``strip`` (drop mismatched pairs) or ``off``.

    ``modality`` optionally scales each LoRA's adaLN modulation per modality;
    see :mod:`h3lora.modality`.
    """
    report = report or StackReport()
    patcher = model.clone()

    model_sd = patcher.model.state_dict()
    index = keymap.build_module_index(model_sd.keys())
    key_map = {k[: -len(".weight")]: k for k in model_sd
               if k.startswith("diffusion_model.") and k.endswith(".weight")}

    try:
        diffusion_model = patcher.get_model_object("diffusion_model")
    except Exception as exc:
        raise ValueError("H3 Power LoRA Stack requires a MiniMax H3 model") from exc
    video_dim = int(getattr(diffusion_model.final_layer.video_out, "out_features", 0) or 0)
    audio_dim = int(getattr(diffusion_model.final_layer.audio_out, "out_features", 0) or 0)
    hidden = int(getattr(diffusion_model, "hidden_size", 0) or 0)
    pdd_entry = None
    target_dim, table = adaln_mod.read_target(diffusion_model)
    if adaln_mode == "off":
        adaln_ctx = None
    else:
        if not grid_path:
            grid_path = adaln_mod.find_silu_grid()
        adaln_ctx = adaln_mod.AdalnContext(
            target_dim, table, grid_path,
            sidecars=sidecar_mod.collect(patcher, diffusion_model),
            time_embedder=getattr(diffusion_model, "time_embedder", None),
        )

    mod_values = modality_mod.normalize_scales(modality)
    mod_geom = None if modality_mod.is_identity(mod_values) else modality_mod.geometry(diffusion_model)

    report.add(f"base: {detect_quantization(patcher)}")
    report.add(f"adaLN: {'curve' if table is not None else 'dense'} (input dim {target_dim})")
    if adaln_mode != "off":
        upstream = adaln_mod.upstream_adaln_fix_warning(model)
        if upstream:
            report.add(upstream)
            LOG.warning("H3 PowerLoraStack:\n%s", upstream)
    if adaln_ctx is not None and table is not None:
        adaln_ctx.basis()
        if adaln_ctx.source:
            line = f"  basis: {adaln_ctx.source}"
            if adaln_ctx.residual is not None:
                line += f" (residual {adaln_ctx.residual:.1e})"
            report.add(line)
    if adaln_ctx is not None:
        try:
            patcher, attached = adaln_mod.port_attached_patches(
                patcher, adaln_ctx, mode=adaln_mode)
        except Exception as exc:
            LOG.exception("H3 PowerLoraStack: incoming adaLN port failed")
            report.add(f"  ! incoming adaLN port failed ({exc}); left upstream patches")
        else:
            if attached["ported"] or attached["stripped"] or attached["unportable"]:
                line = (f"  incoming adaLN: ported {attached['ported']}, "
                        f"stripped {attached['stripped']}, "
                        f"unportable {attached['unportable']} "
                        f"across {attached['keys']} key(s)")
                if attached["residual"] is not None:
                    line += f" (basis fit {attached['residual']:.1e})"
                report.add(line)
                for note in attached["notes"][:8]:
                    report.add(f"    ! {note}")
    if not modality_mod.is_identity(mod_values) and mod_geom is None:
        report.add("  ! adaLN modality control requested but this model's adaLN "
                   "does not split into the expected modalities - ignored")
        # the header carries the reason; keep the per-LoRA notes from blaming it
        # on the LoRAs, which are not at fault here
        mod_values = None

    per_module: "OrderedDict[str, list]" = OrderedDict()
    compute_dtype = model.model_dtype()

    for fallback_row, entry in enumerate(entries, start=1):
        row_index = int(entry.get("row", fallback_row))
        name = entry.get("name") or entry["path"]
        try:
            strength = float(entry.get("strength", 1.0))
        except (TypeError, ValueError):
            report.rejected += 1
            report.add(f"{name} rejected: strength must be numeric")
            continue
        if not math.isfinite(strength):
            report.rejected += 1
            report.add(f"{name} rejected: strength must be finite")
            continue
        row_schedule = schedule_mod.resolve(schedule, row_index)
        try:
            lora_sd = comfy.utils.load_torch_file(entry["path"], safe_load=True)
        except Exception as exc:
            report.rejected += 1
            report.add(f"{name} rejected: could not load LoRA ({exc})")
            continue

        normalized, unmatched = keymap.normalize(lora_sd, index)
        bank, pdd_note = pdd_mod.peel(normalized, video_dim, audio_dim, hidden)
        row_pdd_entry = ({
            "bank": bank,
            "name": name,
            "strength": strength,
            "schedule": row_schedule,
        } if bank is not None else None)
        measured = gain.measure_state_dict(normalized, name)

        # before porting: the port derives its bias delta as ``B @ const``, so
        # scaling B's rows here carries through to the emitted .diff_b
        normalized, mod_stats = modality_mod.apply_to_state_dict(
            normalized, mod_values, mod_geom)
        mod_note = modality_mod.describe(mod_values, mod_stats)
        adaln_note = ""
        table_key = next((key for key in normalized if key.endswith("adaln_t_table")), None)
        source_table = normalized.pop(table_key) if table_key is not None else None
        if adaln_ctx is not None and target_dim:
            try:
                normalized, stats = adaln_mod.port_adaln_pairs(
                    normalized, adaln_ctx, source_table=source_table,
                    mode=adaln_mode)
            except Exception as exc:
                LOG.exception("H3 PowerLoraStack: adaLN port failed for %s", name)
                report.add(f"{name}: adaLN port failed ({exc}); pairs left as-is")
                stats = {"ported": 0, "skipped": 0, "ok": 0, "rebased": 0, "residual": None}
            if stats["ported"]:
                adaln_note = f", adaLN ported x{stats['ported']}"
            if stats["rebased"]:
                adaln_note += f", adaLN rebased x{stats['rebased']}"
            if stats["skipped"]:
                adaln_note += f", adaLN dropped x{stats['skipped']}"
            if stats["residual"] is not None:
                adaln_note += f" (basis fit {stats['residual']:.1e})"

        passengers = [key for key in normalized if keymap.split_suffix(key)[1] is None]
        unsupported_passengers = [key for key in passengers
                                  if not key.endswith("adaln_t_table")]
        if unsupported_passengers:
            report.rejected += len(unsupported_passengers)
            example = ", ".join(unsupported_passengers[:2])
            if len(unsupported_passengers) > 2:
                example += ", ..."
            report.add(f"{name} rejected: {len(unsupported_passengers)} unsupported "
                       f"passenger tensor(s) ({example}); no partial application")
            continue
        if row_pdd_entry is not None:
            pdd_entry = row_pdd_entry
        try:
            loaded = comfy.lora.load_lora(normalized, key_map)
        except Exception as exc:
            report.rejected += 1
            report.add(f"{name} rejected: adapter load failed ({exc})")
            continue

        merge: dict = dict(loaded)
        branched_here = 0
        skipped_here = 0
        branch_rejected_here = 0
        for weight_key, patch in loaded.items():
            if (not weight_key.endswith(".weight") or not isinstance(patch, tuple)
                    or len(patch) <= 1 or patch[0] not in {"diff", "set"}
                    or not patch[1]):
                continue
            candidate = patch[1][0]
            module_path = weight_key[:-len(".weight")]
            weight = _module_weight(patcher, module_path)
            if (not torch.is_tensor(candidate) or weight is None
                    or tuple(candidate.shape) != tuple(weight.shape)):
                LOG.warning("H3 PowerLoraStack: shape mismatch on %s, skipped", weight_key)
                report.skipped += 1
                skipped_here += 1
                merge.pop(weight_key, None)
        for bias_key, bias_patch in loaded.items():
            if (not bias_key.endswith(".bias") or not isinstance(bias_patch, tuple)
                    or len(bias_patch) <= 1 or bias_patch[0] != "diff"
                    or not bias_patch[1]):
                continue
            module_path = bias_key[:-len(".bias")]
            candidate = bias_patch[1][0]
            weight = _module_weight(patcher, module_path)
            if (not torch.is_tensor(candidate) or weight is None
                    or candidate.ndim != 1 or candidate.shape[0] != weight.shape[0]):
                LOG.warning("H3 PowerLoraStack: shape mismatch on %s, skipped", bias_key)
                report.skipped += 1
                skipped_here += 1
                merge.pop(bias_key, None)
        for weight_key, adapter in loaded.items():
            if not weight_key.endswith(".weight"):
                continue
            module_path = weight_key[: -len(".weight")] if weight_key.endswith(".weight") else None
            if module_path is None or not _is_plain_lora(adapter):
                continue
            weight = _module_weight(patcher, module_path)
            up, down = adapter.weights[0], adapter.weights[1]
            if (not torch.is_tensor(up) or not torch.is_tensor(down)
                    or up.ndim != 2 or down.ndim != 2 or up.shape[1] != down.shape[0]
                    or weight is None or up.shape[0] != weight.shape[0]
                    or down.shape[1] != weight.shape[1]):
                LOG.warning("H3 PowerLoraStack: shape mismatch on %s, skipped", weight_key)
                report.skipped += 1
                skipped_here += 1
                merge.pop(weight_key, None)
                continue
            effective_mode = "branch" if row_schedule is not None else mode
            if not _branchable(weight, module_path, effective_mode):
                continue
            alpha = adapter.weights[2]
            alpha_scale = float(alpha) / down.shape[0] if alpha is not None else 1.0
            bias = None
            bias_key = f"{module_path}.bias"
            bias_patch = loaded.get(bias_key)
            if (isinstance(bias_patch, tuple) and len(bias_patch) > 1
                    and bias_patch[0] == "diff" and bias_patch[1]):
                candidate = bias_patch[1][0]
                if getattr(candidate, "ndim", None) == 1 and candidate.shape[0] == weight.shape[0]:
                    bias = candidate.to(torch.float32) * strength
                    merge.pop(bias_key, None)
            if row_schedule is not None or bias is not None:
                per_module.setdefault(module_path, []).append(
                    (up, down, strength * alpha_scale, row_schedule, bias))
            else:
                # Preserve the original strength-folded contribution exactly.
                per_module.setdefault(module_path, []).append(
                    (up, down, strength * alpha_scale))
            merge.pop(weight_key, None)
            branched_here += 1

        if mode == "branch" and merge:
            # Branch mode must not silently fall back to a weight patch for an
            # adapter or passenger that has no runtime branch implementation.
            branch_rejected_here = len(merge)
            report.rejected += branch_rejected_here
            merge.clear()

        accepted = patcher.add_patches(merge, strength) if merge else []
        rejected_here = len(merge) - len(accepted)
        quantized_merge = []
        for patch_key in accepted:
            weight_key = patch_key[0] if isinstance(patch_key, tuple) else patch_key
            if not isinstance(weight_key, str) or not weight_key.endswith(".weight"):
                continue
            module_path = weight_key[:-len(".weight")]
            if isinstance(_module_weight(patcher, module_path), QuantizedTensor):
                quantized_merge.append(weight_key)
        report.rejected += rejected_here
        report.merged += len(accepted)
        report.branched += branched_here

        detail = f"{len(accepted)} merged, {branched_here} branched"
        if unmatched:
            detail += f", {len(unmatched)} unmatched"
        if passengers:
            detail += f", {len(passengers)} ignored"
        if skipped_here:
            detail += f", {skipped_here} skipped"
        if rejected_here:
            detail += f", {rejected_here} rejected"
        if branch_rejected_here:
            detail += f", {branch_rejected_here} rejected in branch mode"
        if quantized_merge:
            detail += f", {len(quantized_merge)} quantized merge"
        if pdd_note:
            detail += f", {pdd_note}"
        report.add(f"{name} @ {strength:g}: {detail}{adaln_note}{mod_note}")
        if quantized_merge:
            report.add("  ! quantized targets were merged because no exact runtime "
                       "adapter path was available")
        if row_schedule is not None:
            arrow = "\u2192"
            report.add(
                f"  sched: {row_schedule.start_strength:.2f} {arrow} "
                f"{row_schedule.end_strength:.2f} {row_schedule.curve} "
                f"({row_schedule.domain} {row_schedule.start_percent:g}\u2013"
                f"{row_schedule.end_percent:g}%)"
            )
            if accepted:
                report.add(
                    f"  ! {len(accepted)} patches on {name} cannot be scheduled "
                    f"(merged at {strength:.2f})"
                )
        if measured.get("rel"):
            # what this LoRA actually does to the weights, so a strength that is
            # far off the calibrated unit is visible without the UI button
            note = f"  rel dW {measured['rel'] * 100:.3f}%"
            factor = measured["factor"]
            suggested = strength * factor
            if abs(strength - suggested) > 0.1 * max(abs(suggested), 1e-6):
                note += f" (auto-balance would use {suggested:.2f}, factor {factor:.2f})"
            report.add(note)
        if not accepted and not branched_here and row_pdd_entry is None:
            report.add(f"  ! {name} matched no layers on this model")

    if per_module:
        report.bank_bytes = branch_mod.bank_bytes(per_module, compute_dtype)
        # a chain of stack nodes must not overwrite each other's bank
        n = 0
        while patcher.get_additional_models_with_key(f"h3_power_lora_bank_{n}"):
            n += 1
        tag = f"h3_power_lora_bank_{n}"
        branch_notes: list[str] = []
        _count, controller = branch_mod.attach(
            patcher, per_module, compute_dtype, tag, notes=branch_notes)
        if controller is not None:
            patcher.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.APPLY_MODEL,
                f"h3_lora_schedule_{n}",
                controller,
            )
        report.add(f"branch bank: {len(per_module)} layers, "
                   f"{report.bank_bytes / (1024 ** 2):.0f} MB")
        for note in branch_notes:
            report.add(f"  ! {note} (counted as branched because a hook was registered)")

    if pdd_entry is not None:
        pdd_bank = pdd_entry["bank"]
        pdd_mod.attach(patcher, pdd_bank, pdd_entry["strength"], pdd_entry["schedule"])
        report.add(f"PDD: {pdd_bank['n']} heads from {pdd_entry['name']} "
                   f"(strength {pdd_entry['strength']:g}, "
                   f"{pdd_mod.bank_bytes(pdd_bank) / (1024 ** 2):.0f} MB, shifts 12/3)")

    adaln_mod.mark_stack_adaln(patcher)
    return patcher, report
