# ComfyUI-H3-PowerLoraStack

A stacked multi-LoRA loader for **MiniMax H3**, in the spirit of rgthree's Power
Lora Loader but built around the three things that actually break H3 LoRAs.

<img width="1533" height="487" alt="Screenshot 2026-08-08 211513" src="https://github.com/user-attachments/assets/cf7ba1dc-96d9-42ae-8c77-89905319b816" />

## Nodes

| Node                            | Purpose                                                                                               |
| ------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **MiniMax H3 Power LoRA Stack** | Any number of LoRAs on one node, each with a toggle and strength, plus one-click strength calibration |
| **MiniMax H3 adaLN Modality**   | Scales stacked LoRAs' adaLN modulation per modality (video / text / audio)                            |
| **MiniMax H3 LoRA Schedule**    | Varies selected stack rows' strength over denoising steps or normalized sigma                         |
| **MiniMax H3 LoRA Inspector**   | Reports a LoRA's format, rank and adaLN basis without loading it                                      |

## Why not just use a normal LoRA loader

### 1. Quantized weights: merging destroys the LoRA

ComfyUI's stock path is dequantize → add delta → `requantize_from_float(scale="recalculate")`.
That round trip is **not idempotent**: re-fitting the codebook and re-rounding to
int4 injects ~1.5% relative weight noise, while a typical H3 LoRA delta is
0.01–0.08% of the weight. The merge therefore replaces the adapter with noise.
Measured on a w4a8 checkpoint, the stock merge recovers under a third of the
LoRA magnitude at cos 0.12–0.14 against the correct result, and takes 128 s
versus 6.5 s.

This node routes quantized layers through an exact runtime low-rank branch
(`y = W_q(x) + B @ A @ x`) instead, keeping the quantized kernel and costing
~1.5% extra FLOPs at rank 64.

`quantized_layers` controls this:

- `auto` (default) — branch quantized layers, merge unquantized ones
- `branch` — never modify a weight, even in bf16 (fast strength A/B testing)
- `merge` — stock behaviour; only useful for comparison

One layer is special-cased: `mlp.fc2` under `TensorWiseINT8Layout` is reached
through `comfy.ops.linear_input_act`, which fuses the activation into the INT8
kernel and never calls `fc2.forward`. A branch there would be silently dropped,
so those layers merge even in `branch` mode.

### 2. adaLN basis mismatch

H3 ships in two forms. A *dense* checkpoint feeds `silu(time_embedder(t))`, a
2688-dim vector, into every `adaln_proj.linear`. A *curve* (pruned) checkpoint
drops the time embedder and stores an `adaln_t_table` of shape `[grid, 8]`.

A LoRA trained against one form has the wrong `lora_A` width for the other, so
ComfyUI logs `shape '[96768, 8]' is invalid for input of size 260112384` and
skips the layer. **Dropping those pairs is not an acceptable fix** — on the
turbo distillation LoRAs the constant term alone is ~100% of the magnitude of
`dW @ S(t)`, so it discards essentially the whole adapter.

This node changes basis instead, which preserves rank:

```
dense -> curve   A' = A @ V,       bias delta  B @ (A @ c)
curve -> dense   A' = A @ pinv(V), bias delta -B @ (A' @ c)
```

where `S(t) = c + V @ table(t)`, recovered by least squares of `[1 | table]`
against the silu grid. The bias delta is emitted as `.diff_b`, which comfy
applies as a `("diff",)` patch on the sibling `.bias`; without it the port is
nearly worthless. Measured end to end, the ported adapter reproduces the dense
contribution at **cos 0.999998**.

The fit uses the *target checkpoint's own* table, because bakes differ — of two
local bakes, one ships an uncentered basis (column norms 22.98, 2.67, 1.66, …)
while the other ships comfy's mean-centered one (7.08, 2.09, 0.75, …).

**This also catches a silent failure the stock loader cannot see.** Two curve
bakes have the same adaLN width, so a LoRA trained against one loads without
complaint on the other and is simply wrong — measured at **cos −0.375**, worse
than not applying it at all. When a LoRA ships its own `adaln_t_table`, this
node rebases table-to-table (exactly, cos 1.000000, no grid needed). When it
does not, the width matches and the mismatch is undetectable — so prefer LoRAs
trained against the checkpoint you are running.

#### The silu grid

Dense↔curve porting needs `h3_silu_temb_grid.safetensors`. It is searched for in
`models/h3_adaln/`, `models/loras/`, `models/diffusion_models/` and one level
under `custom_nodes/`. Table-to-table rebasing does not need it. Set
`adaln_port` to `off` to disable porting entirely.

With a grid from a different build the fit bottoms out at ~1.7e-3 relative,
because the 7th and 8th curve directions are near-degenerate (σ₇ ≈ σ₈) and so
differ between bakes. That is still ~6× below the int8 quantization floor.

### 3. Key conventions

Every H3 LoRA naming convention resolves against the model's own key set rather
than by guessing where underscores split, so `qkv_proj` is never mistaken for
two tokens:

| Convention               | Example                                                        |
| ------------------------ | -------------------------------------------------------------- |
| ai-toolkit / diffusers   | `diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight`         |
| bare (no prefix)         | `blocks.0.attn.qkv_proj.lora_A.weight`                         |
| kohya / musubi           | `lora_unet_blocks_0_attn_qkv_proj.lora_down.weight` + `.alpha` |
| lycoris                  | `lycoris_blocks_0_...`                                         |
| peft / diffusers trainer | `base_model.model.blocks.0...`, `transformer.blocks.0...`      |

Verified against all 37 H3 LoRAs in `models/loras/h3`: **zero unmatched keys.**

## Stacking

Multiple LoRAs on the same layer are fused into a single pair by concatenating
along the rank axis:

```
sum_i s_i * B_i @ A_i @ x  ==  [s_1 B_1 | ... | s_N B_N] @ [A_1; ...; A_N] @ x
```

so a ten-LoRA stack costs one extra matmul pair per layer, not ten. The factors
live in a `_LoraBank` registered via `set_additional_models`, so its VRAM is
accounted for and comfy's weakref bookkeeping stays quiet.

## Denoising schedules

**MiniMax H3 LoRA Schedule** changes selected stack rows' strength during
sampling. Wire its `schedule` output into the stack, select rows with `all`,
`1,3`, or `2-4`, then choose a linear, cosine, smoothstep, power, step, or
explicit curve. `start_percent` and `end_percent` limit the transition to part
of the trajectory. Chain schedule nodes for different row groups; the later
node wins where selectors overlap.

The `steps` domain follows model-call indices. The `sigma` domain follows the
scheduler's actual normalized noise values, which can produce a different
shape with non-linear schedulers. The stack reads `sample_sigmas` supplied by
ComfyUI automatically, so plain KSampler works and **no SIGMAS wire is needed**.

Scheduled rows always use the live branch path, including on unquantized bases.
Any adapter feature or fused layer that cannot run as a branch is merged at the
row's ordinary static strength and called out in the report. Ported adaLN bias
deltas are scheduled with their LoRA rather than being left at a fixed value.

## Auto-balance

**Strength 1.0 is not a unit.** Measured across the 27 non-distillation H3 LoRAs
in `models/loras/h3`, the perturbation produced at strength 1.0 spans **65×** —
0.054% of the base weights at one end, 5.24% at the other. Neither rank nor file
size predicts it: a rank-128 adapter sits at 0.088% while a rank-16 one sits at
0.40%. So a strength that worked on one LoRA carries no information about the
next, and the sweet spot has to be rediscovered per file.

`⚖ Auto-balance strengths` measures what each active LoRA actually does and puts
them all on one scale:

```
rel = sqrt( sum_l ||dW_l||_F^2 / sum_l ||W_l||_F^2 )
```

The factor multiplies the strength you already chose, so your relative intent
between rows survives — what changes is that a LoRA perturbing the model 18×
harder than usual stops arriving at full force. `↺ Restore manual strengths`
puts every row back exactly as it was; the pre-balance value is stashed on the
row, so it survives saving and reloading the workflow. Editing a strength by
hand overrides that row and is not clobbered by a later recompute.

**The factor only ever trims** (clamped to ≤ 1). A LoRA measuring *below* the
reference may be quiet deliberately — distillation adapters sit an order of
magnitude down and are correct at 1.0 — whereas one measuring far above it
essentially never is. That asymmetry means no classifier is needed: every turbo
LoRA in the collection lands on ×1.00 by itself.

Computing this is only affordable because `dW` is never formed. It is up to
28672 × 5376 and there are ~260 per file, but

```
||B A||_F^2 = tr((B^T B)(A A^T))
```

needs only r×r matrices, so a 2.4 GB rank-128 adapter measures in a few seconds,
almost all of it disk. Results cache on (path, mtime, size). LoKr is handled too
— `||W1 ⊗ W2||_F = ||W1||_F · ||W2||_F`.

The reference is the *median* of the collection rather than a hand-picked
target, so the calibration agrees with trainer defaults on ordinary files and
only moves outliers. Base norms come from a per-group constant (measured base
weight RMS is uniform to ~2× within a group and the four linear groups agree to
20%), which avoids reading the 20 GB checkpoint.

adaLN is excluded from the measurement: its basis is checkpoint-dependent — the
same adapter shipped dense and curve8 differs 5.7× there — and it is where a
distillation LoRA keeps the schedule change that must not be normalised away.

Two honest limits: Frobenius energy is not perceptual strength, and it says
nothing about *contention*. Distinct LoRAs are near-orthogonal in weight space
(measured |cos| ≤ 0.03) yet overlap 3–15× above chance in the feature subspaces
they read and write, so two adapters can still fight over the same features at
perfectly balanced magnitudes.

Because the deltas really are near-orthogonal, stack energy adds in quadrature:
holding a stack at the "one LoRA at 1.0" budget wants `1/√N`, not the `1/N` that
gets recommended. Auto-balance does **not** apply that — it calibrates each LoRA
and leaves the total to you, so adding a row never silently weakens the others.

## adaLN modality control

**MiniMax H3 is not built like LTX 2.3.** LTX duplicates the tower per modality
(`audio_attn`, `audio_ff`, `audio_patchify_proj`, `audio_to_video_attn`), so a
LoRA can be steered by picking layers. H3 packs audio and video into one token
sequence and pushes both through the same 50 blocks. The only modality-specific
weights in the entire checkpoint are four tensors — `video_patch_proj`,
`audio_patch_proj`, `final_layer.video_out`, `final_layer.audio_out` — and none
of the 46 H3 LoRAs checked touches any of them. There is no layer-name axis.

One pathway does separate cleanly: **adaLN**. `AdalnProj.forward` computes
`linear(t) -> [M, expand*hidden*modalities]` then `view(M*modalities,
expand*hidden)`, so output feature `j` belongs to modality `j //
(expand*hidden)`. The 96768 rows are three contiguous blocks of 32256, and
`comfy/ldm/minimax/model.py` tags segments `{video: 0, text: 1, audio: 2}`.
Scaling a slice of `lora_B`'s rows scales that modality's modulation exactly,
with no runtime hook.

Wire **MiniMax H3 adaLN Modality** into the stack's `adaln_modality` input. All
three at 1.0 is a no-op; 0.0 removes that modality's share of every stacked
adapter.

This holds for future LoRAs *by construction*: the row order is a property of the
trained weights, not of ComfyUI. Any adapter that loads onto `adaln_proj.linear`
at all must match it, whatever its rank, alpha, trainer convention, or adaLN
basis (dense/curve affects only the `A` side). All seven local H3 bakes — fl2va,
ref2va, int8-convrot, w4a8-mixed, int4-BQ — carry identical geometry.

The geometry is read off the model's own `AdalnProj` rather than hardcoded, and
every layer is shape-checked before it is touched, so a future H3 variant either
adapts or keeps today's behaviour — it cannot slice at the wrong offsets.
`final_layer.adaln_proj` is `AdalnProj(t_dim, hidden, 2, 1)` — one modality,
differentiated only by timestep — so it fails that check and is left alone.

Ordering matters and is handled: the scaling runs *before* adaLN porting, which
derives its bias delta as `B @ const`, so the emitted `.diff_b` inherits it.



Where adaLN *is* present it is not a marginal knob. Measured on the 14 LoRAs
whose adaLN basis matches the checkpoint, it carries 89–99.7% of the
weight-space perturbation for content LoRAs (median ~96%) and 16–23% for the
curve8 turbo adapters. Caveat: relative Frobenius across differently-shaped
matrices is an imperfect proxy for perceptual impact, and adaLN's input is only
8-dimensional, so read that as "where most of the weight change lives", not "96%
of what you see".

## Output

The `report` string output accounts for every LoRA (wire it to a preview/show-text
node to read it; it is also written to the console log):

```
base: ConvRotW4A4 x300, INT8 x50
adaLN: curve (input dim 8)
turbo dense-adaLN @ 1: 26 merged, 275 branched, adaLN ported x51 (basis fit 1.7e-03)
  rel dW 0.047%
style_lora curve-adaLN @ 0.8: 0 merged, 258 branched, adaLN modality video/text/audio = 1/1/0.25 x50
  rel dW 0.054%
motion_lora @ 1: 0 merged, 104 branched, adaLN modality 1/1/0.25 INACTIVE (LoRA has no adaLN pairs)
  rel dW 0.316%
branch bank: 300 layers, 412 MB
```

`rel dW` is the measurement above, reported for every LoRA whether or not
auto-balance was used, so an out-of-scale strength is visible from the API too.
When the applied strength is far from the calibrated one the line says what
auto-balance would have used.

## Limitations

- A curve-trained LoRA on a **dense** checkpoint can only be ported if the LoRA
  carries its own `adaln_t_table`; otherwise there is no record of which bake it
  was trained against and the adaLN pairs are dropped with a warning.
- Runtime branches apply to `MODEL` from ComfyUI's native H3 loader. The
  streaming loader in `minimaxh3chinkloader` uses its own `MINIMAX_H3_MODEL`
  handle and its own LoRA path.
- DoRA, LoHa, LoKr and locon adapters always merge — only plain rank
  decompositions can be branched.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
