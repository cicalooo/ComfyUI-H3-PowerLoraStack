# ComfyUI-H3-PowerLoraStack

A stacked multi-LoRA loader for **MiniMax H3**, in the spirit of rgthree's Power
Lora Loader but built around the three things that actually break H3 LoRAs.

<img width="1426" height="642" alt="image" src="https://github.com/user-attachments/assets/522e8637-d82e-4753-92e1-f0543f9f5525" />


## Nodes

| Node | Purpose |
|---|---|
| **MiniMax H3 Power LoRA Stack** | Any number of LoRAs on one node, each with a toggle and strength |
| **MiniMax H3 LoRA Inspector** | Reports a LoRA's format, rank and adaLN basis without loading it |

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

The fit uses the *target checkpoint's own* table, because bakes differ — locally,
`PinkCherry` ships an uncentered basis (column norms 22.98, 2.67, 1.66, …) while
`10Eros_Max` ships comfy's mean-centered one (7.08, 2.09, 0.75, …).

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

| Convention | Example |
|---|---|
| ai-toolkit / diffusers | `diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight` |
| bare (no prefix) | `blocks.0.attn.qkv_proj.lora_A.weight` |
| kohya / musubi | `lora_unet_blocks_0_attn_qkv_proj.lora_down.weight` + `.alpha` |
| lycoris | `lycoris_blocks_0_...` |
| peft / diffusers trainer | `base_model.model.blocks.0...`, `transformer.blocks.0...` |

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

## Output

The `report` string output accounts for every LoRA (wire it to a preview/show-text
node to read it; it is also written to the console log):

```
base: ConvRotW4A4 x300, INT8 x50
adaLN: curve (input dim 8)
turbo dense-adaLN @ 1: 26 merged, 275 branched, adaLN ported x51 (basis fit 1.7e-03)
GalaxyAce curve-adaLN @ 0.8: 0 merged, 258 branched
branch bank: 300 layers, 412 MB
```

## Limitations

- A curve-trained LoRA on a **dense** checkpoint can only be ported if the LoRA
  carries its own `adaln_t_table`; otherwise there is no record of which bake it
  was trained against and the adaLN pairs are dropped with a warning.
- Runtime branches apply to `MODEL` from ComfyUI's native H3 loader. The
  streaming loader in `minimaxh3chinkloader` uses its own `MINIMAX_H3_MODEL`
  handle and its own LoRA path.
- DoRA, LoHa, LoKr and locon adapters always merge — only plain rank
  decompositions can be branched.
