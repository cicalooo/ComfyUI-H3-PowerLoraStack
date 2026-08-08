# MiniMax H3 Power LoRA Stack — Technical Audit

Audit date: 2026-08-08  
Audited revision: `400770a`  
ComfyUI revision used for compatibility review: `dd79c643` (2026-08-08)

## Executive assessment

The project has a strong technical core. It identifies three real H3-specific
problems—quantized-weight merging, adaLN basis changes, and inconsistent LoRA
key conventions—and addresses them with a sensible module split. The plain-LoRA
runtime branch and the dense-to-curve math both held up under targeted numerical
checks. The key normalizer also mapped all recognized keys in the 46 locally
installed H3 LoRAs to a representative H3 checkpoint with zero unmatched keys.

The node should still be considered **experimental rather than production-ready**.
The main blockers are not syntax or basic mathematics; they are edge-path
correctness, unsupported adapter behavior on quantized models, incomplete
accounting of what was actually applied, and the complete absence of automated
tests. Several README claims are stronger than the implementation can currently
guarantee.

Recommended release posture: keep the current version marked beta/experimental
until the high-priority findings below are fixed and covered by tests.

## Scope and evidence

The audit covered:

- Python node registration and ComfyUI integration.
- Dynamic frontend widgets, picker, serialization, and auto-balance behavior.
- Key normalization and adapter-format handling.
- Quantized runtime branches and stack fusion.
- Dense/curve and curve/curve adaLN conversion.
- Per-modality adaLN scaling.
- Gain measurement, HTTP route, diagnostics, packaging, and documentation.
- The installed H3 library: 46 safetensors files totaling about 29.7 GB.

Checks performed:

- All 11 Python files passed AST parsing.
- The package imported against the installed ComfyUI and registered all three
  nodes successfully.
- `web/h3_power_lora_stack.js` passed `node --check`.
- All 46 local H3 safetensors headers were readable: 32 comfy, 5 kohya, and 9
  bare-format files.
- The local set contains 14 curve-adaLN files, 8 dense-adaLN files, and 24 files
  without adaLN pairs.
- All recognized keys from all 46 files normalized against the sampled H3 model
  with zero unmatched keys.
- Synthetic dense→curve conversion had maximum absolute error around `1.5e-5`.
- Synthetic two-LoRA fusion had maximum absolute error around `2.0e-6`.
- The discovered silu grid is `[1025, 2688]`; the two local curve tables both
  fitted at approximately `1.68e-3` relative residual.
- A CUDA table-to-table fit reproduced the device mismatch described in finding
  H5 below.

Not covered by an existing test harness: full H3 sampling, VRAM/offload stress,
workflow round-trip testing in the browser, older ComfyUI versions, and every
quantization layout. No test suite exists in this repository.

## Priority findings

| ID | Severity | Area | Summary |
|---|---|---|---|
| H1 | High | adaLN | A shared dense-target context caches the first source LoRA's curve basis and can use it for later LoRAs from a different bake. |
| H2 | High | Loading | A local combined LoRA/ALS file contains 102 full adaLN tensors that are silently ignored by the apply path. |
| H3 | High | Quantization | LoKr, LoHa, DoRA, LoCon, and other non-plain adapters are merged into quantized weights, defeating the node's main correctness guarantee. |
| H4 | High | Reporting | “Merged” and “unmatched” counts do not prove that ComfyUI accepted or applied the patches. |
| H5 | High | Device support | Curve-to-curve fitting fails when model tables are on CUDA. |
| M1 | Medium | Formats | PEFT/default and alternate LoRA suffixes normalize but are not handled consistently by adaLN porting, modality scaling, detection, and gain measurement. |
| M2 | Medium | adaLN | Curve→dense computes the same pseudoinverse once per adaLN layer instead of once per basis. |
| M3 | Medium | Reproducibility | Grid selection is implicit and first-match; the chosen grid path/fingerprint is not reported or user-selectable. |
| M4 | Medium | Auto-balance | Calibration is a collection-specific heuristic presented too much like an exact measurement. |
| M5 | Medium | Input handling | Missing/ambiguous files and non-finite strengths can be skipped or propagated without a useful node report. |
| M6 | Medium | Frontend | The LoRA list is cached for the whole browser session and balance state has edge-case drift when replacing the last balanced row. |
| M7 | Medium | Release quality | There are no tests or CI, compatibility is undeclared, and repository metadata/legal files are incomplete. |

## Detailed findings

### H1 — Dense-target multi-LoRA basis cache can corrupt later adapters

`apply_stack()` creates one `AdalnContext` and reuses it for every LoRA. On a
dense target, `AdalnContext.basis(curve_table=...)` must use the source table
shipped by each curve-trained LoRA. Instead, `_basis` and `_failed` are stored
once on the shared context. After the first source table is fitted, later source
tables receive that first fit. If the first LoRA lacks a table and marks the
context failed, later convertible LoRAs are also prevented from fitting.

This is a latent but serious numerical error: it produces plausible-shaped
tensors and therefore may not raise an exception.

Relevant code: `h3lora/apply.py:128`, `h3lora/apply.py:145`,
`h3lora/adaln.py:190`, and `h3lora/adaln.py:200`.

Required work:

- Cache dense-target bases by a stable source-table identity/fingerprint, not
  once per stack.
- Keep target-table basis caching separate; sharing that case is valid.
- Do not let failure for one LoRA poison later LoRAs.
- Add a test stacking two curve LoRAs with different source tables onto one
  dense target, plus a “missing table then valid table” ordering test.

### H2 — Full adaLN passenger tensors are ignored

`keymap.normalize()` deliberately passes full replacement tensors through and
says the caller will handle them. `apply_stack()` only removes
`adaln_t_table`; it then sends the remaining dictionary to
`comfy.lora.load_lora()`, which does not interpret plain
`blocks.N.adaln_proj.linear.weight` or `.bias` tensors as patches.

This is not hypothetical. The local
`minimax_h3_pruned_turbo_ckpt850_loraWithALS.safetensors` contains:

- 208 A tensors and 208 B tensors for ordinary LoRA layers.
- One `adaln_t_table`.
- 51 full adaLN weights and 51 full adaLN biases.

The 102 full tensors are currently omitted while the ordinary layers still
report success. For a file whose schedule behavior depends on those ALS tensors,
that changes the adapter's meaning.

Required work:

- Decide and document whether full replacements are supported.
- If supported, canonicalize them and apply them with explicit `set` semantics,
  including a clear policy for strength values other than 1.0 and for stacking.
- If not supported, reject the file with an actionable error rather than
  partially applying it.
- Include passenger/applied/rejected counts in the node report.

### H3 — Unsupported adapter types merge into quantized weights

Only a plain 2-D `LoRAAdapter` is eligible for the runtime branch. Everything
else is put in `merge`, including LoKr, LoHa, DoRA, LoCon, and full-delta
adapters. On w4a8/int4 this uses the same destructive merge path that the node
exists to avoid. The README lists this as a limitation, but the runtime report
only says “merged”; it does not warn that accuracy may be lost.

The local collection already contains one LoKr file, so this affects a current
user path rather than only a future format.

Required work:

- Preferred: integrate or adapt ComfyUI's bypass-adapter machinery for supported
  non-plain adapters, retaining stack fusion where it is mathematically safe.
- Minimum safe behavior: detect a quantized target and reject or prominently
  warn before merging a non-plain adapter.
- Report forced INT8 `mlp.fc2` merges separately for the same reason.
- Test plain LoRA, LoKr, LoHa/LoCon, DoRA, and full-diff behavior on plain, INT8,
  and w4a8 targets.

### H4 — The report overstates what was applied

There are three accounting gaps:

1. `unmatched` only covers failures in key normalization. Passenger tensors,
   orphaned A/B components, and suffixes Comfy does not consume are not included.
2. `patcher.add_patches()` returns the keys accepted by the model state dict,
   but the return value is ignored. `len(merge)` is reported instead.
3. Branch shape failures increment `report.skipped`, but that count is never
   included in the per-LoRA detail or final summary.

As a result, “0 unmatched” and “N merged” are not equivalent to “everything was
applied,” despite the README promising an account for every LoRA.

Required work:

- Track normalized, consumed-by-Comfy, accepted-by-patcher, branched, and
  rejected keys independently.
- Capture and surface shape mismatch reasons.
- Include missing files from `_collect()` in the returned report instead of
  logging them before report creation.
- Add a strict mode that fails if any material tensor is unconsumed.

### H5 — Table-to-table fitting is not device-safe

`fit_table_to_table()` converts source and target tables to float64 on their
current device, then creates the intercept column on CPU. If a model table is on
CUDA, `torch.cat()` fails with a CPU/CUDA mismatch. This was reproduced on the
installed CUDA runtime.

The usual pre-sampling path may keep model buffers offloaded on CPU, so the bug
will not trigger in every workflow. It remains a real failure under resident,
already-loaded, or nonstandard patcher configurations.

Required work: explicitly move both tables to CPU for fitting, as `fit_basis()`
already does, or create every temporary on `tgt.device`. CPU is preferable for
predictable memory use and cache behavior.

### M1 — Format support is broader in key mapping than downstream handling

The key mapper recognizes `.lora_A.default.weight`, `.lora_B.default.weight`,
`.lora_A`, and `.lora_B`, but `port_adaln_pairs()` only pairs standard
`lora_A.weight`/`lora_B.weight` and kohya down/up names. Modality scaling also
omits the default and bare B suffixes. The inspector misses those rank forms.
Other suffixes supported by current ComfyUI, such as `_lora.up.weight`,
`lora_linear_layer.up.weight`, and `reshape_weight`, are not uniformly
canonicalized when they arrive without the canonical prefix.

This means a PEFT file may load ordinary layers while silently losing adaLN
porting or modality control.

Required work: define one adapter-suffix registry and reuse it in key mapping,
adaLN pairing, modality scaling, inspection, and gain measurement. Add a fixture
for every advertised convention.

### M2 — Curve→dense repeats an expensive pseudoinverse

`torch.linalg.pinv(v)` is inside the per-module loop. A typical H3 LoRA has 51
adaLN projections, so the same `[2688, 8]` pseudoinverse may be computed 51
times. Compute and cache it once with the fitted basis.

Also add caching to table-to-table maps and a residual policy. At present a
high table-to-table residual is logged at info level and the mapping is still
applied. A bad fit should produce a visible warning or fail under strict mode.

### M3 — Grid discovery is convenient but not reproducible

The implementation uses the first generic
`h3_silu_temb_grid.safetensors` found across several locations. `grid_path`
exists in the Python API but is not exposed by the node. The report prints the
fit residual but not which file was selected.

This matters because the README correctly notes that bakes differ. Two users
can run the same workflow and select different grids based on filesystem order.

Required work:

- Expose auto/explicit grid selection, or bind known grid fingerprints to model
  metadata.
- Report the selected path, tensor shape, checksum, and residual.
- Search all configured LoRA/model paths, not only index zero.
- Invalidate the grid cache on mtime/size changes.

### M4 — Auto-balance is useful, but its claims need tighter boundaries

The low-rank Frobenius identity is implemented correctly for ordinary LoRA, but
the final `rel` is not an exact measurement against the loaded model. It uses
RMS constants from one INT8 checkpoint, excludes adaLN, is calculated before
modality/port transformations, and anchors to one local collection median.
Those are defensible product choices, but they make the result a calibration
heuristic rather than a universal unit.

Additional correctness/reporting issues:

- The “exact duplicate fingerprint” is only layer count plus aggregate energy;
  different adapters can collide.
- Tucker LoKr is not measured according to ComfyUI's reconstruction, and one
  decomposed-side alpha layout is scaled incorrectly.
- The backend report calls `factor` the strength auto-balance “would use.” It is
  a multiplier; if the user's manual strength was not 1.0, that suggestion is
  wrong.
- Header-only inspection and auto-balance only support safetensors, while the
  picker can expose other LoRA file types.

Required work: label the feature as heuristic calibration, report both factor
and resulting strength where the manual value is known, use a real content hash
for duplicate detection, and either implement or explicitly disable unsupported
measurement formats.

### M5 — File and numeric input handling can mislead

`_resolve_lora()` falls back to a case-insensitive basename match and returns
the first candidate. If two directories contain the same basename, a moved
workflow can silently select the wrong file. Missing files are logged and
removed before report generation, so a node can return “no LoRAs enabled” when
the real problem is that every requested file is missing.

Strength parsing accepts arbitrary workflow data. `NaN` and infinity are not
rejected and can propagate into bank weights or model patches.

Required work: reject ambiguous basename matches, preserve unresolved rows in
the report, require finite strengths, and bound them consistently with the UI.

### M6 — Frontend lifecycle and feedback need hardening

- `loraListPromise` caches the folder listing for the entire browser session,
  with no refresh action; newly installed LoRAs require a page reload.
- When the only balanced row is repointed, its balance metadata is cleared
  before `isBalanced()` is checked, so the replacement is not automatically
  balanced and the node exits balanced mode.
- Balance failures are console-only and the button has no busy/error state.
- Strengths are rounded to two decimals; small intentional strengths can round
  to zero after calibration.
- Serialization has recently been repaired, but there is no automated workflow
  save/load/clone regression test to protect it.

Required work: add refresh/invalidation, store node-level balance intent rather
than inferring it solely from rows, show user-visible request status, preserve
more numeric precision, and test format-1 migration plus format-2 round trips.

### M7 — Release engineering and repository hygiene are incomplete

- No unit, integration, frontend, or numerical regression tests.
- No CI configuration.
- No minimum ComfyUI revision/API capability check despite imports of recent
  `quant_ops` and `weight_adapter` APIs.
- No `requires-python` declaration.
- `pyproject.toml` declares MIT, but there is no tracked `LICENSE` file.
- Python bytecode and `__pycache__` files are committed, and there is no
  `.gitignore`.
- `pyproject.toml` points to the `doggeddalle` repository/publisher while the
  configured Git origin is `cicalooo`; ownership and release metadata should be
  made consistent.
- The README says key mapping was verified against 37 LoRAs, while later
  sections and the current library use 46. Numerical benchmark claims have no
  checked-in reproduction script or fixture manifest.

Required work: add the legal file and ignore rules immediately, then establish a
small CI matrix and document supported ComfyUI/Python revisions before public
release.

## What is already working well

- The module boundaries are clear: normalization, application, branching,
  adaLN conversion, modality control, gain, detection, and server handling are
  independently understandable.
- Key normalization resolves underscore conventions against the actual model
  key set instead of guessing token boundaries. The 46-file local scan supports
  this design.
- The fused branch identity is correct and avoids one branch call per LoRA.
- Branch parameters are registered as an additional model for ComfyUI memory
  accounting rather than hidden in closures.
- The special INT8 `mlp.fc2` fused path is explicitly recognized.
- Dense→curve conversion includes the required bias delta and passed a targeted
  synthetic test.
- Modality scaling happens before porting, which correctly carries scaling into
  the generated bias delta.
- The frontend escapes LoRA names before inserting highlighted HTML.
- The node provides unusually useful human-readable diagnostics; the remaining
  task is making those diagnostics authoritative.

## Recommended implementation sequence

### Phase 1 — Correctness gate

1. Fix per-source adaLN basis caching and compute `pinv(V)` once.
2. Make full passenger tensor handling explicit: support or reject, never
   silently partially apply.
3. Prevent destructive non-plain merges on quantized weights, with explicit
   warnings for forced fused-layer merges.
4. Rebuild reporting around consumed and accepted keys.
5. Make table fitting device-safe and add residual thresholds/strict mode.
6. Centralize suffix handling across every subsystem.

### Phase 2 — Regression suite

Add CPU-fast tests for:

- Every advertised key convention and alternate suffix.
- Dense→curve, curve→dense, and curve→curve output equivalence, including alpha
  and bias deltas.
- Two source bakes in one dense-target stack and failure isolation.
- Modality scaling before and after basis conversion.
- Branch fusion with positive, zero, and negative strengths.
- Quantized routing decisions for each layout and adapter type.
- Passenger, orphan, malformed shape, missing file, ambiguous file, `NaN`, and
  infinity behavior.
- Gain formulas against explicitly materialized small matrices.
- Frontend create/save/load/clone/migrate and async balance behavior.

Use tiny synthetic modules and safetensors fixtures; the core suite should not
need a 20 GB checkpoint or GPU. Add one optional GPU/offload smoke workflow.

### Phase 3 — Product and release polish

1. Expose/report grid provenance and compatibility.
2. Add LoRA-list refresh and visible balance status.
3. Document installation, compatibility, grid acquisition, example workflows,
   and adapter-type limitations near the top of the README.
4. Add reproducible benchmark scripts and a fixture manifest for numerical
   claims.
5. Add CI, license, ignore rules, version policy, and consistent repository
   metadata.

## Suggested release gates

Do not call the node stable until all of the following are true:

- No material LoRA tensor can be silently unconsumed.
- Quantized mode never silently sends an unsupported adapter through a known
  destructive merge.
- Multi-LoRA adaLN conversion is correct regardless of row order or source bake.
- The report is derived from patches actually accepted by the model patcher.
- Format, math, serialization, and failure behavior have automated coverage.
- A documented ComfyUI version range passes CI and one real H3 smoke workflow.

With those gates met, the project would move from an impressive specialist
prototype to a credible production loader. The underlying approach is worth
continuing; the largest remaining need is to make every fallback explicit and
testable.
