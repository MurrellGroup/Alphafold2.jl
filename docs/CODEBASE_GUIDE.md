# Alphafold2.jl Codebase Guide

Last updated: 2026-02-10

This is a full expository guide to the Julia AF2 implementation. It is intentionally both:
- user-facing (what to run, what to expect), and
- developer-facing (how data moves through code and where each subsystem lives).

## 1) What This Repository Implements

`Alphafold2.jl` is a feature-first (`C, ... , B`) Julia implementation of the major AF2 model components:
- Evoformer trunk,
- Structure module,
- Sidechain/atom projection utilities,
- Output heads and confidence metrics,
- In-process feature pipeline for monomer and multimer from user-provided sequence/MSA/template inputs.

Primary module entrypoint:
- [`src/Alphafold2.jl`](../src/Alphafold2.jl)

## 2) Repository Map

Core library:
- [`src/layers.jl`](../src/layers.jl): `LinearFirst`, `LayerNormFirst`
- [`src/rigid.jl`](../src/rigid.jl): rigid-body and quaternion math
- [`src/openfold_feats.jl`](../src/openfold_feats.jl): frame/atom transforms (`atom14_to_atom37`, torsion frame expansion)
- [`src/openfold_infer_utils.jl`](../src/openfold_infer_utils.jl): confidence metrics (`compute_plddt`, `compute_predicted_aligned_error`, `compute_tm`), atom mask tables
- [`src/training_utils.jl`](../src/training_utils.jl): differentiable sequence feature builders
- [`src/modules/`](../src/modules): AF2 module implementations and NPZ loaders

Feature pipeline:
- [`src/feature_pipeline.jl`](../src/feature_pipeline.jl): constants and includes
- [`src/feature_pipeline/common.jl`](../src/feature_pipeline/common.jl): shared feature-building functions
- [`src/feature_pipeline/monomer.jl`](../src/feature_pipeline/monomer.jl): `build_monomer_features()`
- [`src/feature_pipeline/multimer.jl`](../src/feature_pipeline/multimer.jl): `build_multimer_features()`

Model building and inference:
- [`src/model_builder.jl`](../src/model_builder.jl): `AF2Config`, `AF2Model`, `_build_af2_model()`
- [`src/inference.jl`](../src/inference.jl): pipeline stages (`prepare_inputs`, `run_evoformer`, `run_heads`, `run_inference`), output writers
- [`src/high_level_api.jl`](../src/high_level_api.jl): `load_monomer()`, `load_multimer()`, `fold()`

Script entrypoints (CLI wrappers):
- [`scripts/end_to_end/build_monomer_input_jl.jl`](../scripts/end_to_end/build_monomer_input_jl.jl)
- [`scripts/end_to_end/build_multimer_input_jl.jl`](../scripts/end_to_end/build_multimer_input_jl.jl)

Validation:
- [`scripts/regression`](../scripts/regression): regression test cases and helpers
- [`scripts/gradients/check_full_model_zygote.jl`](../scripts/gradients/check_full_model_zygote.jl)

## 3) Tensor Convention and Layout Boundaries

Internal convention:
- feature-first, batch-last (`C, ..., B`)

Canonical conversion helpers:
- [`src/tensor_utils.jl`](../src/tensor_utils.jl)
  - `af2_to_first_2d`, `first_to_af2_2d`
  - `af2_to_first_3d`, `first_to_af2_3d`

In practice:
- library modules consume feature-first tensors,
- feature pipeline produces feature-first tensors directly.

## 4) End-to-End Pathways

There are three levels of API for running inference:

1. **High-level**: `fold(model, sequence; ...)` — builds features, runs inference, writes PDB/NPZ
2. **Mid-level**: `_infer(model, features)` — takes a raw feature Dict, returns `AF2InferenceResult`
3. **Composable**: individual pipeline stages for research workflows:

```
build_monomer_features / build_multimer_features → Dict
    ↓
prepare_inputs(model, dict) → AF2PreparedInputs
    ↓
initial_recycle_state(model, inputs) → RecycleState
    ↓
run_evoformer(model, inputs, prev) → (msa_act, pair_act)
    ↓
run_heads(model, msa_act, pair_act, inputs) → NamedTuple
    ↓
(update RecycleState, loop back or stop)
```

`run_inference(model, inputs)` wraps the recycle loop and returns `AF2InferenceResult`.

## 4.1 Monomer Path

1. Build features from user inputs (sequence, optional A3M, optional template PDB):
- `Alphafold2.build_monomer_features()` in [`src/feature_pipeline/monomer.jl`](../src/feature_pipeline/monomer.jl)

2. Run model inference via composable stages or the convenience wrapper:
- `prepare_inputs` → `run_inference` in [`src/inference.jl`](../src/inference.jl)
- Or `_infer(model, features)` which wraps both

3. Or use the high-level API:
- `fold(model, sequence; ...)` in [`src/high_level_api.jl`](../src/high_level_api.jl)

## 4.2 Multimer Path

1. Build multimer features from chain sequences plus optional per-chain A3M/template inputs:
- `Alphafold2.build_multimer_features()` in [`src/feature_pipeline/multimer.jl`](../src/feature_pipeline/multimer.jl)

2. Same pipeline stages handle monomer and multimer:
- `prepare_inputs` → `run_inference` in [`src/inference.jl`](../src/inference.jl)

3. Multimer-specific metadata included:
- `asym_id`, `entity_id`, `sym_id`, `cluster_bias_mask`

## 5) Input Processing and Feature Construction

## 5.1 Monomer Builder

Main outputs include:
- sequence features (`aatype`, `seq_mask`, `residue_index`)
- MSA stacks (`msa`, `deletion_matrix`, masks)
- model-entry features (`target_feat`, `msa_feat`)
- extra MSA stack
- optional template tensors

Implementation:
- [`src/feature_pipeline/monomer.jl`](../src/feature_pipeline/monomer.jl)

## 5.2 Multimer Builder

Main outputs include:
- sequence and chain IDs (`aatype`, `asym_id`, `entity_id`, `sym_id`)
- paired/unpaired MSA according to pairing mode
- deletion/mask stacks
- optional multimer template stack

Implementation:
- [`src/feature_pipeline/multimer.jl`](../src/feature_pipeline/multimer.jl)

Pairing modes and semantics are documented in:
- [`README.md`](../README.md) (`Multimer Pairing Modes`)
- [`docs/INTERNAL_REPRESENTATIONS.md`](./INTERNAL_REPRESENTATIONS.md)

## 6) Forward Pipeline in Detail

Forward execution is orchestrated via composable pipeline stages in
[`src/inference.jl`](../src/inference.jl).

### Model construction (one-time)

Model construction is separate from inference:
- `_build_af2_model(arrs)` in [`src/model_builder.jl`](../src/model_builder.jl) creates an `AF2Model`
  from checkpoint arrays, instantiating all modules:
  - Evoformer blocks: [`src/modules/evoformer_iteration.jl`](../src/modules/evoformer_iteration.jl)
  - Extra MSA stack: same block type with global-column attention variant
  - Template embedding: [`src/modules/template_embedding.jl`](../src/modules/template_embedding.jl)
  - Structure module: [`src/modules/structure_module_core.jl`](../src/modules/structure_module_core.jl)
  - Output heads: [`src/modules/output_heads.jl`](../src/modules/output_heads.jl)
  - Confidence heads: [`src/modules/confidence_heads.jl`](../src/modules/confidence_heads.jl)

### Pipeline stages

**Stage 1: `prepare_inputs(model, dump)`** — non-differentiable preprocessing:
- Parse raw feature Dict into typed tensors
- Build/normalize `target_feat`, `msa_feat`, `extra_msa_feat` in feature-first layout
- Transfer to GPU if model is on GPU
- Returns `AF2PreparedInputs` (30 fields)

**Stage 2: `initial_recycle_state(model, inputs)`** — zero-init:
- Allocates zero tensors for `msa_first_row`, `pair`, `atom37`
- Returns `RecycleState`

**Stage 3: `run_evoformer(model, inputs, prev)`** — one evoformer cycle:
- Preprocessing projections (1D target, MSA, pair)
- Recycling features (previous distogram, MSA first row, pair activations)
- Relative position encoding
- Template pair embedding
- Extra MSA stack
- Full evoformer block stack
- Returns `(msa_act, pair_act)` tensors

**Stage 4: `run_heads(model, msa_act, pair_act, inputs)`** — structure + heads:
- Single projection from first MSA row
- Structure module (IPA, backbone update, sidechain)
- atom14 → atom37 conversion
- Output heads: masked MSA, distogram, experimentally resolved
- Confidence heads: pLDDT, PAE, pTM (multimer)
- Returns NamedTuple with all outputs

**Stage 5: `run_inference(model, inputs)`** — full recycle loop:
- Calls stages 2–4 for `num_iters` iterations
- Updates `RecycleState` between iterations
- Computes Cα distance metrics
- Returns `AF2InferenceResult`

### Legacy entry point

`_infer(model, dump)` wraps `prepare_inputs` → `run_inference` for backward compatibility.

## 7) Layer/Module Breakdown

Attention and pair/MSA logic:
- [`src/modules/attention.jl`](../src/modules/attention.jl)
- [`src/modules/msa_attention.jl`](../src/modules/msa_attention.jl)
- [`src/modules/triangle_attention.jl`](../src/modules/triangle_attention.jl)
- [`src/modules/triangle.jl`](../src/modules/triangle.jl)
- [`src/modules/outer_product_mean.jl`](../src/modules/outer_product_mean.jl)
- [`src/modules/transition.jl`](../src/modules/transition.jl)

Template stack:
- [`src/modules/template_pair_stack.jl`](../src/modules/template_pair_stack.jl)
- [`src/modules/template_embedding.jl`](../src/modules/template_embedding.jl)
- [`src/modules/template_single_rows.jl`](../src/modules/template_single_rows.jl)

Structure path:
- [`src/modules/ipa.jl`](../src/modules/ipa.jl)
- [`src/modules/fold_iteration_core.jl`](../src/modules/fold_iteration_core.jl)
- [`src/modules/sidechain.jl`](../src/modules/sidechain.jl)
- [`src/modules/structure_module_core.jl`](../src/modules/structure_module_core.jl)

Heads:
- [`src/modules/output_heads.jl`](../src/modules/output_heads.jl)
- [`src/modules/confidence_heads.jl`](../src/modules/confidence_heads.jl)

## 8) Recycling Semantics

Recycling is explicit via the `RecycleState` type. Each iteration carries forward:
- `msa_first_row`: first-row MSA activations `(c_m, L, 1)`
- `pair`: pair activations `(c_z, L, L, 1)`
- `atom37`: predicted atom coordinates `(1, L, 37, 3)` on CPU (used for distogram)

The recycle loop runs in `run_inference()`:

```julia
prev = initial_recycle_state(model, inputs)
for i in 1:inputs.num_iters
    msa_act, pair_act = run_evoformer(model, inputs, prev)
    heads = run_heads(model, msa_act, pair_act, inputs)
    prev = RecycleState(
        msa_first_row = view(msa_act, :, 1, :, :),
        pair = pair_act,
        atom37 = heads.atom37,
    )
end
```

Users can replicate this loop manually to inspect or modify state between iterations.

Implementation:
- [`src/inference.jl`](../src/inference.jl)

## 9) Coordinates: How Tensors Become Structure

Core path:
1. `StructureModuleCore` predicts rigid trajectories and sidechain torsions.
2. Sidechain module (`MultiRigidSidechain`) produces atom14 coordinates in local/global frames.
3. `make_atom14_masks!` builds atom14/atom37 index/mask maps from residue type.
4. `atom14_to_atom37` maps atom14 predictions to atom37 layout.

Implementation references:
- [`src/modules/structure_module_core.jl`](../src/modules/structure_module_core.jl)
- [`src/modules/sidechain.jl`](../src/modules/sidechain.jl)
- [`src/openfold_feats.jl`](../src/openfold_feats.jl)
- [`src/openfold_infer_utils.jl`](../src/openfold_infer_utils.jl)

PDB export:
- chain assignment from `asym_id`
- residue numbering from `residue_index` (+1 for PDB)
- pLDDT written as per-residue B-factor

Code:
- `_write_fold_pdb` in [`src/inference.jl`](../src/inference.jl)

## 10) Confidence Outputs

Head modules:
- `PredictedLDDTHead`
- `PredictedAlignedErrorHead`

Utilities:
- `compute_plddt`
- `compute_predicted_aligned_error`
- `compute_tm`

Code:
- [`src/modules/confidence_heads.jl`](../src/modules/confidence_heads.jl)
- [`src/openfold_infer_utils.jl`](../src/openfold_infer_utils.jl)

Notes:
- `pLDDT` is computed from logits softmax-weighted bin centers and scaled to `[0,100]`.
- PAE uses predicted aligned error distributions over distance bins.
- `compute_tm` supports optional interface mode using chain IDs.

## 11) Monomer vs Multimer Separation in Code

Checkpoint/model-family branching is handled in `_build_af2_model()` and `_infer()`:
- multimer detection uses checkpoint key structure.
- key branch points:
1. relative position features (`_relpos_one_hot` vs `_multimer_relpos_features`)
2. template embedding implementation (`TemplateEmbedding` vs `TemplateEmbeddingMultimer`)
3. IPA variant (`InvariantPointAttention` vs `MultimerInvariantPointAttention`)
4. structure position scale differences
5. multimer-native MSA row sampling path before feature build

Primary files:
- [`src/model_builder.jl`](../src/model_builder.jl)
- [`src/inference.jl`](../src/inference.jl)

## 12) Validation Surfaces

Unit/module checks:
- `test/runtests.jl`

Regression (pure Julia end-to-end inputs->PDB):
- cases in [`scripts/regression/regression_cases.jl`](../scripts/regression/regression_cases.jl)
- generator in [`scripts/regression/generate_reference_pdbs.jl`](../scripts/regression/generate_reference_pdbs.jl)

Gradient checks:
- [`scripts/gradients/check_full_model_zygote.jl`](../scripts/gradients/check_full_model_zygote.jl)

## 13) User-Facing Doc Entry Points

- Getting started and status: [`README.md`](../README.md)
- Internal feature contracts: [`docs/INTERNAL_REPRESENTATIONS.md`](./INTERNAL_REPRESENTATIONS.md)
- Template deep dive: [`docs/TEMPLATE_PROCESSING.md`](./TEMPLATE_PROCESSING.md)
