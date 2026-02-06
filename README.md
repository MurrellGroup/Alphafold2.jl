# Alphafold2.jl

WIP Julia port of AlphaFold2 with feature-first, batch-last tensor conventions.

Current milestone:
- Core tensor and layer utilities (`LinearFirst`, `LayerNormFirst`)
- Geometry utilities (`rigid.jl`)
- AF2 `TriangleMultiplication` module in Julia
- AF2 `TriangleAttention` module in Julia
- AF2 `Transition` module in Julia
- AF2 `OuterProductMean` module in Julia
- AF2 `MSARowAttentionWithPairBias` module in Julia
- AF2 `MSAColumnAttention` module in Julia
- AF2 `MSAColumnGlobalAttention` module in Julia
- AF2 `InvariantPointAttention` module in Julia
- AF2 deterministic `EvoformerIteration` module in Julia
- AF2 `TemplatePairStack` module in Julia
- AF2 `TemplateEmbedding` / `SingleTemplateEmbedding` module in Julia
- AF2 template torsion + `TemplateSingleRows` module in Julia
- AF2 `FoldIterationCore` module in Julia
- AF2 `GenerateAffinesCore` module in Julia
- AF2 `MultiRigidSidechain` module in Julia
- AF2 `StructureModuleCore` (affine + sidechain trajectories) in Julia
- OpenFold/AF2 geometry helpers (`torsion_angles_to_frames`, `frames_and_literature_positions_to_atom14_pos`, `atom14_to_atom37`) and residue constants
- Python-to-Julia parity harness for each of the modules above
- Real-checkpoint parity harness (`params_model_1.npz`) for structure/evoformer modules

## Run tests

```bash
JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/test/runtests.jl
```

## TriangleMultiplication parity check

1) Dump official AF2 module activations and parameters:

```bash
python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/dump_triangle_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/triangle_outgoing_dump.npz
```

2) Check Julia parity against that dump:

```bash
JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/check_triangle_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/triangle_outgoing_dump.npz
```

Use `--incoming` on the Python script to check the incoming equation variant.

## TriangleAttention parity check

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/dump_triangle_attention_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/triangle_attention_row_dump.npz

JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/check_triangle_attention_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/triangle_attention_row_dump.npz
```

Use `--column` on the Python script to check the per-column orientation.

## Transition parity check

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/dump_transition_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/transition_dump.npz

JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/check_transition_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/transition_dump.npz
```

## OuterProductMean parity check

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/dump_outer_product_mean_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/opm_dump.npz

JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/check_outer_product_mean_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/opm_dump.npz
```

## MSA attention parity checks

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/dump_msa_row_attention_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/msa_row_dump.npz

JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/check_msa_row_attention_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/msa_row_dump.npz

JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/dump_msa_column_attention_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/msa_col_dump.npz

JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/check_msa_column_attention_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/msa_col_dump.npz

JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/dump_msa_column_global_attention_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/msa_col_global_dump.npz

JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/check_msa_column_global_attention_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/msa_col_global_dump.npz
```

## TemplatePairStack parity check

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/dump_template_pair_stack_from_params_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --params /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/template_pair_stack_real_dump.npz

JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/check_template_pair_stack_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/template_pair_stack_real_dump.npz
```

## TemplateSingleRows parity check

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/dump_template_single_rows_from_params_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --params /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  --template-npz /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_py_r1.npz \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/template_single_rows_real_dump.npz

JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/check_template_single_rows_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/template_single_rows_real_dump.npz
```

## TemplateEmbedding parity check

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/dump_template_embedding_from_params_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --params /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  --input-npz /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_pre_evo_r1_fresh2.npz \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/template_embedding_real_dump.npz

JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/check_template_embedding_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/template_embedding_real_dump.npz
```

## PredictedLDDT parity check (real weights)

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/dump_predicted_lddt_from_params_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --params /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/predicted_lddt_real_dump.npz

JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/check_predicted_lddt_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/predicted_lddt_real_dump.npz
```

## PredictedAlignedError parity check (PTM real weights or synthetic-init fallback)

`params_model_1.npz` does not include `predicted_aligned_error_head` weights.  
Use `params_model_1_ptm.npz` for real-weight parity, or omit `--params` for synthetic-init parity.

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/dump_predicted_aligned_error_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --params /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1_ptm.npz \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/predicted_aligned_error_real_dump.npz

JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/check_predicted_aligned_error_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/predicted_aligned_error_real_dump.npz
```

## Distogram / MaskedMSA / ExperimentallyResolved parity checks (real weights)

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/dump_distogram_from_params_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --params /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/distogram_real_dump.npz

JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/dump_masked_msa_from_params_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --params /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/masked_msa_real_dump.npz

JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/dump_experimentally_resolved_from_params_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --params /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/experimentally_resolved_real_dump.npz

JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/check_distogram_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/distogram_real_dump.npz

JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/check_masked_msa_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/masked_msa_real_dump.npz

JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/check_experimentally_resolved_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/experimentally_resolved_real_dump.npz
```

## InvariantPointAttention parity check

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/dump_ipa_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/ipa_dump.npz

JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/check_ipa_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/ipa_dump.npz
```

## EvoformerIteration parity check

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/dump_evoformer_iteration_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --out-dir /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/evo_dump \
  --outer-first

JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/check_evoformer_iteration_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/evo_dump
```

## Real AF2 Weights (small-machine checks)

This workflow uses a single checkpoint file (`params_model_1.npz`, ~356MB) and tiny random tensors.

Download one checkpoint:

```bash
mkdir -p /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights
curl -L --fail -o /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  https://huggingface.co/josephdviviano/openfold/resolve/main/alphafold_params/params_model_1.npz
```

Run real-weight parity checks end-to-end:

```bash
/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/run_real_weight_checks.sh
```

This validates:
- Structure `InvariantPointAttention` against real AF2 weights.
- Evoformer iteration parity for blocks `0`, `23`, and `47` from the 48-block stack.
- `TemplateSingleRows` against real AF2 weights and real template atom inputs.
- `TemplateEmbedding` against real AF2 weights and real template atom inputs.
- `PredictedLDDTHead` against real AF2 weights and pLDDT helper parity.
- `DistogramHead`, `MaskedMsaHead`, and `ExperimentallyResolvedHead` against real AF2 weights.
- `PredictedAlignedErrorHead` + confidence helpers (`PAE`, `pTM`) with PTM real weights when available (otherwise synthetic-init fallback).
- Structure `FoldIterationCore` (single structure step minus sidechain feedback).
- `GenerateAffinesCore` (8-step structure loop core; tolerant to iterative FP accumulation).
- `MultiRigidSidechain` (real AF2 sidechain weights and outputs).
- `StructureModuleCore` end-to-end (affine + sidechain trajectories with per-metric tolerances).

## End-to-end core fold run (small sequence)

This repository now includes a minimal end-to-end AF2 core path:

- sequence -> minimal AF2-style `target_feat`/`msa_feat`
- 48-block Evoformer trunk
- structure module fold loop + sidechain
- final `atom14` and `atom37` coordinates

Run a tiny-sequence Python reference (official AF2 modules, same checkpoint):

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_core_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --params /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  --sequence ACDEFGHIKLMN \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_core_py_ref.npz
```

Run the Julia end-to-end fold:

```bash
JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_OFFLINE=true \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_core_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  ACDEFGHIKLMN \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_core_jl_out.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_core_py_ref.npz
```

The Julia script writes:

- `out_atom14` with shape `(L, 14, 3)`
- `out_atom37` with shape `(L, 37, 3)`
- `atom37_mask` with shape `(L, 37)`

## Template-conditioned plausible case (glucagon)

This run uses a real local template (`alphafold/common/testdata/glucagon.pdb`, chain `A`) and gives physically plausible consecutive C-alpha spacing.

Python (official AF2 modules, template + extra-MSA + recycle, and pre-evo dump):

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_template_case_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --params /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  --template-pdb /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold/alphafold/common/testdata/glucagon.pdb \
  --template-chain A \
  --num-recycle 1 \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_py_r1.npz \
  --dump-pre-evo /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_pre_evo_r1.npz
```

`run_af2_template_case_py.py` auto-selects `model_1_ptm` when the params filename contains `_ptm` (or use `--model-name model_1_ptm` explicitly). PTM dumps include predicted aligned error and pTM outputs for parity checks.

Julia parity on the same complex per-iteration pre-evo activations:

```bash
JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_OFFLINE=true \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_from_pre_evo_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_pre_evo_r1.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_jl_from_preevo_r1.npz
```

Julia hybrid parity path (native recycle embedding + relpos + extra-MSA feature creation + template embedding + template single-row construction):

```bash
JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_OFFLINE=true \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_template_hybrid_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_pre_evo_r1.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_jl_hybrid_r1.npz
```

Julia native template run (same script, no pre-evo parity tensors required; uses only `aatype`, `seq_mask`, and template atom inputs):

```bash
JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_OFFLINE=true \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_template_hybrid_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_py_r1_fresh2.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_jl_native_r1.npz
```

Julia-only template input build from a PDB chain (no Python input preparation):

```bash
JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/build_template_input_from_pdb_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold/alphafold/common/testdata/glucagon.pdb \
  A \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_input_native.npz \
  "" \
  1 \
  ""
```

You can also pass comma-separated template PDBs and chains (for multiple templates), e.g.  
`<pdb1,pdb2>` and `<A,B>`.

Pass an A3M/FASTA MSA file as the 6th argument to include user-provided MSA rows:

```bash
JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/build_template_input_from_pdb_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold/alphafold/common/testdata/glucagon.pdb \
  A \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_input_msa_native.npz \
  "" \
  1 \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/glucagon_test.a3m
```

If the provided query sequence length differs from the template chain length, the builder performs a global sequence alignment and maps template residues onto query positions (unmapped query positions get unknown-template placeholders).

```bash
JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
JULIA_PKG_OFFLINE=true \
JULIA_PKG_PRECOMPILE_AUTO=0 \
/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_template_hybrid_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_input_native.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_jl_native_from_pdb.npz
```

This writes both:
- `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_jl_hybrid_r1.npz`
- `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_jl_hybrid_r1.pdb`

The output NPZ now also includes confidence outputs:
- `out_predicted_lddt_logits` (shape `(L, num_bins)`)
- `out_plddt` (shape `(L,)`)
- `mean_plddt`, `min_plddt`, `max_plddt`

And AF2 head outputs:
- `out_masked_msa_logits`
- `out_distogram_logits`
- `out_distogram_bin_edges`
- `out_experimentally_resolved_logits`

If `predicted_aligned_error_head` weights are present in the checkpoint (e.g., PTM models), it also writes:
- `predicted_aligned_error`
- `max_predicted_aligned_error`
- `predicted_tm_score`

PDB files written by `run_af2_template_hybrid_jl.jl` carry per-residue pLDDT in the B-factor column.

`run_af2_template_hybrid_jl.jl` also supports user-provided MSA inputs in the input NPZ:
- `msa` (shape `(N_seq, L)`, HHblits-style IDs `0..21`)
- `deletion_matrix` (shape `(N_seq, L)`, deletion counts)
- optional `extra_msa` and `extra_deletion_matrix`
- optional `residue_index`

If these are missing, it falls back to single-sequence features.
