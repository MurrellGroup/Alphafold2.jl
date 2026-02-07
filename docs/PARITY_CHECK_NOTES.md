## Parity Check Notes

This file is the parity-notes section migrated from `README.md`.
The original notes were preserved and copied here verbatim for maintenance.

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

## Multimer end-to-end parity check (recycle loop)

Use the multimer runner to dump Python pre-evo activations, then run Julia hybrid parity
on the exact same dump.

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_multimer_case_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --params /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/params_npz/params_model_1_multimer_v3.npz \
  --sequences MKQLEDKVEELLSKNYHLENEVARLKKLV,MKQLEDKVEELLSKNYHLENEVARLKKLV \
  --num-recycle 5 \
  --num-msa 64 \
  --num-extra-msa 128 \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_py_r5.npz \
  --dump-pre-evo /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_pre_evo_r5.npz

env JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
  JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
  JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  /Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_template_hybrid_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/params_npz/params_model_1_multimer_v3.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_pre_evo_r5.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_jl_hybrid_r5.npz
```

Multimer parity with user-provided per-chain A3Ms (same recycle-5 loop):

```bash
JAX_PLATFORMS=cpu PYTHONPATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_multimer_case_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --params /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/params_npz/params_model_1_multimer_v3.npz \
  --sequences MKQLEDKVEELLSKNYHLENEVARLKKLV,MKQLEDKVEELLSKNYHLENEVARLKKLV \
  --msa-files /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/gcn4_chainA_test.a3m,/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/gcn4_chainB_test.a3m \
  --num-recycle 5 \
  --num-msa 64 \
  --num-extra-msa 128 \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_with_msa_py_r5.npz \
  --dump-pre-evo /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_with_msa_pre_evo_r5.npz

env JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
  JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
  JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  /Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_template_hybrid_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/params_npz/params_model_1_multimer_v3.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_with_msa_pre_evo_r5.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_with_msa_jl_hybrid_r5.npz
```

Observed parity envelope for this recycle-5 + per-chain-MSA case:
- `single_max_abs <= 0.0025634766`
- `pair_max_abs <= 0.00048828125`
- `out_atom37 max_abs <= 2.2053719e-05`
- `out_plddt max_abs <= 5.3405762e-05`
- `predicted_aligned_error max_abs <= 1.0490417e-05`

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

This workflow uses a single official AF2 checkpoint (`params_model_1.npz`, ~356MB) and tiny random tensors.

If your weights are staged under `/Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/params_npz`,
you can copy one checkpoint to the expected local parity path with:

```bash
mkdir -p /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights
cp /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/params_npz/params_model_1.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz
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
