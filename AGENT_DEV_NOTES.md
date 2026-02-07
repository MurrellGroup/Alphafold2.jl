# Agent Dev Notes

Internal runbook for this specific workstation/repo layout.

This file holds environment-specific commands that were previously in `README.md`.

## Pinned Local Environment

```bash
export JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM
export JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot
export JULIA_PKG_OFFLINE=true
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_BIN=/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia
```

## Fast Commands

### Julia tests

```bash
env JULIA_PROJECT=$JULIA_PROJECT JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH JULIA_PKG_OFFLINE=$JULIA_PKG_OFFLINE JULIA_PKG_PRECOMPILE_AUTO=$JULIA_PKG_PRECOMPILE_AUTO \
  $JULIA_BIN --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/test/runtests.jl
```

### Full real-weight parity sweep

```bash
bash /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/run_real_weight_checks.sh
```

Expected final line:

```text
All configured parity checks passed.
```

### Full-model Zygote gradients

```bash
env JULIA_PROJECT=$JULIA_PROJECT JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH JULIA_PKG_OFFLINE=$JULIA_PKG_OFFLINE JULIA_PKG_PRECOMPILE_AUTO=$JULIA_PKG_PRECOMPILE_AUTO \
  $JULIA_BIN --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/gradients/check_full_model_zygote.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  ACDEFGHIK
```

## End-to-End Template-Conditioned Run (Python -> Julia)

Python reference + pre-evo dump:

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_template_case_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --params /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1_ptm.npz \
  --template-pdb /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold/alphafold/common/testdata/glucagon.pdb \
  --template-chain A \
  --sequence HSQGTFTSDYSKYLDSRRAQDFVQWLMNT \
  --msa-file /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/glucagon_test.a3m \
  --num-recycle 1 \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_py_docs_r1_ptm.npz \
  --dump-pre-evo /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_pre_evo_docs_r1_ptm.npz
```

Julia hybrid parity run on that dump:

```bash
env JULIA_PROJECT=$JULIA_PROJECT JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH JULIA_PKG_OFFLINE=$JULIA_PKG_OFFLINE JULIA_PKG_PRECOMPILE_AUTO=$JULIA_PKG_PRECOMPILE_AUTO \
  $JULIA_BIN --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_template_hybrid_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1_ptm.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_pre_evo_docs_r1_ptm.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_jl_hybrid_docs_r1_ptm.npz
```

PDB output from this run:

- `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_jl_hybrid_docs_r1_ptm.pdb`

## Related Internal Docs

- `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/docs/PARITY_CHECK_NOTES.md`
- `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/docs/RUNNABLE_EXAMPLES.md`

## Pure Julia PDB Regression Suite

Status update (2026-02-07):
- End-to-end pure-Julia reference PDB fixtures are generated for:
  - monomer: seq-only, MSA-only, template-only, template+MSA
  - multimer: seq-only, MSA-only, template+MSA
- Fixtures are stored in:
  - `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/test/regression/reference_pdbs`
- Regression runner:
  - `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/test/pure_julia_regression_pdb.jl`
  - enabled with `AF2_RUN_PURE_JULIA_REGRESSION=1`
- Reference checks are strict on PDB atom identity/order and coordinate deltas.
- Geometry fields in output NPZ remain reported for diagnostics but are not used as
  hard pass/fail gates in the regression test (to avoid false negatives on weak
  small-chain, no-template multimer cases that still match reference exactly).

Implementation note (important):
- `NPZ.jl` currently throws EOF read errors for some NPZ files containing
  zero-sized template arrays (`template_*` with leading dim 0).
- Workaround in `scripts/end_to_end/build_multimer_input_jl.jl`:
  when no templates are provided, omit `template_*` keys entirely.
- `run_af2_template_hybrid_jl.jl` already handles missing template keys by
  synthesizing zero-template tensors internally, preserving model behavior.
