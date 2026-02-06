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
