# Alphafold2.jl

Julia port of AlphaFold2 with feature-first, batch-last tensor conventions.

Current scope includes:
- Evoformer trunk and structure module core (real AF2 weights)
- Output heads (`masked_msa`, `distogram`, `experimentally_resolved`)
- Confidence heads (`predicted_lddt`, `predicted_aligned_error`/`pTM` when PTM weights are present)
- End-to-end template-conditioned inference with user-provided MSA/template inputs
- PDB export plus geometry sanity metrics (consecutive C-alpha distances)
- Python parity harnesses against official AF2 modules
- Zygote gradient checks through full model forward path (parameter grads + soft-sequence input grads)

## Where To Start

- Runnable command cookbook: `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/docs/RUNNABLE_EXAMPLES.md`
- Full parity command reference (migrated from the previous README, preserved verbatim):
  `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/docs/PARITY_CHECK_NOTES.md`

## Environment

Most scripts are run with the same Julia environment used for ESM tooling:

```bash
export JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM
export JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot
export JULIA_PKG_OFFLINE=true
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_BIN=/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia
```

## Quick Commands

### 1) Run Julia module tests

```bash
env JULIA_PROJECT=$JULIA_PROJECT JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH JULIA_PKG_OFFLINE=$JULIA_PKG_OFFLINE JULIA_PKG_PRECOMPILE_AUTO=$JULIA_PKG_PRECOMPILE_AUTO \
  $JULIA_BIN --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/test/runtests.jl
```

### 2) Run full real-weight parity sweep

```bash
bash /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/run_real_weight_checks.sh
```

Expected final line:

```text
All configured parity checks passed.
```

### 3) Run end-to-end template-conditioned inference (Python -> Julia)

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

## Gradients (Zygote)

Full-stack gradient check (48 Evoformer blocks + 8 structure layers) with loss `mean(plddt)`:

```bash
env JULIA_PROJECT=$JULIA_PROJECT JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH JULIA_PKG_OFFLINE=$JULIA_PKG_OFFLINE JULIA_PKG_PRECOMPILE_AUTO=$JULIA_PKG_PRECOMPILE_AUTO \
  $JULIA_BIN --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/gradients/check_full_model_zygote.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  ACDEFGHIK
```

What this checks:
- Non-zero parameter gradients through the full model forward path.
- Non-zero gradients with respect to a soft sequence input (`seq_logits`, not strict one-hot).

Observed passing output (example):

```text
Zygote full-model gradient check
  sequence length: 9
  evoformer blocks used: 48 / 48
  structure layers: 8
  loss (mean pLDDT): 67.299850
  seq_grad_l1: 2832.9614
  param_grad_l1: 4.120176e+10
  ...
PASS
```

## Notes

- MSA/template search is intentionally out of scope; use user-provided MSA/template inputs.
- Amber relax is out of scope.
- PTM/PAE outputs are available when PTM checkpoints are used (e.g., `params_model_1_ptm.npz`).
