# Current Status

Last updated: 2026-02-07

This document is a developer-facing status/checkpoint for AF2 Julia parity work.
It focuses on:
- input tensor parity (Julia builders vs Python reference builders),
- end-to-end PDB parity (Julia inference vs Python official references),
- what is currently passing/failing,
- what to fix next.

## 1) How To Run Parity Checks

### 1.1 Input Tensor Parity (Python + Julia)

This is the canonical check. It runs Julia native builders and compares against Python reference feature builders.

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/regression/check_input_tensor_parity.py \
  --repo-root /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --min-msa-rows-multimer 129
```

Expected final line:

```text
Input tensor parity PASSED for all checked cases.
```

Notes:
- This script supports both TOML and JSON manifests.
- Since the old Julia `manifest.toml` was removed, it now falls back to:
  `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/test/regression/reference_pdbs/manifest_python_official.json`.

### 1.2 Input Builder Debug Examples (Julia-only / Python-only)

Julia monomer builder example:

```bash
env JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
  JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
  JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  /Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/build_monomer_input_jl.jl \
  ACDEFGHIK /tmp/af2_debug_monomer_input.npz 1 \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/test/regression/msa/monomer_short.a3m
```

Python reference builder example (single case):

```bash
python3.11 - <<'PY'
import sys, pathlib, importlib.util, numpy as np, json
repo = pathlib.Path('/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl')
spec = importlib.util.spec_from_file_location('pyref', repo/'scripts/regression/generate_python_official_reference_pdbs.py')
mod = importlib.util.module_from_spec(spec); sys.modules['pyref']=mod; spec.loader.exec_module(mod)
sys.path.insert(0, '/Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold')
from alphafold.common import residue_constants
from alphafold.data import pipeline, parsers
raw = mod._build_monomer_raw_features(
    sequence='ACDEFGHIK',
    msa_file=str(repo/'test/regression/msa/monomer_short.a3m'),
    template_pdb=None,
    template_chain='A',
    pipeline=pipeline,
    parsers=parsers,
    residue_constants=residue_constants,
)
print('raw msa shape:', raw['msa'].shape)
print('raw deletion shape:', raw['deletion_matrix_int'].shape)
PY
```

### 1.3 End-to-End PDB Parity

#### A) Generate Python official reference PDBs

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/regression/generate_python_official_reference_pdbs.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --input-manifest /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/test/regression/reference_pdbs/manifest_python_official.json \
  --output-dir /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/test/regression/reference_pdbs \
  --suffix .python_official \
  --seed 0 \
  --min-msa-rows-multimer 129
```

#### B) Run pure-Julia E2E cases and strict compare to Python official PDBs (Julia)

```bash
env JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
  JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
  JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  AF2_MULTIMER_MIN_MSA_ROWS=129 \
  /Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/regression/check_pdb_parity_against_python.jl \
  /tmp/af2_pdb_parity_run
```

This prints strict atom-key + coordinate deltas case-by-case.

#### C) Relaxed structural compare (Python)

```bash
python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/regression/check_pdb_parity_relaxed.py \
  --run-dir /tmp/af2_pdb_parity_run
```

This prints:
- `strict_identity_ok`: strict atom key match (atom/resname/chain/resseq in order),
- `order_atom_name_match`: only atom/resname order match,
- `max_abs_order`, `rms_order`: raw coordinate deltas in file order,
- `ca_aligned_rmsd`: Kabsch-aligned CA RMSD (best indicator for structural agreement).

## 2) Current Check Status

## 2.1 Input Tensor Parity

Current status: PASS on all regression cases.

Cases passing:
- `monomer_seq_only`
- `monomer_msa_only`
- `monomer_template_only`
- `monomer_template_msa`
- `multimer_seq_only`
- `multimer_msa_only`
- `multimer_template_msa`

Summary:
- `aatype`, `residue_index`, `msa`, `deletion_matrix`, `msa_mask` match.
- Multimer `asym_id`/`entity_id`/`sym_id`/`cluster_bias_mask` match.
- Template tensors match for template cases.

## 2.2 End-to-End PDB Parity (Latest Rerun)

Latest run dir used:
- `/tmp/af2_compare_rerun_20260207_1845`

Strict comparison (Julia helper):
- `monomer_seq_only`: strict key match `true`, but large coordinate deltas (`max_abs=15.265`, `rms=7.693`)
- `monomer_msa_only`: strict key match `true`, large deltas (`max_abs=16.469`, `rms=8.042`)
- `monomer_template_only`: strict key match `true`, much closer (`max_abs=2.127`, `rms=0.366`)
- `monomer_template_msa`: strict key match `true`, close (`max_abs=1.169`, `rms=0.406`)
- `multimer_seq_only`: strict key match `false` (`atom_identity`)
- `multimer_msa_only`: strict key match `false` (`atom_identity`)
- `multimer_template_msa`: strict key match `false` (`atom_identity`)

Relaxed comparison (Python helper):

```text
case,strict_identity_ok,order_atom_name_match,max_abs_order,rms_order,ca_aligned_rmsd
monomer_msa_only,true,true,16.469,13.93,1.38668
monomer_seq_only,true,true,15.265,13.3247,1.51195
monomer_template_msa,true,true,1.169,0.703438,0.0684627
monomer_template_only,true,true,2.127,0.634522,0.0401543
multimer_msa_only,false,true,22.456,12.5502,0.0868915
multimer_seq_only,false,true,27.308,13.1535,0.0565358
multimer_template_msa,false,true,21.754,12.4831,0.0659382
```

Interpretation:
- Multimer structures are close after alignment (low CA-aligned RMSD), but strict identity fails due chain label/order mismatch.
- Monomer no-template (`seq_only`, `msa_only`) is still not at end-to-end parity (large geometric delta).
- Monomer template-conditioned cases are substantially closer.

## 2.3 What Was Fixed Recently

- Input parity script now handles missing TOML manifest and falls back to JSON manifest.
- Monomer/template builders now use HHblits query-token logic + AF2 remap semantics and all-ones MSA mask convention at model-entry parity stage.
- MSA dedup behavior aligned with Python (`make_msa_features`-style dedup by aligned sequence).
- Multimer padding/mask behavior aligned to Python multimer `pad_msa` + `make_msa_mask`.
- Multimer template stack layout aligned to Python merged-template stack behavior used in current reference path.

## 3) Next Fixes (Priority)

1. Fix monomer no-template end-to-end parity gap (`monomer_seq_only`, `monomer_msa_only`).
   - Focus on monomer inference path behavior after input build (feature processing / evoformer entry / recycle update path).
   - Add additional intermediate dumps for monomer official path and compare against Julia step-by-step.

2. Fix multimer strict identity mismatch in exported PDB chain labeling/order.
   - Align Julia chain ID ordering with Python reference output ordering so strict compare passes.
   - Keep structural parity (already strong in aligned RMSD).

3. Make parity reporting less ambiguous.
   - Replace confusing wording with explicit field names in output/reporting:
     - atom-key match vs coordinate-delta metrics.

4. Re-run full parity sweep after fixes.
   - input tensors,
   - PDB strict + relaxed,
   - confidence outputs (`pLDDT` / `PAE` / `pTM`) in end-to-end comparisons.
