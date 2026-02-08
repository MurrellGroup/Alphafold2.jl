## Runnable Examples

Last validated on: 2026-02-07

These commands were run successfully on this machine and cover the full in-scope AF2 port:
- core/module tests
- real-weight parity checks (all configured modules/heads)
- end-to-end template-conditioned inference (Python + Julia hybrid/native)
- end-to-end multimer inference (Python reference, Julia hybrid parity, Julia-native input)
- PDB export and geometry checks
- full-model Zygote gradients (parameter grads + soft-sequence input grads)

### 1) Module Test Sweep

```bash
env JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
  JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
  JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  /Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/test/runtests.jl
```

### 2) Full Real-Weight Parity Sweep

```bash
bash /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/parity/run_real_weight_checks.sh
```

Expected final line:

```text
All configured parity checks passed.
```

For the detailed parity command catalog, see `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/docs/PARITY_CHECK_NOTES.md`.

### 3) Build Template+MSA Input (Julia)

```bash
env JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
  JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
  JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  /Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/build_template_input_from_pdb_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold/alphafold/common/testdata/glucagon.pdb \
  A \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_input_docs_r1.npz \
  "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT" \
  1 \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/glucagon_test.a3m
```

### 4) Python AF2 Reference Run + Pre-Evo Dump

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

### 5) Julia Hybrid Parity Run (same pre-evo activations)

```bash
env JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
  JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
  JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  /Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_template_hybrid_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1_ptm.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_pre_evo_docs_r1_ptm.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_jl_hybrid_docs_r1_ptm.npz
```

### 6) Julia Native Run (from Julia-built input NPZ)

```bash
env JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
  JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
  JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  /Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_template_hybrid_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1_ptm.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_input_docs_r1.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_jl_native_docs_r1_ptm.npz
```

Generated PDBs:
- `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_jl_hybrid_docs_r1_ptm.pdb`
- `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_template_glucagon_jl_native_docs_r1_ptm.pdb`

### 7) Full-Model Zygote Gradient Check

```bash
env JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
  JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
  JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  /Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/gradients/check_full_model_zygote.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz \
  ACDEFGHIK
```

Expected terminal footer:

```text
Zygote full-model gradient check
...
seq_grad_l1: (positive)
param_grad_l1: (positive)
PASS
```

This script now supports both monomer-family and multimer checkpoints.

Multimer gradient check example:

```bash
env JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
  JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
  JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  /Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/gradients/check_full_model_zygote.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/params_npz/params_model_1_multimer_v3.npz \
  MKQLEDKVEELLSKNYHLENEVARLKKLV,MKQLEDKVEELLSKNYHLENEVARLKKLV \
  1
```

### 8) Multimer Python Reference Run (2-chain, no external A3Ms)

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_multimer_case_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --params /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/params_npz/params_model_1_multimer_v3.npz \
  --sequences MKQLEDKVEELLSKNYHLENEVARLKKLV,MKQLEDKVEELLSKNYHLENEVARLKKLV \
  --num-recycle 1 \
  --num-msa 64 \
  --num-extra-msa 128 \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_py_r1.npz \
  --dump-pre-evo /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_pre_evo_r1.npz
```

### 9) Multimer Python Run with User-Provided Per-Chain A3Ms

Sample A3Ms included in the repo:
- `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/gcn4_chainA_test.a3m`
- `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/gcn4_chainB_test.a3m`

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_multimer_case_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --params /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/params_npz/params_model_1_multimer_v3.npz \
  --sequences MKQLEDKVEELLSKNYHLENEVARLKKLV,MKQLEDKVEELLSKNYHLENEVARLKKLV \
  --msa-files /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/gcn4_chainA_test.a3m,/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/gcn4_chainB_test.a3m \
  --num-recycle 1 \
  --num-msa 64 \
  --num-extra-msa 128 \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_with_msa_py_r1.npz \
  --dump-pre-evo /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_with_msa_pre_evo_r1.npz
```

### 10) Multimer Julia Hybrid Parity Run (same pre-evo multimer dump)

```bash
env JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
  JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
  JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  /Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_template_hybrid_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/params_npz/params_model_1_multimer_v3.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_pre_evo_r1.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_jl_hybrid_r1.npz
```

Generated PDB:
- `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_jl_hybrid_r1.pdb`

### 11) Multimer Recycle-5 Parity Check (Python + Julia)

Python recycle-5 reference and dump:

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
```

Julia recycle-5 hybrid parity:

```bash
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

Validated parity at recycle 5 from this run:
- `single_max_abs`: `0.0035400391`
- `pair_max_abs`: `0.00033569336`
- `atom14_max_abs`: `1.2397766e-05`
- `traj3x4_max_abs`: `1.1205673e-05`

### 12) Multimer PDB Chain-ID and Interface Contact Sanity Check

Check chain IDs and `TER` records in exported Julia multimer PDB:

```bash
python3.11 - <<'PY'
from pathlib import Path
p = Path('/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_jl_hybrid_r5.pdb')
chains = sorted({line[21] for line in p.read_text().splitlines() if line.startswith('ATOM')})
ters = sum(1 for line in p.read_text().splitlines() if line.startswith('TER'))
print('chains', chains)
print('TER', ters)
PY
```

Check Python-vs-Julia interface-contact agreement for the same recycle-5 case:

```bash
python3.11 - <<'PY'
import numpy as np
py = np.load('/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_py_r5.npz')
jl = np.load('/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_jl_hybrid_r5.npz')
dump = np.load('/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_pre_evo_r5.npz')
asym = dump['asym_id'].astype(int)
ca_idx = 1

def contacts(atom37, mask):
    ca = atom37[:, ca_idx, :]
    m = mask[:, ca_idx] > 0.5
    a = np.where(asym == asym.min())[0]
    b = np.where(asym != asym.min())[0]
    d = []
    for i in a:
        if not m[i]:
            continue
        for j in b:
            if not m[j]:
                continue
            d.append(float(np.linalg.norm(ca[i] - ca[j])))
    d = np.array(d, np.float32)
    return float(d.min()), int((d < 8.0).sum()), int((d < 5.0).sum())

pmin, p8, p5 = contacts(py['out_atom37'], py['atom37_mask'])
jmin, j8, j5 = contacts(jl['out_atom37'], jl['atom37_mask'])
print('python min/p8/p5', pmin, p8, p5)
print('julia  min/p8/p5', jmin, j8, j5)
print('atom37 max abs', float(np.abs(py['out_atom37'] - jl['out_atom37']).max()))
PY
```

### 13) Build Native Multimer Input in Julia (from sequences + per-chain A3Ms)

```bash
env JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
  JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
  JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  /Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/build_multimer_input_jl.jl \
  "MKQLEDKVEELLSKNYHLENEVARLKKLV,MKQLEDKVEELLSKNYHLENEVARLKKLV" \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_native_input_r5.npz \
  5 \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/gcn4_chainA_test.a3m,/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/gcn4_chainB_test.a3m
```

Per-chain multi-template groups are supported using `+` inside each chain entry.
Example (chain A has 2 template rows, chain B has 1 template row):

```bash
env JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
  JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
  JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  /Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/build_multimer_input_jl.jl \
  "MKQLEDKVEELLSKNYHLENEVARLKKLV,MKQLEDKVEELLSKNYHLENEVARLKKLV" \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_native_input_template_uneven_r5.npz \
  5 \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/gcn4_chainA_test.a3m,/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/gcn4_chainB_test.a3m \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/test/regression/templates/gcn4_dimer_template.pdb+/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/test/regression/templates/gcn4_dimer_template.pdb,/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/test/regression/templates/gcn4_dimer_template.pdb \
  A+B,B
```

Pairing mode examples (`build_multimer_input_jl.jl` arg 7):

```bash
# Pair by row index
.../build_multimer_input_jl.jl <seqs_csv> <out.npz> <recycles> <msa_csv> '' '' 'pair by row index' 0

# Pair by taxon labels
.../build_multimer_input_jl.jl <seqs_csv> <out.npz> <recycles> <msa_csv> '' '' 'taxon labels matched' 0

# Random pairing with deterministic seed
.../build_multimer_input_jl.jl <seqs_csv> <out.npz> <recycles> <msa_csv> '' '' 'random pairing' 7
```

Partial per-chain inputs are accepted:
- Missing MSA for one chain: pass an empty CSV slot, e.g. `msa_chainA.a3m,`
- Missing template for one chain: pass an empty CSV slot in both template args, e.g. `templateA.pdb,` and `A,`

### 14) Multimer Julia Native Run (recycle=5, from Julia-built multimer input)

```bash
env JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
  JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
  JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  /Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_template_hybrid_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/params_npz/params_model_1_multimer_v3.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_native_input_r5.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_jl_native_r5.npz
```

Generated PDB:
- `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_jl_native_r5.pdb`

Expected geometry output now includes both overall and intra-chain C-alpha metrics for multimers:
- `mean`, `std`, `min`, `max`, `outlier_fraction`
- `intra_chain_mean`, `intra_chain_std`, `intra_chain_min`, `intra_chain_max`, `intra_chain_outlier_fraction`

### 15) Audit and Convert All Official AF2 Weights to Safetensors

```bash
python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/weights/audit_and_convert_af2_weights.py \
  --weights-root /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official \
  --skip-download \
  --skip-extract
```

This writes:
- `/Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/params_safetensors` (15 safetensors files)
- `/Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/AF2_WEIGHTS_AUDIT.json`
- `/Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/AF2_WEIGHTS_AUDIT.md`

### 16) Multimer + Templates Hybrid Parity (Python r=5 -> Julia)

Use an existing 2-chain multimer PDB as per-chain template source (`A` and `B`).

```bash
JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_multimer_case_py.py \
  --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold \
  --params /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/params_npz/params_model_1_multimer_v3.npz \
  --sequences MKQLEDKVEELLSKNYHLENEVARLKKLV,MKQLEDKVEELLSKNYHLENEVARLKKLV \
  --msa-files /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/gcn4_chainA_test.a3m,/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/gcn4_chainB_test.a3m \
  --template-pdbs /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_jl_native_r5.pdb,/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_jl_native_r5.pdb \
  --template-chains A,B \
  --num-recycle 5 \
  --out /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_template_py_r5.npz \
  --dump-pre-evo /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_template_pre_evo_r5.npz

env JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
  JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
  JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  /Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_template_hybrid_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/params_npz/params_model_1_multimer_v3.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_template_pre_evo_r5.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_template_jl_hybrid_r5.npz
```

Observed parity envelope for this case:
- `template_embed_max_abs <= 1.7e-5`
- `pair_max_abs <= 4.3e-4`
- `out_atom37 max_abs <= 2.3e-5`

### 17) Pure Julia Native Multimer + Templates (input + run)

Build multimer input in Julia (sequences + per-chain A3Ms + per-chain templates):

```bash
env JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
  JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
  JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  /Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/build_multimer_input_jl.jl \
  "MKQLEDKVEELLSKNYHLENEVARLKKLV,MKQLEDKVEELLSKNYHLENEVARLKKLV" \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_template_native_input_r1.npz \
  1 \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/gcn4_chainA_test.a3m,/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/gcn4_chainB_test.a3m \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_jl_native_r5.pdb,/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_jl_native_r5.pdb \
  A,B
```

Run native Julia end-to-end:

```bash
env JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM \
  JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot \
  JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  /Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia \
  --startup-file=no --history-file=no \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_template_hybrid_jl.jl \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/params_npz/params_model_1_multimer_v3.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_template_native_input_r1.npz \
  /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_template_jl_native_r1.npz
```

Generated PDB:
- `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/af2_multimer_gcn4_template_jl_native_r1.pdb`
