# Alphafold2.jl

`Alphafold2.jl` is a pure Julia implementation of AlphaFold2 using feature-first, batch-last tensor layout.

## Status

Implemented:
- Evoformer trunk modules
- Structure module core and sidechain path
- Output heads: `MaskedMsaHead`, `DistogramHead`, `ExperimentallyResolvedHead`
- Confidence heads: `PredictedLDDTHead`, `PredictedAlignedErrorHead`
- Confidence utilities: `compute_plddt`, `compute_predicted_aligned_error`, `compute_tm`
- Multimer structure path (including recycle loop and chain-aware PDB export)
- Multimer template embedding + template-row integration (recycle loop)
- In-process feature pipeline (monomer + multimer, no subprocess or intermediate files)

In scope:
- User-provided MSAs and template structures

Out of scope:
- MSA/template search pipelines
- Amber relax

## Documentation

See `docs/INDEX.md` for the full documentation map.

Key docs:
1. Full codebase walkthrough: `docs/CODEBASE_GUIDE.md`
2. Template internals: `docs/TEMPLATE_PROCESSING.md`
3. Internal feature contracts: `docs/INTERNAL_REPRESENTATIONS.md`

## Installation

From a local checkout:

```julia
using Pkg
Pkg.develop(path="path/to/Alphafold2.jl")
Pkg.instantiate()
```

If your resolver cannot find `Onion`, add it as a path dependency first:

```julia
using Pkg
Pkg.develop(path="path/to/Onion.jl")
Pkg.develop(path="path/to/Alphafold2.jl")
Pkg.instantiate()
```

## Quickstart

### REPL Workflow

Load a model once, then call `fold(...)` repeatedly:

```julia
using Alphafold2

# Monomer
mono_model = load_monomer()
r = fold("ACDEFGHIK"; num_recycle=3, out_prefix="/tmp/af2_mono")
@show r.out_pdb r.mean_plddt

# Multimer
multi_model = load_multimer()
r = fold(
    ["MKQLEDKVEELLSKNYHLENEVARLKKLV", "MKQLEDKVEELLSKNYHLENEVARLKKLV"];
    msas=["chainA.a3m", "chainB.a3m"],
    pairing_mode="block diagonal",
    num_recycle=5,
    out_prefix="/tmp/af2_multi",
)
@show r.out_pdb r.mean_plddt r.ptm
```

Notes:
- `load_monomer()` / `load_multimer()` download weights from HuggingFace on first use.
- `fold(...)` writes both `*_out.npz` and `*_out.pdb`; `FoldResult` returns both paths plus summary confidence values.
- You can also call `fold(model, sequence_or_sequences; ...)` explicitly.

### Confidence utilities

```julia
using Alphafold2

L, B = 16, 1

pae_logits = randn(Float32, 64, L, L, B)
pae = compute_predicted_aligned_error(pae_logits)
tm_score = compute_tm(pae_logits)

lddt_logits = randn(Float32, 50, L, B)
plddt = compute_plddt(lddt_logits)

@show size(pae[:predicted_aligned_error])  # (L, L, B)
@show tm_score
@show size(plddt)                          # (L, B)
```

### Pipeline API

The inference pipeline is decomposed into composable stages for research workflows.
Each stage can be called independently.

```
build_monomer_features / build_multimer_features
    ↓
prepare_inputs  →  initial_recycle_state
    ↓                    ↓
    └──────→ run_evoformer(model, inputs, prev) → (msa_act, pair_act)
                 ↓
             run_heads(model, msa_act, pair_act, inputs) → NamedTuple
                 ↓
             (update RecycleState, loop back)
                 ↓
             run_inference(model, inputs) → AF2InferenceResult   # wraps the loop
```

**Stage reference:**

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `prepare_inputs(model, features)` | raw feature Dict | `AF2PreparedInputs` | Parse + device transfer |
| `initial_recycle_state(model, inputs)` | prepared inputs | `RecycleState` | Zero-init recycle state |
| `run_evoformer(model, inputs, prev)` | inputs + prev state | `(msa_act, pair_act)` | One evoformer cycle |
| `run_heads(model, msa_act, pair_act, inputs)` | evoformer output | NamedTuple | Structure module + all heads |
| `run_inference(model, inputs)` | prepared inputs | `AF2InferenceResult` | Full recycle loop |

**Example: manual recycle loop with inspection:**

```julia
using Alphafold2

model = load_monomer()
features = build_monomer_features("MKQLLED...")
inputs = prepare_inputs(model, features; num_recycle=3)
prev = initial_recycle_state(model, inputs)

for i in 1:inputs.num_iters
    msa_act, pair_act = run_evoformer(model, inputs, prev)
    heads = run_heads(model, msa_act, pair_act, inputs)
    @show i, mean(heads.plddt)  # inspect confidence per iteration
    prev = RecycleState(
        msa_first_row = view(msa_act, :, 1, :, :),
        pair = pair_act,
        atom37 = heads.atom37,
    )
end
```

**Key types:**
- `AF2PreparedInputs` — device-placed tensors ready for the evoformer (30 fields)
- `RecycleState` — carries `msa_first_row`, `pair`, `atom37` between iterations
- `AF2InferenceResult` — final output with coordinates, logits, confidence metrics

### Differentiable sequence features

```julia
using Alphafold2

L, B = 20, 1
seq_logits = randn(Float32, 21, L, B)
feats = build_soft_sequence_features(seq_logits)

aatype = fill(0, L, B)
seq_mask, msa_mask, residue_index = build_basic_masks(aatype; n_msa_seq=1)

@show size(feats[:target_feat])   # (22, L, B)
@show size(feats[:msa_feat])      # (49, 1, L, B)
```

## Multimer Pairing Modes

The `build_multimer_features()` function supports explicit MSA pairing modes:

Supported modes:
1. `block diagonal` (default) — no cross-chain pairing
2. `taxon labels matched` — pairs rows by taxon labels from A3M headers
3. `pair by row index` — pairs rows by position across chains
4. `random pairing` — shuffled row-index pairing

Set via `pairing_mode` kwarg to `fold()` or `AF2_MULTIMER_PAIRING_MODE` env var.

## Scripts

- `scripts/end_to_end/build_monomer_input_jl.jl` — CLI wrapper for monomer feature building
- `scripts/end_to_end/build_multimer_input_jl.jl` — CLI wrapper for multimer feature building
- `scripts/end_to_end/build_template_input_from_pdb_jl.jl` — template feature extraction
- `scripts/regression/` — regression test cases and helpers
- `scripts/gradients/check_full_model_zygote.jl` — full-model Zygote gradient check
- `scripts/examples/` — example workflows

## Known Gaps

1. **Monomer MSA preprocessing**: Current Julia implementation uses deterministic row truncation
   and simplified `msa_feat` construction. The official pipeline applies stochastic MSA sampling,
   masked-MSA corruption, nearest-neighbor clustering, and cluster-derived profile/deletion means.

2. **Multimer in-model MSA sampling/masking**: Official multimer performs stochastic Gumbel-argsort
   sampling with `cluster_bias_mask` and BERT-style corruption inside forward. Current Julia uses
   deterministic sampling.

3. **Heteromer MSA pairing**: Official pipeline uses species-identifier-based pairing with
   deduplication. Current Julia builder supports simpler pairing modes (block diagonal, taxon
   labels, row index, random).
