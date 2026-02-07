# Alphafold2.jl

`Alphafold2.jl` is a Julia port of AlphaFold2 modules using feature-first, batch-last tensor layout.

## Status

Implemented and parity-checked (against official AlphaFold Python components):
- Evoformer trunk modules
- Structure module core and sidechain path
- Output heads: `MaskedMsaHead`, `DistogramHead`, `ExperimentallyResolvedHead`
- Confidence heads: `PredictedLDDTHead`, `PredictedAlignedErrorHead`
- Confidence utilities: `compute_plddt`, `compute_predicted_aligned_error`, `compute_tm`
- Multimer structure path parity (including recycle loop and chain-aware PDB export)

In scope:
- User-provided MSAs and template structures

Out of scope:
- MSA/template search pipelines
- Amber relax

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

### Output and confidence heads

```julia
using Alphafold2

L, B = 32, 1

c_s = 384
single = randn(Float32, c_s, L, B)
plddt_head = PredictedLDDTHead(c_s)
plddt_logits = plddt_head(single)[:logits]

c_z = 128
pair = randn(Float32, c_z, L, L, B)
dist_head = DistogramHead(c_z)
dist = dist_head(pair)

@show size(plddt_logits)      # (50, L, B)
@show size(dist[:logits])     # (64, L, L, B)
```

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
@show size(seq_mask), size(msa_mask), size(residue_index)
```

## Scripts

This repository also includes script-based workflows for:
- End-to-end template-conditioned runs
- End-to-end multimer runs (Python reference + Julia hybrid parity)
- Python-vs-Julia parity checks
- Full-model Zygote gradient checks

See:
- `scripts/end_to_end`
- `scripts/parity`
- `scripts/gradients`

## Developer Notes

Machine-specific parity/runbook commands were moved out of this README.
For internal developer setup and exact command lines, see:
- `AGENT_DEV_NOTES.md`
- `docs/PARITY_CHECK_NOTES.md`
- `docs/RUNNABLE_EXAMPLES.md`

## Multimer Example Commands

Runnable, validated commands are in:
- `docs/RUNNABLE_EXAMPLES.md`

That file includes:
- Multimer Python reference runs (with and without chain-specific A3Ms)
- Julia hybrid parity runs on identical pre-evo multimer dumps
- Recycle-5 multimer parity checks
- Chain-ID and interface-contact sanity checks on exported multimer PDBs
