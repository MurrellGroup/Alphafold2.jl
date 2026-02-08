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
- Multimer template embedding + template-row integration parity (recycle loop)
- Multimer template-stack row semantics in native builders (including multi-template and uneven per-chain template counts for validated regression cases)

In scope:
- User-provided MSAs and template structures

Out of scope:
- MSA/template search pipelines
- Amber relax

## Documentation

For a full user/developer documentation map, see:
- `docs/INDEX.md`

Key expository docs:
1. Full codebase walkthrough (data paths, layers, recycling, coordinates, confidence, monomer/multimer split):
- `docs/CODEBASE_GUIDE.md`
2. Template internals (exact extraction/alignment/feature semantics):
- `docs/TEMPLATE_PROCESSING.md`
3. Internal feature contracts and in-memory representation plan:
- `docs/INTERNAL_REPRESENTATIONS.md`

## Multimer Pairing Modes

`scripts/end_to_end/build_multimer_input_jl.jl` now supports explicit MSA pairing modes.

Default mode:
- `block diagonal`
- This keeps each chain MSA unpaired (rows are embedded chain-by-chain with gaps outside each chain segment).

Supported modes:
1. `block diagonal`
2. `taxon labels matched`
3. `pair by row index`
4. `random pairing`

How to set:
- Optional positional argument 7 to `build_multimer_input_jl.jl`, or
- `AF2_MULTIMER_PAIRING_MODE` environment variable.

Random seed:
- Optional positional argument 8, or `AF2_MULTIMER_PAIRING_SEED` env var.

Example builder invocations:
1. Default (`block diagonal`):
```bash
julia --startup-file=no --history-file=no \
  scripts/end_to_end/build_multimer_input_jl.jl \
  MKQLEDKVEELLSKNYHLENEVARLKKLV,MKQLEDKVEELLSKNYHLENEVARLKKLV \
  /tmp/af2_multimer_blockdiag.npz \
  1 \
  test/regression/msa/multimer_chainA_gcn4.a3m,test/regression/msa/multimer_chainB_gcn4.a3m
```

2. Pair rows by index:
```bash
julia scripts/end_to_end/build_multimer_input_jl.jl <seqs_csv> <out.npz> <recycles> <msa_csv> '' '' 'pair by row index' 0
```

3. Pair rows by taxon labels:
```bash
julia scripts/end_to_end/build_multimer_input_jl.jl <seqs_csv> <out.npz> <recycles> <msa_csv> '' '' 'taxon labels matched' 0
```

4. Random pairing:
```bash
julia scripts/end_to_end/build_multimer_input_jl.jl <seqs_csv> <out.npz> <recycles> <msa_csv> '' '' 'random pairing' 7
```

Mode behavior:
1. `block diagonal`:
- No cross-chain pairing. Query and non-query rows are all kept as unpaired block-diagonal rows.
- `cluster_bias_mask` marks the first row of each chain block.

2. `taxon labels matched`:
- Pairs non-query rows across chains using taxon labels parsed from A3M headers (e.g. `OX=...`, `TaxID=...`, `taxon=...`, `OS=...`).
- Adds one full concatenated query row first, then taxon-paired rows, then leftover unpaired rows.
- Includes AF2-like guards (species present in only one chain are skipped; very deep species buckets are skipped).

3. `pair by row index`:
- Adds one full concatenated query row first.
- Pairs non-query rows strictly by index across chains, then appends leftovers as unpaired rows.
- Important: this mode intentionally does not deduplicate MSA rows before pairing, so row indices remain aligned to the input file order.

4. `random pairing`:
- Same structure as row-index pairing, but non-query rows are shuffled first (query row is never shuffled).
- Pairing then uses shuffled row index order.

Partial per-chain inputs:
1. MSA:
- You can provide an MSA for some chains and leave others empty by passing empty CSV entries (query-only fallback for empty entries).
- Example: `msaA.a3m,` for a 2-chain case.

2. Templates:
- You can provide per-chain template groups with `+` inside each chain slot and allow empty chain slots.
- Example: `a1.pdb+a2.pdb,` with chain spec `A+B,`.

## Features not yet implemented

The items below are known, concrete gaps between current Julia behavior and official AlphaFold Python behavior in the path from user inputs to model-layer inputs.

Implementation note for all items below:
- Target official Python behavior first.
- Prefer implementing preprocessing-style logic in one consistent place (monomer-style input processing) when this does not change semantics.
- If official behavior is forward-time (notably multimer sampling/masking/clustering tied to ensemble/recycle execution), keep it in forward and match semantics there rather than relocating it.

1. Monomer official MSA preprocessing parity is not complete.
- Affected path: `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/build_monomer_input_jl.jl`.
- Official behavior (monomer) applies these steps inside `RunModel.process_features` TF preprocessing:
  `sample_msa` (randomized non-query row shuffle/select), `make_masked_msa`, `nearest_neighbor_clusters`, `summarize_clusters`, then `make_msa_feat`.
- Current Julia behavior uses deterministic row truncation/order and simplified `msa_feat` construction that does not fully reproduce masked-MSA corruption + cluster-derived profile/deletion means.
- Current validated mismatch: official model-input parity fails for monomer MSA cases:
  `monomer_msa_only` (`msa_feat`) and `monomer_template_msa` (`msa_feat`, `msa_mask`).
- Validation command:
  `JAX_PLATFORMS=cpu python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/regression/check_model_input_parity_official.py --repo-root /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl --alphafold-repo /Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold --manifest /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/test/regression/reference_pdbs/manifest_python_official.json --min-msa-rows-multimer 512`

2. Multimer native Julia path does not yet reproduce official in-model MSA sampling/masking semantics.
- Affected path: `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_template_hybrid_jl.jl`.
- Official behavior (multimer) performs sampling/masking/clustering inside model forward (`modules_multimer`):
  `sample_msa` (Gumbel argsort with `cluster_bias_mask`), `make_masked_msa` (BERT corruption), `nearest_neighbor_clusters`, `create_msa_feat`.
- Current Julia native multimer path uses `_sample_msa_rows_deterministic` and does not implement the official stochastic/Gumbel masking/sampling path.
- Note: this does not contradict current raw-input parity checks, because official multimer `RunModel.process_features` returns raw features unchanged; these differences appear inside forward.

3. Full official heteromer MSA pairing/dedup pipeline is not implemented in Julia builders.
- Affected path: `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/build_multimer_input_jl.jl`.
- Official behavior pairs rows by species identifiers and deduplicates against paired rows (`msa_pairing.create_paired_features`, `deduplicate_unpaired_sequences`).
- Current Julia builder uses row-index pairing + unpaired row appends, which can differ on realistic heteromer MSAs.

4. Validation status is case-specific, not universal.
- Current official model-input parity status:
  monomer (`seq_only`, `template_only`) pass;
  monomer (`msa_only`, `template_msa`) fail as described above;
  current multimer raw-feature cases pass.
- Any item above should be considered unresolved until both implementation and parity validation are done and this section is updated.

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
- Native multimer input construction in Julia (from chain sequences/A3Ms/templates)
- Official AF2 weight audit + NPZ->safetensors conversion
- Python-vs-Julia parity checks
- Full-model Zygote gradient checks (monomer-family and multimer checkpoints)

See:
- `scripts/end_to_end`
- `scripts/parity`
- `scripts/gradients`

## REPL Workflows (Load Once, Run Many Folds)

The end-to-end runner script now exposes a callable function:
- `run_af2_template_hybrid(params_source, input_dump_npz, output_npz)`

You can include it once in a REPL session, load safetensors weights once, and run multiple folds without re-reading weights from disk.

### Monomer REPL Session

1. Build one or more monomer input dumps:
```bash
julia --startup-file=no --history-file=no scripts/end_to_end/build_monomer_input_jl.jl \
  ACDEFGHIK /tmp/mono_case1_input.npz 3

julia --startup-file=no --history-file=no scripts/end_to_end/build_monomer_input_jl.jl \
  ACDEFGHIK /tmp/mono_case2_input.npz 3 test/regression/msa/monomer_short.a3m
```

2. Start Julia REPL and reuse one loaded parameter dictionary:
```julia
include("scripts/end_to_end/run_af2_template_hybrid_jl.jl")

mono_path = Main.Alphafold2.resolve_af2_params_path("alphafold2_model_1_ptm_dm_2022-12-06.safetensors")
mono_params = Main.Alphafold2.af2_params_read(mono_path)  # load once

run_af2_template_hybrid(mono_params, "/tmp/mono_case1_input.npz", "/tmp/mono_case1_out.npz")
run_af2_template_hybrid(mono_params, "/tmp/mono_case2_input.npz", "/tmp/mono_case2_out.npz")
```

### Multimer REPL Session

1. Build one or more multimer input dumps:
```bash
julia --startup-file=no --history-file=no scripts/end_to_end/build_multimer_input_jl.jl \
  MKQLEDKVEELLSKNYHLENEVARLKKLV,MKQLEDKVEELLSKNYHLENEVARLKKLV \
  /tmp/multi_case1_input.npz 5

julia --startup-file=no --history-file=no scripts/end_to_end/build_multimer_input_jl.jl \
  MKQLEDKVEELLSKNYHLENEVARLKKLV,MKQLEDKVEELLSKNYHLENEVARLKKLV \
  /tmp/multi_case2_input.npz 5 \
  test/regression/msa/multimer_chainA_gcn4.a3m,test/regression/msa/multimer_chainB_gcn4.a3m
```

2. Reuse loaded multimer weights in REPL:
```julia
include("scripts/end_to_end/run_af2_template_hybrid_jl.jl")

multi_path = Main.Alphafold2.resolve_af2_params_path("alphafold2_model_1_multimer_v3_dm_2022-12-06.safetensors")
multi_params = Main.Alphafold2.af2_params_read(multi_path)  # load once

run_af2_template_hybrid(multi_params, "/tmp/multi_case1_input.npz", "/tmp/multi_case1_out.npz")
run_af2_template_hybrid(multi_params, "/tmp/multi_case2_input.npz", "/tmp/multi_case2_out.npz")
```

Notes:
- Run the REPL examples from the repository root.
- The runner writes both `*.npz` outputs and `*.pdb` files.
- `resolve_af2_params_path` fetches from `MurrellLab/AlphaFold2.jl` if the safetensors file is not already cached.

## Developer Notes

Machine-specific parity/runbook commands were moved out of this README.
For internal developer setup and exact command lines, see:
- `AGENT_DEV_NOTES.md`
- `docs/PARITY_CHECK_NOTES.md`
- `docs/RUNNABLE_EXAMPLES.md`
- `docs/WEIGHTS_AUDIT.md`

## Multimer Example Commands

Runnable, validated commands are in:
- `docs/RUNNABLE_EXAMPLES.md`

That file includes:
- Multimer Python reference runs (with and without chain-specific A3Ms)
- Julia hybrid parity runs on identical pre-evo multimer dumps
- Recycle-5 multimer parity checks
- Native recycle-5 multimer run from Julia-built input features
- Chain-ID and interface-contact sanity checks on exported multimer PDBs
- Intra-chain C-alpha geometry checks for multimer plausibility
