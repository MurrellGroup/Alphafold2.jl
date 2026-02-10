# Internal Representations

Last updated: 2026-02-10

This document describes:
1. The internal feature representations used by `Alphafold2.jl` after input processing.
2. Exactly what is extracted from MSA/template user inputs.

This is documentation only. It does not define any behavior changes.

Related docs:
- [`CODEBASE_GUIDE.md`](./CODEBASE_GUIDE.md)
- [`TEMPLATE_PROCESSING.md`](./TEMPLATE_PROCESSING.md)

## 1) Dataflow

The feature pipeline builds input features in-process:
1. `build_monomer_features()` or `build_multimer_features()` returns a `Dict{String,Any}`.
2. `_infer()` consumes the Dict, converting to feature-first tensors internally.

The model path itself is feature-first, batch-last (`C, ... , B`) internally.

## 2) What Is Extracted From Templates

Template parsing reads `ATOM` records for the requested chain(s), and for atoms recognized in `Alphafold2.atom_order`.

For each template row, the feature pipeline produces:
1. `template_aatype`: integer residue types per aligned query position.
2. `template_all_atom_positions`: full 37-atom coordinates per residue (`x,y,z`).
3. `template_all_atom_masks`: full 37-atom presence mask per residue.

Important clarification:
- This is not backbone-only. Sidechain atoms are included when present in the input PDB.
- Missing atoms remain zeroed in coordinates and masked out in `template_all_atom_masks`.
- Query-template length mismatch is handled by global alignment; unmatched query positions remain masked/unknown.

## 3) Feature Payload Contracts

## 3.1 Monomer Features

Produced by `Alphafold2.build_monomer_features()` in `src/feature_pipeline/monomer.jl`.

Core keys:
- `aatype`: `Int32[L,1]`
- `seq_mask`: `Float32[L]`
- `residue_index`: `Int32[L]` (0-based)
- `msa`: `Int32[S_raw,L]`
- `deletion_matrix`: `Float32[S_raw,L]`
- `msa_mask`: `Float32[S_raw,L]` (raw)
- `msa_mask_model`: `Float32[S_cluster,L]` (cluster stack mask)
- `target_feat`: `Float32[L,22]`
- `msa_feat`: `Float32[S_cluster,L,49]`
- `extra_msa`: `Int32[S_extra,L]`
- `extra_deletion_matrix`: `Float32[S_extra,L]`
- `extra_msa_mask`: `Float32[S_extra,L]`
- `extra_has_deletion`: `Float32[S_extra,L]`
- `extra_deletion_value`: `Float32[S_extra,L]`
- `num_recycle`: `Int32[]`

Optional template keys:
- `template_aatype`: `Int32[T,L]`
- `template_all_atom_positions`: `Float32[T,L,37,3]`
- `template_all_atom_masks`: `Float32[T,L,37]`
- `template_mask`: `Float32[T]`
- `template_sum_probs`: `Float32[T]`

## 3.2 Multimer Features

Produced by `Alphafold2.build_multimer_features()` in `src/feature_pipeline/multimer.jl`.

Core keys:
- `aatype`: `Int32[L_total,1]`
- `seq_mask`: `Float32[L_total]`
- `residue_index`: `Int32[L_total]` (per-chain 0-based index)
- `asym_id`: `Int32[L_total]`
- `entity_id`: `Int32[L_total]`
- `sym_id`: `Int32[L_total]`
- `msa`: `Int32[S,L_total]`
- `deletion_matrix`: `Float32[S,L_total]`
- `msa_mask`: `Float32[S,L_total]`
- `cluster_bias_mask`: `Float32[S]`
- `extra_msa`: `Int32[S,L_total]`
- `extra_deletion_matrix`: `Float32[S,L_total]`
- `extra_msa_mask`: `Float32[S,L_total]`
- `num_recycle`: `Int32[]`

Optional template keys:
- `template_aatype`: `Int32[T,L_total]`
- `template_all_atom_positions`: `Float32[T,L_total,37,3]`
- `template_all_atom_masks`: `Float32[T,L_total,37]`

## 4) Forward-Time Internal Feature Forms

Inside `_infer()`, features are converted to feature-first tensors:

1. Query/cluster features:
- `target_feat`: `(C_target,L,1)`
- `msa_feat`: `(49,S_cluster,L,1)`
- `extra_msa_feat`: `(25,S_extra,L,1)`

2. Template row features:
- `template_rows_first`: `(C_m,T,L,1)`
- `template_row_mask`: `(T,L)`

3. Evoformer states per recycle:
- `msa_act`: `(C_m,S_total,L,1)`
- `pair_act`: `(C_z,L,L,1)`
- `single`: `(C_s,L,1)`

4. Structure/confidence outputs:
- atom coordinates/masks (`atom14`, `atom37` paths)
- heads (`masked_msa`, `distogram`, `experimentally_resolved`, `pLDDT`, optional `PAE`/`pTM`)

## 5) Notes

1. Template atom representation already contains full sidechain-capable 37-atom tensors.
2. The key architectural boundary is the semantic contract of the feature keys and shapes listed in Sections 3 and 4.
3. Multimer pairing policy is explicit via `pairing_mode` kwarg (block-diagonal, taxon-matched, row-index, random).
