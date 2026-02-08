# Internal Representations (Current + Proposed)

Last updated: 2026-02-08

This document describes:
1. The current internal feature representations used by `Alphafold2.jl` after input processing.
2. Exactly what is extracted from MSA/template user inputs.
3. A concrete in-memory representation plan for moving toward a Julia-native IPA pipeline (without NPZ handoff files).

This is documentation only. It does not define any behavior changes.

## 1) Current Dataflow (As Implemented)

Current end-to-end scripts follow this pattern:
1. Build input features (`build_monomer_input_jl.jl` or `build_multimer_input_jl.jl`).
2. Serialize a feature payload to `.npz`.
3. Run model (`run_af2_template_hybrid_jl.jl`) by loading that payload.

The model path itself is feature-first, batch-last (`C, ... , B`) internally, with explicit conversions when comparing to Python tensors.

## 2) What Is Extracted From Templates Today

Template parsing in both builders reads `ATOM` records for the requested chain(s), and for atoms recognized in `Alphafold2.atom_order`.

For each template row, the builders produce:
1. `template_aatype`: integer residue types per aligned query position.
2. `template_all_atom_positions`: full 37-atom coordinates per residue (`x,y,z`).
3. `template_all_atom_masks`: full 37-atom presence mask per residue.

Important clarification:
- This is not backbone-only. Sidechain atoms are included when present in the input PDB.
- Missing atoms remain zeroed in coordinates and masked out in `template_all_atom_masks`.
- Query-template length mismatch is handled by global alignment; unmatched query positions remain masked/unknown.

## 3) Current Builder Payload Contracts

## 3.1 Monomer Builder Payload

Produced by `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/build_monomer_input_jl.jl`.

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

## 3.2 Multimer Builder Payload

Produced by `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/build_multimer_input_jl.jl`.

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

Inside `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/end_to_end/run_af2_template_hybrid_jl.jl`, these are converted to feature-first tensors:

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

## 5) Minimal In-Memory Representation Set For Julia-Native IPA

To remove the NPZ handoff and run programmatically, the smallest complete in-memory contract is:

1. Sequence and chain metadata
- `aatype`, `residue_index`, `seq_mask`
- multimer IDs: `asym_id`, `entity_id`, `sym_id`

2. MSA stacks
- raw/cluster MSA int tokens
- deletion matrices
- row masks
- `cluster_bias_mask` (multimer)
- optional precomputed `target_feat`/`msa_feat` overrides

3. Template stacks
- `template_aatype`
- `template_all_atom_positions`
- `template_all_atom_masks`
- optional `template_mask`

4. Runtime config
- `num_recycle`
- checkpoint/model kind metadata (monomer-family vs multimer)

5. Optional parity/reference fields
- pre-evo dumps and parity-only references should be separate optional attachments, not required by production inference structs.

A concrete typed layout would be a small group of immutable structs, e.g.:
1. `AF2SequenceFeatures`
2. `AF2MSAFeatures`
3. `AF2TemplateFeatures`
4. `AF2ModelInput`
5. `AF2RunConfig`

with one adapter that converts these into current feature-first tensors consumed by Evoformer/StructureModule.

## 6) Suggested Migration Steps (No Behavior Change)

1. Add typed in-memory feature structs mirroring the payload keys above.
2. Refactor builder scripts to return these structs in-process.
3. Keep NPZ writer/reader as a thin I/O adapter around the structs.
4. Refactor `run_af2_template_hybrid_jl.jl` entry to accept either:
- NPZ path (legacy path), or
- `AF2ModelInput` directly (new path).
5. Reuse existing parity checks by serializing struct->NPZ in tests to confirm byte-level contract stability during migration.

## 7) Practical Notes For Internal IPA Work

1. Template atom representation already contains full sidechain-capable 37-atom tensors, so no new template atom schema is required for IPA migration.
2. The key architectural boundary to preserve is not NPZ format; it is the semantic contract of the feature keys and shapes listed in Sections 3 and 4.
3. Multimer pairing policy should remain explicit in config (block-diagonal, taxon-matched, row-index, random) so programmatic callers do not depend on script-only env variables.

