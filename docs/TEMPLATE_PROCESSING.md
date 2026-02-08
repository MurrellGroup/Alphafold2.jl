# Template Processing: End-to-End Semantics

Last updated: 2026-02-08

This document explains exactly what happens to template inputs in the current codebase, from PDB parsing through model features.

## 1) Scope

Template handling appears in three layers:
1. Builder stage (script-level parsing/alignment into template tensors).
2. Feature stage (template-derived row/pair features inside model runner).
3. Model stage (template embedding modules in Evoformer path).

Primary files:
- [`scripts/end_to_end/build_monomer_input_jl.jl`](../scripts/end_to_end/build_monomer_input_jl.jl)
- [`scripts/end_to_end/build_multimer_input_jl.jl`](../scripts/end_to_end/build_multimer_input_jl.jl)
- [`scripts/end_to_end/run_af2_template_hybrid_jl.jl`](../scripts/end_to_end/run_af2_template_hybrid_jl.jl)
- [`src/modules/template_single_rows.jl`](../src/modules/template_single_rows.jl)
- [`src/modules/template_embedding.jl`](../src/modules/template_embedding.jl)

## 2) Builder Stage: Parsing Raw PDB Templates

Both monomer and multimer builders parse `ATOM` lines for specified chains.

Parsing logic:
- retain residues keyed by `(resseq, insertion_code)`
- collect atom coordinates for recognized names in `Alphafold2.atom_order`
- infer residue one-letter sequence from `resname`

Outputs per parsed template chain:
1. `template_seq`
2. `template_aatype` (`Int32[L_template]`)
3. `template_all_atom_positions` (`Float32[1,L_template,37,3]`)
4. `template_all_atom_masks` (`Float32[1,L_template,37]`)

Important:
- This is full atom37-capable extraction, not backbone-only.
- Sidechain atoms are included when present.
- Missing atoms are represented by zero coordinates with mask `0`.

Code references:
- monomer `_parse_template_chain`: [`scripts/end_to_end/build_monomer_input_jl.jl`](../scripts/end_to_end/build_monomer_input_jl.jl)
- multimer `_parse_template_chain`: [`scripts/end_to_end/build_multimer_input_jl.jl`](../scripts/end_to_end/build_multimer_input_jl.jl)

## 3) Alignment to Query Index Space

After parsing, template residues are aligned to query sequence positions.

Mechanism:
1. global sequence alignment (`_global_align_query_to_template`)
2. projection of template residue/atom tensors into query length `L_query`
3. unmatched query positions remain unknown/masked

Aligned outputs:
- `template_aatype_aligned[L_query]`
- `template_pos_aligned[L_query,37,3]`
- `template_mask_aligned[L_query,37]`

Code references:
- `_global_align_query_to_template`
- `_align_template_to_query`
in both builder scripts.

## 4) Monomer Template Stack Assembly

Monomer builder accepts one or multiple template PDBs (CSV list).

Assembly rules:
1. each input template is aligned to query length
2. rows are stacked as template axis `T`
3. stack is padded/truncated to `AF2_MONOMER_MAX_TEMPLATES` (default `4`)

Produced keys:
- `template_aatype[T,L]`
- `template_all_atom_positions[T,L,37,3]`
- `template_all_atom_masks[T,L,37]`
- `template_mask[T]`
- `template_sum_probs[T]`

Code:
- [`scripts/end_to_end/build_monomer_input_jl.jl`](../scripts/end_to_end/build_monomer_input_jl.jl)

## 5) Multimer Template Stack Assembly

Multimer builder accepts per-chain template groups:
- chain entries separated by `,`
- multiple template rows per chain separated by `+`

Example:
- template PDB arg: `a1.pdb+a2.pdb,b1.pdb`
- chain arg: `A+B,B`

Semantics:
1. each chain has 0..N template rows
2. global stack is built by template row index (`ti`) across chains
3. each row stores concatenated complex-length features (`L_total`)
4. chain segments without that row stay zero-masked
5. total rows capped by `AF2_MULTIMER_MAX_TEMPLATES` (default `4`)

Produced keys:
- `template_aatype[T,L_total]`
- `template_all_atom_positions[T,L_total,37,3]`
- `template_all_atom_masks[T,L_total,37]`

Code:
- [`scripts/end_to_end/build_multimer_input_jl.jl`](../scripts/end_to_end/build_multimer_input_jl.jl)

## 6) Template Featureization in Runner

Runner consumes template tensors and builds two distinct feature paths:

1. Template single rows (`TemplateSingleRows`)
- produces per-template per-residue row features for Evoformer MSA stack augmentation
- uses atom37-derived torsion features (`atom37_to_torsion_angles`)
- feature dimensionality differs by checkpoint family:
  - monomer-style: 57
  - multimer-style: 34

2. Template pair embedding
- monomer: `TemplateEmbedding` via attention over stacked template pair reps
- multimer: `TemplateEmbeddingMultimer` via weighted averaging of per-template multimer pair embeddings

Code:
- [`src/modules/template_single_rows.jl`](../src/modules/template_single_rows.jl)
- [`src/modules/template_embedding.jl`](../src/modules/template_embedding.jl)

## 7) What Features Are Built from Template Atoms

Common geometric ingredients in template embedding:
1. pseudo-beta coordinates (CB; glycine uses CA)
2. pairwise distance gram bins (`dgram`)
3. residue-type one-hot row/column features
4. local frame-derived directional features (unit vectors optionally enabled)
5. backbone-validity masks

Monomer template embedding:
- stacks per-template pair reps then attends with query pair embedding.

Multimer template embedding:
- includes multichain masking,
- injects normalized query embedding via `template_pair_embedding_8`,
- aggregates template rows with `template_mask` weights.

Code details:
- `_single_template_embedding` and `_single_template_embedding_multimer` in [`src/modules/template_embedding.jl`](../src/modules/template_embedding.jl)

## 8) Interaction with Recycling

Template features are recomputed/used inside each recycle iteration in the runner.

Per-iteration flow:
1. build recycle-updated query pair representation
2. apply template embedding to that pair representation
3. continue through extra MSA + Evoformer blocks

Code:
- recycle loop in [`scripts/end_to_end/run_af2_template_hybrid_jl.jl`](../scripts/end_to_end/run_af2_template_hybrid_jl.jl)

## 9) Programmatic Representation Requirements (No File I/O)

If templates are provided programmatically (no PDB/NPZ file path), minimal required representation is:

1. Aligned template stack:
- `template_aatype[T,L]`
- `template_all_atom_positions[T,L,37,3]`
- `template_all_atom_masks[T,L,37]`
- optional `template_mask[T]`

2. Build-time mapping metadata (only needed if constructing from raw chains):
- query sequence(s)
- template sequence(s)
- alignment mapping(s) to query index space
- chain segment offsets for multimer concatenation

As long as aligned template tensors are correctly formed, the downstream model path does not require raw PDB records.

## 10) Current Constraints and Edge Cases

1. Chain/template specs must have matching cardinalities (except supported broadcast forms); invalid specs error.
2. Missing chains/templates in a row are represented by zero-masked segments.
3. Unknown residues default to `X`/unknown type and remain mask-gated.
4. Template behavior can differ between monomer and multimer model families due to architecture-specific embedding implementations.

See also:
- [`docs/INTERNAL_REPRESENTATIONS.md`](./INTERNAL_REPRESENTATIONS.md)
- [`docs/CODEBASE_GUIDE.md`](./CODEBASE_GUIDE.md)

