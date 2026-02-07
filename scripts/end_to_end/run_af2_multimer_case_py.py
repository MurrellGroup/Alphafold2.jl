#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict, List, Tuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def _slice_prefix(params, prefix):
    out = {}
    for scope, vars_dict in params.items():
        if scope.startswith(prefix):
            key = scope[len(prefix):]
            key = key.replace("/~_relative_encoding/", "/relative_encoding/")
            out[key] = {k: np.asarray(v) for k, v in vars_dict.items()}
    return out


def _aatype_from_sequence(seq: str, residue_constants) -> np.ndarray:
    seq = seq.strip().upper()
    idxs = [residue_constants.restype_order_with_x.get(ch, 20) for ch in seq]
    return np.asarray(idxs, dtype=np.int32)


def _parse_fasta_sequences(path: str) -> List[str]:
    seqs: List[str] = []
    cur: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
                continue
            cur.append(line)
    if cur:
        seqs.append("".join(cur))
    return seqs


def _a3m_row_to_aligned_and_deletions(seq: str) -> Tuple[str, np.ndarray]:
    aligned: List[str] = []
    deletion: List[int] = []
    d = 0
    for ch in seq:
        if ch.islower():
            d += 1
        else:
            aligned.append(ch.upper())
            deletion.append(d)
            d = 0
    return "".join(aligned), np.asarray(deletion, dtype=np.float32)


def _msa_ids_from_aligned_seq(seq: str, residue_constants) -> np.ndarray:
    mapper = residue_constants.HHBLITS_AA_TO_ID
    return np.asarray([mapper.get(ch.upper(), 20) for ch in seq], dtype=np.int32)


def _load_msa_file(msa_path: str, query_aatype: np.ndarray, residue_constants) -> Tuple[np.ndarray, np.ndarray]:
    seqs = _parse_fasta_sequences(msa_path)
    if not seqs:
        raise ValueError(f"No sequences found in MSA file: {msa_path}")
    l = int(query_aatype.shape[0])
    rows: List[np.ndarray] = []
    dels: List[np.ndarray] = []
    for s in seqs:
        aligned, deletion = _a3m_row_to_aligned_and_deletions(s)
        if len(aligned) != l:
            raise ValueError(f"MSA row length {len(aligned)} does not match query length {l}")
        rows.append(_msa_ids_from_aligned_seq(aligned, residue_constants))
        dels.append(deletion)

    if not np.array_equal(rows[0], query_aatype):
        rows = [query_aatype.copy()] + rows
        dels = [np.zeros((l,), dtype=np.float32)] + dels

    msa = np.stack(rows, axis=0).astype(np.int32)
    deletion_matrix = np.stack(dels, axis=0).astype(np.float32)
    return msa, deletion_matrix


def _build_chain_features(
    seq: str,
    chain_id: str,
    msa_ids: np.ndarray,
    deletion_matrix: np.ndarray,
    pipeline,
    pipeline_multimer,
) -> Dict[str, np.ndarray]:
    l = len(seq)
    seq_feat = pipeline.make_sequence_features(seq, f"chain_{chain_id}", l)

    monomer_feat = dict(seq_feat)
    monomer_feat["msa"] = np.asarray(msa_ids, dtype=np.int32)
    monomer_feat["deletion_matrix_int"] = np.asarray(np.rint(deletion_matrix), dtype=np.int32)
    monomer_feat["num_alignments"] = np.asarray([msa_ids.shape[0]] * l, dtype=np.int32)
    monomer_feat["msa_species_identifiers"] = np.asarray([b""] * msa_ids.shape[0], dtype=np.object_)

    converted = pipeline_multimer.convert_monomer_features(monomer_feat, chain_id=chain_id)
    converted["msa_all_seq"] = converted["msa"].copy()
    converted["deletion_matrix_int_all_seq"] = monomer_feat["deletion_matrix_int"].copy()
    converted["msa_species_identifiers_all_seq"] = monomer_feat["msa_species_identifiers"].copy()
    return converted


def _ca_distance_metrics(
    atom37,
    atom37_mask,
    residue_constants,
    asym_id=None,
    intra_chain_only=False,
):
    ca_idx = residue_constants.atom_order["CA"]
    valid = atom37_mask[:, ca_idx] > 0.5
    dists = []
    for i in range(atom37.shape[0] - 1):
        if intra_chain_only and asym_id is not None and int(asym_id[i]) != int(asym_id[i + 1]):
            continue
        if valid[i] and valid[i + 1]:
            d = np.linalg.norm(atom37[i + 1, ca_idx] - atom37[i, ca_idx])
            dists.append(float(d))
    if not dists:
        return np.zeros((0,), dtype=np.float32), np.nan, np.nan, np.nan, np.nan, np.nan
    d = np.asarray(dists, dtype=np.float32)
    outlier_frac = float(np.mean((d < 3.2) | (d > 4.4)))
    return d, float(np.mean(d)), float(np.std(d)), float(np.min(d)), float(np.max(d)), outlier_frac


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphafold-repo", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--sequences", required=True, help="Comma-separated chain sequences, e.g. ACDE,FGHI")
    parser.add_argument("--msa-files", default="", help="Optional comma-separated A3M files, one per chain")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dump-pre-evo", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-recycle", type=int, default=1)
    parser.add_argument("--num-msa", type=int, default=64)
    parser.add_argument("--num-extra-msa", type=int, default=128)
    parser.add_argument("--min-msa-rows", type=int, default=512)
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.common import confidence  # pylint: disable=import-outside-toplevel
    from alphafold.common import residue_constants  # pylint: disable=import-outside-toplevel
    from alphafold.data import feature_processing  # pylint: disable=import-outside-toplevel
    from alphafold.data import pipeline  # pylint: disable=import-outside-toplevel
    from alphafold.data import pipeline_multimer  # pylint: disable=import-outside-toplevel
    from alphafold.model import common_modules  # pylint: disable=import-outside-toplevel
    from alphafold.model import config as af_config  # pylint: disable=import-outside-toplevel
    from alphafold.model import folding_multimer  # pylint: disable=import-outside-toplevel
    from alphafold.model import layer_stack  # pylint: disable=import-outside-toplevel
    from alphafold.model import modules  # pylint: disable=import-outside-toplevel
    from alphafold.model import modules_multimer  # pylint: disable=import-outside-toplevel
    from alphafold.model import prng  # pylint: disable=import-outside-toplevel
    from alphafold.model import utils  # pylint: disable=import-outside-toplevel

    seqs = [s.strip().upper() for s in args.sequences.split(",") if s.strip()]
    if len(seqs) < 2:
        raise ValueError("Multimer run requires at least 2 chain sequences.")

    msa_files = [s.strip() for s in args.msa_files.split(",")] if args.msa_files.strip() else []
    if msa_files and len(msa_files) != len(seqs):
        raise ValueError(f"--msa-files count ({len(msa_files)}) must match sequence count ({len(seqs)})")

    model_name = args.model_name.strip()
    if not model_name:
        b = os.path.basename(args.params)
        model_name = "model_1_multimer_v3" if "multimer" in b else "model_1_multimer_v3"

    cfg = af_config.model_config(model_name)
    if not cfg.model.global_config.multimer_mode:
        raise ValueError(f"Model {model_name} is not a multimer config.")

    gc = cfg.model.global_config
    gc.deterministic = True
    gc.eval_dropout = False
    gc.bfloat16 = False
    gc.bfloat16_output = False
    cfg.model.num_recycle = int(args.num_recycle)
    cfg.model.resample_msa_in_recycling = False
    c = cfg.model.embeddings_and_evoformer
    c.num_msa = int(args.num_msa)
    c.num_extra_msa = int(args.num_extra_msa)

    # Julia multimer parity path currently focuses on template-free multimer.
    c.template.enabled = False
    c.template.embed_torsion_angles = False

    all_chain_features = {}
    for i, seq in enumerate(seqs):
        chain_id = chr(ord("A") + i)
        q = _aatype_from_sequence(seq, residue_constants)
        if msa_files:
            msa_ids, deletion_matrix = _load_msa_file(msa_files[i], q, residue_constants)
        else:
            msa_ids = q[None, :]
            deletion_matrix = np.zeros((1, len(seq)), dtype=np.float32)

        all_chain_features[chain_id] = _build_chain_features(
            seq,
            chain_id,
            msa_ids,
            deletion_matrix,
            pipeline,
            pipeline_multimer,
        )

    all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)
    np_example = feature_processing.pair_and_merge(all_chain_features)
    np_example = pipeline_multimer.pad_msa(np_example, int(args.min_msa_rows))

    l = int(np_example["aatype"].shape[0])
    np_example["aatype"] = np.asarray(np_example["aatype"], dtype=np.int32)
    np_example["residue_index"] = np.asarray(np_example["residue_index"], dtype=np.int32)
    np_example["msa"] = np.asarray(np_example["msa"], dtype=np.int32)
    np_example["deletion_matrix"] = np.asarray(np_example["deletion_matrix"], dtype=np.float32)
    np_example["msa_mask"] = np.asarray(np_example["msa_mask"], dtype=np.float32)
    np_example["seq_mask"] = np.asarray(np_example["seq_mask"], dtype=np.float32)
    np_example["asym_id"] = np.asarray(np_example["asym_id"], dtype=np.int32)
    np_example["entity_id"] = np.asarray(np_example["entity_id"], dtype=np.int32)
    np_example["sym_id"] = np.asarray(np_example["sym_id"], dtype=np.int32)
    np_example["cluster_bias_mask"] = np.asarray(np_example["cluster_bias_mask"], dtype=np.float32)

    np_example["template_aatype"] = np.zeros((0, l), dtype=np.int32)
    np_example["template_all_atom_positions"] = np.zeros((0, l, 37, 3), dtype=np.float32)
    np_example["template_all_atom_mask"] = np.zeros((0, l, 37), dtype=np.float32)

    with open(args.params, "rb") as f:
        flat = np.load(f, allow_pickle=False)
        full_params = utils.flat_params_to_haiku(flat)
    params = _slice_prefix(full_params, "alphafold/alphafold_iteration/")
    has_pae_head = "predicted_aligned_error_head/logits" in params

    def fn(batch, prev_pos, prev_msa_first_row, prev_pair):
        safe_key = prng.SafeKey(hk.next_rng_key())
        batch = dict(batch)
        c = cfg.model.embeddings_and_evoformer
        gc = cfg.model.global_config
        dtype = jnp.bfloat16 if gc.bfloat16 else jnp.float32

        batch["prev_pos"] = prev_pos
        batch["prev_msa_first_row"] = prev_msa_first_row
        batch["prev_pair"] = prev_pair

        with hk.experimental.name_scope("evoformer"):
            batch["msa_profile"] = modules_multimer.make_msa_profile(batch)
            target_feat = jax.nn.one_hot(batch["aatype"], 21).astype(dtype)
            preprocess_1d = common_modules.Linear(c.msa_channel, name="preprocess_1d")(target_feat)

            safe_key, sample_key, mask_key = safe_key.split(3)
            batch = modules_multimer.sample_msa(sample_key, batch, c.num_msa)
            batch = modules_multimer.make_masked_msa(batch, mask_key, c.masked_msa)

            (batch["cluster_profile"], batch["cluster_deletion_mean"]) = modules_multimer.nearest_neighbor_clusters(batch)
            msa_feat = modules_multimer.create_msa_feat(batch).astype(dtype)
            preprocess_msa = common_modules.Linear(c.msa_channel, name="preprocess_msa")(msa_feat)
            msa_activations = jnp.expand_dims(preprocess_1d, axis=0) + preprocess_msa

            left_single = common_modules.Linear(c.pair_channel, name="left_single")(target_feat)
            right_single = common_modules.Linear(c.pair_channel, name="right_single")(target_feat)
            pair_activations = left_single[:, None] + right_single[None, :]
            mask_2d = batch["seq_mask"][:, None] * batch["seq_mask"][None, :]
            mask_2d = mask_2d.astype(dtype)

            if c.recycle_pos:
                prev_pseudo_beta = modules.pseudo_beta_fn(batch["aatype"], batch["prev_pos"], None)
                dgram = modules.dgram_from_positions(prev_pseudo_beta, **c.prev_pos).astype(dtype)
                pair_activations += common_modules.Linear(c.pair_channel, name="prev_pos_linear")(dgram)

            if c.recycle_features:
                prev_msa_first_row_n = common_modules.LayerNorm(
                    axis=[-1], create_scale=True, create_offset=True, name="prev_msa_first_row_norm"
                )(batch["prev_msa_first_row"]).astype(dtype)
                msa_activations = msa_activations.at[0].add(prev_msa_first_row_n)
                pair_activations += common_modules.LayerNorm(
                    axis=[-1], create_scale=True, create_offset=True, name="prev_pair_norm"
                )(batch["prev_pair"]).astype(dtype)

            if c.max_relative_idx:
                rel_feats = []
                pos = batch["residue_index"]
                asym_id = batch["asym_id"]
                asym_id_same = jnp.equal(asym_id[:, None], asym_id[None, :])
                offset = pos[:, None] - pos[None, :]
                clipped_offset = jnp.clip(offset + c.max_relative_idx, a_min=0, a_max=2 * c.max_relative_idx)

                if c.use_chain_relative:
                    final_offset = jnp.where(
                        asym_id_same,
                        clipped_offset,
                        (2 * c.max_relative_idx + 1) * jnp.ones_like(clipped_offset),
                    )
                    rel_pos = jax.nn.one_hot(final_offset, 2 * c.max_relative_idx + 2)
                    rel_feats.append(rel_pos)

                    entity_id_same = jnp.equal(batch["entity_id"][:, None], batch["entity_id"][None, :])
                    rel_feats.append(entity_id_same.astype(rel_pos.dtype)[..., None])

                    rel_sym_id = batch["sym_id"][:, None] - batch["sym_id"][None, :]
                    max_rel_chain = c.max_relative_chain
                    clipped_rel_chain = jnp.clip(rel_sym_id + max_rel_chain, a_min=0, a_max=2 * max_rel_chain)
                    final_rel_chain = jnp.where(
                        entity_id_same,
                        clipped_rel_chain,
                        (2 * max_rel_chain + 1) * jnp.ones_like(clipped_rel_chain),
                    )
                    rel_chain = jax.nn.one_hot(final_rel_chain, 2 * c.max_relative_chain + 2)
                    rel_feats.append(rel_chain)
                else:
                    rel_pos = jax.nn.one_hot(clipped_offset, 2 * c.max_relative_idx + 1)
                    rel_feats.append(rel_pos)

                rel_feat = jnp.concatenate(rel_feats, axis=-1).astype(dtype)
                with hk.experimental.name_scope("relative_encoding"):
                    pair_activations += common_modules.Linear(c.pair_channel, name="position_activations")(rel_feat)

            pair_after_recycle_relpos = pair_activations
            template_pair_representation = jnp.zeros_like(pair_activations)
            pair_after_template = pair_activations
            template_single_rows = jnp.zeros((0, target_feat.shape[0], c.msa_channel), dtype=pair_activations.dtype)

            (extra_msa_feat, extra_msa_mask) = modules_multimer.create_extra_msa_feature(batch, c.num_extra_msa)
            extra_msa_activations = common_modules.Linear(c.extra_msa_channel, name="extra_msa_activations")(extra_msa_feat).astype(dtype)
            extra_msa_mask = extra_msa_mask.astype(dtype)

            extra_evoformer_input = {"msa": extra_msa_activations, "pair": pair_activations}
            extra_masks = {"msa": extra_msa_mask, "pair": mask_2d}
            extra_evoformer_iteration = modules.EvoformerIteration(c.evoformer, gc, is_extra_msa=True, name="extra_msa_stack")

            def extra_evoformer_fn(x):
                act, safe_key = x
                safe_key, safe_subkey = safe_key.split()
                out = extra_evoformer_iteration(
                    activations=act,
                    masks=extra_masks,
                    is_training=False,
                    safe_key=safe_subkey,
                )
                return (out, safe_key)

            if gc.use_remat:
                extra_evoformer_fn = hk.remat(extra_evoformer_fn)

            safe_key, safe_subkey = safe_key.split()
            extra_evoformer_stack = layer_stack.layer_stack(c.extra_msa_stack_num_block)(extra_evoformer_fn)
            extra_evoformer_output, safe_key = extra_evoformer_stack((extra_evoformer_input, safe_subkey))
            pair_activations = extra_evoformer_output["pair"]
            pair_after_extra = pair_activations

            num_msa_sequences = msa_activations.shape[0]
            evoformer_input = {"msa": msa_activations, "pair": pair_activations}
            evoformer_masks = {"msa": batch["msa_mask"].astype(dtype), "pair": mask_2d}

            pre_msa = evoformer_input["msa"]
            pre_pair = evoformer_input["pair"]
            pre_msa_mask = evoformer_masks["msa"]
            pre_pair_mask = evoformer_masks["pair"]

            evoformer_iteration = modules.EvoformerIteration(c.evoformer, gc, is_extra_msa=False, name="evoformer_iteration")

            def evoformer_fn(x):
                act, safe_key = x
                safe_key, safe_subkey = safe_key.split()
                out = evoformer_iteration(
                    activations=act,
                    masks=evoformer_masks,
                    is_training=False,
                    safe_key=safe_subkey,
                )
                return (out, safe_key)

            if gc.use_remat:
                evoformer_fn = hk.remat(evoformer_fn)

            safe_key, safe_subkey = safe_key.split()
            evoformer_stack = layer_stack.layer_stack(c.evoformer_num_block)(evoformer_fn)
            evoformer_output, _ = evoformer_stack((evoformer_input, safe_subkey))

            msa_activations = evoformer_output["msa"]
            pair_activations = evoformer_output["pair"]
            single_activations = common_modules.Linear(c.seq_channel, name="single_activations")(msa_activations[0])

        representations = {
            "single": single_activations,
            "pair": pair_activations,
            "msa": msa_activations[:num_msa_sequences, :, :],
            "msa_first_row": msa_activations[0],
        }

        struct_out = folding_multimer.StructureModule(
            cfg.model.heads.structure_module,
            gc,
            name="structure_module",
        )(representations, batch, is_training=False, compute_loss=True)

        out_masked_msa = modules.MaskedMsaHead(
            cfg.model.heads.masked_msa,
            gc,
            name="masked_msa_head",
        )(representations, batch=None, is_training=False)["logits"]

        out_distogram = modules.DistogramHead(
            cfg.model.heads.distogram,
            gc,
            name="distogram_head",
        )(representations, batch=None, is_training=False)

        out_experimentally_resolved = modules.ExperimentallyResolvedHead(
            cfg.model.heads.experimentally_resolved,
            gc,
            name="experimentally_resolved_head",
        )({"single": single_activations}, batch=None, is_training=False)["logits"]

        out_lddt = modules.PredictedLDDTHead(
            cfg.model.heads.predicted_lddt,
            gc,
            name="predicted_lddt_head",
        )({"structure_module": struct_out["act"]}, batch=None, is_training=False)

        if has_pae_head:
            out_pae = modules.PredictedAlignedErrorHead(
                cfg.model.heads.predicted_aligned_error,
                gc,
                name="predicted_aligned_error_head",
            )({"pair": pair_activations}, batch=None, is_training=False)
            pae_logits = out_pae["logits"]
            pae_breaks = out_pae["breaks"]
        else:
            pae_logits = jnp.zeros((single_activations.shape[0], single_activations.shape[0], 0), dtype=single_activations.dtype)
            pae_breaks = jnp.zeros((0,), dtype=jnp.float32)

        return (
            batch["msa"],
            batch["deletion_matrix"],
            batch["msa_mask"],
            batch["extra_msa"],
            batch["extra_deletion_matrix"],
            batch["extra_msa_mask"],
            target_feat,
            msa_feat,
            pre_msa,
            pre_pair,
            pre_msa_mask,
            pre_pair_mask,
            pair_after_recycle_relpos,
            template_pair_representation,
            pair_after_template,
            pair_after_extra,
            template_single_rows,
            extra_msa_feat,
            single_activations,
            pair_activations,
            out_masked_msa,
            out_distogram["logits"],
            out_distogram["bin_edges"],
            out_experimentally_resolved,
            out_lddt["logits"],
            pae_logits,
            pae_breaks,
            struct_out["final_atom14_positions"],
            struct_out["traj"],
            struct_out["sidechains"]["angles_sin_cos"],
            struct_out["sidechains"]["unnormalized_angles_sin_cos"],
            struct_out["final_atom_positions"],
            struct_out["final_atom_mask"],
            representations["msa_first_row"],
            representations["pair"],
            batch["residue_index"],
            batch["asym_id"],
            batch["entity_id"],
            batch["sym_id"],
        )

    transformed = hk.transform(fn)
    key = jax.random.PRNGKey(args.seed)

    batch_j = {
        "aatype": jnp.asarray(np_example["aatype"], dtype=jnp.int32),
        "residue_index": jnp.asarray(np_example["residue_index"], dtype=jnp.int32),
        "msa": jnp.asarray(np_example["msa"], dtype=jnp.int32),
        "deletion_matrix": jnp.asarray(np_example["deletion_matrix"], dtype=jnp.float32),
        "msa_mask": jnp.asarray(np_example["msa_mask"], dtype=jnp.float32),
        "seq_mask": jnp.asarray(np_example["seq_mask"], dtype=jnp.float32),
        "asym_id": jnp.asarray(np_example["asym_id"], dtype=jnp.int32),
        "entity_id": jnp.asarray(np_example["entity_id"], dtype=jnp.int32),
        "sym_id": jnp.asarray(np_example["sym_id"], dtype=jnp.int32),
        "cluster_bias_mask": jnp.asarray(np_example["cluster_bias_mask"], dtype=jnp.float32),
        "all_atom_positions": jnp.asarray(np_example["all_atom_positions"], dtype=jnp.float32),
        "all_atom_mask": jnp.asarray(np_example["all_atom_mask"], dtype=jnp.float32),
        "template_aatype": jnp.asarray(np_example["template_aatype"], dtype=jnp.int32),
        "template_all_atom_positions": jnp.asarray(np_example["template_all_atom_positions"], dtype=jnp.float32),
        "template_all_atom_mask": jnp.asarray(np_example["template_all_atom_mask"], dtype=jnp.float32),
    }

    n = int(np_example["aatype"].shape[0])
    prev_pos = jnp.zeros((n, 37, 3), dtype=jnp.float32)
    prev_msa_first_row = jnp.zeros((n, c.msa_channel), dtype=jnp.float32)
    prev_pair = jnp.zeros((n, n, c.pair_channel), dtype=jnp.float32)

    msa_final = None
    deletion_matrix_final = None
    msa_mask_final = None
    extra_msa_final = None
    extra_deletion_matrix_final = None
    extra_msa_mask_final = None
    target_feat_final = None
    msa_feat_final = None

    pre_msa_iters = []
    pre_pair_iters = []
    pre_msa_mask_iters = []
    pair_after_recycle_relpos_iters = []
    template_pair_representation_iters = []
    pair_after_template_iters = []
    pair_after_extra_iters = []
    template_single_rows_iters = []
    extra_msa_feat_iters = []

    out_single_iters = []
    out_pair_iters = []
    out_masked_msa_iters = []
    out_distogram_iters = []
    out_experimentally_resolved_iters = []
    out_lddt_logits_iters = []
    out_plddt_iters = []
    out_pae_logits_iters = []
    out_pae_iters = []
    out_ptm_iters = []
    out_atom14_iters = []
    out_affine_iters = []
    out_angles_iters = []
    out_unnorm_angles_iters = []

    out_single = None
    out_pair = None
    out_masked_msa = None
    out_distogram = None
    out_distogram_bin_edges = None
    out_experimentally_resolved = None
    out_lddt_logits = None
    out_plddt = None
    out_pae_logits = None
    out_pae_breaks = None
    out_pae = None
    out_ptm = None
    out_atom14 = None
    out_affine = None
    out_angles = None
    out_unnorm_angles = None
    out_atom37 = None
    out_atom37_mask = None
    residue_index_out = None
    asym_id_out = None
    entity_id_out = None
    sym_id_out = None

    for _ in range(args.num_recycle + 1):
        (
            msa_cur,
            deletion_matrix_cur,
            msa_mask_cur,
            extra_msa_cur,
            extra_deletion_matrix_cur,
            extra_msa_mask_cur,
            target_feat_cur,
            msa_feat_cur,
            pre_msa,
            pre_pair,
            pre_msa_mask,
            _pre_pair_mask,
            pair_after_recycle_relpos,
            template_pair_representation,
            pair_after_template,
            pair_after_extra,
            template_single_rows,
            extra_msa_feat,
            out_single,
            out_pair,
            out_masked_msa,
            out_distogram,
            out_distogram_bin_edges,
            out_experimentally_resolved,
            out_lddt_logits,
            out_pae_logits,
            out_pae_breaks,
            out_atom14,
            out_affine,
            out_angles,
            out_unnorm_angles,
            out_atom37,
            out_atom37_mask,
            prev_msa_first_row,
            prev_pair,
            residue_index_out,
            asym_id_out,
            entity_id_out,
            sym_id_out,
        ) = transformed.apply(params, key, batch_j, prev_pos, prev_msa_first_row, prev_pair)

        prev_pos = out_atom37

        out_plddt = confidence.compute_plddt(np.asarray(out_lddt_logits, dtype=np.float32))
        if out_pae_logits.shape[-1] > 0:
            pae_stats = confidence.compute_predicted_aligned_error(
                logits=np.asarray(out_pae_logits, dtype=np.float32),
                breaks=np.asarray(out_pae_breaks, dtype=np.float32),
            )
            out_pae = np.asarray(pae_stats["predicted_aligned_error"], dtype=np.float32)
            out_ptm = np.asarray(
                confidence.predicted_tm_score(
                    logits=np.asarray(out_pae_logits, dtype=np.float32),
                    breaks=np.asarray(out_pae_breaks, dtype=np.float32),
                ),
                dtype=np.float32,
            )
        else:
            out_pae = np.zeros((out_atom37.shape[0], out_atom37.shape[0]), dtype=np.float32)
            out_ptm = np.asarray(np.nan, dtype=np.float32)

        msa_final = np.asarray(msa_cur, dtype=np.int32)
        deletion_matrix_final = np.asarray(deletion_matrix_cur, dtype=np.float32)
        msa_mask_final = np.asarray(msa_mask_cur, dtype=np.float32)
        extra_msa_final = np.asarray(extra_msa_cur, dtype=np.int32)[: c.num_extra_msa]
        extra_deletion_matrix_final = np.asarray(extra_deletion_matrix_cur, dtype=np.float32)[: c.num_extra_msa]
        extra_msa_mask_final = np.asarray(extra_msa_mask_cur, dtype=np.float32)[: c.num_extra_msa]
        target_feat_final = np.asarray(target_feat_cur, dtype=np.float32)
        msa_feat_final = np.asarray(msa_feat_cur, dtype=np.float32)

        pre_msa_iters.append(np.asarray(pre_msa, dtype=np.float32))
        pre_pair_iters.append(np.asarray(pre_pair, dtype=np.float32))
        pre_msa_mask_iters.append(np.asarray(pre_msa_mask, dtype=np.float32))
        pair_after_recycle_relpos_iters.append(np.asarray(pair_after_recycle_relpos, dtype=np.float32))
        template_pair_representation_iters.append(np.asarray(template_pair_representation, dtype=np.float32))
        pair_after_template_iters.append(np.asarray(pair_after_template, dtype=np.float32))
        pair_after_extra_iters.append(np.asarray(pair_after_extra, dtype=np.float32))
        template_single_rows_iters.append(np.asarray(template_single_rows, dtype=np.float32))
        extra_msa_feat_iters.append(np.asarray(extra_msa_feat, dtype=np.float32))

        out_single_iters.append(np.asarray(out_single, dtype=np.float32))
        out_pair_iters.append(np.asarray(out_pair, dtype=np.float32))
        out_masked_msa_iters.append(np.asarray(out_masked_msa, dtype=np.float32))
        out_distogram_iters.append(np.asarray(out_distogram, dtype=np.float32))
        out_experimentally_resolved_iters.append(np.asarray(out_experimentally_resolved, dtype=np.float32))
        out_lddt_logits_iters.append(np.asarray(out_lddt_logits, dtype=np.float32))
        out_plddt_iters.append(np.asarray(out_plddt, dtype=np.float32))
        out_pae_logits_iters.append(np.asarray(out_pae_logits, dtype=np.float32))
        out_pae_iters.append(np.asarray(out_pae, dtype=np.float32))
        out_ptm_iters.append(np.asarray(out_ptm, dtype=np.float32))
        out_atom14_iters.append(np.asarray(out_atom14, dtype=np.float32))
        out_affine_iters.append(np.asarray(out_affine, dtype=np.float32))
        out_angles_iters.append(np.asarray(out_angles, dtype=np.float32))
        out_unnorm_angles_iters.append(np.asarray(out_unnorm_angles, dtype=np.float32))

    out_single = np.asarray(out_single, dtype=np.float32)
    out_pair = np.asarray(out_pair, dtype=np.float32)
    out_masked_msa = np.asarray(out_masked_msa, dtype=np.float32)
    out_distogram = np.asarray(out_distogram, dtype=np.float32)
    out_distogram_bin_edges = np.asarray(out_distogram_bin_edges, dtype=np.float32)
    out_experimentally_resolved = np.asarray(out_experimentally_resolved, dtype=np.float32)
    out_lddt_logits = np.asarray(out_lddt_logits, dtype=np.float32)
    out_plddt = np.asarray(out_plddt, dtype=np.float32)
    out_pae_logits = np.asarray(out_pae_logits, dtype=np.float32)
    out_pae_breaks = np.asarray(out_pae_breaks, dtype=np.float32)
    out_pae = np.asarray(out_pae, dtype=np.float32)
    out_ptm = np.asarray(out_ptm, dtype=np.float32)
    out_atom14 = np.asarray(out_atom14, dtype=np.float32)
    out_affine = np.asarray(out_affine, dtype=np.float32)
    out_angles = np.asarray(out_angles, dtype=np.float32)
    out_unnorm_angles = np.asarray(out_unnorm_angles, dtype=np.float32)
    out_atom37 = np.asarray(out_atom37, dtype=np.float32)
    out_atom37_mask = np.asarray(out_atom37_mask, dtype=np.float32)

    asym_id_out = np.asarray(asym_id_out, dtype=np.int32)
    d, dmean, dstd, dmin, dmax, dout = _ca_distance_metrics(
        out_atom37,
        out_atom37_mask,
        residue_constants,
        asym_id=asym_id_out,
        intra_chain_only=False,
    )
    d_intra, dmean_intra, dstd_intra, dmin_intra, dmax_intra, dout_intra = _ca_distance_metrics(
        out_atom37,
        out_atom37_mask,
        residue_constants,
        asym_id=asym_id_out,
        intra_chain_only=True,
    )
    print("Geometry check (consecutive C-alpha distances)")
    print(f"  count: {d.shape[0]}")
    print(f"  mean: {dmean:.6f} A")
    print(f"  std:  {dstd:.6f} A")
    print(f"  min:  {dmin:.6f} A")
    print(f"  max:  {dmax:.6f} A")
    print(f"  outliers (<3.2 or >4.4 A): {np.sum((d < 3.2) | (d > 4.4))} ({dout:.3f})")
    if np.unique(asym_id_out).size > 1:
        print("Geometry check (intra-chain consecutive C-alpha distances)")
        print(f"  count: {d_intra.shape[0]}")
        print(f"  mean: {dmean_intra:.6f} A")
        print(f"  std:  {dstd_intra:.6f} A")
        print(f"  min:  {dmin_intra:.6f} A")
        print(f"  max:  {dmax_intra:.6f} A")
        print(f"  outliers (<3.2 or >4.4 A): {np.sum((d_intra < 3.2) | (d_intra > 4.4))} ({dout_intra:.3f})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_arrays = dict(
        aatype=np.asarray(np_example["aatype"], dtype=np.int32),
        residue_index=np.asarray(residue_index_out, dtype=np.int32),
        asym_id=asym_id_out,
        entity_id=np.asarray(entity_id_out, dtype=np.int32),
        sym_id=np.asarray(sym_id_out, dtype=np.int32),
        seq_mask=np.asarray(np_example["seq_mask"], dtype=np.float32),
        target_feat=np.asarray(target_feat_final, dtype=np.float32),
        msa_feat=np.asarray(msa_feat_final, dtype=np.float32),
        msa=np.asarray(msa_final, dtype=np.int32),
        deletion_matrix=np.asarray(deletion_matrix_final, dtype=np.float32),
        msa_mask=np.asarray(msa_mask_final, dtype=np.float32),
        extra_msa=np.asarray(extra_msa_final, dtype=np.int32),
        extra_deletion_matrix=np.asarray(extra_deletion_matrix_final, dtype=np.float32),
        extra_msa_mask=np.asarray(extra_msa_mask_final, dtype=np.float32),
        out_single=out_single,
        out_pair=out_pair,
        out_masked_msa_logits=out_masked_msa,
        out_distogram_logits=out_distogram,
        out_distogram_bin_edges=out_distogram_bin_edges,
        out_experimentally_resolved_logits=out_experimentally_resolved,
        out_predicted_lddt_logits=out_lddt_logits,
        out_plddt=out_plddt,
        mean_plddt=np.float32(np.mean(out_plddt)),
        out_atom14=out_atom14,
        out_affine=out_affine,
        out_angles=out_angles,
        out_unnormalized_angles=out_unnorm_angles,
        out_atom37=out_atom37,
        atom37_mask=out_atom37_mask,
        ca_consecutive_distances=d.astype(np.float32),
        ca_distance_mean=np.float32(dmean),
        ca_distance_std=np.float32(dstd),
        ca_distance_min=np.float32(dmin),
        ca_distance_max=np.float32(dmax),
        ca_distance_outlier_fraction=np.float32(dout),
        ca_consecutive_distances_intra_chain=d_intra.astype(np.float32),
        ca_distance_intra_chain_mean=np.float32(dmean_intra),
        ca_distance_intra_chain_std=np.float32(dstd_intra),
        ca_distance_intra_chain_min=np.float32(dmin_intra),
        ca_distance_intra_chain_max=np.float32(dmax_intra),
        ca_distance_intra_chain_outlier_fraction=np.float32(dout_intra),
        num_recycle=np.int32(args.num_recycle),
        has_pae_head=np.int32(1 if has_pae_head else 0),
    )
    if has_pae_head:
        out_arrays["out_predicted_aligned_error_logits"] = out_pae_logits
        out_arrays["out_predicted_aligned_error_breaks"] = out_pae_breaks
        out_arrays["out_predicted_aligned_error"] = out_pae
        out_arrays["out_predicted_tm_score"] = out_ptm
    np.savez(args.out, **out_arrays)
    print(f"Saved AF2 multimer run to {args.out}")

    if args.dump_pre_evo:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_pre_evo)), exist_ok=True)
        dump_arrays = dict(
            aatype=np.asarray(np_example["aatype"], dtype=np.int32),
            residue_index=np.asarray(residue_index_out, dtype=np.int32),
            asym_id=np.asarray(asym_id_out, dtype=np.int32),
            entity_id=np.asarray(entity_id_out, dtype=np.int32),
            sym_id=np.asarray(sym_id_out, dtype=np.int32),
            seq_mask=np.asarray(np_example["seq_mask"], dtype=np.float32),
            target_feat=np.asarray(target_feat_final, dtype=np.float32),
            msa_feat=np.asarray(msa_feat_final, dtype=np.float32),
            msa=np.asarray(msa_final, dtype=np.int32),
            deletion_matrix=np.asarray(deletion_matrix_final, dtype=np.float32),
            msa_mask=np.asarray(msa_mask_final, dtype=np.float32),
            extra_msa=np.asarray(extra_msa_final, dtype=np.int32),
            extra_deletion_matrix=np.asarray(extra_deletion_matrix_final, dtype=np.float32),
            extra_msa_mask=np.asarray(extra_msa_mask_final, dtype=np.float32),
            pre_msa=np.stack(pre_msa_iters, axis=0),
            pre_pair=np.stack(pre_pair_iters, axis=0),
            pre_msa_mask=np.stack(pre_msa_mask_iters, axis=0),
            pair_after_recycle_relpos=np.stack(pair_after_recycle_relpos_iters, axis=0),
            template_pair_representation=np.stack(template_pair_representation_iters, axis=0),
            pair_after_template=np.stack(pair_after_template_iters, axis=0),
            pair_after_extra=np.stack(pair_after_extra_iters, axis=0),
            extra_msa_feat=np.stack(extra_msa_feat_iters, axis=0),
            out_single=np.stack(out_single_iters, axis=0),
            out_pair=np.stack(out_pair_iters, axis=0),
            out_masked_msa_logits=np.stack(out_masked_msa_iters, axis=0),
            out_distogram_logits=np.stack(out_distogram_iters, axis=0),
            out_distogram_bin_edges=np.asarray(out_distogram_bin_edges, dtype=np.float32),
            out_experimentally_resolved_logits=np.stack(out_experimentally_resolved_iters, axis=0),
            out_predicted_lddt_logits=np.stack(out_lddt_logits_iters, axis=0),
            out_plddt=np.stack(out_plddt_iters, axis=0),
            out_atom14=np.stack(out_atom14_iters, axis=0),
            out_affine=np.stack(out_affine_iters, axis=0),
            out_angles=np.stack(out_angles_iters, axis=0),
            out_unnormalized_angles=np.stack(out_unnorm_angles_iters, axis=0),
            out_atom37=np.asarray(out_atom37, dtype=np.float32),
            atom37_mask=np.asarray(out_atom37_mask, dtype=np.float32),
            num_recycle=np.int32(args.num_recycle),
            has_pae_head=np.int32(1 if has_pae_head else 0),
        )
        if has_pae_head:
            dump_arrays["out_predicted_aligned_error_logits"] = np.stack(out_pae_logits_iters, axis=0)
            dump_arrays["out_predicted_aligned_error_breaks"] = np.asarray(out_pae_breaks, dtype=np.float32)
            dump_arrays["out_predicted_aligned_error"] = np.stack(out_pae_iters, axis=0)
            dump_arrays["out_predicted_tm_score"] = np.stack(out_ptm_iters, axis=0)
        np.savez(args.dump_pre_evo, **dump_arrays)
        print(f"Saved pre-evo dump to {args.dump_pre_evo}")


if __name__ == "__main__":
    main()
