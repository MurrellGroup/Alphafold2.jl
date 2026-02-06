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


RESTYPE_3TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _slice_prefix(params, prefix):
    out = {}
    for scope, vars_dict in params.items():
        if scope.startswith(prefix):
            out[scope[len(prefix):]] = {k: np.asarray(v) for k, v in vars_dict.items()}
    return out


def _aatype_from_sequence(seq, residue_constants):
    seq = seq.strip().upper()
    idxs = [residue_constants.restype_order.get(ch, 20) for ch in seq]
    return np.asarray(idxs, dtype=np.int32)


def _deletion_value_transform(x: np.ndarray) -> np.ndarray:
    return np.arctan(x / 3.0) * (2.0 / np.pi)


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


def _parse_template_chain(
    pdb_path: str,
    chain_id: str,
    residue_constants,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    atom_order = residue_constants.atom_order
    residues: List[Dict] = []
    residue_index = {}

    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain_id:
                continue

            key = (line[22:26], line[26])
            if key not in residue_index:
                residue_index[key] = len(residues)
                residues.append({"resname": line[17:20].strip(), "atoms": {}})

            atom_name = line[12:16].strip()
            if atom_name not in atom_order:
                continue

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            residues[residue_index[key]]["atoms"][atom_name] = (x, y, z)

    if not residues:
        raise ValueError(f"No residues parsed from {pdb_path} chain {chain_id!r}")

    seq = "".join(RESTYPE_3TO1.get(r["resname"], "X") for r in residues)
    n = len(seq)
    aatype = _aatype_from_sequence(seq, residue_constants)

    positions = np.zeros((1, n, 37, 3), dtype=np.float32)
    masks = np.zeros((1, n, 37), dtype=np.float32)
    for i, r in enumerate(residues):
        for atom_name, xyz in r["atoms"].items():
            aidx = atom_order[atom_name]
            positions[0, i, aidx, :] = np.asarray(xyz, dtype=np.float32)
            masks[0, i, aidx] = 1.0

    return seq, aatype, positions, masks


def _make_features_with_template(
    seq: str,
    aatype: np.ndarray,
    template_aatype: np.ndarray,
    template_positions: np.ndarray,
    template_masks: np.ndarray,
    residue_constants,
    msa: np.ndarray | None = None,
    deletion_matrix: np.ndarray | None = None,
):
    n = int(aatype.shape[0])
    if msa is None:
        msa = aatype[None, :]  # (1, N)
    if deletion_matrix is None:
        deletion_matrix = np.zeros((msa.shape[0], n), dtype=np.float32)
    msa = np.asarray(msa, dtype=np.int32)
    deletion_matrix = np.asarray(deletion_matrix, dtype=np.float32)
    if msa.shape[1] != n:
        raise ValueError(f"MSA length mismatch: {msa.shape[1]} != {n}")
    if deletion_matrix.shape != msa.shape:
        raise ValueError(f"Deletion matrix shape mismatch: {deletion_matrix.shape} vs {msa.shape}")

    has_break = np.zeros((n,), dtype=np.float32)
    aatype_1hot = np.eye(21, dtype=np.float32)[aatype]  # (N, 21)
    target_feat = np.concatenate([has_break[:, None], aatype_1hot], axis=-1)  # (N, 22)

    msa_1hot = np.eye(23, dtype=np.float32)[msa]  # (N_seq, N, 23)
    has_del = (deletion_matrix > 0).astype(np.float32)
    del_val = _deletion_value_transform(deletion_matrix).astype(np.float32)
    cluster_profile = msa_1hot.copy()
    deletion_mean = del_val.copy()
    msa_feat = np.concatenate(
        [
            msa_1hot,
            has_del[..., None],
            del_val[..., None],
            cluster_profile,
            deletion_mean[..., None],
        ],
        axis=-1,
    )  # (N_seq, N, 49)

    extra_msa = msa.copy().astype(np.int32)
    extra_deletion_matrix = deletion_matrix.copy().astype(np.float32)
    extra_has_deletion = has_del.copy().astype(np.float32)
    extra_deletion_value = del_val.copy().astype(np.float32)
    gap_id = residue_constants.HHBLITS_AA_TO_ID["-"]
    extra_msa_mask = (extra_msa != gap_id).astype(np.float32)

    seq_mask = np.ones((n,), dtype=np.float32)
    msa_mask = (msa != gap_id).astype(np.float32)
    residue_index = np.arange(n, dtype=np.int32)

    return {
        "target_feat": target_feat,
        "msa_feat": msa_feat,
        "seq_mask": seq_mask,
        "msa_mask": msa_mask,
        "residue_index": residue_index,
        "aatype": aatype,
        "msa": msa,
        "deletion_matrix": deletion_matrix,
        "extra_msa": extra_msa,
        "extra_deletion_matrix": extra_deletion_matrix,
        "extra_has_deletion": extra_has_deletion,
        "extra_deletion_value": extra_deletion_value,
        "extra_msa_mask": extra_msa_mask,
        "template_aatype": template_aatype[None, :],
        "template_all_atom_positions": template_positions,
        "template_all_atom_masks": template_masks,
        "template_sum_probs": np.ones((1, 1), dtype=np.float32),
        "template_mask": np.ones((1,), dtype=np.float32),
        "sequence": seq,
    }


def _build_atom14_to_atom37_tables(aatype, residue_constants):
    restypes = residue_constants.restypes
    atom_types = residue_constants.atom_types
    restype_name_to_atom14_names = residue_constants.restype_name_to_atom14_names
    restype_1to3 = residue_constants.restype_1to3

    restype_atom37_to_atom14 = np.zeros((21, len(atom_types)), dtype=np.int32)
    restype_atom37_mask = np.zeros((21, len(atom_types)), dtype=np.float32)

    for i, rt in enumerate(restypes):
        atom14_names = restype_name_to_atom14_names[restype_1to3[rt]]
        atom14_idx = {name: idx for idx, name in enumerate(atom14_names) if name}
        for a, atom_name in enumerate(atom_types):
            if atom_name in atom14_idx:
                restype_atom37_to_atom14[i, a] = atom14_idx[atom_name]
                restype_atom37_mask[i, a] = 1.0

    idx = np.clip(aatype, 0, 20)
    residx_atom37_to_atom14 = restype_atom37_to_atom14[idx]  # (N, 37)
    atom37_exists = restype_atom37_mask[idx]  # (N, 37)
    return residx_atom37_to_atom14, atom37_exists


def _ca_distance_metrics(atom37, atom37_mask, residue_constants):
    ca_idx = residue_constants.atom_order["CA"]
    valid = atom37_mask[:, ca_idx] > 0.5
    dists = []
    for i in range(atom37.shape[0] - 1):
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
    parser.add_argument("--template-pdb", required=True)
    parser.add_argument("--template-chain", default="A")
    parser.add_argument("--sequence", default="")
    parser.add_argument("--msa-file", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dump-pre-evo", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-recycle", type=int, default=1)
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.common import confidence  # pylint: disable=import-outside-toplevel
    from alphafold.common import residue_constants  # pylint: disable=import-outside-toplevel
    from alphafold.model import all_atom  # pylint: disable=import-outside-toplevel
    from alphafold.model import common_modules  # pylint: disable=import-outside-toplevel
    from alphafold.model import config as af_config  # pylint: disable=import-outside-toplevel
    from alphafold.model import folding  # pylint: disable=import-outside-toplevel
    from alphafold.model import layer_stack  # pylint: disable=import-outside-toplevel
    from alphafold.model import modules  # pylint: disable=import-outside-toplevel
    from alphafold.model import prng  # pylint: disable=import-outside-toplevel
    from alphafold.model import r3  # pylint: disable=import-outside-toplevel
    from alphafold.model import utils  # pylint: disable=import-outside-toplevel

    template_seq, template_aatype, template_pos, template_mask = _parse_template_chain(
        args.template_pdb, args.template_chain, residue_constants
    )
    seq = args.sequence.strip().upper() if args.sequence.strip() else template_seq
    aatype = _aatype_from_sequence(seq, residue_constants)
    if aatype.shape[0] != template_aatype.shape[0]:
        raise ValueError(
            "Template and query sequence lengths differ; this script expects a direct per-residue template."
        )

    if args.msa_file.strip():
        msa_ids, deletion_matrix = _load_msa_file(args.msa_file.strip(), aatype, residue_constants)
    else:
        msa_ids, deletion_matrix = None, None
    feats = _make_features_with_template(
        seq,
        aatype,
        template_aatype,
        template_pos,
        template_mask,
        residue_constants,
        msa=msa_ids,
        deletion_matrix=deletion_matrix,
    )
    residx_atom37_to_atom14_np, atom37_exists_np = _build_atom14_to_atom37_tables(
        feats["aatype"], residue_constants
    )

    model_name = args.model_name.strip()
    if not model_name:
        model_name = "model_1_ptm" if "_ptm" in os.path.basename(args.params) else "model_1"
    cfg = af_config.model_config(model_name)
    gc = cfg.model.global_config
    gc.deterministic = True
    gc.eval_dropout = False
    c = cfg.model.embeddings_and_evoformer
    template_placeholder_for_undefined = not gc.zero_init

    with open(args.params, "rb") as f:
        flat = np.load(f, allow_pickle=False)
        full_params = utils.flat_params_to_haiku(flat)
    params = _slice_prefix(full_params, "alphafold/alphafold_iteration/")
    has_pae_head = "predicted_aligned_error_head/logits" in params

    residx_atom37_to_atom14 = jnp.asarray(residx_atom37_to_atom14_np, dtype=jnp.int32)
    atom37_exists = jnp.asarray(atom37_exists_np, dtype=jnp.float32)

    def fn(
        target_feat,
        msa_feat,
        seq_mask,
        msa_mask,
        residue_index,
        aatype_in,
        extra_msa,
        extra_has_deletion,
        extra_deletion_value,
        extra_msa_mask,
        template_aatype_in,
        template_all_atom_positions,
        template_all_atom_masks,
        template_sum_probs,
        template_mask_in,
        prev_pos,
        prev_msa_first_row,
        prev_pair,
    ):
        safe_key = prng.SafeKey(hk.next_rng_key())

        # Equivalent to EmbeddingsAndEvoformer, but returns the pre-evoformer
        # activations so Julia can consume exactly the same complex inputs.
        with hk.experimental.name_scope("evoformer"):
            preprocess_1d = common_modules.Linear(c.msa_channel, name="preprocess_1d")(target_feat)
            preprocess_msa = common_modules.Linear(c.msa_channel, name="preprocess_msa")(msa_feat)
            msa_activations = jnp.expand_dims(preprocess_1d, axis=0) + preprocess_msa

            left_single = common_modules.Linear(c.pair_channel, name="left_single")(target_feat)
            right_single = common_modules.Linear(c.pair_channel, name="right_single")(target_feat)
            pair_activations = left_single[:, None] + right_single[None]
            mask_2d = seq_mask[:, None] * seq_mask[None, :]

            prev_pseudo_beta = modules.pseudo_beta_fn(aatype_in, prev_pos, None)
            dgram = modules.dgram_from_positions(prev_pseudo_beta, **c.prev_pos)
            pair_activations += common_modules.Linear(c.pair_channel, name="prev_pos_linear")(dgram)

            prev_msa_first_row_n = common_modules.LayerNorm(
                axis=[-1], create_scale=True, create_offset=True, name="prev_msa_first_row_norm"
            )(prev_msa_first_row)
            msa_activations = msa_activations.at[0].add(prev_msa_first_row_n)

            prev_pair_n = common_modules.LayerNorm(
                axis=[-1], create_scale=True, create_offset=True, name="prev_pair_norm"
            )(prev_pair)
            pair_activations += prev_pair_n

            if c.max_relative_feature:
                offset = residue_index[:, None] - residue_index[None, :]
                rel_pos = jax.nn.one_hot(
                    jnp.clip(offset + c.max_relative_feature, a_min=0, a_max=2 * c.max_relative_feature),
                    2 * c.max_relative_feature + 1,
                )
                pair_activations += common_modules.Linear(c.pair_channel, name="pair_activiations")(rel_pos)
            pair_after_recycle_relpos = pair_activations

            template_pseudo_beta, template_pseudo_beta_mask = modules.pseudo_beta_fn(
                template_aatype_in, template_all_atom_positions, template_all_atom_masks
            )
            template_batch = {
                "template_aatype": template_aatype_in,
                "template_all_atom_positions": template_all_atom_positions,
                "template_all_atom_masks": template_all_atom_masks,
                "template_sum_probs": template_sum_probs,
                "template_mask": template_mask_in,
                "template_pseudo_beta": template_pseudo_beta,
                "template_pseudo_beta_mask": template_pseudo_beta_mask,
            }
            template_pair_representation = modules.TemplateEmbedding(c.template, gc)(
                pair_activations, template_batch, mask_2d, is_training=False
            )
            pair_activations += template_pair_representation
            pair_after_template = pair_activations
            template_single_rows = jnp.zeros((0, target_feat.shape[0], c.msa_channel), dtype=pair_activations.dtype)

            extra_msa_feat = modules.create_extra_msa_feature(
                {
                    "extra_msa": extra_msa,
                    "extra_has_deletion": extra_has_deletion,
                    "extra_deletion_value": extra_deletion_value,
                }
            )
            extra_msa_activations = common_modules.Linear(c.extra_msa_channel, name="extra_msa_activations")(
                extra_msa_feat
            )
            extra_msa_stack_input = {"msa": extra_msa_activations, "pair": pair_activations}
            extra_msa_stack_iteration = modules.EvoformerIteration(
                c.evoformer, gc, is_extra_msa=True, name="extra_msa_stack"
            )

            def extra_msa_stack_fn(x):
                act, key = x
                key, sub = key.split()
                out = extra_msa_stack_iteration(
                    activations=act,
                    masks={"msa": extra_msa_mask, "pair": mask_2d},
                    is_training=False,
                    safe_key=sub,
                )
                return out, key

            extra_msa_stack = layer_stack.layer_stack(c.extra_msa_stack_num_block)(extra_msa_stack_fn)
            extra_msa_output, safe_key = extra_msa_stack((extra_msa_stack_input, safe_key))
            pair_activations = extra_msa_output["pair"]
            pair_after_extra = pair_activations

            evoformer_input = {"msa": msa_activations, "pair": pair_activations}
            evoformer_masks = {"msa": msa_mask, "pair": mask_2d}

            if c.template.enabled and c.template.embed_torsion_angles:
                num_templ, num_res = template_aatype_in.shape
                aatype_one_hot = jax.nn.one_hot(template_aatype_in, 22, axis=-1)
                ret = all_atom.atom37_to_torsion_angles(
                    aatype=template_aatype_in,
                    all_atom_pos=template_all_atom_positions,
                    all_atom_mask=template_all_atom_masks,
                    placeholder_for_undefined=template_placeholder_for_undefined,
                )
                template_features = jnp.concatenate(
                    [
                        aatype_one_hot,
                        jnp.reshape(ret["torsion_angles_sin_cos"], [num_templ, num_res, 14]),
                        jnp.reshape(ret["alt_torsion_angles_sin_cos"], [num_templ, num_res, 14]),
                        ret["torsion_angles_mask"],
                    ],
                    axis=-1,
                )
                template_activations = common_modules.Linear(
                    c.msa_channel, initializer="relu", name="template_single_embedding"
                )(template_features)
                template_activations = jax.nn.relu(template_activations)
                template_activations = common_modules.Linear(
                    c.msa_channel, initializer="relu", name="template_projection"
                )(template_activations)
                template_single_rows = template_activations
                evoformer_input["msa"] = jnp.concatenate([evoformer_input["msa"], template_activations], axis=0)
                torsion_angle_mask = ret["torsion_angles_mask"][:, :, 2].astype(evoformer_masks["msa"].dtype)
                evoformer_masks["msa"] = jnp.concatenate([evoformer_masks["msa"], torsion_angle_mask], axis=0)

            pre_msa = evoformer_input["msa"]
            pre_pair = evoformer_input["pair"]
            pre_msa_mask = evoformer_masks["msa"]
            pre_pair_mask = evoformer_masks["pair"]

            evoformer_iteration = modules.EvoformerIteration(
                c.evoformer, gc, is_extra_msa=False, name="evoformer_iteration"
            )

            def evoformer_fn(x):
                act, key = x
                key, sub = key.split()
                out = evoformer_iteration(
                    activations=act,
                    masks=evoformer_masks,
                    is_training=False,
                    safe_key=sub,
                )
                return out, key

            evoformer_stack = layer_stack.layer_stack(c.evoformer_num_block)(evoformer_fn)
            evoformer_output, safe_key = evoformer_stack((evoformer_input, safe_key))
            single_activations = common_modules.Linear(c.seq_channel, name="single_activations")(
                evoformer_output["msa"][0]
            )

        with hk.experimental.name_scope("structure_module"):
            struct_out = folding.generate_affines(
                representations={"single": single_activations, "pair": evoformer_output["pair"]},
                batch={"seq_mask": seq_mask, "aatype": aatype_in},
                config=cfg.model.heads.structure_module,
                global_config=gc,
                is_training=False,
                safe_key=safe_key,
            )
        out_masked_msa = modules.MaskedMsaHead(
            cfg.model.heads.masked_msa,
            gc,
            name="masked_msa_head",
        )({"msa": evoformer_output["msa"]}, batch=None, is_training=False)["logits"]
        out_distogram = modules.DistogramHead(
            cfg.model.heads.distogram,
            gc,
            name="distogram_head",
        )({"pair": evoformer_output["pair"]}, batch=None, is_training=False)
        out_experimentally_resolved = modules.ExperimentallyResolvedHead(
            cfg.model.heads.experimentally_resolved,
            gc,
            name="experimentally_resolved_head",
        )({"single": single_activations}, batch=None, is_training=False)["logits"]
        predicted_lddt = modules.PredictedLDDTHead(
            cfg.model.heads.predicted_lddt,
            gc,
            name="predicted_lddt_head",
        )({"structure_module": struct_out["act"]}, batch=None, is_training=False)
        lddt_logits = predicted_lddt["logits"]
        if has_pae_head:
            predicted_aligned_error = modules.PredictedAlignedErrorHead(
                cfg.model.heads.predicted_aligned_error,
                gc,
                name="predicted_aligned_error_head",
            )({"pair": evoformer_output["pair"]}, batch=None, is_training=False)
            pae_logits = predicted_aligned_error["logits"]
            pae_breaks = predicted_aligned_error["breaks"]
        else:
            pae_logits = jnp.zeros(
                (single_activations.shape[0], single_activations.shape[0], 0),
                dtype=single_activations.dtype,
            )
            pae_breaks = jnp.zeros((0,), dtype=jnp.float32)

        atom14 = r3.vecs_to_tensor(struct_out["sc"]["atom_pos"])[-1]
        affine = struct_out["affine"][-1]
        atom37 = jnp.take_along_axis(atom14, residx_atom37_to_atom14[..., None], axis=1)
        atom37 = atom37 * atom37_exists[..., None]

        return (
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
            evoformer_output["pair"],
            out_masked_msa,
            out_distogram["logits"],
            out_distogram["bin_edges"],
            out_experimentally_resolved,
            lddt_logits,
            pae_logits,
            pae_breaks,
            atom14,
            affine,
            atom37,
            atom37_exists,
            evoformer_output["msa"][0],
            evoformer_output["pair"],
        )

    transformed = hk.transform(fn)
    key = jax.random.PRNGKey(args.seed)

    batch_j = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in feats.items()}
    n = int(feats["aatype"].shape[0])
    prev_pos = jnp.zeros((n, 37, 3), dtype=jnp.float32)
    prev_msa_first_row = jnp.zeros((n, c.msa_channel), dtype=jnp.float32)
    prev_pair = jnp.zeros((n, n, c.pair_channel), dtype=jnp.float32)

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
    out_atom37 = None
    out_atom37_mask = None

    for _ in range(args.num_recycle + 1):
        (
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
            out_atom37,
            out_atom37_mask,
            prev_msa_first_row,
            prev_pair,
        ) = transformed.apply(
            params,
            key,
            batch_j["target_feat"],
            batch_j["msa_feat"],
            batch_j["seq_mask"],
            batch_j["msa_mask"],
            batch_j["residue_index"],
            batch_j["aatype"],
            batch_j["extra_msa"],
            batch_j["extra_has_deletion"],
            batch_j["extra_deletion_value"],
            batch_j["extra_msa_mask"],
            batch_j["template_aatype"],
            batch_j["template_all_atom_positions"],
            batch_j["template_all_atom_masks"],
            batch_j["template_sum_probs"],
            batch_j["template_mask"],
            prev_pos,
            prev_msa_first_row,
            prev_pair,
        )
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
    out_atom37 = np.asarray(out_atom37, dtype=np.float32)
    out_atom37_mask = np.asarray(out_atom37_mask, dtype=np.float32)

    d, dmean, dstd, dmin, dmax, dout = _ca_distance_metrics(out_atom37, out_atom37_mask, residue_constants)
    print("Geometry check (consecutive C-alpha distances)")
    print(f"  count: {d.shape[0]}")
    print(f"  mean: {dmean:.6f} A")
    print(f"  std:  {dstd:.6f} A")
    print(f"  min:  {dmin:.6f} A")
    print(f"  max:  {dmax:.6f} A")
    print(f"  outliers (<3.2 or >4.4 A): {np.sum((d < 3.2) | (d > 4.4))} ({dout:.3f})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_arrays = dict(
        target_feat=np.asarray(feats["target_feat"], dtype=np.float32),
        msa_feat=np.asarray(feats["msa_feat"], dtype=np.float32),
        seq_mask=np.asarray(feats["seq_mask"], dtype=np.float32),
        msa_mask=np.asarray(feats["msa_mask"], dtype=np.float32),
        residue_index=np.asarray(feats["residue_index"], dtype=np.int32),
        aatype=np.asarray(feats["aatype"], dtype=np.int32),
        msa=np.asarray(feats["msa"], dtype=np.int32),
        deletion_matrix=np.asarray(feats["deletion_matrix"], dtype=np.float32),
        template_aatype=np.asarray(feats["template_aatype"], dtype=np.int32),
        template_all_atom_positions=np.asarray(feats["template_all_atom_positions"], dtype=np.float32),
        template_all_atom_masks=np.asarray(feats["template_all_atom_masks"], dtype=np.float32),
        extra_msa=np.asarray(feats["extra_msa"], dtype=np.int32),
        extra_deletion_matrix=np.asarray(feats["extra_deletion_matrix"], dtype=np.float32),
        extra_msa_mask=np.asarray(feats["extra_msa_mask"], dtype=np.float32),
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
        out_atom37=out_atom37,
        atom37_mask=out_atom37_mask,
        ca_consecutive_distances=d.astype(np.float32),
        ca_distance_mean=np.float32(dmean),
        ca_distance_std=np.float32(dstd),
        ca_distance_min=np.float32(dmin),
        ca_distance_max=np.float32(dmax),
        ca_distance_outlier_fraction=np.float32(dout),
        num_recycle=np.int32(args.num_recycle),
        has_pae_head=np.int32(1 if has_pae_head else 0),
    )
    if has_pae_head:
        out_arrays["out_predicted_aligned_error_logits"] = out_pae_logits
        out_arrays["out_predicted_aligned_error_breaks"] = out_pae_breaks
        out_arrays["out_predicted_aligned_error"] = out_pae
        out_arrays["out_predicted_tm_score"] = out_ptm
    np.savez(args.out, **out_arrays)
    print(f"Saved AF2 template case run to {args.out}")

    if args.dump_pre_evo:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_pre_evo)), exist_ok=True)
        dump_arrays = dict(
            aatype=np.asarray(feats["aatype"], dtype=np.int32),
            residue_index=np.asarray(feats["residue_index"], dtype=np.int32),
            msa=np.asarray(feats["msa"], dtype=np.int32),
            deletion_matrix=np.asarray(feats["deletion_matrix"], dtype=np.float32),
            msa_mask=np.asarray(feats["msa_mask"], dtype=np.float32),
            extra_msa=np.asarray(feats["extra_msa"], dtype=np.int32),
            extra_deletion_matrix=np.asarray(feats["extra_deletion_matrix"], dtype=np.float32),
            extra_msa_mask=np.asarray(feats["extra_msa_mask"], dtype=np.float32),
            seq_mask=np.asarray(feats["seq_mask"], dtype=np.float32),
            template_aatype=np.asarray(feats["template_aatype"], dtype=np.int32),
            template_all_atom_positions=np.asarray(feats["template_all_atom_positions"], dtype=np.float32),
            template_all_atom_masks=np.asarray(feats["template_all_atom_masks"], dtype=np.float32),
            template_placeholder_for_undefined=np.int32(1 if template_placeholder_for_undefined else 0),
            pre_msa=np.stack(pre_msa_iters, axis=0),
            pre_pair=np.stack(pre_pair_iters, axis=0),
            pre_msa_mask=np.stack(pre_msa_mask_iters, axis=0),
            pair_after_recycle_relpos=np.stack(pair_after_recycle_relpos_iters, axis=0),
            template_pair_representation=np.stack(template_pair_representation_iters, axis=0),
            pair_after_template=np.stack(pair_after_template_iters, axis=0),
            pair_after_extra=np.stack(pair_after_extra_iters, axis=0),
            template_single_rows=np.stack(template_single_rows_iters, axis=0),
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
