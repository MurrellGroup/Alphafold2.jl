#!/usr/bin/env python3
import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def _slice_prefix(params, prefix):
    out = {}
    for scope, vars_dict in params.items():
        if scope.startswith(prefix):
            out[scope[len(prefix):]] = {k: np.asarray(v) for k, v in vars_dict.items()}
    return out


def _aatype_from_sequence(seq, residue_constants):
    seq = seq.strip().upper()
    idxs = []
    for ch in seq:
        idxs.append(residue_constants.restype_order.get(ch, 20))
    return np.asarray(idxs, dtype=np.int32)


def _make_minimal_features(aatype):
    n = int(aatype.shape[0])
    msa = aatype[None, :]  # (1, N)

    has_break = np.zeros((n,), dtype=np.float32)
    aatype_1hot = np.eye(21, dtype=np.float32)[aatype]  # (N, 21)
    target_feat = np.concatenate([has_break[:, None], aatype_1hot], axis=-1)  # (N, 22)

    msa_1hot = np.eye(23, dtype=np.float32)[msa]  # (1, N, 23)
    has_del = np.zeros((1, n), dtype=np.float32)
    del_val = np.zeros((1, n), dtype=np.float32)
    cluster_profile = msa_1hot.copy()
    deletion_mean = np.zeros((1, n), dtype=np.float32)
    msa_feat = np.concatenate(
        [
            msa_1hot,
            has_del[..., None],
            del_val[..., None],
            cluster_profile,
            deletion_mean[..., None],
        ],
        axis=-1,
    )  # (1, N, 49)

    seq_mask = np.ones((n,), dtype=np.float32)
    msa_mask = np.ones((1, n), dtype=np.float32)
    residue_index = np.arange(n, dtype=np.int32)

    return {
        "target_feat": target_feat,
        "msa_feat": msa_feat,
        "seq_mask": seq_mask,
        "msa_mask": msa_mask,
        "residue_index": residue_index,
        "aatype": aatype,
    }


def _build_atom14_to_atom37_tables(aatype, residue_constants):
    restypes = residue_constants.restypes
    atom_types = residue_constants.atom_types
    atom_order = residue_constants.atom_order
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
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-recycle", type=int, default=1)
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.common import residue_constants  # pylint: disable=import-outside-toplevel
    from alphafold.model import common_modules  # pylint: disable=import-outside-toplevel
    from alphafold.model import config as af_config  # pylint: disable=import-outside-toplevel
    from alphafold.model import folding  # pylint: disable=import-outside-toplevel
    from alphafold.model import layer_stack  # pylint: disable=import-outside-toplevel
    from alphafold.model import modules  # pylint: disable=import-outside-toplevel
    from alphafold.model import prng  # pylint: disable=import-outside-toplevel
    from alphafold.model import r3  # pylint: disable=import-outside-toplevel
    from alphafold.model import utils  # pylint: disable=import-outside-toplevel

    cfg = af_config.model_config("model_1")
    gc = cfg.model.global_config
    gc.deterministic = True
    gc.eval_dropout = False

    aatype = _aatype_from_sequence(args.sequence, residue_constants)
    feats = _make_minimal_features(aatype)
    residx_atom37_to_atom14_np, atom37_exists_np = _build_atom14_to_atom37_tables(
        feats["aatype"], residue_constants
    )

    with open(args.params, "rb") as f:
        flat = np.load(f, allow_pickle=False)
        full_params = utils.flat_params_to_haiku(flat)
    params = _slice_prefix(full_params, "alphafold/alphafold_iteration/")

    c = cfg.model.embeddings_and_evoformer

    residx_atom37_to_atom14 = jnp.asarray(residx_atom37_to_atom14_np, dtype=jnp.int32)
    atom37_exists = jnp.asarray(atom37_exists_np, dtype=jnp.float32)

    def fn(
        target_feat,
        msa_feat,
        seq_mask,
        msa_mask,
        residue_index,
        aatype,
        prev_pos,
        prev_msa_first_row,
        prev_pair,
    ):
        safe_key = prng.SafeKey(hk.next_rng_key())
        with hk.experimental.name_scope("evoformer"):
            preprocess_1d = common_modules.Linear(c.msa_channel, name="preprocess_1d")(target_feat)
            preprocess_msa = common_modules.Linear(c.msa_channel, name="preprocess_msa")(msa_feat)
            msa_activations = jnp.expand_dims(preprocess_1d, axis=0) + preprocess_msa

            left_single = common_modules.Linear(c.pair_channel, name="left_single")(target_feat)
            right_single = common_modules.Linear(c.pair_channel, name="right_single")(target_feat)
            pair_activations = left_single[:, None] + right_single[None]

            prev_pseudo_beta = modules.pseudo_beta_fn(aatype, prev_pos, None)
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

            mask_2d = seq_mask[:, None] * seq_mask[None, :]
            if c.max_relative_feature:
                offset = residue_index[:, None] - residue_index[None, :]
                rel_pos = jax.nn.one_hot(
                    jnp.clip(offset + c.max_relative_feature, a_min=0, a_max=2 * c.max_relative_feature),
                    2 * c.max_relative_feature + 1,
                )
                pair_activations += common_modules.Linear(c.pair_channel, name="pair_activiations")(rel_pos)

            evoformer_input = {"msa": msa_activations, "pair": pair_activations}
            evoformer_masks = {"msa": msa_mask, "pair": mask_2d}
            evoformer_iteration = modules.EvoformerIteration(
                c.evoformer, gc, is_extra_msa=False, name="evoformer_iteration"
            )

            def evoformer_fn(x):
                act, k = x
                k, sub = k.split()
                out = evoformer_iteration(
                    activations=act,
                    masks=evoformer_masks,
                    is_training=False,
                    safe_key=sub,
                )
                return out, k

            evoformer_stack = layer_stack.layer_stack(c.evoformer_num_block)(evoformer_fn)
            evoformer_output, safe_key = evoformer_stack((evoformer_input, safe_key))
            single_activations = common_modules.Linear(c.seq_channel, name="single_activations")(
                evoformer_output["msa"][0]
            )

        with hk.experimental.name_scope("structure_module"):
            struct_out = folding.generate_affines(
                representations={"single": single_activations, "pair": evoformer_output["pair"]},
                batch={"seq_mask": seq_mask, "aatype": aatype},
                config=cfg.model.heads.structure_module,
                global_config=gc,
                is_training=False,
                safe_key=safe_key,
            )

        atom14 = r3.vecs_to_tensor(struct_out["sc"]["atom_pos"])[-1]
        affine = struct_out["affine"][-1]
        atom37 = jnp.take_along_axis(atom14, residx_atom37_to_atom14[..., None], axis=1)
        atom37 = atom37 * atom37_exists[..., None]
        return (
            single_activations,
            evoformer_output["pair"],
            atom14,
            affine,
            atom37,
            evoformer_output["msa"][0],
            evoformer_output["pair"],
            atom37_exists,
        )

    transformed = hk.transform(fn)
    key = jax.random.PRNGKey(args.seed)
    target_feat_j = jnp.asarray(feats["target_feat"], dtype=jnp.float32)
    msa_feat_j = jnp.asarray(feats["msa_feat"], dtype=jnp.float32)
    seq_mask_j = jnp.asarray(feats["seq_mask"], dtype=jnp.float32)
    msa_mask_j = jnp.asarray(feats["msa_mask"], dtype=jnp.float32)
    residue_index_j = jnp.asarray(feats["residue_index"], dtype=jnp.int32)
    aatype_j = jnp.asarray(feats["aatype"], dtype=jnp.int32)

    n = int(feats["aatype"].shape[0])
    prev_pos = jnp.zeros((n, 37, 3), dtype=jnp.float32)
    prev_msa_first_row = jnp.zeros((n, c.msa_channel), dtype=jnp.float32)
    prev_pair = jnp.zeros((n, n, c.pair_channel), dtype=jnp.float32)

    out_single = None
    out_pair = None
    out_atom14 = None
    out_affine = None
    out_atom37 = None
    out_atom37_mask = None

    for _ in range(args.num_recycle + 1):
        (
            out_single,
            out_pair,
            out_atom14,
            out_affine,
            out_atom37,
            prev_msa_first_row,
            prev_pair,
            out_atom37_mask,
        ) = transformed.apply(
            params,
            key,
            target_feat_j,
            msa_feat_j,
            seq_mask_j,
            msa_mask_j,
            residue_index_j,
            aatype_j,
            prev_pos,
            prev_msa_first_row,
            prev_pair,
        )
        prev_pos = out_atom37

    out_single = np.asarray(out_single, dtype=np.float32)
    out_pair = np.asarray(out_pair, dtype=np.float32)
    out_atom14 = np.asarray(out_atom14, dtype=np.float32)
    out_affine = np.asarray(out_affine, dtype=np.float32)
    out_atom37 = np.asarray(out_atom37, dtype=np.float32)
    out_atom37_mask = np.asarray(out_atom37_mask, dtype=np.float32)

    d, dmean, dstd, dmin, dmax, dout = _ca_distance_metrics(
        out_atom37, out_atom37_mask, residue_constants
    )
    print("Geometry check (consecutive C-alpha distances)")
    print(f"  count: {d.shape[0]}")
    print(f"  mean: {dmean:.6f} A")
    print(f"  std:  {dstd:.6f} A")
    print(f"  min:  {dmin:.6f} A")
    print(f"  max:  {dmax:.6f} A")
    print(f"  outliers (<3.2 or >4.4 A): {np.sum((d < 3.2) | (d > 4.4))} ({dout:.3f})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        target_feat=np.asarray(feats["target_feat"], dtype=np.float32),
        msa_feat=np.asarray(feats["msa_feat"], dtype=np.float32),
        seq_mask=np.asarray(feats["seq_mask"], dtype=np.float32),
        msa_mask=np.asarray(feats["msa_mask"], dtype=np.float32),
        residue_index=np.asarray(feats["residue_index"], dtype=np.int32),
        aatype=np.asarray(feats["aatype"], dtype=np.int32),
        out_single=out_single,
        out_pair=out_pair,
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
    )
    print(f"Saved AF2 core reference run to {args.out}")


if __name__ == "__main__":
    main()
