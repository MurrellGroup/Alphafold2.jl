#!/usr/bin/env python3
import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def _extract(params, scope, name):
    return np.asarray(params[scope][name], dtype=np.float32)


def _save_npz(path, arrays):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **arrays)


def _slice_structure_params(full_params: hk.Params) -> hk.Params:
    prefix = "alphafold/alphafold_iteration/structure_module/"
    out = {}
    for scope, vars_dict in full_params.items():
        if not scope.startswith(prefix):
            continue
        new_scope = scope[len(prefix):]
        out[new_scope] = {k: np.asarray(v) for k, v in vars_dict.items()}
    return out


def _side_key(base: str, i: int) -> str:
    return base if i == 1 else f"{base}_{i-1}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphafold-repo", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--num-layer-override", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.model import common_modules  # pylint: disable=import-outside-toplevel
    from alphafold.model import config as af_config  # pylint: disable=import-outside-toplevel
    from alphafold.model import folding  # pylint: disable=import-outside-toplevel
    from alphafold.model import prng  # pylint: disable=import-outside-toplevel
    from alphafold.model import r3  # pylint: disable=import-outside-toplevel
    from alphafold.model import utils  # pylint: disable=import-outside-toplevel

    cfg = af_config.model_config("model_1")
    sm_cfg = cfg.model.heads.structure_module
    global_cfg = cfg.model.global_config
    global_cfg.deterministic = True
    global_cfg.eval_dropout = False

    with open(args.params, "rb") as f:
        flat = np.load(f, allow_pickle=False)
        full_params = utils.flat_params_to_haiku(flat)

    params = _slice_structure_params(full_params)

    c_s = int(params["initial_projection"]["weights"].shape[0])
    c_z = int(params["pair_layer_norm"]["scale"].shape[0])

    transition_scopes = []
    if "fold_iteration/transition" in params:
        transition_scopes.append("fold_iteration/transition")
    i = 1
    while f"fold_iteration/transition_{i}" in params:
        transition_scopes.append(f"fold_iteration/transition_{i}")
        i += 1
    num_transition_layers = len(transition_scopes)

    side_resblock_scopes = []
    if "fold_iteration/rigid_sidechain/resblock1" in params:
        side_resblock_scopes.append("fold_iteration/rigid_sidechain/resblock1")
    i = 1
    while f"fold_iteration/rigid_sidechain/resblock1_{i}" in params:
        side_resblock_scopes.append(f"fold_iteration/rigid_sidechain/resblock1_{i}")
        i += 1
    num_sidechain_residual_block = len(side_resblock_scopes)

    num_layer = int(args.num_layer_override) if args.num_layer_override > 0 else int(sm_cfg.num_layer)

    def fn(single, pair, seq_mask, aatype):
        act = common_modules.LayerNorm(
            axis=[-1],
            create_scale=True,
            create_offset=True,
            name="single_layer_norm",
        )(single)

        initial_act = act
        act = common_modules.Linear(sm_cfg.num_channel, name="initial_projection")(act)

        affine = folding.generate_new_affine(seq_mask[:, None])
        activations = {"act": act, "affine": affine.to_tensor()}

        act2d = common_modules.LayerNorm(
            axis=[-1],
            create_scale=True,
            create_offset=True,
            name="pair_layer_norm",
        )(pair)

        fold_iteration = folding.FoldIteration(sm_cfg, global_cfg, name="fold_iteration")

        outputs = []
        safe_keys = prng.SafeKey(hk.next_rng_key()).split(num_layer)
        for sub_key in safe_keys:
            activations, output = fold_iteration(
                activations,
                sequence_mask=seq_mask[:, None],
                update_affine=True,
                is_training=False,
                initial_act=initial_act,
                safe_key=sub_key,
                static_feat_2d=act2d,
                aatype=aatype,
            )
            outputs.append(output)

        out_act = activations["act"]
        out_affine = jnp.stack([o["affine"] for o in outputs], axis=0)
        out_angles = jnp.stack([o["sc"]["angles_sin_cos"] for o in outputs], axis=0)
        out_unnorm = jnp.stack([o["sc"]["unnormalized_angles_sin_cos"] for o in outputs], axis=0)
        out_atom_pos = jnp.stack([r3.vecs_to_tensor(o["sc"]["atom_pos"]) for o in outputs], axis=0)

        return out_act, out_affine, out_angles, out_unnorm, out_atom_pos

    transformed = hk.transform(fn)

    key = jax.random.PRNGKey(args.seed)
    k_single, k_pair, k_mask, k_aatype, k_apply = jax.random.split(key, 5)

    single = jax.random.normal(k_single, (args.n, c_s), dtype=jnp.float32)
    pair = jax.random.normal(k_pair, (args.n, args.n, c_z), dtype=jnp.float32)
    seq_mask = (jax.random.uniform(k_mask, (args.n,)) > 0.2).astype(jnp.float32)
    aatype = jax.random.randint(k_aatype, (args.n,), minval=0, maxval=20, dtype=jnp.int32)

    out_act, out_affine, out_angles, out_unnorm, out_atom_pos = transformed.apply(
        params,
        k_apply,
        single,
        pair,
        seq_mask,
        aatype,
    )

    arrays = {
        "single": np.asarray(single, dtype=np.float32),
        "pair": np.asarray(pair, dtype=np.float32),
        "seq_mask": np.asarray(seq_mask, dtype=np.float32),
        "aatype": np.asarray(aatype, dtype=np.int32),
        "out_act": np.asarray(out_act, dtype=np.float32),
        "out_affine": np.asarray(out_affine, dtype=np.float32),
        "out_angles": np.asarray(out_angles, dtype=np.float32),
        "out_unnormalized_angles": np.asarray(out_unnorm, dtype=np.float32),
        "out_atom_pos": np.asarray(out_atom_pos, dtype=np.float32),
        "num_head": np.asarray(int(sm_cfg.num_head), dtype=np.int32),
        "c_hidden": np.asarray(int(sm_cfg.num_scalar_qk), dtype=np.int32),
        "num_point_qk": np.asarray(int(sm_cfg.num_point_qk), dtype=np.int32),
        "num_point_v": np.asarray(int(sm_cfg.num_point_v), dtype=np.int32),
        "num_transition_layers": np.asarray(num_transition_layers, dtype=np.int32),
        "num_layer": np.asarray(num_layer, dtype=np.int32),
        "position_scale": np.asarray(float(sm_cfg.position_scale), dtype=np.float32),
        "num_residual_block": np.asarray(num_sidechain_residual_block, dtype=np.int32),
        "single_layer_norm_scale": _extract(params, "single_layer_norm", "scale"),
        "single_layer_norm_offset": _extract(params, "single_layer_norm", "offset"),
        "initial_projection_weights": _extract(params, "initial_projection", "weights"),
        "initial_projection_bias": _extract(params, "initial_projection", "bias"),
        "pair_layer_norm_scale": _extract(params, "pair_layer_norm", "scale"),
        "pair_layer_norm_offset": _extract(params, "pair_layer_norm", "offset"),
        "q_scalar_weights": _extract(params, "fold_iteration/invariant_point_attention/q_scalar", "weights"),
        "q_scalar_bias": _extract(params, "fold_iteration/invariant_point_attention/q_scalar", "bias"),
        "q_point_local_weights": _extract(params, "fold_iteration/invariant_point_attention/q_point_local", "weights"),
        "q_point_local_bias": _extract(params, "fold_iteration/invariant_point_attention/q_point_local", "bias"),
        "kv_scalar_weights": _extract(params, "fold_iteration/invariant_point_attention/kv_scalar", "weights"),
        "kv_scalar_bias": _extract(params, "fold_iteration/invariant_point_attention/kv_scalar", "bias"),
        "kv_point_local_weights": _extract(params, "fold_iteration/invariant_point_attention/kv_point_local", "weights"),
        "kv_point_local_bias": _extract(params, "fold_iteration/invariant_point_attention/kv_point_local", "bias"),
        "attention_2d_weights": _extract(params, "fold_iteration/invariant_point_attention/attention_2d", "weights"),
        "attention_2d_bias": _extract(params, "fold_iteration/invariant_point_attention/attention_2d", "bias"),
        "trainable_point_weights": _extract(params, "fold_iteration/invariant_point_attention", "trainable_point_weights"),
        "output_projection_weights": _extract(params, "fold_iteration/invariant_point_attention/output_projection", "weights"),
        "output_projection_bias": _extract(params, "fold_iteration/invariant_point_attention/output_projection", "bias"),
        "attention_layer_norm_scale": _extract(params, "fold_iteration/attention_layer_norm", "scale"),
        "attention_layer_norm_offset": _extract(params, "fold_iteration/attention_layer_norm", "offset"),
        "transition_layer_norm_scale": _extract(params, "fold_iteration/transition_layer_norm", "scale"),
        "transition_layer_norm_offset": _extract(params, "fold_iteration/transition_layer_norm", "offset"),
        "affine_update_weights": _extract(params, "fold_iteration/affine_update", "weights"),
        "affine_update_bias": _extract(params, "fold_iteration/affine_update", "bias"),
    }

    for scope in transition_scopes:
        key_name = scope.split("/")[-1]
        arrays[f"{key_name}_weights"] = _extract(params, scope, "weights")
        arrays[f"{key_name}_bias"] = _extract(params, scope, "bias")

    for i in range(1, 3):
        p = _side_key("input_projection", i)
        arrays[f"{p}_weights"] = _extract(params, f"fold_iteration/rigid_sidechain/{p}", "weights")
        arrays[f"{p}_bias"] = _extract(params, f"fold_iteration/rigid_sidechain/{p}", "bias")

    for i in range(1, num_sidechain_residual_block + 1):
        p1 = _side_key("resblock1", i)
        p2 = _side_key("resblock2", i)
        arrays[f"{p1}_weights"] = _extract(params, f"fold_iteration/rigid_sidechain/{p1}", "weights")
        arrays[f"{p1}_bias"] = _extract(params, f"fold_iteration/rigid_sidechain/{p1}", "bias")
        arrays[f"{p2}_weights"] = _extract(params, f"fold_iteration/rigid_sidechain/{p2}", "weights")
        arrays[f"{p2}_bias"] = _extract(params, f"fold_iteration/rigid_sidechain/{p2}", "bias")

    arrays["unnormalized_angles_weights"] = _extract(
        params,
        "fold_iteration/rigid_sidechain/unnormalized_angles",
        "weights",
    )
    arrays["unnormalized_angles_bias"] = _extract(
        params,
        "fold_iteration/rigid_sidechain/unnormalized_angles",
        "bias",
    )

    _save_npz(args.out, arrays)
    print(f"Saved real-weight structure-module-core dump: {args.out}")


if __name__ == "__main__":
    main()
