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


def _slice_fold_params(full_params: hk.Params) -> hk.Params:
    prefix = "alphafold/alphafold_iteration/structure_module/fold_iteration/"
    out = {}
    for scope, vars_dict in full_params.items():
        if not scope.startswith(prefix):
            continue
        new_scope = scope[len(prefix):]
        out[new_scope] = {k: np.asarray(v) for k, v in vars_dict.items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphafold-repo", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.model import common_modules  # pylint: disable=import-outside-toplevel
    from alphafold.model import config as af_config  # pylint: disable=import-outside-toplevel
    from alphafold.model import folding  # pylint: disable=import-outside-toplevel
    from alphafold.model import utils  # pylint: disable=import-outside-toplevel

    cfg = af_config.model_config("model_1")
    sm_cfg = cfg.model.heads.structure_module
    global_cfg = cfg.model.global_config

    with open(args.params, "rb") as f:
        flat = np.load(f, allow_pickle=False)
        full_params = utils.flat_params_to_haiku(flat)

    params = _slice_fold_params(full_params)

    c_s = int(params["transition"]["weights"].shape[0])
    c_z = int(params["invariant_point_attention/attention_2d"]["weights"].shape[0])

    # Determine transition depth from checkpoint keys.
    transition_scopes = []
    if "transition" in params:
        transition_scopes.append("transition")
    i = 1
    while f"transition_{i}" in params:
        transition_scopes.append(f"transition_{i}")
        i += 1
    num_transition_layers = len(transition_scopes)

    def fn(act, z, mask):
        affine = folding.generate_new_affine(mask)

        ipa = folding.InvariantPointAttention(sm_cfg, global_cfg, name="invariant_point_attention")
        attn = ipa(act, z, mask, affine)

        act = act + attn
        act = common_modules.LayerNorm(
            axis=[-1],
            create_scale=True,
            create_offset=True,
            name="attention_layer_norm",
        )(act)

        input_act = act
        final_init = "zeros" if global_cfg.zero_init else "linear"
        for i in range(sm_cfg.num_layer_in_transition):
            init = "relu" if i < sm_cfg.num_layer_in_transition - 1 else final_init
            act = common_modules.Linear(sm_cfg.num_channel, initializer=init, name="transition")(act)
            if i < sm_cfg.num_layer_in_transition - 1:
                act = jax.nn.relu(act)

        act = act + input_act
        act = common_modules.LayerNorm(
            axis=[-1],
            create_scale=True,
            create_offset=True,
            name="transition_layer_norm",
        )(act)

        affine_update = common_modules.Linear(6, initializer=final_init, name="affine_update")(act)
        affine = affine.pre_compose(affine_update)

        return act, affine_update, affine.to_tensor()

    transformed = hk.transform(fn)

    key = jax.random.PRNGKey(args.seed)
    k_act, k_z, k_mask, k_apply = jax.random.split(key, 4)

    act = jax.random.normal(k_act, (args.n, c_s), dtype=jnp.float32)
    z = jax.random.normal(k_z, (args.n, args.n, c_z), dtype=jnp.float32)
    mask = (jax.random.uniform(k_mask, (args.n, 1)) > 0.2).astype(jnp.float32)

    out_act, out_affine_update, out_affine_tensor = transformed.apply(params, k_apply, act, z, mask)

    arrays = {
        "act": np.asarray(act, dtype=np.float32),
        "z": np.asarray(z, dtype=np.float32),
        "mask": np.asarray(mask, dtype=np.float32),
        "out_act": np.asarray(out_act, dtype=np.float32),
        "out_affine_update": np.asarray(out_affine_update, dtype=np.float32),
        "out_affine_tensor": np.asarray(out_affine_tensor, dtype=np.float32),
        "num_head": np.asarray(int(sm_cfg.num_head), dtype=np.int32),
        "c_hidden": np.asarray(int(sm_cfg.num_scalar_qk), dtype=np.int32),
        "num_point_qk": np.asarray(int(sm_cfg.num_point_qk), dtype=np.int32),
        "num_point_v": np.asarray(int(sm_cfg.num_point_v), dtype=np.int32),
        "num_transition_layers": np.asarray(num_transition_layers, dtype=np.int32),
        "q_scalar_weights": _extract(params, "invariant_point_attention/q_scalar", "weights"),
        "q_scalar_bias": _extract(params, "invariant_point_attention/q_scalar", "bias"),
        "q_point_local_weights": _extract(params, "invariant_point_attention/q_point_local", "weights"),
        "q_point_local_bias": _extract(params, "invariant_point_attention/q_point_local", "bias"),
        "kv_scalar_weights": _extract(params, "invariant_point_attention/kv_scalar", "weights"),
        "kv_scalar_bias": _extract(params, "invariant_point_attention/kv_scalar", "bias"),
        "kv_point_local_weights": _extract(params, "invariant_point_attention/kv_point_local", "weights"),
        "kv_point_local_bias": _extract(params, "invariant_point_attention/kv_point_local", "bias"),
        "attention_2d_weights": _extract(params, "invariant_point_attention/attention_2d", "weights"),
        "attention_2d_bias": _extract(params, "invariant_point_attention/attention_2d", "bias"),
        "trainable_point_weights": _extract(params, "invariant_point_attention", "trainable_point_weights"),
        "output_projection_weights": _extract(params, "invariant_point_attention/output_projection", "weights"),
        "output_projection_bias": _extract(params, "invariant_point_attention/output_projection", "bias"),
        "attention_layer_norm_scale": _extract(params, "attention_layer_norm", "scale"),
        "attention_layer_norm_offset": _extract(params, "attention_layer_norm", "offset"),
        "transition_layer_norm_scale": _extract(params, "transition_layer_norm", "scale"),
        "transition_layer_norm_offset": _extract(params, "transition_layer_norm", "offset"),
        "affine_update_weights": _extract(params, "affine_update", "weights"),
        "affine_update_bias": _extract(params, "affine_update", "bias"),
    }

    for scope in transition_scopes:
        arrays[f"{scope}_weights"] = _extract(params, scope, "weights")
        arrays[f"{scope}_bias"] = _extract(params, scope, "bias")

    _save_npz(args.out, arrays)
    print(f"Saved real-weight fold-iteration-core dump: {args.out}")


if __name__ == "__main__":
    main()
