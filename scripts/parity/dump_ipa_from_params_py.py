#!/usr/bin/env python3
import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def _save_npz(path, arrays):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **arrays)


def _extract(params, scope, name):
    return np.asarray(params[scope][name], dtype=np.float32)


def _slice_ipa_params(full_params: hk.Params) -> hk.Params:
    prefix = "alphafold/alphafold_iteration/structure_module/fold_iteration/"
    out = {}
    for scope, vars_dict in full_params.items():
        if not scope.startswith(prefix):
            continue
        if not scope.startswith(prefix + "invariant_point_attention"):
            continue
        new_scope = scope[len(prefix):]
        out[new_scope] = {k: np.asarray(v) for k, v in vars_dict.items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphafold-repo", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.model import config as af_config  # pylint: disable=import-outside-toplevel
    from alphafold.model import folding  # pylint: disable=import-outside-toplevel
    from alphafold.model import utils  # pylint: disable=import-outside-toplevel

    model_cfg = af_config.model_config("model_1")
    sm_cfg = model_cfg.model.heads.structure_module
    global_cfg = model_cfg.model.global_config

    with open(args.params, "rb") as f:
        flat = np.load(f, allow_pickle=False)
        full_params = utils.flat_params_to_haiku(flat)

    params = _slice_ipa_params(full_params)

    c_s = int(params["invariant_point_attention/q_scalar"]["weights"].shape[0])
    c_z = int(params["invariant_point_attention/attention_2d"]["weights"].shape[0])

    def fn(s, z, mask):
        ipa = folding.InvariantPointAttention(sm_cfg, global_cfg, name="invariant_point_attention")
        affine = folding.generate_new_affine(mask)
        return ipa(s, z, mask, affine)

    transformed = hk.transform(fn)

    key = jax.random.PRNGKey(args.seed)
    k_s, k_z, k_mask, k_apply = jax.random.split(key, 4)

    s = jax.random.normal(k_s, (args.n, c_s), dtype=jnp.float32)
    z = jax.random.normal(k_z, (args.n, args.n, c_z), dtype=jnp.float32)
    mask = (jax.random.uniform(k_mask, (args.n, 1)) > 0.15).astype(jnp.float32)

    out = transformed.apply(params, k_apply, s, z, mask)

    arrays = {
        "s": np.asarray(s, dtype=np.float32),
        "z": np.asarray(z, dtype=np.float32),
        "mask": np.asarray(mask, dtype=np.float32),
        "out": np.asarray(out, dtype=np.float32),
        "num_head": np.asarray(int(sm_cfg.num_head), dtype=np.int32),
        "c_hidden": np.asarray(int(sm_cfg.num_scalar_qk), dtype=np.int32),
        "num_point_qk": np.asarray(int(sm_cfg.num_point_qk), dtype=np.int32),
        "num_point_v": np.asarray(int(sm_cfg.num_point_v), dtype=np.int32),
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
    }

    _save_npz(args.out, arrays)
    print(f"Saved real-weight IPA dump: {args.out}")


if __name__ == "__main__":
    main()
