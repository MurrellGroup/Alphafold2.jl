#!/usr/bin/env python3
import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict


def _cfg(num_head: int, c_hidden: int, num_point_qk: int, num_point_v: int, c_out: int):
    cfg = ConfigDict()
    cfg.num_head = num_head
    cfg.num_scalar_qk = c_hidden
    cfg.num_scalar_v = c_hidden
    cfg.num_point_qk = num_point_qk
    cfg.num_point_v = num_point_v
    cfg.num_channel = c_out

    gc = ConfigDict()
    gc.zero_init = False
    return cfg, gc


def _extract(params, key, var):
    return np.asarray(params[f"invariant_point_attention/{key}"][var], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphafold-repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=23)
    parser.add_argument("--c-s", type=int, default=31)
    parser.add_argument("--c-z", type=int, default=17)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--c-hidden", type=int, default=7)
    parser.add_argument("--qk-points", type=int, default=3)
    parser.add_argument("--v-points", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.model import folding  # pylint: disable=import-outside-toplevel

    cfg, gc = _cfg(args.heads, args.c_hidden, args.qk_points, args.v_points, args.c_s)

    def fn(s, z, mask):
        ipa = folding.InvariantPointAttention(cfg, gc, name="invariant_point_attention")
        affine = folding.generate_new_affine(mask)
        return ipa(s, z, mask, affine)

    transformed = hk.transform(fn)

    key = jax.random.PRNGKey(args.seed)
    k_s, k_z, k_mask, k_init = jax.random.split(key, 4)

    s = jax.random.normal(k_s, (args.n, args.c_s), dtype=jnp.float32)
    z = jax.random.normal(k_z, (args.n, args.n, args.c_z), dtype=jnp.float32)
    mask = (jax.random.uniform(k_mask, (args.n, 1)) > 0.15).astype(jnp.float32)

    params = transformed.init(k_init, s, z, mask)
    out = transformed.apply(params, None, s, z, mask)

    arrays = {
        "s": np.asarray(s, dtype=np.float32),
        "z": np.asarray(z, dtype=np.float32),
        "mask": np.asarray(mask, dtype=np.float32),
        "out": np.asarray(out, dtype=np.float32),
        "num_head": np.asarray(args.heads, dtype=np.int32),
        "c_hidden": np.asarray(args.c_hidden, dtype=np.int32),
        "num_point_qk": np.asarray(args.qk_points, dtype=np.int32),
        "num_point_v": np.asarray(args.v_points, dtype=np.int32),
        "q_scalar_weights": _extract(params, "q_scalar", "weights"),
        "q_scalar_bias": _extract(params, "q_scalar", "bias"),
        "q_point_local_weights": _extract(params, "q_point_local", "weights"),
        "q_point_local_bias": _extract(params, "q_point_local", "bias"),
        "kv_scalar_weights": _extract(params, "kv_scalar", "weights"),
        "kv_scalar_bias": _extract(params, "kv_scalar", "bias"),
        "kv_point_local_weights": _extract(params, "kv_point_local", "weights"),
        "kv_point_local_bias": _extract(params, "kv_point_local", "bias"),
        "attention_2d_weights": _extract(params, "attention_2d", "weights"),
        "attention_2d_bias": _extract(params, "attention_2d", "bias"),
        "trainable_point_weights": np.asarray(params["invariant_point_attention"]["trainable_point_weights"], dtype=np.float32),
        "output_projection_weights": _extract(params, "output_projection", "weights"),
        "output_projection_bias": _extract(params, "output_projection", "bias"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **arrays)

    print(f"Saved: {args.out}")
    print(f"s={arrays['s'].shape} z={arrays['z'].shape} out={arrays['out'].shape}")


if __name__ == "__main__":
    main()
