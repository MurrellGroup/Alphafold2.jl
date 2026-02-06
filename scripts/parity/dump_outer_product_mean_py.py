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


def _cfg(c_outer: int, chunk_size: int):
    cfg = ConfigDict()
    cfg.num_outer_channel = c_outer
    cfg.chunk_size = chunk_size

    gc = ConfigDict()
    gc.zero_init = False
    return cfg, gc


def _extract(params, key, var):
    return np.asarray(params[f"outer_product_mean/{key}"][var], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphafold-repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-seq", type=int, default=8)
    parser.add_argument("--n-res", type=int, default=13)
    parser.add_argument("--c-m", type=int, default=19)
    parser.add_argument("--c-outer", type=int, default=11)
    parser.add_argument("--c-z", type=int, default=23)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.model import modules  # pylint: disable=import-outside-toplevel

    cfg, gc = _cfg(args.c_outer, args.chunk_size)

    def fn(act, mask):
        mod = modules.OuterProductMean(
            config=cfg,
            global_config=gc,
            num_output_channel=args.c_z,
            name="outer_product_mean",
        )
        return mod(act, mask, is_training=False)

    transformed = hk.transform(fn)

    key = jax.random.PRNGKey(args.seed)
    k_act, k_mask, k_init = jax.random.split(key, 3)

    act = jax.random.normal(k_act, (args.n_seq, args.n_res, args.c_m), dtype=jnp.float32)
    mask = (jax.random.uniform(k_mask, (args.n_seq, args.n_res)) > 0.2).astype(jnp.float32)

    params = transformed.init(k_init, act, mask)
    out = transformed.apply(params, None, act, mask)

    arrays = {
        "act": np.asarray(act, dtype=np.float32),
        "mask": np.asarray(mask, dtype=np.float32),
        "out": np.asarray(out, dtype=np.float32),
        "layer_norm_input_scale": _extract(params, "layer_norm_input", "scale"),
        "layer_norm_input_offset": _extract(params, "layer_norm_input", "offset"),
        "left_projection_weights": _extract(params, "left_projection", "weights"),
        "left_projection_bias": _extract(params, "left_projection", "bias"),
        "right_projection_weights": _extract(params, "right_projection", "weights"),
        "right_projection_bias": _extract(params, "right_projection", "bias"),
        "output_w": np.asarray(params["outer_product_mean"]["output_w"], dtype=np.float32),
        "output_b": np.asarray(params["outer_product_mean"]["output_b"], dtype=np.float32),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **arrays)

    print(f"Saved: {args.out}")
    print(f"act={arrays['act'].shape} out={arrays['out'].shape}")


if __name__ == "__main__":
    main()
