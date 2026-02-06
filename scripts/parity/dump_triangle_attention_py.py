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


def _config(column: bool, num_head: int, key_dim: int, value_dim: int):
    cfg = ConfigDict()
    cfg.orientation = "per_column" if column else "per_row"
    cfg.num_head = num_head
    cfg.gating = True
    cfg.key_dim = key_dim
    cfg.value_dim = value_dim

    gc = ConfigDict()
    gc.zero_init = False
    gc.subbatch_size = 256
    return cfg, gc


def _extract(params, key, var):
    return np.asarray(params[f"triangle_attention/{key}"][var], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphafold-repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=29)
    parser.add_argument("--cz", type=int, default=47)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--head-dim", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--column", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.model import modules  # pylint: disable=import-outside-toplevel

    key_dim = args.heads * args.head_dim
    value_dim = args.heads * args.head_dim
    cfg, gc = _config(args.column, args.heads, key_dim, value_dim)

    def fn(pair_act, pair_mask):
        mod = modules.TriangleAttention(cfg, gc, name="triangle_attention")
        return mod(pair_act, pair_mask, is_training=False)

    transformed = hk.transform(fn)

    key = jax.random.PRNGKey(args.seed)
    k_act, k_mask, k_init = jax.random.split(key, 3)

    pair_act = jax.random.normal(k_act, (args.n, args.n, args.cz), dtype=jnp.float32)
    pair_mask = (jax.random.uniform(k_mask, (args.n, args.n)) > 0.1).astype(jnp.float32)

    params = transformed.init(k_init, pair_act, pair_mask)
    out = transformed.apply(params, None, pair_act, pair_mask)

    arrays = {
        "pair_act": np.asarray(pair_act, dtype=np.float32),
        "pair_mask": np.asarray(pair_mask, dtype=np.float32),
        "out": np.asarray(out, dtype=np.float32),
        "column": np.asarray(1 if args.column else 0, dtype=np.int32),
        "num_head": np.asarray(args.heads, dtype=np.int32),
        "head_dim": np.asarray(args.head_dim, dtype=np.int32),
        "query_norm_scale": _extract(params, "query_norm", "scale"),
        "query_norm_offset": _extract(params, "query_norm", "offset"),
        "feat_2d_weights": np.asarray(params["triangle_attention"]["feat_2d_weights"], dtype=np.float32),
        "attention_query_w": _extract(params, "attention", "query_w"),
        "attention_key_w": _extract(params, "attention", "key_w"),
        "attention_value_w": _extract(params, "attention", "value_w"),
        "attention_gating_w": _extract(params, "attention", "gating_w"),
        "attention_gating_b": _extract(params, "attention", "gating_b"),
        "attention_output_w": _extract(params, "attention", "output_w"),
        "attention_output_b": _extract(params, "attention", "output_b"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **arrays)

    orientation = "per_column" if args.column else "per_row"
    print(f"Saved: {args.out}")
    print(f"orientation={orientation} pair_act={arrays['pair_act'].shape} out={arrays['out'].shape}")


if __name__ == "__main__":
    main()
