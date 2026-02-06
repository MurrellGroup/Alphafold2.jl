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


def _cfg(num_head: int, key_dim: int, value_dim: int):
    cfg = ConfigDict()
    cfg.orientation = "per_column"
    cfg.num_head = num_head
    cfg.gating = True
    cfg.key_dim = key_dim
    cfg.value_dim = value_dim

    gc = ConfigDict()
    gc.zero_init = False
    gc.subbatch_size = 256
    return cfg, gc


def _extract(params, key, var):
    return np.asarray(params[f"msa_column_global_attention/{key}"][var], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphafold-repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-seq", type=int, default=7)
    parser.add_argument("--n-res", type=int, default=13)
    parser.add_argument("--c-m", type=int, default=17)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.model import modules  # pylint: disable=import-outside-toplevel

    key_dim = args.heads * args.head_dim
    value_dim = args.heads * args.head_dim
    cfg, gc = _cfg(args.heads, key_dim, value_dim)

    def fn(msa_act, msa_mask):
        mod = modules.MSAColumnGlobalAttention(cfg, gc, name="msa_column_global_attention")
        return mod(msa_act, msa_mask, is_training=False)

    transformed = hk.transform(fn)

    key = jax.random.PRNGKey(args.seed)
    k_msa, k_mask, k_init = jax.random.split(key, 3)

    msa_act = jax.random.normal(k_msa, (args.n_seq, args.n_res, args.c_m), dtype=jnp.float32)
    msa_mask = (jax.random.uniform(k_mask, (args.n_seq, args.n_res)) > 0.2).astype(jnp.float32)

    params = transformed.init(k_init, msa_act, msa_mask)
    out = transformed.apply(params, None, msa_act, msa_mask)

    arrays = {
        "msa_act": np.asarray(msa_act, dtype=np.float32),
        "msa_mask": np.asarray(msa_mask, dtype=np.float32),
        "out": np.asarray(out, dtype=np.float32),
        "num_head": np.asarray(args.heads, dtype=np.int32),
        "head_dim": np.asarray(args.head_dim, dtype=np.int32),
        "query_norm_scale": _extract(params, "query_norm", "scale"),
        "query_norm_offset": _extract(params, "query_norm", "offset"),
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

    print(f"Saved: {args.out}")
    print(f"msa_act={arrays['msa_act'].shape} out={arrays['out'].shape}")


if __name__ == "__main__":
    main()
