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


def _cfg(num_intermediate_factor: float):
    cfg = ConfigDict()
    cfg.num_intermediate_factor = num_intermediate_factor

    gc = ConfigDict()
    gc.zero_init = False
    gc.subbatch_size = 256
    return cfg, gc


def _extract(params, key, var):
    return np.asarray(params[f"transition/{key}"][var], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphafold-repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--n", type=int, default=23)
    parser.add_argument("--c", type=int, default=31)
    parser.add_argument("--factor", type=float, default=2.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.model import modules  # pylint: disable=import-outside-toplevel

    cfg, gc = _cfg(args.factor)

    def fn(act, mask):
        mod = modules.Transition(cfg, gc, name="transition")
        return mod(act, mask, is_training=False)

    transformed = hk.transform(fn)

    key = jax.random.PRNGKey(args.seed)
    k_act, k_mask, k_init = jax.random.split(key, 3)

    act = jax.random.normal(k_act, (args.batch, args.n, args.c), dtype=jnp.float32)
    mask = (jax.random.uniform(k_mask, (args.batch, args.n)) > 0.15).astype(jnp.float32)

    params = transformed.init(k_init, act, mask)
    out = transformed.apply(params, None, act, mask)

    arrays = {
        "act": np.asarray(act, dtype=np.float32),
        "mask": np.asarray(mask, dtype=np.float32),
        "out": np.asarray(out, dtype=np.float32),
        "num_intermediate": np.asarray(int(act.shape[-1] * args.factor), dtype=np.int32),
        "input_layer_norm_scale": _extract(params, "input_layer_norm", "scale"),
        "input_layer_norm_offset": _extract(params, "input_layer_norm", "offset"),
        "transition1_weights": _extract(params, "transition1", "weights"),
        "transition1_bias": _extract(params, "transition1", "bias"),
        "transition2_weights": _extract(params, "transition2", "weights"),
        "transition2_bias": _extract(params, "transition2", "bias"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **arrays)

    print(f"Saved: {args.out}")
    print(f"act={arrays['act'].shape} out={arrays['out'].shape}")


if __name__ == "__main__":
    main()
