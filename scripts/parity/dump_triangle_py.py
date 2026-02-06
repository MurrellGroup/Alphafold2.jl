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


def _config(outgoing: bool):
    cfg = ConfigDict()
    cfg.num_intermediate_channel = 17
    cfg.equation = "ikc,jkc->ijc" if outgoing else "kjc,kic->ijc"
    cfg.fuse_projection_weights = False

    gc = ConfigDict()
    gc.zero_init = False
    gc.subbatch_size = 4
    return cfg, gc


def _extract(params, key, var):
    return np.asarray(params[f"triangle_multiplication/{key}"][var], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphafold-repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=31)
    parser.add_argument("--cz", type=int, default=47)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--incoming", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.model import modules  # pylint: disable=import-outside-toplevel

    outgoing = not args.incoming
    cfg, gc = _config(outgoing)

    def fn(act, mask):
        mod = modules.TriangleMultiplication(cfg, gc, name="triangle_multiplication")
        return mod(act, mask, is_training=False)

    transformed = hk.transform(fn)

    key = jax.random.PRNGKey(args.seed)
    key_act, key_mask, key_init = jax.random.split(key, 3)

    act = jax.random.normal(key_act, (args.n, args.n, args.cz), dtype=jnp.float32)
    mask = (jax.random.uniform(key_mask, (args.n, args.n)) > 0.15).astype(jnp.float32)

    params = transformed.init(key_init, act, mask)
    out = transformed.apply(params, None, act, mask)

    arrays = {
        "act": np.asarray(act, dtype=np.float32),
        "mask": np.asarray(mask, dtype=np.float32),
        "out": np.asarray(out, dtype=np.float32),
        "outgoing": np.asarray(1 if outgoing else 0, dtype=np.int32),
        "layer_norm_input_scale": _extract(params, "layer_norm_input", "scale"),
        "layer_norm_input_offset": _extract(params, "layer_norm_input", "offset"),
        "left_projection_weights": _extract(params, "left_projection", "weights"),
        "left_projection_bias": _extract(params, "left_projection", "bias"),
        "right_projection_weights": _extract(params, "right_projection", "weights"),
        "right_projection_bias": _extract(params, "right_projection", "bias"),
        "left_gate_weights": _extract(params, "left_gate", "weights"),
        "left_gate_bias": _extract(params, "left_gate", "bias"),
        "right_gate_weights": _extract(params, "right_gate", "weights"),
        "right_gate_bias": _extract(params, "right_gate", "bias"),
        "center_layer_norm_scale": _extract(params, "center_layer_norm", "scale"),
        "center_layer_norm_offset": _extract(params, "center_layer_norm", "offset"),
        "output_projection_weights": _extract(params, "output_projection", "weights"),
        "output_projection_bias": _extract(params, "output_projection", "bias"),
        "gating_linear_weights": _extract(params, "gating_linear", "weights"),
        "gating_linear_bias": _extract(params, "gating_linear", "bias"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **arrays)

    print(f"Saved: {args.out}")
    print(f"shape act={arrays['act'].shape} out={arrays['out'].shape} outgoing={outgoing}")


if __name__ == "__main__":
    main()
