#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def _slice_prefix(params: hk.Params, prefix: str) -> hk.Params:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for scope, vars_dict in params.items():
        if scope.startswith(prefix):
            out[scope[len(prefix):]] = {k: np.asarray(v) for k, v in vars_dict.items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphafold-repo", required=True)
    parser.add_argument("--params", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-res", type=int, default=17)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.common import confidence  # pylint: disable=import-outside-toplevel
    from alphafold.model import config as af_config  # pylint: disable=import-outside-toplevel
    from alphafold.model import modules  # pylint: disable=import-outside-toplevel
    from alphafold.model import utils  # pylint: disable=import-outside-toplevel

    cfg = af_config.model_config("model_1_ptm")
    gc = cfg.model.global_config
    gc.deterministic = True
    gc.eval_dropout = False
    hc = cfg.model.heads.predicted_aligned_error

    def fn(pair_repr):
        mod = modules.PredictedAlignedErrorHead(hc, gc, name="predicted_aligned_error_head")
        return mod({"pair": pair_repr}, batch=None, is_training=False)

    transformed = hk.transform(fn)
    key = jax.random.PRNGKey(args.seed)
    key_pair, key_init, key_apply = jax.random.split(key, 3)

    if args.params:
        with open(args.params, "rb") as f:
            flat = np.load(f, allow_pickle=False)
            full_params = utils.flat_params_to_haiku(flat)
        params = _slice_prefix(full_params, "alphafold/alphafold_iteration/")
        c_z = int(params["predicted_aligned_error_head/logits"]["weights"].shape[0])
    else:
        params = None
        c_z = int(args.c_z)

    pair_repr = jax.random.normal(key_pair, (args.n_res, args.n_res, c_z), dtype=jnp.float32)
    if params is None:
        params = transformed.init(key_init, pair_repr)
    out = transformed.apply(params, key_apply, pair_repr)
    logits = np.asarray(out["logits"], dtype=np.float32)
    breaks = np.asarray(out["breaks"], dtype=np.float32)

    pae = confidence.compute_predicted_aligned_error(logits=logits, breaks=breaks)
    ptm = confidence.predicted_tm_score(logits=logits, breaks=breaks)

    scope = None
    for s in params:
        if s.endswith("predicted_aligned_error_head/logits"):
            scope = s
            break
    if scope is None:
        raise ValueError("Failed to locate predicted_aligned_error_head/logits params scope.")

    arrays: Dict[str, np.ndarray] = {
        "pair_repr": np.asarray(pair_repr, dtype=np.float32),
        "logits": logits,
        "breaks": breaks,
        "logits_weights": np.asarray(params[scope]["weights"], dtype=np.float32),
        "logits_bias": np.asarray(params[scope]["bias"], dtype=np.float32),
        "num_bins": np.asarray(int(hc.num_bins), dtype=np.int32),
        "max_error_bin": np.asarray(float(hc.max_error_bin), dtype=np.float32),
        "aligned_confidence_probs": np.asarray(pae["aligned_confidence_probs"], dtype=np.float32),
        "predicted_aligned_error": np.asarray(pae["predicted_aligned_error"], dtype=np.float32),
        "max_predicted_aligned_error": np.asarray(float(pae["max_predicted_aligned_error"]), dtype=np.float32),
        "predicted_tm_score": np.asarray(float(ptm), dtype=np.float32),
        "used_params": np.asarray(1 if args.params else 0, dtype=np.int32),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **arrays)
    print(f"Saved predicted aligned error dump: {args.out}")


if __name__ == "__main__":
    main()
