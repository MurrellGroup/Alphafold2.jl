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
    parser.add_argument("--params", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-res", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.common import confidence  # pylint: disable=import-outside-toplevel
    from alphafold.model import config as af_config  # pylint: disable=import-outside-toplevel
    from alphafold.model import modules  # pylint: disable=import-outside-toplevel
    from alphafold.model import utils  # pylint: disable=import-outside-toplevel

    cfg = af_config.model_config("model_1")
    gc = cfg.model.global_config
    gc.deterministic = True
    gc.eval_dropout = False
    hc = cfg.model.heads.predicted_lddt

    with open(args.params, "rb") as f:
        flat = np.load(f, allow_pickle=False)
        full_params = utils.flat_params_to_haiku(flat)
    params = _slice_prefix(full_params, "alphafold/alphafold_iteration/")

    c_s = int(params["predicted_lddt_head/input_layer_norm"]["scale"].shape[0])

    def fn(structure_module_repr):
        mod = modules.PredictedLDDTHead(hc, gc, name="predicted_lddt_head")
        return mod({"structure_module": structure_module_repr}, batch=None, is_training=False)["logits"]

    transformed = hk.transform(fn)
    key = jax.random.PRNGKey(args.seed)
    key_x, key_apply = jax.random.split(key)
    structure_module = jax.random.normal(key_x, (args.n_res, c_s), dtype=jnp.float32)
    logits = transformed.apply(params, key_apply, structure_module)
    plddt = confidence.compute_plddt(np.asarray(logits, dtype=np.float32))

    arrays: Dict[str, np.ndarray] = {
        "structure_module": np.asarray(structure_module, dtype=np.float32),
        "logits": np.asarray(logits, dtype=np.float32),
        "plddt": np.asarray(plddt, dtype=np.float32),
        "c_s": np.asarray(c_s, dtype=np.int32),
        "num_channels": np.asarray(int(hc.num_channels), dtype=np.int32),
        "num_bins": np.asarray(int(hc.num_bins), dtype=np.int32),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **arrays)
    print(f"Saved predicted LDDT dump: {args.out}")


if __name__ == "__main__":
    main()
