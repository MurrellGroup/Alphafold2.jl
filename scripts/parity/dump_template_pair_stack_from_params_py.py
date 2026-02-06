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


PREFIX_SCOPE = "alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/"
FULL_PREFIX = PREFIX_SCOPE + "template_pair_stack/__layer_stack_no_state"


def _save_npz(path: str, arrays: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **arrays)


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
    parser.add_argument("--n-res", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.model import config as af_config  # pylint: disable=import-outside-toplevel
    from alphafold.model import modules  # pylint: disable=import-outside-toplevel
    from alphafold.model import utils  # pylint: disable=import-outside-toplevel

    model_cfg = af_config.model_config("model_1")
    tcfg = model_cfg.model.embeddings_and_evoformer.template.template_pair_stack
    global_cfg = model_cfg.model.global_config
    global_cfg.deterministic = True
    global_cfg.eval_dropout = False

    with open(args.params, "rb") as f:
        flat = np.load(f, allow_pickle=False)
        full_params = utils.flat_params_to_haiku(flat)

    params = _slice_prefix(full_params, PREFIX_SCOPE)

    c_t = int(params["template_pair_stack/__layer_stack_no_state/pair_transition/transition2"]["bias"].shape[-1])

    def fn(pair, pair_mask):
        mod = modules.TemplatePairStack(tcfg, global_cfg, name="template_pair_stack")
        return mod(pair, pair_mask, is_training=False, safe_key=None)

    transformed = hk.transform(fn)

    key = jax.random.PRNGKey(args.seed)
    k_pair, k_mask, k_apply = jax.random.split(key, 3)
    pair = jax.random.normal(k_pair, (args.n_res, args.n_res, c_t), dtype=jnp.float32)
    pair_mask = (jax.random.uniform(k_mask, (args.n_res, args.n_res)) > 0.2).astype(jnp.float32)

    out = transformed.apply(params, k_apply, pair, pair_mask)

    arrays: Dict[str, np.ndarray] = {
        "pair": np.asarray(pair, dtype=np.float32),
        "pair_mask": np.asarray(pair_mask, dtype=np.float32),
        "out": np.asarray(out, dtype=np.float32),
        "num_block": np.asarray(int(params["template_pair_stack/__layer_stack_no_state/pair_transition/transition2"]["bias"].shape[0]), dtype=np.int32),
        "num_head_pair": np.asarray(int(params["template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node"]["feat_2d_weights"].shape[-1]), dtype=np.int32),
        "pair_head_dim": np.asarray(int(params["template_pair_stack/__layer_stack_no_state/triangle_attention_starting_node/attention"]["query_w"].shape[-1]), dtype=np.int32),
        "c_tri_mul": np.asarray(int(params["template_pair_stack/__layer_stack_no_state/triangle_multiplication_outgoing/left_projection"]["bias"].shape[-1]), dtype=np.int32),
        "pair_transition_factor": np.asarray(
            float(
                params["template_pair_stack/__layer_stack_no_state/pair_transition/transition1"]["bias"].shape[-1]
                / c_t
            ),
            dtype=np.float32,
        ),
    }

    # Save raw template pair stack weights with full checkpoint-style keys.
    for scope, vars_dict in full_params.items():
        if not scope.startswith(FULL_PREFIX):
            continue
        for name, arr in vars_dict.items():
            arrays[f"{scope}/{name}"] = np.asarray(arr, dtype=np.float32)

    _save_npz(args.out, arrays)

    print(f"Saved template pair stack dump: {args.out}")
    print(f"pair={arrays['pair'].shape} out={arrays['out'].shape}")


if __name__ == "__main__":
    main()
