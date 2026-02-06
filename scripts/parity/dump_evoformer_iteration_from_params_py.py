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


def _save_npz(path: str, arrays: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **arrays)


def _extract(params, scope, name):
    return np.asarray(params[scope][name], dtype=np.float32)


def _slice_evo_params(full_params: hk.Params, block_idx: int) -> hk.Params:
    prefix = "alphafold/alphafold_iteration/evoformer/"
    out: Dict[str, Dict[str, np.ndarray]] = {}

    for scope, vars_dict in full_params.items():
        if not scope.startswith(prefix):
            continue

        new_scope = scope[len(prefix):]
        out[new_scope] = {}
        for name, arr in vars_dict.items():
            x = np.asarray(arr)
            if x.ndim > 0 and x.shape[0] == 48:
                x = x[block_idx]
            out[new_scope][name] = x

    return out


def _save_transition(params, base, out_path):
    arrays = {
        "input_layer_norm_scale": _extract(params, f"{base}/input_layer_norm", "scale"),
        "input_layer_norm_offset": _extract(params, f"{base}/input_layer_norm", "offset"),
        "transition1_weights": _extract(params, f"{base}/transition1", "weights"),
        "transition1_bias": _extract(params, f"{base}/transition1", "bias"),
        "transition2_weights": _extract(params, f"{base}/transition2", "weights"),
        "transition2_bias": _extract(params, f"{base}/transition2", "bias"),
    }
    _save_npz(out_path, arrays)


def _save_triangle_mult(params, base, out_path):
    arrays = {
        "layer_norm_input_scale": _extract(params, f"{base}/layer_norm_input", "scale"),
        "layer_norm_input_offset": _extract(params, f"{base}/layer_norm_input", "offset"),
        "left_projection_weights": _extract(params, f"{base}/left_projection", "weights"),
        "left_projection_bias": _extract(params, f"{base}/left_projection", "bias"),
        "right_projection_weights": _extract(params, f"{base}/right_projection", "weights"),
        "right_projection_bias": _extract(params, f"{base}/right_projection", "bias"),
        "left_gate_weights": _extract(params, f"{base}/left_gate", "weights"),
        "left_gate_bias": _extract(params, f"{base}/left_gate", "bias"),
        "right_gate_weights": _extract(params, f"{base}/right_gate", "weights"),
        "right_gate_bias": _extract(params, f"{base}/right_gate", "bias"),
        "center_layer_norm_scale": _extract(params, f"{base}/center_layer_norm", "scale"),
        "center_layer_norm_offset": _extract(params, f"{base}/center_layer_norm", "offset"),
        "output_projection_weights": _extract(params, f"{base}/output_projection", "weights"),
        "output_projection_bias": _extract(params, f"{base}/output_projection", "bias"),
        "gating_linear_weights": _extract(params, f"{base}/gating_linear", "weights"),
        "gating_linear_bias": _extract(params, f"{base}/gating_linear", "bias"),
    }
    _save_npz(out_path, arrays)


def _save_attention(params, base, out_path):
    arrays = {
        "query_norm_scale": _extract(params, f"{base}/query_norm", "scale"),
        "query_norm_offset": _extract(params, f"{base}/query_norm", "offset"),
        "attention_query_w": _extract(params, f"{base}/attention", "query_w"),
        "attention_key_w": _extract(params, f"{base}/attention", "key_w"),
        "attention_value_w": _extract(params, f"{base}/attention", "value_w"),
        "attention_gating_w": _extract(params, f"{base}/attention", "gating_w"),
        "attention_gating_b": _extract(params, f"{base}/attention", "gating_b"),
        "attention_output_w": _extract(params, f"{base}/attention", "output_w"),
        "attention_output_b": _extract(params, f"{base}/attention", "output_b"),
    }
    _save_npz(out_path, arrays)


def _save_triangle_attention(params, base, out_path):
    arrays = {
        "query_norm_scale": _extract(params, f"{base}/query_norm", "scale"),
        "query_norm_offset": _extract(params, f"{base}/query_norm", "offset"),
        "feat_2d_weights": _extract(params, base, "feat_2d_weights"),
        "attention_query_w": _extract(params, f"{base}/attention", "query_w"),
        "attention_key_w": _extract(params, f"{base}/attention", "key_w"),
        "attention_value_w": _extract(params, f"{base}/attention", "value_w"),
        "attention_gating_w": _extract(params, f"{base}/attention", "gating_w"),
        "attention_gating_b": _extract(params, f"{base}/attention", "gating_b"),
        "attention_output_w": _extract(params, f"{base}/attention", "output_w"),
        "attention_output_b": _extract(params, f"{base}/attention", "output_b"),
    }
    _save_npz(out_path, arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphafold-repo", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--n-seq", type=int, default=4)
    parser.add_argument("--n-res", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.model import config as af_config  # pylint: disable=import-outside-toplevel
    from alphafold.model import modules  # pylint: disable=import-outside-toplevel
    from alphafold.model import utils  # pylint: disable=import-outside-toplevel

    model_cfg = af_config.model_config("model_1")
    evo_cfg = model_cfg.model.embeddings_and_evoformer.evoformer
    global_cfg = model_cfg.model.global_config

    # Disable dropout deterministically.
    global_cfg.deterministic = True
    global_cfg.eval_dropout = False

    with open(args.params, "rb") as f:
        flat = np.load(f, allow_pickle=False)
        full_params = utils.flat_params_to_haiku(flat)

    params = _slice_evo_params(full_params, args.block)

    c_m = int(params["evoformer_iteration/msa_transition/transition2"]["bias"].shape[0])
    c_z = int(params["evoformer_iteration/pair_transition/transition2"]["bias"].shape[0])

    def fn(msa, pair, msa_mask, pair_mask):
        evo = modules.EvoformerIteration(evo_cfg, global_cfg, is_extra_msa=False, name="evoformer_iteration")
        activations = {"msa": msa, "pair": pair}
        masks = {"msa": msa_mask, "pair": pair_mask}
        out = evo(activations, masks, is_training=False, safe_key=None)
        return out["msa"], out["pair"]

    transformed = hk.transform(fn)

    key = jax.random.PRNGKey(args.seed)
    k_msa, k_pair, k_mmask, k_pmask, k_apply = jax.random.split(key, 5)

    msa = jax.random.normal(k_msa, (args.n_seq, args.n_res, c_m), dtype=jnp.float32)
    pair = jax.random.normal(k_pair, (args.n_res, args.n_res, c_z), dtype=jnp.float32)
    msa_mask = (jax.random.uniform(k_mmask, (args.n_seq, args.n_res)) > 0.2).astype(jnp.float32)
    pair_mask = (jax.random.uniform(k_pmask, (args.n_res, args.n_res)) > 0.2).astype(jnp.float32)

    out_msa, out_pair = transformed.apply(params, k_apply, msa, pair, msa_mask, pair_mask)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    io_arrays = {
        "msa": np.asarray(msa, dtype=np.float32),
        "pair": np.asarray(pair, dtype=np.float32),
        "msa_mask": np.asarray(msa_mask, dtype=np.float32),
        "pair_mask": np.asarray(pair_mask, dtype=np.float32),
        "out_msa": np.asarray(out_msa, dtype=np.float32),
        "out_pair": np.asarray(out_pair, dtype=np.float32),
        "num_head_msa": np.asarray(int(evo_cfg.msa_row_attention_with_pair_bias.num_head), dtype=np.int32),
        "msa_head_dim": np.asarray(
            int(params["evoformer_iteration/msa_row_attention_with_pair_bias/attention"]["query_w"].shape[-1]),
            dtype=np.int32,
        ),
        "num_head_pair": np.asarray(int(evo_cfg.triangle_attention_starting_node.num_head), dtype=np.int32),
        "pair_head_dim": np.asarray(
            int(params["evoformer_iteration/triangle_attention_starting_node/attention"]["query_w"].shape[-1]),
            dtype=np.int32,
        ),
        "c_outer": np.asarray(int(evo_cfg.outer_product_mean.num_outer_channel), dtype=np.int32),
        "c_tri_mul": np.asarray(int(evo_cfg.triangle_multiplication_outgoing.num_intermediate_channel), dtype=np.int32),
        "outer_first": np.asarray(1 if bool(evo_cfg.outer_product_mean.first) else 0, dtype=np.int32),
        "block": np.asarray(args.block, dtype=np.int32),
    }
    _save_npz(os.path.join(out_dir, "evo_io.npz"), io_arrays)

    base = "evoformer_iteration"

    _save_npz(
        os.path.join(out_dir, "outer_product_mean.npz"),
        {
            "layer_norm_input_scale": _extract(params, f"{base}/outer_product_mean/layer_norm_input", "scale"),
            "layer_norm_input_offset": _extract(params, f"{base}/outer_product_mean/layer_norm_input", "offset"),
            "left_projection_weights": _extract(params, f"{base}/outer_product_mean/left_projection", "weights"),
            "left_projection_bias": _extract(params, f"{base}/outer_product_mean/left_projection", "bias"),
            "right_projection_weights": _extract(params, f"{base}/outer_product_mean/right_projection", "weights"),
            "right_projection_bias": _extract(params, f"{base}/outer_product_mean/right_projection", "bias"),
            "output_w": _extract(params, f"{base}/outer_product_mean", "output_w"),
            "output_b": _extract(params, f"{base}/outer_product_mean", "output_b"),
        },
    )

    _save_npz(
        os.path.join(out_dir, "msa_row_attention.npz"),
        {
            "query_norm_scale": _extract(params, f"{base}/msa_row_attention_with_pair_bias/query_norm", "scale"),
            "query_norm_offset": _extract(params, f"{base}/msa_row_attention_with_pair_bias/query_norm", "offset"),
            "feat_2d_norm_scale": _extract(params, f"{base}/msa_row_attention_with_pair_bias/feat_2d_norm", "scale"),
            "feat_2d_norm_offset": _extract(params, f"{base}/msa_row_attention_with_pair_bias/feat_2d_norm", "offset"),
            "feat_2d_weights": _extract(params, f"{base}/msa_row_attention_with_pair_bias", "feat_2d_weights"),
            "attention_query_w": _extract(params, f"{base}/msa_row_attention_with_pair_bias/attention", "query_w"),
            "attention_key_w": _extract(params, f"{base}/msa_row_attention_with_pair_bias/attention", "key_w"),
            "attention_value_w": _extract(params, f"{base}/msa_row_attention_with_pair_bias/attention", "value_w"),
            "attention_gating_w": _extract(params, f"{base}/msa_row_attention_with_pair_bias/attention", "gating_w"),
            "attention_gating_b": _extract(params, f"{base}/msa_row_attention_with_pair_bias/attention", "gating_b"),
            "attention_output_w": _extract(params, f"{base}/msa_row_attention_with_pair_bias/attention", "output_w"),
            "attention_output_b": _extract(params, f"{base}/msa_row_attention_with_pair_bias/attention", "output_b"),
        },
    )

    _save_attention(params, f"{base}/msa_column_attention", os.path.join(out_dir, "msa_column_attention.npz"))
    _save_transition(params, f"{base}/msa_transition", os.path.join(out_dir, "msa_transition.npz"))
    _save_transition(params, f"{base}/pair_transition", os.path.join(out_dir, "pair_transition.npz"))

    _save_triangle_mult(params, f"{base}/triangle_multiplication_outgoing", os.path.join(out_dir, "tri_mul_out.npz"))
    _save_triangle_mult(params, f"{base}/triangle_multiplication_incoming", os.path.join(out_dir, "tri_mul_in.npz"))

    _save_triangle_attention(params, f"{base}/triangle_attention_starting_node", os.path.join(out_dir, "tri_att_start.npz"))
    _save_triangle_attention(params, f"{base}/triangle_attention_ending_node", os.path.join(out_dir, "tri_att_end.npz"))

    print(f"Saved real-weight evoformer dump directory: {out_dir}")


if __name__ == "__main__":
    main()
