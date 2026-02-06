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
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.model import modules  # pylint: disable=import-outside-toplevel
    from alphafold.model import config as af_config  # pylint: disable=import-outside-toplevel
    from alphafold.model import utils  # pylint: disable=import-outside-toplevel

    cfg = af_config.model_config("model_1")
    gc = cfg.model.global_config
    gc.deterministic = True
    gc.eval_dropout = False
    c = cfg.model.embeddings_and_evoformer

    with open(args.params, "rb") as f:
        flat = np.load(f, allow_pickle=False)
        full_params = utils.flat_params_to_haiku(flat)
    params = _slice_prefix(full_params, "alphafold/alphafold_iteration/evoformer/")

    ref = np.load(args.input_npz)
    query_embedding = np.asarray(ref["pair_after_recycle_relpos"][0], dtype=np.float32)  # (L,L,C)
    seq_mask = np.asarray(ref["seq_mask"], dtype=np.float32)  # (L,)
    mask_2d = seq_mask[:, None] * seq_mask[None, :]

    template_aatype = np.asarray(ref["template_aatype"], dtype=np.int32)  # (T,L)
    template_all_atom_positions = np.asarray(ref["template_all_atom_positions"], dtype=np.float32)  # (T,L,37,3)
    template_all_atom_masks = np.asarray(ref["template_all_atom_masks"], dtype=np.float32)  # (T,L,37)
    template_mask = np.ones((template_aatype.shape[0],), dtype=np.float32)

    def fn_single(query_embedding_in, template_aatype_in, template_all_atom_positions_in, template_all_atom_masks_in, mask_2d_in):
        template_pseudo_beta, template_pseudo_beta_mask = modules.pseudo_beta_fn(
            template_aatype_in, template_all_atom_positions_in, template_all_atom_masks_in
        )
        single_batch = {
            "template_aatype": template_aatype_in[0],
            "template_all_atom_positions": template_all_atom_positions_in[0],
            "template_all_atom_masks": template_all_atom_masks_in[0],
            "template_pseudo_beta": template_pseudo_beta[0],
            "template_pseudo_beta_mask": template_pseudo_beta_mask[0],
        }
        with hk.experimental.name_scope("template_embedding"):
            single_mod = modules.SingleTemplateEmbedding(c.template, gc, name="single_template_embedding")
            single_out = single_mod(query_embedding_in, single_batch, mask_2d_in, is_training=False)
        return single_out

    def fn_full(query_embedding_in, template_aatype_in, template_all_atom_positions_in, template_all_atom_masks_in, mask_2d_in, template_mask_in):
        template_pseudo_beta, template_pseudo_beta_mask = modules.pseudo_beta_fn(
            template_aatype_in, template_all_atom_positions_in, template_all_atom_masks_in
        )
        template_batch = {
            "template_aatype": template_aatype_in,
            "template_all_atom_positions": template_all_atom_positions_in,
            "template_all_atom_masks": template_all_atom_masks_in,
            "template_sum_probs": jnp.ones((template_aatype_in.shape[0], 1), dtype=query_embedding_in.dtype),
            "template_mask": template_mask_in,
            "template_pseudo_beta": template_pseudo_beta,
            "template_pseudo_beta_mask": template_pseudo_beta_mask,
        }
        mod = modules.TemplateEmbedding(c.template, gc, name="template_embedding")
        return mod(query_embedding_in, template_batch, mask_2d_in, is_training=False)

    transformed_single = hk.transform(fn_single)
    transformed_full = hk.transform(fn_full)
    key = jax.random.PRNGKey(args.seed)

    single_out = transformed_single.apply(
        params,
        key,
        jnp.asarray(query_embedding),
        jnp.asarray(template_aatype),
        jnp.asarray(template_all_atom_positions),
        jnp.asarray(template_all_atom_masks),
        jnp.asarray(mask_2d),
    )

    out = transformed_full.apply(
        params,
        key,
        jnp.asarray(query_embedding),
        jnp.asarray(template_aatype),
        jnp.asarray(template_all_atom_positions),
        jnp.asarray(template_all_atom_masks),
        jnp.asarray(mask_2d),
        jnp.asarray(template_mask),
    )

    arrays: Dict[str, np.ndarray] = {
        "query_embedding": np.asarray(query_embedding, dtype=np.float32),
        "mask_2d": np.asarray(mask_2d, dtype=np.float32),
        "template_mask": np.asarray(template_mask, dtype=np.float32),
        "template_aatype": np.asarray(template_aatype, dtype=np.int32),
        "template_all_atom_positions": np.asarray(template_all_atom_positions, dtype=np.float32),
        "template_all_atom_masks": np.asarray(template_all_atom_masks, dtype=np.float32),
        "single_out": np.asarray(single_out, dtype=np.float32),
        "out": np.asarray(out, dtype=np.float32),
    }

    full_prefix = "alphafold/alphafold_iteration/evoformer/template_embedding"
    for scope, vars_dict in full_params.items():
        if not scope.startswith(full_prefix):
            continue
        for name, arr in vars_dict.items():
            arrays[f"{scope}/{name}"] = np.asarray(arr, dtype=np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **arrays)
    print(f"Saved template embedding dump: {args.out}")


if __name__ == "__main__":
    main()
