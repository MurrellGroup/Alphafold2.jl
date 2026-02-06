#!/usr/bin/env python3
import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def _slice_prefix(params, prefix):
    out = {}
    for scope, vars_dict in params.items():
        if scope.startswith(prefix):
            out[scope[len(prefix):]] = {k: np.asarray(v) for k, v in vars_dict.items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphafold-repo", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--template-npz", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.model import all_atom  # pylint: disable=import-outside-toplevel
    from alphafold.model import common_modules  # pylint: disable=import-outside-toplevel
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

    ref = np.load(args.template_npz)
    template_aatype = np.asarray(ref["template_aatype"], dtype=np.int32)  # (T,L)
    template_all_atom_positions = np.asarray(ref["template_all_atom_positions"], dtype=np.float32)  # (T,L,37,3)
    template_all_atom_masks = np.asarray(ref["template_all_atom_masks"], dtype=np.float32)  # (T,L,37)

    placeholder_for_undefined = not gc.zero_init

    def fn(template_aatype_in, template_all_atom_positions_in, template_all_atom_masks_in):
        num_templ, num_res = template_aatype_in.shape
        aatype_one_hot = jax.nn.one_hot(template_aatype_in, 22, axis=-1)

        ret = all_atom.atom37_to_torsion_angles(
            aatype=template_aatype_in,
            all_atom_pos=template_all_atom_positions_in,
            all_atom_mask=template_all_atom_masks_in,
            placeholder_for_undefined=placeholder_for_undefined,
        )

        template_features = jnp.concatenate(
            [
                aatype_one_hot,
                jnp.reshape(ret["torsion_angles_sin_cos"], [num_templ, num_res, 14]),
                jnp.reshape(ret["alt_torsion_angles_sin_cos"], [num_templ, num_res, 14]),
                ret["torsion_angles_mask"],
            ],
            axis=-1,
        )

        template_activations = common_modules.Linear(
            c.msa_channel, initializer="relu", name="template_single_embedding"
        )(template_features)
        template_activations = jax.nn.relu(template_activations)
        template_activations = common_modules.Linear(
            c.msa_channel, initializer="relu", name="template_projection"
        )(template_activations)

        torsion_angle_mask = ret["torsion_angles_mask"][:, :, 2].astype(template_activations.dtype)

        return (
            ret["torsion_angles_sin_cos"],
            ret["alt_torsion_angles_sin_cos"],
            ret["torsion_angles_mask"],
            template_activations,
            torsion_angle_mask,
        )

    transformed = hk.transform(fn)
    key = jax.random.PRNGKey(0)

    (
        torsion_angles_sin_cos,
        alt_torsion_angles_sin_cos,
        torsion_angles_mask,
        template_single_rows,
        torsion_row_mask,
    ) = transformed.apply(
        params,
        key,
        jnp.asarray(template_aatype),
        jnp.asarray(template_all_atom_positions),
        jnp.asarray(template_all_atom_masks),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        template_aatype=template_aatype,
        template_all_atom_positions=template_all_atom_positions,
        template_all_atom_masks=template_all_atom_masks,
        torsion_angles_sin_cos=np.asarray(torsion_angles_sin_cos, dtype=np.float32),
        alt_torsion_angles_sin_cos=np.asarray(alt_torsion_angles_sin_cos, dtype=np.float32),
        torsion_angles_mask=np.asarray(torsion_angles_mask, dtype=np.float32),
        template_single_rows=np.asarray(template_single_rows, dtype=np.float32),
        torsion_row_mask=np.asarray(torsion_row_mask, dtype=np.float32),
        placeholder_for_undefined=np.asarray([1 if placeholder_for_undefined else 0], dtype=np.int32),
    )

    print(f"Saved template single rows dump: {args.out}")


if __name__ == "__main__":
    main()
