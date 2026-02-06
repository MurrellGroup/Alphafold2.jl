#!/usr/bin/env python3
import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def _extract(params, scope, name):
    return np.asarray(params[scope][name], dtype=np.float32)


def _save_npz(path, arrays):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **arrays)


def _slice_fold_params(full_params: hk.Params) -> hk.Params:
    prefix = "alphafold/alphafold_iteration/structure_module/fold_iteration/"
    out = {}
    for scope, vars_dict in full_params.items():
        if not scope.startswith(prefix):
            continue
        new_scope = scope[len(prefix):]
        out[new_scope] = {k: np.asarray(v) for k, v in vars_dict.items()}
    return out


def _side_key(base: str, i: int) -> str:
    return base if i == 1 else f"{base}_{i-1}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphafold-repo", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.alphafold_repo)
    from alphafold.model import config as af_config  # pylint: disable=import-outside-toplevel
    from alphafold.model import folding  # pylint: disable=import-outside-toplevel
    from alphafold.model import quat_affine  # pylint: disable=import-outside-toplevel
    from alphafold.model import r3  # pylint: disable=import-outside-toplevel
    from alphafold.model import utils  # pylint: disable=import-outside-toplevel

    cfg = af_config.model_config("model_1")
    sm_cfg = cfg.model.heads.structure_module
    side_cfg = sm_cfg.sidechain
    global_cfg = cfg.model.global_config

    with open(args.params, "rb") as f:
        flat = np.load(f, allow_pickle=False)
        full_params = utils.flat_params_to_haiku(flat)

    params = _slice_fold_params(full_params)

    c_in = int(params["rigid_sidechain/input_projection"]["weights"].shape[0])

    def fn(affine_tensor, act, initial_act, aatype):
        affine = quat_affine.QuatAffine.from_tensor(affine_tensor)
        out = folding.MultiRigidSidechain(side_cfg, global_cfg, name="rigid_sidechain")(
            affine,
            [act, initial_act],
            aatype,
        )
        atom_pos = r3.vecs_to_tensor(out["atom_pos"])
        return out["angles_sin_cos"], out["unnormalized_angles_sin_cos"], atom_pos

    transformed = hk.transform(fn)

    key = jax.random.PRNGKey(args.seed)
    k_act, k_init, k_mask, k_update, k_aatype, k_apply = jax.random.split(key, 6)

    act = jax.random.normal(k_act, (args.n, c_in), dtype=jnp.float32)
    initial_act = jax.random.normal(k_init, (args.n, c_in), dtype=jnp.float32)

    seq_mask = (jax.random.uniform(k_mask, (args.n, 1)) > 0.2).astype(jnp.float32)
    affine = folding.generate_new_affine(seq_mask)
    affine_update = jax.random.normal(k_update, (args.n, 6), dtype=jnp.float32) * 0.15
    affine = affine.pre_compose(affine_update).scale_translation(sm_cfg.position_scale)
    affine_tensor = affine.to_tensor()

    aatype = jax.random.randint(k_aatype, (args.n,), minval=0, maxval=20, dtype=jnp.int32)

    out_angles, out_unnorm, out_atom_pos = transformed.apply(
        params,
        k_apply,
        affine_tensor,
        act,
        initial_act,
        aatype,
    )

    arrays = {
        "affine_tensor": np.asarray(affine_tensor, dtype=np.float32),
        "act": np.asarray(act, dtype=np.float32),
        "initial_act": np.asarray(initial_act, dtype=np.float32),
        "aatype": np.asarray(aatype, dtype=np.int32),
        "out_angles": np.asarray(out_angles, dtype=np.float32),
        "out_unnormalized_angles": np.asarray(out_unnorm, dtype=np.float32),
        "out_atom_pos": np.asarray(out_atom_pos, dtype=np.float32),
        "num_residual_block": np.asarray(int(side_cfg.num_residual_block), dtype=np.int32),
    }

    for i in range(1, 3):
        p = _side_key("input_projection", i)
        arrays[f"{p}_weights"] = _extract(params, f"rigid_sidechain/{p}", "weights")
        arrays[f"{p}_bias"] = _extract(params, f"rigid_sidechain/{p}", "bias")

    for i in range(1, int(side_cfg.num_residual_block) + 1):
        p1 = _side_key("resblock1", i)
        p2 = _side_key("resblock2", i)
        arrays[f"{p1}_weights"] = _extract(params, f"rigid_sidechain/{p1}", "weights")
        arrays[f"{p1}_bias"] = _extract(params, f"rigid_sidechain/{p1}", "bias")
        arrays[f"{p2}_weights"] = _extract(params, f"rigid_sidechain/{p2}", "weights")
        arrays[f"{p2}_bias"] = _extract(params, f"rigid_sidechain/{p2}", "bias")

    arrays["unnormalized_angles_weights"] = _extract(params, "rigid_sidechain/unnormalized_angles", "weights")
    arrays["unnormalized_angles_bias"] = _extract(params, "rigid_sidechain/unnormalized_angles", "bias")

    _save_npz(args.out, arrays)
    print(f"Saved real-weight sidechain dump: {args.out}")


if __name__ == "__main__":
    main()
