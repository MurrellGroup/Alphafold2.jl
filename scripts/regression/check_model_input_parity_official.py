#!/usr/bin/env python3
"""Check Julia model-input tensors against official Python process_features outputs.

This compares the tensors that are fed to model layers:
- Monomer: outputs of official RunModel.process_features.
- Multimer: official RunModel.process_features returns raw features unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


JULIA_BIN_DEFAULT = (
    "/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia"
)
JULIA_PROJECT_DEFAULT = "/Users/benmurrell/JuliaM3/juliaESM"
JULIA_DEPOT_DEFAULT = (
    "/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:"
    "/Users/benmurrell/JuliaM3/juliaESM/.julia_depot"
)


@dataclass
class CompareResult:
    key: str
    ok: bool
    detail: str


def _load_pyref_module(repo_root: pathlib.Path):
    mod_path = (
        repo_root
        / "scripts"
        / "regression"
        / "generate_python_reference_pdbs_legacy_do_not_use.py"
    )
    spec = importlib.util.spec_from_file_location("pyref_legacy", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pyref_legacy"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_manifest(manifest_path: pathlib.Path) -> Dict[str, object]:
    suffix = manifest_path.suffix.lower()
    if suffix == ".toml":
        import tomllib

        with open(manifest_path, "rb") as handle:
            return dict(tomllib.load(handle))
    if suffix == ".json":
        with open(manifest_path, "r", encoding="utf-8") as handle:
            return dict(json.load(handle))
    raise ValueError(f"Unsupported manifest format for {manifest_path}")


def _iter_selected_cases(
    cases: Dict[str, Dict[str, object]], selected: Iterable[str] | None
):
    if selected is None:
        for name in cases:
            yield name
    else:
        sel = set(selected)
        for name in cases:
            if name in sel:
                yield name


def _run_julia_builder(
    repo_root: pathlib.Path,
    case_name: str,
    case: Dict[str, object],
    out_npz: pathlib.Path,
    julia_bin: str,
    env: Dict[str, str],
) -> None:
    model = str(case["model"])
    seq = str(case["sequence_arg"])
    num_recycle = int(case["num_recycle"])
    msa_files = [str(x) for x in case.get("msa_files", [])]
    template_pdbs = [str(x) for x in case.get("template_pdbs", [])]
    template_chains = [str(x) for x in case.get("template_chains", [])]

    if model == "monomer":
        script = repo_root / "scripts" / "end_to_end" / "build_monomer_input_jl.jl"
        args: List[str] = [seq, str(out_npz), str(num_recycle)]
        has_msa = len(msa_files) > 0
        has_templates = len(template_pdbs) > 0
        if has_msa or has_templates:
            args.append(msa_files[0] if has_msa else "")
            if has_templates:
                args.append(",".join(template_pdbs))
                args.append(",".join(template_chains))
    elif model == "multimer":
        script = repo_root / "scripts" / "end_to_end" / "build_multimer_input_jl.jl"
        args = [seq, str(out_npz), str(num_recycle)]
        has_msa = len(msa_files) > 0
        has_templates = len(template_pdbs) > 0
        if has_msa or has_templates:
            args.append(",".join(msa_files) if has_msa else "")
            if has_templates:
                args.append(",".join(template_pdbs))
                args.append(",".join(template_chains))
    else:
        raise ValueError(f"Unsupported model kind in case {case_name}: {model}")

    cmd = [julia_bin, "--startup-file=no", "--history-file=no", str(script), *args]
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _compare_int(key: str, a: np.ndarray, b: np.ndarray) -> CompareResult:
    ai = np.asarray(a, dtype=np.int32)
    bi = np.asarray(b, dtype=np.int32)
    if ai.shape != bi.shape:
        return CompareResult(key, False, f"shape {ai.shape} vs {bi.shape}")
    mismatch = int(np.sum(ai != bi))
    if mismatch != 0:
        return CompareResult(key, False, f"mismatch_count={mismatch}")
    return CompareResult(key, True, "exact")


def _compare_float(key: str, a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> CompareResult:
    af = np.asarray(a, dtype=np.float32)
    bf = np.asarray(b, dtype=np.float32)
    if af.shape != bf.shape:
        return CompareResult(key, False, f"shape {af.shape} vs {bf.shape}")
    max_abs = float(np.max(np.abs(af - bf))) if af.size else 0.0
    if max_abs > atol:
        return CompareResult(key, False, f"max_abs={max_abs:.8g} > {atol:g}")
    return CompareResult(key, True, f"max_abs={max_abs:.8g}")


def _trim_padded_templates(py_arr: np.ndarray, jl_arr: np.ndarray) -> Tuple[np.ndarray, CompareResult | None]:
    if jl_arr.shape == py_arr.shape:
        return jl_arr, None
    if jl_arr.ndim < 1 or py_arr.ndim < 1:
        return jl_arr, CompareResult("template_stack", False, f"shape {py_arr.shape} vs {jl_arr.shape}")
    if jl_arr.shape[1:] != py_arr.shape[1:] or jl_arr.shape[0] < py_arr.shape[0]:
        return jl_arr, CompareResult("template_stack", False, f"shape {py_arr.shape} vs {jl_arr.shape}")
    tail = jl_arr[py_arr.shape[0] :]
    if np.any(np.asarray(tail) != 0):
        return jl_arr, CompareResult("template_stack", False, "padded template tail is non-zero")
    return jl_arr[: py_arr.shape[0]], None


def _load_params(params_path: str):
    from alphafold.model import utils  # pylint: disable=import-outside-toplevel

    with open(params_path, "rb") as handle:
        flat = np.load(handle, allow_pickle=False)
        return utils.flat_params_to_haiku(flat)


def _build_official_processed_features(
    mod,
    case: Dict[str, object],
    monomer_params: str,
    multimer_params: str,
    min_msa_rows_multimer: int,
):
    from alphafold.common import residue_constants  # pylint: disable=import-outside-toplevel
    from alphafold.data import parsers  # pylint: disable=import-outside-toplevel
    from alphafold.data import pipeline  # pylint: disable=import-outside-toplevel
    from alphafold.data import pipeline_multimer  # pylint: disable=import-outside-toplevel
    from alphafold.model import config as af_config  # pylint: disable=import-outside-toplevel
    from alphafold.model import model as af_model  # pylint: disable=import-outside-toplevel

    model_kind = str(case["model"])
    sequence_arg = str(case["sequence_arg"])
    num_recycle = int(case["num_recycle"])
    msa_files = [str(x) for x in case.get("msa_files", [])]
    template_pdbs = [str(x) for x in case.get("template_pdbs", [])]
    template_chains = [str(x) for x in case.get("template_chains", [])]

    if model_kind == "monomer":
        params_path = monomer_params
        model_name = "model_1_ptm" if "_ptm" in os.path.basename(params_path) else "model_1"
        cfg = af_config.model_config(model_name)
        cfg.model.global_config.deterministic = True
        cfg.model.global_config.eval_dropout = False
        cfg.model.global_config.bfloat16 = False
        cfg.model.global_config.bfloat16_output = False
        cfg.model.num_recycle = num_recycle
        cfg.model.resample_msa_in_recycling = False
        cfg.data.common.num_recycle = num_recycle
        cfg.data.common.resample_msa_in_recycling = False
        cfg.data.eval.num_ensemble = 1
        has_templates = len(template_pdbs) > 0
        cfg.model.embeddings_and_evoformer.template.enabled = has_templates
        if hasattr(cfg.model.embeddings_and_evoformer.template, "embed_torsion_angles"):
            cfg.model.embeddings_and_evoformer.template.embed_torsion_angles = has_templates
        cfg.data.common.use_templates = has_templates
        raw = mod._build_monomer_raw_features(
            sequence=sequence_arg,
            msa_file=msa_files[0] if msa_files else None,
            template_pdb=template_pdbs[0] if template_pdbs else None,
            template_chain=template_chains[0] if template_chains else "A",
            pipeline=pipeline,
            parsers=parsers,
            residue_constants=residue_constants,
        )
    elif model_kind == "multimer":
        params_path = multimer_params
        cfg = af_config.model_config("model_1_multimer_v3")
        cfg.model.global_config.deterministic = True
        cfg.model.global_config.eval_dropout = False
        cfg.model.global_config.bfloat16 = False
        cfg.model.global_config.bfloat16_output = False
        cfg.model.num_recycle = num_recycle
        cfg.model.resample_msa_in_recycling = False
        data_cfg = getattr(cfg, "data", None)
        if data_cfg is not None:
            if hasattr(data_cfg, "common"):
                data_cfg.common.num_recycle = num_recycle
                data_cfg.common.resample_msa_in_recycling = False
            if hasattr(data_cfg, "eval"):
                data_cfg.eval.num_ensemble = 1
        has_templates = len(template_pdbs) > 0
        cfg.model.embeddings_and_evoformer.template.enabled = has_templates
        if hasattr(cfg.model.embeddings_and_evoformer.template, "embed_torsion_angles"):
            cfg.model.embeddings_and_evoformer.template.embed_torsion_angles = has_templates

        seqs = [s.strip().upper() for s in sequence_arg.split(",") if s.strip()]
        if has_templates:
            if len(template_pdbs) == 1 and len(seqs) > 1:
                template_pdbs = template_pdbs * len(seqs)
            if len(template_chains) == 1 and len(seqs) > 1:
                template_chains = template_chains * len(seqs)

        raw = mod._build_multimer_raw_features(
            sequences=seqs,
            msa_files=msa_files,
            template_pdbs=template_pdbs,
            template_chains=template_chains,
            min_msa_rows=min_msa_rows_multimer,
            pipeline=pipeline,
            pipeline_multimer=pipeline_multimer,
            parsers=parsers,
            residue_constants=residue_constants,
        )
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")

    params = _load_params(params_path)
    runner = af_model.RunModel(cfg, params)
    return runner.process_features(raw, random_seed=0)


def _compare_case_monomer(jl: Dict[str, np.ndarray], py: Dict[str, np.ndarray]) -> List[CompareResult]:
    checks: List[CompareResult] = []
    checks.append(_compare_int("aatype", np.asarray(py["aatype"])[0], np.asarray(jl["aatype"]).reshape(-1)))
    checks.append(
        _compare_int(
            "residue_index",
            np.asarray(py["residue_index"])[0],
            np.asarray(jl["residue_index"]).reshape(-1),
        )
    )
    checks.append(_compare_float("seq_mask", np.asarray(py["seq_mask"])[0], np.asarray(jl["seq_mask"]).reshape(-1)))
    checks.append(_compare_float("target_feat", np.asarray(py["target_feat"])[0], np.asarray(jl["target_feat"])))
    checks.append(_compare_float("msa_feat", np.asarray(py["msa_feat"])[0], np.asarray(jl["msa_feat"])))
    py_msa_mask = np.asarray(py["msa_mask"])[0]
    jl_msa_mask = np.asarray(jl["msa_mask_model"] if "msa_mask_model" in jl else jl["msa_mask"])
    checks.append(_compare_float("msa_mask", py_msa_mask, jl_msa_mask))
    checks.append(_compare_int("extra_msa", np.asarray(py["extra_msa"])[0], np.asarray(jl["extra_msa"])))
    checks.append(
        _compare_float(
            "extra_msa_mask",
            np.asarray(py["extra_msa_mask"])[0],
            np.asarray(jl["extra_msa_mask"]),
        )
    )
    if "extra_deletion_value" in py and "extra_deletion_value" in jl:
        checks.append(
            _compare_float(
                "extra_deletion_value",
                np.asarray(py["extra_deletion_value"])[0],
                np.asarray(jl["extra_deletion_value"]),
            )
        )

    if "template_aatype" in py and "template_aatype" in jl:
        py_template_aatype = np.asarray(py["template_aatype"])[0]
        py_template_pos = np.asarray(py["template_all_atom_positions"])[0]
        py_template_mask = np.asarray(py["template_all_atom_masks"])[0]
        jl_template_aatype = np.asarray(jl["template_aatype"])
        jl_template_pos = np.asarray(jl["template_all_atom_positions"])
        jl_template_mask = np.asarray(jl["template_all_atom_masks"])

        jl_template_aatype, err = _trim_padded_templates(py_template_aatype, jl_template_aatype)
        if err is not None:
            checks.append(CompareResult("template_aatype", False, err.detail))
        jl_template_pos, err = _trim_padded_templates(py_template_pos, jl_template_pos)
        if err is not None:
            checks.append(CompareResult("template_all_atom_positions", False, err.detail))
        jl_template_mask, err = _trim_padded_templates(py_template_mask, jl_template_mask)
        if err is not None:
            checks.append(CompareResult("template_all_atom_masks", False, err.detail))
        checks.append(_compare_int("template_aatype", py_template_aatype, jl_template_aatype))
        checks.append(_compare_float("template_all_atom_positions", py_template_pos, jl_template_pos))
        checks.append(_compare_float("template_all_atom_masks", py_template_mask, jl_template_mask))
        if "template_mask" in py and "template_mask" in jl:
            checks.append(_compare_float("template_mask", np.asarray(py["template_mask"])[0], np.asarray(jl["template_mask"])))

    return checks


def _compare_case_multimer(jl: Dict[str, np.ndarray], py: Dict[str, np.ndarray]) -> List[CompareResult]:
    checks: List[CompareResult] = []
    checks.append(_compare_int("aatype", np.asarray(py["aatype"]), np.asarray(jl["aatype"]).reshape(-1)))
    checks.append(
        _compare_int(
            "residue_index",
            np.asarray(py["residue_index"]),
            np.asarray(jl["residue_index"]).reshape(-1),
        )
    )
    checks.append(_compare_int("asym_id", np.asarray(py["asym_id"]), np.asarray(jl["asym_id"]).reshape(-1)))
    checks.append(_compare_int("entity_id", np.asarray(py["entity_id"]), np.asarray(jl["entity_id"]).reshape(-1)))
    checks.append(_compare_int("sym_id", np.asarray(py["sym_id"]), np.asarray(jl["sym_id"]).reshape(-1)))
    if "seq_mask" in py and "seq_mask" in jl:
        checks.append(_compare_float("seq_mask", np.asarray(py["seq_mask"]), np.asarray(jl["seq_mask"]).reshape(-1)))
    checks.append(_compare_int("msa", np.asarray(py["msa"]), np.asarray(jl["msa"])))
    checks.append(_compare_float("deletion_matrix", np.asarray(py["deletion_matrix"]), np.asarray(jl["deletion_matrix"])))
    checks.append(_compare_float("msa_mask", np.asarray(py["msa_mask"]), np.asarray(jl["msa_mask"])))
    checks.append(
        _compare_float(
            "cluster_bias_mask",
            np.asarray(py["cluster_bias_mask"]),
            np.asarray(jl["cluster_bias_mask"]),
        )
    )

    if "template_aatype" in py and "template_aatype" in jl and np.asarray(py["template_aatype"]).size > 0:
        py_taa = np.asarray(py["template_aatype"])
        py_tpos = np.asarray(py["template_all_atom_positions"])
        py_tmask = np.asarray(py["template_all_atom_mask"])
        jl_taa = np.asarray(jl["template_aatype"])
        jl_tpos = np.asarray(jl["template_all_atom_positions"])
        jl_tmask = np.asarray(jl["template_all_atom_masks"])
        checks.append(_compare_int("template_aatype", py_taa, jl_taa))
        checks.append(_compare_float("template_all_atom_positions", py_tpos, jl_tpos))
        checks.append(_compare_float("template_all_atom_mask", py_tmask, jl_tmask))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl")
    parser.add_argument("--alphafold-repo", default="/Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold")
    parser.add_argument(
        "--manifest",
        default="/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/test/regression/reference_pdbs/manifest_python_official.json",
    )
    parser.add_argument("--cases", default="")
    parser.add_argument("--min-msa-rows-multimer", type=int, default=512)
    parser.add_argument("--julia-bin", default=JULIA_BIN_DEFAULT)
    args = parser.parse_args()

    repo_root = pathlib.Path(args.repo_root).resolve()
    manifest_path = pathlib.Path(args.manifest).resolve()
    manifest = _load_manifest(manifest_path)
    cases = dict(manifest["cases"])
    monomer_params = str(manifest["monomer_params"])
    multimer_params = str(manifest["multimer_params"])
    selected = [x.strip() for x in args.cases.split(",") if x.strip()] if args.cases else None

    sys.path.insert(0, str(pathlib.Path(args.alphafold_repo).resolve()))
    mod = _load_pyref_module(repo_root)

    env = os.environ.copy()
    env.setdefault("JULIA_PROJECT", JULIA_PROJECT_DEFAULT)
    env.setdefault("JULIA_DEPOT_PATH", JULIA_DEPOT_DEFAULT)
    env.setdefault("JULIA_PKG_OFFLINE", "true")
    env.setdefault("JULIA_PKG_PRECOMPILE_AUTO", "0")
    env.setdefault("AF2_MULTIMER_MIN_MSA_ROWS", str(args.min_msa_rows_multimer))
    # Keep official-comparison parity stable regardless of the native default.
    env.setdefault("AF2_MULTIMER_PAIRING_MODE", "pair by row index")

    failures = 0
    with tempfile.TemporaryDirectory(prefix="af2_model_input_parity_") as tmpdir:
        tmpdir_path = pathlib.Path(tmpdir)
        for case_name in _iter_selected_cases(cases, selected):
            case = cases[case_name]
            print(f"\n=== {case_name} ===")
            out_npz = tmpdir_path / f"{case_name}.jl_input.npz"
            _run_julia_builder(repo_root, case_name, case, out_npz, args.julia_bin, env)
            jl = dict(np.load(out_npz, allow_pickle=False))
            py = _build_official_processed_features(
                mod,
                case,
                monomer_params=monomer_params,
                multimer_params=multimer_params,
                min_msa_rows_multimer=args.min_msa_rows_multimer,
            )

            model = str(case["model"])
            if model == "monomer":
                checks = _compare_case_monomer(jl, py)
            elif model == "multimer":
                checks = _compare_case_multimer(jl, py)
            else:
                raise ValueError(f"Unsupported case model: {model}")

            case_fail = False
            for check in checks:
                status = "OK" if check.ok else "FAIL"
                print(f"  [{status}] {check.key}: {check.detail}")
                if not check.ok:
                    case_fail = True
            if case_fail:
                failures += 1

    if failures:
        print(f"\nModel-input parity FAILED for {failures} case(s).")
        return 1
    print("\nModel-input parity PASSED for all checked cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
