#!/usr/bin/env python3
"""Check Julia native input-builder tensors against Python reference builders."""

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
    mod_path = repo_root / "scripts" / "regression" / "generate_python_official_reference_pdbs.py"
    spec = importlib.util.spec_from_file_location("pyref", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pyref"] = mod
    spec.loader.exec_module(mod)
    return mod


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


def _monomer_py_features(mod, case: Dict[str, object], residue_constants, pipeline, parsers):
    sequence = str(case["sequence_arg"])
    msa_files = [str(x) for x in case.get("msa_files", [])]
    template_pdbs = [str(x) for x in case.get("template_pdbs", [])]
    template_chains = [str(x) for x in case.get("template_chains", [])]

    raw = mod._build_monomer_raw_features(
        sequence=sequence,
        msa_file=msa_files[0] if msa_files else None,
        template_pdb=template_pdbs[0] if template_pdbs else None,
        template_chain=template_chains[0] if template_chains else "A",
        pipeline=pipeline,
        parsers=parsers,
        residue_constants=residue_constants,
    )

    hh_to_af2 = np.asarray(residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE, dtype=np.int32)
    out: Dict[str, np.ndarray] = {
        "aatype": np.argmax(np.asarray(raw["aatype"], dtype=np.float32), axis=-1).astype(np.int32),
        "residue_index": np.asarray(raw["residue_index"], dtype=np.int32),
        "msa": np.take(hh_to_af2, np.asarray(raw["msa"], dtype=np.int32), axis=0),
        "deletion_matrix": np.asarray(raw["deletion_matrix_int"], dtype=np.float32),
    }
    out["msa_mask"] = np.ones_like(out["msa"], dtype=np.float32)

    if "template_aatype" in raw:
        tmpl_aatype_hh = np.argmax(np.asarray(raw["template_aatype"], dtype=np.float32), axis=-1).astype(np.int32)
        out["template_aatype"] = np.take(hh_to_af2, tmpl_aatype_hh, axis=0)
        out["template_all_atom_positions"] = np.asarray(raw["template_all_atom_positions"], dtype=np.float32)
        out["template_all_atom_masks"] = np.asarray(raw["template_all_atom_masks"], dtype=np.float32)

    return out


def _multimer_py_features(
    mod,
    case: Dict[str, object],
    residue_constants,
    pipeline,
    pipeline_multimer,
    parsers,
    min_msa_rows: int,
):
    sequences = [s.strip().upper() for s in str(case["sequence_arg"]).split(",") if s.strip()]
    msa_files = [str(x) for x in case.get("msa_files", [])]
    template_pdbs = [str(x) for x in case.get("template_pdbs", [])]
    template_chains = [str(x) for x in case.get("template_chains", [])]

    np_example = mod._build_multimer_raw_features(
        sequences=sequences,
        msa_files=msa_files,
        template_pdbs=template_pdbs,
        template_chains=template_chains,
        min_msa_rows=min_msa_rows,
        pipeline=pipeline,
        pipeline_multimer=pipeline_multimer,
        parsers=parsers,
        residue_constants=residue_constants,
    )

    out = {
        "aatype": np.asarray(np_example["aatype"], dtype=np.int32),
        "residue_index": np.asarray(np_example["residue_index"], dtype=np.int32),
        "asym_id": np.asarray(np_example["asym_id"], dtype=np.int32),
        "entity_id": np.asarray(np_example["entity_id"], dtype=np.int32),
        "sym_id": np.asarray(np_example["sym_id"], dtype=np.int32),
        "msa": np.asarray(np_example["msa"], dtype=np.int32),
        "deletion_matrix": np.asarray(np_example["deletion_matrix"], dtype=np.float32),
        "msa_mask": np.asarray(np_example["msa_mask"], dtype=np.float32),
        "cluster_bias_mask": np.asarray(np_example["cluster_bias_mask"], dtype=np.float32),
    }

    if "template_aatype" in np_example and np.asarray(np_example["template_aatype"]).size > 0:
        out["template_aatype"] = np.asarray(np_example["template_aatype"], dtype=np.int32)
        out["template_all_atom_positions"] = np.asarray(np_example["template_all_atom_positions"], dtype=np.float32)
        out["template_all_atom_masks"] = np.asarray(np_example["template_all_atom_mask"], dtype=np.float32)

    return out


def _load_manifest_cases(manifest_path: pathlib.Path) -> Dict[str, Dict[str, object]]:
    suffix = manifest_path.suffix.lower()
    if suffix == ".toml":
        import tomllib

        with open(manifest_path, "rb") as handle:
            manifest = tomllib.load(handle)
    elif suffix == ".json":
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        raise ValueError(f"Unsupported manifest format for {manifest_path}")
    return dict(manifest["cases"])


def _iter_selected_cases(cases: Dict[str, Dict[str, object]], selected: Iterable[str] | None):
    if selected is None:
        for name in cases:
            yield name
    else:
        sel = set(selected)
        for name in cases:
            if name in sel:
                yield name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl")
    parser.add_argument("--alphafold-repo", default="/Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--cases", default="")
    parser.add_argument("--min-msa-rows-multimer", type=int, default=129)
    parser.add_argument("--julia-bin", default=JULIA_BIN_DEFAULT)
    args = parser.parse_args()

    repo_root = pathlib.Path(args.repo_root).resolve()
    if args.manifest:
        manifest_path = pathlib.Path(args.manifest).resolve()
    else:
        default_toml = repo_root / "test" / "regression" / "reference_pdbs" / "manifest.toml"
        default_json = repo_root / "test" / "regression" / "reference_pdbs" / "manifest_python_official.json"
        manifest_path = default_toml if default_toml.is_file() else default_json
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    cases = _load_manifest_cases(manifest_path)
    selected = [x.strip() for x in args.cases.split(",") if x.strip()] if args.cases else None

    sys.path.insert(0, str(pathlib.Path(args.alphafold_repo).resolve()))
    from alphafold.common import residue_constants  # pylint: disable=import-outside-toplevel
    from alphafold.data import parsers  # pylint: disable=import-outside-toplevel
    from alphafold.data import pipeline  # pylint: disable=import-outside-toplevel
    from alphafold.data import pipeline_multimer  # pylint: disable=import-outside-toplevel

    mod = _load_pyref_module(repo_root)

    env = os.environ.copy()
    env.setdefault("JULIA_PROJECT", JULIA_PROJECT_DEFAULT)
    env.setdefault("JULIA_DEPOT_PATH", JULIA_DEPOT_DEFAULT)
    env.setdefault("JULIA_PKG_OFFLINE", "true")
    env.setdefault("JULIA_PKG_PRECOMPILE_AUTO", "0")
    env.setdefault("AF2_MULTIMER_MIN_MSA_ROWS", str(args.min_msa_rows_multimer))

    failures = 0

    with tempfile.TemporaryDirectory(prefix="af2_input_parity_") as tmpdir:
        tmpdir_path = pathlib.Path(tmpdir)
        for case_name in _iter_selected_cases(cases, selected):
            case = cases[case_name]
            print(f"\n=== {case_name} ===")
            out_npz = tmpdir_path / f"{case_name}.jl_input.npz"
            _run_julia_builder(repo_root, case_name, case, out_npz, args.julia_bin, env)
            jl = dict(np.load(out_npz, allow_pickle=False))

            model = str(case["model"])
            if model == "monomer":
                py = _monomer_py_features(mod, case, residue_constants, pipeline, parsers)
                checks = [
                    _compare_int("aatype", py["aatype"], np.asarray(jl["aatype"]).reshape(-1)),
                    _compare_int("residue_index", py["residue_index"], np.asarray(jl["residue_index"]).reshape(-1)),
                    _compare_int("msa", py["msa"], np.asarray(jl["msa"])),
                    _compare_float("deletion_matrix", py["deletion_matrix"], np.asarray(jl["deletion_matrix"])),
                    _compare_float("msa_mask", py["msa_mask"], np.asarray(jl["msa_mask"])),
                ]
                if "template_aatype" in py and "template_aatype" in jl:
                    checks.extend(
                        [
                            _compare_int("template_aatype", py["template_aatype"], np.asarray(jl["template_aatype"])),
                            _compare_float(
                                "template_all_atom_positions",
                                py["template_all_atom_positions"],
                                np.asarray(jl["template_all_atom_positions"]),
                            ),
                            _compare_float(
                                "template_all_atom_masks",
                                py["template_all_atom_masks"],
                                np.asarray(jl["template_all_atom_masks"]),
                            ),
                        ]
                    )
            elif model == "multimer":
                py = _multimer_py_features(
                    mod,
                    case,
                    residue_constants,
                    pipeline,
                    pipeline_multimer,
                    parsers,
                    min_msa_rows=args.min_msa_rows_multimer,
                )
                checks = [
                    _compare_int("aatype", py["aatype"], np.asarray(jl["aatype"]).reshape(-1)),
                    _compare_int("residue_index", py["residue_index"], np.asarray(jl["residue_index"]).reshape(-1)),
                    _compare_int("asym_id", py["asym_id"], np.asarray(jl["asym_id"]).reshape(-1)),
                    _compare_int("entity_id", py["entity_id"], np.asarray(jl["entity_id"]).reshape(-1)),
                    _compare_int("sym_id", py["sym_id"], np.asarray(jl["sym_id"]).reshape(-1)),
                    _compare_int("msa", py["msa"], np.asarray(jl["msa"])),
                    _compare_float("deletion_matrix", py["deletion_matrix"], np.asarray(jl["deletion_matrix"])),
                    _compare_float("msa_mask", py["msa_mask"], np.asarray(jl["msa_mask"])),
                    _compare_float("cluster_bias_mask", py["cluster_bias_mask"], np.asarray(jl["cluster_bias_mask"])),
                ]
                if "template_aatype" in py and "template_aatype" in jl:
                    checks.extend(
                        [
                            _compare_int("template_aatype", py["template_aatype"], np.asarray(jl["template_aatype"])),
                            _compare_float(
                                "template_all_atom_positions",
                                py["template_all_atom_positions"],
                                np.asarray(jl["template_all_atom_positions"]),
                            ),
                            _compare_float(
                                "template_all_atom_masks",
                                py["template_all_atom_masks"],
                                np.asarray(jl["template_all_atom_masks"]),
                            ),
                        ]
                    )
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
        print(f"\nInput tensor parity FAILED for {failures} case(s).")
        return 1

    print("\nInput tensor parity PASSED for all checked cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
