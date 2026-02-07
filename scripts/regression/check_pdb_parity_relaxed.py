#!/usr/bin/env python3
"""Relaxed PDB parity report for Julia-vs-Python reference outputs."""

from __future__ import annotations

import argparse
import math
import pathlib
from typing import Dict, List

import numpy as np


def _parse_atoms(pdb_path: pathlib.Path) -> List[Dict[str, object]]:
    atoms: List[Dict[str, object]] = []
    with open(pdb_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.startswith("ATOM"):
                continue
            line = raw.rstrip("\n").ljust(80)
            atoms.append(
                {
                    "atom": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21],
                    "resseq": int(line[22:26].strip()),
                    "xyz": np.asarray(
                        [
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        ],
                        dtype=np.float64,
                    ),
                }
            )
    return atoms


def _kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    p_centered = p - p.mean(axis=0, keepdims=True)
    q_centered = q - q.mean(axis=0, keepdims=True)
    cov = p_centered.T @ q_centered
    v, _, wt = np.linalg.svd(cov)
    if np.linalg.det(v @ wt) < 0:
        v[:, -1] *= -1
    rot = v @ wt
    p_aligned = p_centered @ rot
    diff = p_aligned - q_centered
    return float(math.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Directory containing <case>/<case>_out.pdb outputs.")
    parser.add_argument(
        "--ref-dir",
        default="/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/test/regression/reference_pdbs",
    )
    parser.add_argument("--cases", default="")
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir).resolve()
    ref_dir = pathlib.Path(args.ref_dir).resolve()
    if args.cases:
        case_names = [x.strip() for x in args.cases.split(",") if x.strip()]
    else:
        case_names = sorted([p.name for p in run_dir.iterdir() if p.is_dir()])

    print("case,strict_identity_ok,order_atom_name_match,max_abs_order,rms_order,ca_aligned_rmsd")
    for case in case_names:
        julia_pdb = run_dir / case / f"{case}_out.pdb"
        python_pdb = ref_dir / f"{case}.python_official.pdb"
        if not julia_pdb.is_file() or not python_pdb.is_file():
            print(f"{case},false,false,NaN,NaN,NaN")
            continue

        jl = _parse_atoms(julia_pdb)
        py = _parse_atoms(python_pdb)
        if len(jl) != len(py):
            print(f"{case},false,false,NaN,NaN,NaN")
            continue

        strict_identity_ok = all(
            (a["atom"], a["resname"], a["chain"], a["resseq"])
            == (b["atom"], b["resname"], b["chain"], b["resseq"])
            for a, b in zip(jl, py)
        )
        order_atom_name_match = all(
            (a["atom"], a["resname"]) == (b["atom"], b["resname"])
            for a, b in zip(jl, py)
        )

        xyz_j = np.stack([x["xyz"] for x in jl], axis=0)
        xyz_p = np.stack([x["xyz"] for x in py], axis=0)
        max_abs_order = float(np.max(np.abs(xyz_j - xyz_p)))
        rms_order = float(np.sqrt(np.mean(np.sum((xyz_j - xyz_p) ** 2, axis=1))))

        ca_j = np.stack([x["xyz"] for x in jl if x["atom"] == "CA"], axis=0)
        ca_p = np.stack([x["xyz"] for x in py if x["atom"] == "CA"], axis=0)
        ca_aligned_rmsd = _kabsch_rmsd(ca_j, ca_p)

        print(
            f"{case},{str(strict_identity_ok).lower()},{str(order_atom_name_match).lower()},"
            f"{max_abs_order:.6g},{rms_order:.6g},{ca_aligned_rmsd:.6g}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
