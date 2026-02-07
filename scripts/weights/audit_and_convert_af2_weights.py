#!/usr/bin/env python3.11
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
from safetensors import safe_open
from safetensors.numpy import save_file


OFFICIAL_TAR_URL = "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"


@dataclass
class AuditEntry:
    source_npz: str
    model_name: str
    variant: str
    supported_by_alphafold2_jl: bool
    support_level: str
    support_notes: str
    num_tensors: int
    num_bytes: int
    has_evoformer: bool
    has_structure_module: bool
    has_predicted_lddt_head: bool
    has_predicted_aligned_error_head: bool
    output_safetensors: str
    source_npz_sha256: str
    output_safetensors_sha256: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", "0"))
        written = 0
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100.0 * written / total
                    print(f"download: {written}/{total} bytes ({pct:.2f}%)", end="\r", flush=True)
    print(f"\ndownload complete: {out_path}")


def extract_npz_members(tar_path: Path, params_dir: Path) -> list[Path]:
    params_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            if not name.endswith(".npz"):
                continue
            out_path = params_dir / name
            if out_path.exists():
                extracted.append(out_path)
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            with out_path.open("wb") as dst:
                dst.write(src.read())
            extracted.append(out_path)
    extracted.sort()
    return extracted


def classify_model(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem
    if re.fullmatch(r"params_model_[1-5]", stem):
        return stem.replace("params_", ""), "monomer"
    if re.fullmatch(r"params_model_[1-5]_ptm", stem):
        return stem.replace("params_", ""), "monomer_ptm"
    if re.fullmatch(r"params_model_[1-5]_multimer(_v[0-9]+)?", stem):
        return stem.replace("params_", ""), "multimer"
    return stem.replace("params_", ""), "unknown"


def support_decision(variant: str) -> tuple[bool, str, str]:
    if variant == "monomer":
        return (
            True,
            "SUPPORTED_PARTIAL_CONFIDENCE",
            "Core monomer inference supported; PAE/pTM head not expected in non-ptm checkpoints.",
        )
    if variant == "monomer_ptm":
        return (
            True,
            "SUPPORTED_FULL_MONOMER",
            "Monomer + confidence heads (pLDDT/PAE/pTM) supported.",
        )
    if variant == "multimer":
        return (
            True,
            "SUPPORTED_FULL_MULTIMER",
            "Multimer structure path and confidence heads supported.",
        )
    return (
        False,
        "UNSUPPORTED_UNKNOWN",
        "Unrecognized checkpoint naming pattern; no support guarantee.",
    )


def build_safetensors_name(model_name: str, variant: str) -> str:
    if variant == "monomer":
        return f"alphafold2_{model_name}_dm_2022-12-06.safetensors"
    if variant == "monomer_ptm":
        return f"alphafold2_{model_name}_dm_2022-12-06.safetensors"
    if variant == "multimer":
        return f"alphafold2_{model_name}_dm_2022-12-06.safetensors"
    return f"alphafold2_{model_name}_dm_2022-12-06_unknown.safetensors"


def convert_npz_to_safetensors(npz_path: Path, safetensors_path: Path) -> tuple[int, int, dict[str, bool]]:
    data: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as z:
        for k in z.files:
            arr = z[k]
            if arr.dtype.kind in {"U", "S", "O"}:
                continue
            data[k] = np.ascontiguousarray(arr)

    metadata = {
        "source_npz": npz_path.name,
        "converted_by": "scripts/weights/audit_and_convert_af2_weights.py",
        "source": OFFICIAL_TAR_URL,
    }
    save_file(data, str(safetensors_path), metadata=metadata)

    with safe_open(str(safetensors_path), framework="np") as f:
        out_keys = list(f.keys())
    with np.load(npz_path, allow_pickle=False) as z:
        in_keys = [k for k in z.files if z[k].dtype.kind not in {"U", "S", "O"}]

        flags = {
            "has_evoformer": any("/evoformer/" in k for k in in_keys),
            "has_structure_module": any("/structure_module/" in k for k in in_keys),
            "has_predicted_lddt_head": any("/predicted_lddt_head/" in k for k in in_keys),
            "has_predicted_aligned_error_head": any("/predicted_aligned_error_head/" in k for k in in_keys),
        }
        if len(in_keys) != len(out_keys):
            raise RuntimeError(
                f"Tensor count mismatch for {npz_path.name}: npz={len(in_keys)} safetensors={len(out_keys)}"
            )
        return len(in_keys), int(sum(z[k].nbytes for k in z.files if z[k].dtype.kind not in {"U", "S", "O"})), flags


def write_markdown_report(path: Path, entries: list[AuditEntry]) -> None:
    lines = [
        "# AlphaFold2 Weight Audit",
        "",
        f"Source: `{OFFICIAL_TAR_URL}`",
        "",
        "| source_npz | model_name | variant | supported | support_level | num_tensors | num_bytes | safetensors |",
        "|---|---|---|---:|---|---:|---:|---|",
    ]
    for e in entries:
        lines.append(
            f"| `{e.source_npz}` | `{e.model_name}` | `{e.variant}` | "
            f"`{str(e.supported_by_alphafold2_jl).lower()}` | `{e.support_level}` | "
            f"{e.num_tensors} | {e.num_bytes} | `{e.output_safetensors}` |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `SUPPORTED_FULL_MONOMER`: expected to run with confidence heads.")
    lines.append("- `SUPPORTED_FULL_MULTIMER`: expected to run with multimer recycle path and confidence heads.")
    lines.append("- `SUPPORTED_PARTIAL_CONFIDENCE`: monomer models without PAE/pTM head.")
    lines.append("- `UNSUPPORTED_UNKNOWN`: naming pattern not recognized by this audit script.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-root", type=Path, required=True, help="Output root for downloaded/extracted weights.")
    parser.add_argument("--tar-url", type=str, default=OFFICIAL_TAR_URL)
    parser.add_argument("--tar-path", type=Path, default=None, help="Optional pre-downloaded tar path.")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--params-dir-name", type=str, default="params_npz")
    parser.add_argument("--safetensors-dir-name", type=str, default="params_safetensors")
    parser.add_argument("--manifest-json", type=str, default="AF2_WEIGHTS_AUDIT.json")
    parser.add_argument("--manifest-md", type=str, default="AF2_WEIGHTS_AUDIT.md")
    args = parser.parse_args()

    weights_root = args.weights_root.resolve()
    weights_root.mkdir(parents=True, exist_ok=True)

    tar_path = args.tar_path.resolve() if args.tar_path else (weights_root / Path(args.tar_url).name)
    params_dir = weights_root / args.params_dir_name
    safetensors_dir = weights_root / args.safetensors_dir_name
    safetensors_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        if tar_path.exists():
            print(f"tar already exists: {tar_path}")
        else:
            print(f"downloading official params tar: {args.tar_url}")
            download_file(args.tar_url, tar_path)

    npz_files: list[Path]
    if not args.skip_extract:
        print(f"extracting npz files from: {tar_path}")
        npz_files = extract_npz_members(tar_path, params_dir)
    else:
        npz_files = sorted(params_dir.glob("*.npz"))

    if not npz_files:
        raise RuntimeError(f"No .npz files found in {params_dir}")

    print(f"found {len(npz_files)} npz files")

    entries: list[AuditEntry] = []
    for npz_path in npz_files:
        model_name, variant = classify_model(npz_path.name)
        supported, support_level, support_notes = support_decision(variant)
        out_name = build_safetensors_name(model_name, variant)
        out_path = safetensors_dir / out_name

        if not args.skip_convert:
            print(f"converting: {npz_path.name} -> {out_name}")
            num_tensors, num_bytes, flags = convert_npz_to_safetensors(npz_path, out_path)
        else:
            with np.load(npz_path, allow_pickle=False) as z:
                keys = z.files
                num_tensors = len(keys)
                num_bytes = int(sum(z[k].nbytes for k in keys))
                flags = {
                    "has_evoformer": any("/evoformer/" in k for k in keys),
                    "has_structure_module": any("/structure_module/" in k for k in keys),
                    "has_predicted_lddt_head": any("/predicted_lddt_head/" in k for k in keys),
                    "has_predicted_aligned_error_head": any("/predicted_aligned_error_head/" in k for k in keys),
                }

        entry = AuditEntry(
            source_npz=npz_path.name,
            model_name=model_name,
            variant=variant,
            supported_by_alphafold2_jl=supported,
            support_level=support_level,
            support_notes=support_notes,
            num_tensors=num_tensors,
            num_bytes=num_bytes,
            has_evoformer=flags["has_evoformer"],
            has_structure_module=flags["has_structure_module"],
            has_predicted_lddt_head=flags["has_predicted_lddt_head"],
            has_predicted_aligned_error_head=flags["has_predicted_aligned_error_head"],
            output_safetensors=out_name,
            source_npz_sha256=sha256_file(npz_path),
            output_safetensors_sha256=sha256_file(out_path) if out_path.exists() else "",
        )
        entries.append(entry)

    entries.sort(key=lambda e: e.source_npz)

    manifest_json_path = weights_root / args.manifest_json
    manifest_md_path = weights_root / args.manifest_md
    manifest_json_path.write_text(
        json.dumps(
            {
                "source_tar_url": args.tar_url,
                "source_tar_path": str(tar_path),
                "params_dir": str(params_dir),
                "safetensors_dir": str(safetensors_dir),
                "num_models": len(entries),
                "entries": [asdict(e) for e in entries],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_markdown_report(manifest_md_path, entries)

    print(f"wrote: {manifest_json_path}")
    print(f"wrote: {manifest_md_path}")
    print(f"safetensors directory: {safetensors_dir}")


if __name__ == "__main__":
    main()
