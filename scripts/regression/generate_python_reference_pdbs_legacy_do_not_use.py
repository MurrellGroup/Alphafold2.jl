#!/usr/bin/env python3
"""LEGACY helper to generate reference PDBs via local orchestration.

DEPRECATED: kept only for internal/debug compatibility.
Do not use this script as the canonical Python reference pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np


RESTYPE_3TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


@dataclass(frozen=True)
class ParsedPdbAtom:
    atom: str
    resname: str
    chain: str
    resseq: int
    x: float
    y: float
    z: float


def _parse_pdb_atoms(pdb_path: str) -> List[ParsedPdbAtom]:
    atoms: List[ParsedPdbAtom] = []
    with open(pdb_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.startswith("ATOM"):
                continue
            line = raw.rstrip("\n")
            if len(line) < 54:
                continue
            atom = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21:22]
            resseq = int(line[22:26].strip())
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            atoms.append(
                ParsedPdbAtom(
                    atom=atom,
                    resname=resname,
                    chain=chain,
                    resseq=resseq,
                    x=x,
                    y=y,
                    z=z,
                )
            )
    return atoms


def _ca_geometry_from_pdb(pdb_path: str) -> Dict[str, float]:
    atoms = _parse_pdb_atoms(pdb_path)
    ca = [a for a in atoms if a.atom == "CA"]
    if len(ca) < 2:
        return {
            "ca_distance_mean": float("nan"),
            "ca_distance_std": float("nan"),
            "ca_distance_min": float("nan"),
            "ca_distance_max": float("nan"),
            "ca_distance_outlier_fraction": float("nan"),
            "ca_distance_intra_chain_mean": float("nan"),
            "ca_distance_intra_chain_std": float("nan"),
            "ca_distance_intra_chain_min": float("nan"),
            "ca_distance_intra_chain_max": float("nan"),
            "ca_distance_intra_chain_outlier_fraction": float("nan"),
        }

    d_all: List[float] = []
    d_intra: List[float] = []
    for i in range(len(ca) - 1):
        dx = ca[i + 1].x - ca[i].x
        dy = ca[i + 1].y - ca[i].y
        dz = ca[i + 1].z - ca[i].z
        d = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        d_all.append(d)
        if ca[i + 1].chain == ca[i].chain:
            d_intra.append(d)

    def _stats(vals: Sequence[float]) -> Tuple[float, float, float, float, float]:
        if not vals:
            return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
        arr = np.asarray(vals, dtype=np.float32)
        outlier = float(np.mean((arr < 3.2) | (arr > 4.4)))
        return (
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.min(arr)),
            float(np.max(arr)),
            outlier,
        )

    mean_all, std_all, min_all, max_all, out_all = _stats(d_all)
    mean_intra, std_intra, min_intra, max_intra, out_intra = _stats(d_intra)
    return {
        "ca_distance_mean": mean_all,
        "ca_distance_std": std_all,
        "ca_distance_min": min_all,
        "ca_distance_max": max_all,
        "ca_distance_outlier_fraction": out_all,
        "ca_distance_intra_chain_mean": mean_intra,
        "ca_distance_intra_chain_std": std_intra,
        "ca_distance_intra_chain_min": min_intra,
        "ca_distance_intra_chain_max": max_intra,
        "ca_distance_intra_chain_outlier_fraction": out_intra,
    }


def _global_align_query_to_template(query_seq: str, template_seq: str) -> List[int]:
    lq = len(query_seq)
    lt = len(template_seq)
    match_score = 1
    mismatch_score = -1
    gap_penalty = -1

    score = np.zeros((lq + 1, lt + 1), dtype=np.int32)
    trace = np.zeros((lq + 1, lt + 1), dtype=np.uint8)  # 1=diag, 2=up, 3=left

    for i in range(lq):
        score[i + 1, 0] = score[i, 0] + gap_penalty
        trace[i + 1, 0] = 2
    for j in range(lt):
        score[0, j + 1] = score[0, j] + gap_penalty
        trace[0, j + 1] = 3

    for i in range(lq):
        qi = query_seq[i].upper()
        for j in range(lt):
            tj = template_seq[j].upper()
            diag = score[i, j] + (match_score if qi == tj else mismatch_score)
            up = score[i, j + 1] + gap_penalty
            left = score[i + 1, j] + gap_penalty
            best = diag
            direction = 1
            if up > best:
                best = up
                direction = 2
            if left > best:
                best = left
                direction = 3
            score[i + 1, j + 1] = best
            trace[i + 1, j + 1] = direction

    mapping = [0] * lq  # query index -> template index (1-based, 0 means gap)
    i = lq
    j = lt
    while i > 0 or j > 0:
        direction = int(trace[i, j])
        if direction == 1:
            mapping[i - 1] = j
            i -= 1
            j -= 1
        elif direction == 2:
            mapping[i - 1] = 0
            i -= 1
        else:
            j -= 1
    return mapping


def _parse_template_chain(
    pdb_path: str,
    chain_id: str,
    residue_constants,
) -> Tuple[str, np.ndarray, np.ndarray]:
    atom_order = residue_constants.atom_order
    residues: List[Dict[str, object]] = []
    residue_index: Dict[Tuple[str, str], int] = {}
    with open(pdb_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain_id:
                continue
            key = (line[22:26], line[26])
            if key not in residue_index:
                residue_index[key] = len(residues)
                residues.append({"resname": line[17:20].strip(), "atoms": {}})
            atom_name = line[12:16].strip()
            if atom_name not in atom_order:
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            atom_map = residues[residue_index[key]]["atoms"]
            assert isinstance(atom_map, dict)
            atom_map[atom_name] = (x, y, z)

    if not residues:
        raise ValueError(f"No residues parsed from template {pdb_path} chain {chain_id}.")

    seq = "".join(RESTYPE_3TO1.get(str(r["resname"]), "X") for r in residues)
    positions = np.zeros((len(residues), residue_constants.atom_type_num, 3), dtype=np.float32)
    masks = np.zeros((len(residues), residue_constants.atom_type_num), dtype=np.float32)
    for i, residue in enumerate(residues):
        atom_map = residue["atoms"]
        assert isinstance(atom_map, dict)
        for atom_name, xyz in atom_map.items():
            atom_idx = atom_order[atom_name]
            positions[i, atom_idx, :] = np.asarray(xyz, dtype=np.float32)
            masks[i, atom_idx] = 1.0
    return seq, positions, masks


def _build_template_features(
    query_seq: str,
    template_pdb: str | None,
    template_chain: str | None,
    residue_constants,
    domain_name: str,
) -> Dict[str, np.ndarray]:
    num_res = len(query_seq)
    if not template_pdb:
        return {
            "template_aatype": np.zeros(
                (1, num_res, len(residue_constants.restypes_with_x_and_gap)),
                dtype=np.float32,
            ),
            "template_all_atom_masks": np.zeros(
                (1, num_res, residue_constants.atom_type_num), dtype=np.float32
            ),
            "template_all_atom_positions": np.zeros(
                (1, num_res, residue_constants.atom_type_num, 3), dtype=np.float32
            ),
            "template_domain_names": np.array(["".encode()], dtype=object),
            "template_sum_probs": np.array([0.0], dtype=np.float32),
        }

    tchain = (template_chain or "A").strip()
    if len(tchain) != 1:
        raise ValueError(f"Template chain must be exactly one character, got {template_chain!r}")
    template_seq, template_pos, template_mask = _parse_template_chain(template_pdb, tchain, residue_constants)
    mapping = _global_align_query_to_template(query_seq, template_seq)

    out_seq_chars = ["-"] * num_res
    out_pos = np.zeros((num_res, residue_constants.atom_type_num, 3), dtype=np.float32)
    out_mask = np.zeros((num_res, residue_constants.atom_type_num), dtype=np.float32)
    mapped = 0
    for qi, tj in enumerate(mapping):
        if tj <= 0:
            continue
        ti = int(tj - 1)
        out_seq_chars[qi] = template_seq[ti]
        out_pos[qi, :, :] = template_pos[ti, :, :]
        out_mask[qi, :] = template_mask[ti, :]
        mapped += 1
    print(
        f"  template map: {os.path.basename(template_pdb)} chain {tchain} "
        f"mapped {mapped}/{num_res}"
    )

    out_seq = "".join(out_seq_chars)
    template_aatype = residue_constants.sequence_to_onehot(
        out_seq, residue_constants.HHBLITS_AA_TO_ID
    ).astype(np.float32)
    return {
        "template_aatype": template_aatype[None, :, :],
        "template_all_atom_masks": out_mask[None, :, :].astype(np.float32),
        "template_all_atom_positions": out_pos[None, :, :, :].astype(np.float32),
        "template_domain_names": np.array([domain_name.encode()], dtype=object),
        "template_sum_probs": np.array([1.0], dtype=np.float32),
    }


def _split_template_group(spec: str | None) -> List[str]:
    if not spec:
        return []
    return [part.strip() for part in str(spec).split("+") if part.strip()]


def _build_template_features_group(
    query_seq: str,
    template_pdb_spec: str | None,
    template_chain_spec: str | None,
    residue_constants,
    domain_prefix: str,
) -> Dict[str, np.ndarray]:
    template_pdbs = _split_template_group(template_pdb_spec)
    template_chains = _split_template_group(template_chain_spec)
    if not template_pdbs:
        return _build_template_features(
            query_seq=query_seq,
            template_pdb=None,
            template_chain=None,
            residue_constants=residue_constants,
            domain_name=f"{domain_prefix}_none_A",
        )

    if not template_chains:
        template_chains = ["A"]
    if len(template_chains) == 1 and len(template_pdbs) > 1:
        template_chains = template_chains * len(template_pdbs)
    if len(template_chains) != len(template_pdbs):
        raise ValueError(
            "Template chain group count must match template pdb group count. "
            f"Got {len(template_chains)} chains for {len(template_pdbs)} templates."
        )

    template_aatype_rows: List[np.ndarray] = []
    template_mask_rows: List[np.ndarray] = []
    template_pos_rows: List[np.ndarray] = []
    domain_names: List[bytes] = []
    sum_probs: List[float] = []

    for ti, (pdb_path, chain_id) in enumerate(zip(template_pdbs, template_chains), start=1):
        if len(chain_id) != 1:
            raise ValueError(f"Template chain must be exactly one character, got {chain_id!r}")
        feat = _build_template_features(
            query_seq=query_seq,
            template_pdb=pdb_path,
            template_chain=chain_id,
            residue_constants=residue_constants,
            domain_name=f"{domain_prefix}_{ti}_{os.path.basename(pdb_path)}_{chain_id}",
        )
        template_aatype_rows.append(np.asarray(feat["template_aatype"], dtype=np.float32))
        template_mask_rows.append(np.asarray(feat["template_all_atom_masks"], dtype=np.float32))
        template_pos_rows.append(np.asarray(feat["template_all_atom_positions"], dtype=np.float32))
        domain_names.extend(np.asarray(feat["template_domain_names"], dtype=object).tolist())
        sum_probs.extend(np.asarray(feat["template_sum_probs"], dtype=np.float32).tolist())

    return {
        "template_aatype": np.concatenate(template_aatype_rows, axis=0),
        "template_all_atom_masks": np.concatenate(template_mask_rows, axis=0),
        "template_all_atom_positions": np.concatenate(template_pos_rows, axis=0),
        "template_domain_names": np.asarray(domain_names, dtype=object),
        "template_sum_probs": np.asarray(sum_probs, dtype=np.float32),
    }


def _parse_msa_from_a3m(msa_path: str, query_seq: str, parsers):
    with open(msa_path, "r", encoding="utf-8") as handle:
        a3m = handle.read()
    msa = parsers.parse_a3m(a3m)
    if not msa.sequences:
        raise ValueError(f"No sequences in MSA file: {msa_path}")
    if msa.sequences[0] != query_seq:
        sequences = [query_seq] + list(msa.sequences)
        deletion = [[0] * len(query_seq)] + [list(row) for row in msa.deletion_matrix]
        descriptions = ["query"] + list(msa.descriptions)
        return parsers.Msa(sequences=sequences, deletion_matrix=deletion, descriptions=descriptions)
    return msa


def _default_query_msa(query_seq: str, parsers):
    return parsers.Msa(
        sequences=[query_seq],
        deletion_matrix=[[0] * len(query_seq)],
        descriptions=["query"],
    )


def _build_monomer_raw_features(
    sequence: str,
    msa_file: str | None,
    template_pdb: str | None,
    template_chain: str | None,
    pipeline,
    parsers,
    residue_constants,
) -> Dict[str, np.ndarray]:
    sequence_features = pipeline.make_sequence_features(sequence, "query", len(sequence))
    msa_obj = _parse_msa_from_a3m(msa_file, sequence, parsers) if msa_file else _default_query_msa(sequence, parsers)
    msa_features = pipeline.make_msa_features([msa_obj])
    raw = dict(sequence_features)
    raw.update(msa_features)
    raw.update(
        _build_template_features(
            sequence,
            template_pdb=template_pdb,
            template_chain=template_chain,
            residue_constants=residue_constants,
            domain_name=f"{os.path.basename(template_pdb) if template_pdb else 'none'}_{template_chain or 'A'}",
        )
    )
    return raw


def _build_multimer_raw_features(
    sequences: Sequence[str],
    msa_files: Sequence[str],
    template_pdbs: Sequence[str],
    template_chains: Sequence[str],
    min_msa_rows: int,
    pipeline,
    pipeline_multimer,
    parsers,
    residue_constants,
) -> Dict[str, np.ndarray]:
    all_chain_features: Dict[str, Dict[str, np.ndarray]] = {}
    has_templates = len(template_pdbs) > 0
    for i, seq in enumerate(sequences):
        chain_id = chr(ord("A") + i)
        sequence_features = pipeline.make_sequence_features(seq, f"chain_{chain_id}", len(seq))
        msa_obj = (
            _parse_msa_from_a3m(msa_files[i], seq, parsers)
            if msa_files
            else _default_query_msa(seq, parsers)
        )
        msa_features = pipeline.make_msa_features([msa_obj])
        monomer_features = dict(sequence_features)
        monomer_features.update(msa_features)
        if has_templates:
            monomer_features.update(
                _build_template_features_group(
                    seq,
                    template_pdb_spec=template_pdbs[i],
                    template_chain_spec=template_chains[i],
                    residue_constants=residue_constants,
                    domain_prefix=f"chain_{chain_id}",
                )
            )
        else:
            monomer_features.update(
                _build_template_features(
                    seq,
                    template_pdb=None,
                    template_chain=None,
                    residue_constants=residue_constants,
                    domain_name="none",
                )
            )

        converted = pipeline_multimer.convert_monomer_features(monomer_features, chain_id=chain_id)
        converted["msa_all_seq"] = converted["msa"].copy()
        converted["deletion_matrix_int_all_seq"] = monomer_features["deletion_matrix_int"].copy()
        converted["msa_species_identifiers_all_seq"] = monomer_features["msa_species_identifiers"].copy()
        all_chain_features[chain_id] = converted

    all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)
    np_example = pipeline_multimer.feature_processing.pair_and_merge(all_chain_features)
    np_example = pipeline_multimer.pad_msa(np_example, int(min_msa_rows))
    return np_example


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sorted_chain_ids_from_pdb(pdb_path: str) -> List[str]:
    ids = []
    seen = set()
    for atom in _parse_pdb_atoms(pdb_path):
        if atom.chain not in seen:
            seen.add(atom.chain)
            ids.append(atom.chain)
    return ids


def _run_case_with_official_runner(
    case_name: str,
    case_info: Dict[str, object],
    monomer_params: str,
    multimer_params: str,
    out_pdb: str,
    seed: int,
    min_msa_rows_multimer: int,
    alphafold_repo: str,
) -> Dict[str, object]:
    sys.path.insert(0, alphafold_repo)
    from alphafold.common import protein  # pylint: disable=import-outside-toplevel
    from alphafold.common import residue_constants  # pylint: disable=import-outside-toplevel
    from alphafold.data import parsers  # pylint: disable=import-outside-toplevel
    from alphafold.data import pipeline  # pylint: disable=import-outside-toplevel
    from alphafold.data import pipeline_multimer  # pylint: disable=import-outside-toplevel
    from alphafold.model import config as af_config  # pylint: disable=import-outside-toplevel
    from alphafold.model import model as af_model  # pylint: disable=import-outside-toplevel
    from alphafold.model import utils  # pylint: disable=import-outside-toplevel

    model_kind = str(case_info["model"])
    sequence_arg = str(case_info["sequence_arg"])
    num_recycle = int(case_info["num_recycle"])
    msa_files = [str(x) for x in case_info.get("msa_files", [])]
    template_pdbs = [str(x) for x in case_info.get("template_pdbs", [])]
    template_chains = [str(x) for x in case_info.get("template_chains", [])]

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

        msa_file = msa_files[0] if msa_files else None
        template_pdb = template_pdbs[0] if template_pdbs else None
        template_chain = template_chains[0] if template_chains else "A"
        raw_features = _build_monomer_raw_features(
            sequence=sequence_arg,
            msa_file=msa_file,
            template_pdb=template_pdb,
            template_chain=template_chain,
            pipeline=pipeline,
            parsers=parsers,
            residue_constants=residue_constants,
        )
    elif model_kind == "multimer":
        params_path = multimer_params
        model_name = "model_1_multimer_v3"
        cfg = af_config.model_config(model_name)
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
        if len(seqs) < 2:
            raise ValueError(f"Multimer case {case_name} needs at least 2 sequences.")
        if msa_files and len(msa_files) != len(seqs):
            raise ValueError(f"Case {case_name}: msa_files count mismatch.")
        if has_templates:
            if len(template_pdbs) == 1 and len(seqs) > 1:
                template_pdbs = template_pdbs * len(seqs)
            if len(template_chains) == 1 and len(seqs) > 1:
                template_chains = template_chains * len(seqs)
            if len(template_pdbs) != len(seqs) or len(template_chains) != len(seqs):
                raise ValueError(f"Case {case_name}: template count mismatch.")
        raw_features = _build_multimer_raw_features(
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

    with open(params_path, "rb") as handle:
        flat = np.load(handle, allow_pickle=False)
        params = utils.flat_params_to_haiku(flat)

    model_runner = af_model.RunModel(cfg, params)
    processed_feature_dict = model_runner.process_features(raw_features, random_seed=seed)
    prediction_result = model_runner.predict(processed_feature_dict, random_seed=seed)

    plddt = np.asarray(prediction_result["plddt"], dtype=np.float32)
    b_factors = np.repeat(plddt[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=b_factors,
        remove_leading_feature_dimension=not model_runner.multimer_mode,
    )
    pdb_text = protein.to_pdb(unrelaxed_protein)
    with open(out_pdb, "w", encoding="utf-8") as handle:
        handle.write(pdb_text)

    geometry = _ca_geometry_from_pdb(out_pdb)
    geometry["mean_plddt"] = float(np.mean(plddt))
    return {
        "model": model_kind,
        "num_recycle": num_recycle,
        "sequence_arg": sequence_arg,
        "msa_files": msa_files,
        "template_pdbs": template_pdbs,
        "template_chains": template_chains,
        "sha256": _sha256(out_pdb),
        "observed_chain_ids": _sorted_chain_ids_from_pdb(out_pdb),
        "geometry": geometry,
        "params_path": params_path,
        "output_pdb": out_pdb,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphafold-repo", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suffix", default=".python_official")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-msa-rows-multimer", type=int, default=512)
    parser.add_argument("--cases", default="")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting existing .python_official reference files (disabled by default).",
    )
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.input_manifest)
    if manifest_path.lower().endswith(".toml"):
        import tomllib

        with open(manifest_path, "rb") as handle:
            input_manifest = tomllib.load(handle)
    elif manifest_path.lower().endswith(".json"):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            input_manifest = json.load(handle)
    else:
        raise ValueError(f"Unsupported manifest format: {manifest_path}")

    monomer_params = str(input_manifest["monomer_params"])
    multimer_params = str(input_manifest["multimer_params"])
    cases = dict(input_manifest["cases"])
    selected = {x.strip() for x in args.cases.split(",") if x.strip()} if args.cases else None

    os.makedirs(args.output_dir, exist_ok=True)
    out_cases: Dict[str, object] = {}
    ordered_case_names = list(cases.keys())
    for case_name in ordered_case_names:
        if selected is not None and case_name not in selected:
            continue
        info = cases[case_name]
        out_pdb = os.path.join(args.output_dir, f"{case_name}{args.suffix}.pdb")
        if os.path.exists(out_pdb) and not args.allow_overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing reference file: {out_pdb}. "
                "Pass --allow-overwrite to override."
            )
        print(f"Running official Python case: {case_name}")
        out_info = _run_case_with_official_runner(
            case_name=case_name,
            case_info=info,
            monomer_params=monomer_params,
            multimer_params=multimer_params,
            out_pdb=out_pdb,
            seed=args.seed,
            min_msa_rows_multimer=args.min_msa_rows_multimer,
            alphafold_repo=args.alphafold_repo,
        )
        g = out_info["geometry"]
        assert isinstance(g, dict)
        print(
            f"  wrote {out_pdb} | mean_pLDDT={g.get('mean_plddt', float('nan')):.4f} "
            f"| intra_outlier={g.get('ca_distance_intra_chain_outlier_fraction', float('nan')):.3f}"
        )
        out_cases[case_name] = out_info

    manifest_out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "scripts/regression/generate_python_reference_pdbs_legacy_do_not_use.py",
        "alphafold_repo": os.path.abspath(args.alphafold_repo),
        "input_manifest": os.path.abspath(args.input_manifest),
        "monomer_params": os.path.abspath(monomer_params),
        "multimer_params": os.path.abspath(multimer_params),
        "seed": int(args.seed),
        "min_msa_rows_multimer": int(args.min_msa_rows_multimer),
        "cases": out_cases,
    }
    manifest_path = os.path.join(args.output_dir, "manifest_python_official.json")
    if os.path.exists(manifest_path) and not args.allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing manifest: {manifest_path}. "
            "Pass --allow-overwrite to override."
        )
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_out, handle, indent=2, sort_keys=True)
    print(f"Saved Python-official manifest: {manifest_path}")


if __name__ == "__main__":
    main()
