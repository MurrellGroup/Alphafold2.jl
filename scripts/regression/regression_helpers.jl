# Regression test helper functions for Alphafold2.jl
#
# Provides:
#   - In-process feature building + inference using pre-built AF2Model
#   - PDB parsing, coordinate comparison, geometry metrics
#   - Clash detection for structural quality validation
#
# Usage: include() this file after regression_cases.jl. Requires Alphafold2 module loaded.

using NPZ
using Statistics

function _regression_pdb_path_from_npz(npz_path::AbstractString)
    if endswith(lowercase(npz_path), ".npz")
        return string(first(npz_path, lastindex(npz_path) - 4), ".pdb")
    end
    return string(npz_path, ".pdb")
end

function _csv_or_empty(values::Vector{String})
    return isempty(values) ? "" : join(values, ",")
end

# In-process regression using pre-built AF2Model (fast: no layer reconstruction, no subprocess)
function run_inprocess_regression_case(
    model,  # AF2Model
    case::NamedTuple,
    workdir::AbstractString;
    repo_root::AbstractString=normpath(joinpath(@__DIR__, "..", "..")),
)
    mkpath(workdir)
    input_npz = joinpath(workdir, string(case.name, "_input.npz"))
    out_npz = joinpath(workdir, string(case.name, "_out.npz"))

    features = if case.model == :monomer
        Alphafold2.build_monomer_features(
            case.sequence_arg;
            num_recycle=Int(case.num_recycle),
            msa_file=isempty(case.msa_files) ? "" : case.msa_files[1],
            template_pdb_arg=_csv_or_empty(case.template_pdbs),
            template_chain_arg=isempty(case.template_chains) ? "A" : _csv_or_empty(case.template_chains),
        )
    else
        seqs = [uppercase(strip(s)) for s in split(case.sequence_arg, ",") if !isempty(strip(s))]
        Alphafold2.build_multimer_features(
            seqs;
            num_recycle=Int(case.num_recycle),
            msa_files=isempty(case.msa_files) ? String[] : case.msa_files,
            template_pdb_arg=_csv_or_empty(case.template_pdbs),
            template_chain_arg=_csv_or_empty(case.template_chains),
        )
    end

    NPZ.npzwrite(input_npz, features)
    result = Alphafold2._infer(model, features; num_recycle=Int(case.num_recycle))

    out_pdb = _regression_pdb_path_from_npz(out_npz)
    Alphafold2._write_fold_pdb(out_pdb, result)
    Alphafold2._write_fold_npz(out_npz, result)

    return (input_npz=input_npz, out_npz=out_npz, out_pdb=out_pdb)
end

function parse_pdb_atoms(pdb_path::AbstractString)
    atoms = NamedTuple[]
    open(pdb_path, "r") do io
        for raw in eachline(io)
            startswith(raw, "ATOM") || continue
            line = rpad(raw, 80)
            atom_name = strip(line[13:16])
            resname = strip(line[18:20])
            chain = line[22]
            resseq = parse(Int, strip(line[23:26]))
            x = parse(Float64, strip(line[31:38]))
            y = parse(Float64, strip(line[39:46]))
            z = parse(Float64, strip(line[47:54]))
            push!(atoms, (atom=atom_name, resname=resname, chain=chain, resseq=resseq, x=x, y=y, z=z))
        end
    end
    return atoms
end

function pdb_chain_ids(pdb_path::AbstractString)
    atoms = parse_pdb_atoms(pdb_path)
    return unique(a.chain for a in atoms)
end

function compare_pdb_coordinates(actual_path::AbstractString, reference_path::AbstractString)
    a = parse_pdb_atoms(actual_path)
    b = parse_pdb_atoms(reference_path)
    length(a) == length(b) || return Dict{Symbol,Any}(:ok => false, :reason => "atom_count", :a => length(a), :b => length(b))

    max_abs = 0.0
    sum_sq = 0.0
    sum_abs = 0.0
    mismatch = 0
    ncoord = 0
    for i in eachindex(a)
        ai = a[i]
        bi = b[i]
        if ai.atom != bi.atom || ai.resname != bi.resname || ai.chain != bi.chain || ai.resseq != bi.resseq
            mismatch += 1
            continue
        end
        dx = abs(ai.x - bi.x)
        dy = abs(ai.y - bi.y)
        dz = abs(ai.z - bi.z)
        max_abs = max(max_abs, dx, dy, dz)
        sum_abs += dx + dy + dz
        sum_sq += dx * dx + dy * dy + dz * dz
        ncoord += 3
    end

    if mismatch > 0
        return Dict{Symbol,Any}(:ok => false, :reason => "atom_identity", :mismatch => mismatch)
    end

    mean_abs = ncoord == 0 ? 0.0 : sum_abs / ncoord
    rms = ncoord == 0 ? 0.0 : sqrt(sum_sq / ncoord)
    return Dict{Symbol,Any}(
        :ok => true,
        :max_abs => max_abs,
        :mean_abs => mean_abs,
        :rms => rms,
        :atom_count => length(a),
    )
end

function npz_geometry_metrics(npz_path::AbstractString)
    arr = NPZ.npzread(npz_path)
    metrics = Dict{String,Any}()
    for key in (
        "ca_distance_mean",
        "ca_distance_std",
        "ca_distance_min",
        "ca_distance_max",
        "ca_distance_outlier_fraction",
        "ca_distance_intra_chain_mean",
        "ca_distance_intra_chain_std",
        "ca_distance_intra_chain_min",
        "ca_distance_intra_chain_max",
        "ca_distance_intra_chain_outlier_fraction",
        "mean_plddt",
    )
        if haskey(arr, key)
            v = arr[key]
            metrics[key] = v isa AbstractArray ? Float64(v[]) : Float64(v)
        end
    end
    return metrics
end

"""
    check_clashes(pdb_path; thresh=2.0) -> (count, worst_distance)

Count atomic clashes in a PDB file. A clash is defined as two atoms from
non-adjacent residues (sequence gap > 1) closer than `thresh` Angstroms.

Returns a named tuple `(count=N, worst=d)` where `worst` is the shortest
offending distance (Inf if no clashes).

Expected results for healthy structures:
  - All multimer cases: 0 clashes
  - Monomer seq_only (9 res): ≤1 clash (marginal, short sequence)
  - Monomer template_msa (29 res): ≤3 clashes (short sequence)
  - Any case with >5 clashes or worst < 1.5 Å indicates a serious problem
"""
function check_clashes(pdb_path::AbstractString; thresh::Float64=2.0)
    atoms = parse_pdb_atoms(pdb_path)
    n = 0
    worst = Inf
    for i in 1:length(atoms), j in (i+1):length(atoms)
        abs(atoms[i].resseq - atoms[j].resseq) <= 1 && continue
        d = sqrt((atoms[i].x - atoms[j].x)^2 +
                 (atoms[i].y - atoms[j].y)^2 +
                 (atoms[i].z - atoms[j].z)^2)
        if d < thresh
            n += 1
            worst = min(worst, d)
        end
    end
    return (count=n, worst=worst)
end
