using NPZ
using Statistics

function _julia_cmd(script_path::AbstractString, args::Vector{String}; project::AbstractString="")
    if isempty(project)
        return `$(Base.julia_cmd()) --startup-file=no --history-file=no $script_path $args`
    else
        return `$(Base.julia_cmd()) --startup-file=no --history-file=no --project=$project $script_path $args`
    end
end

function _regression_pdb_path_from_npz(npz_path::AbstractString)
    if endswith(lowercase(npz_path), ".npz")
        return string(first(npz_path, lastindex(npz_path) - 4), ".pdb")
    end
    return string(npz_path, ".pdb")
end

function _csv_or_empty(values::Vector{String})
    return isempty(values) ? "" : join(values, ",")
end

function _build_features_subprocess(
    repo_root::AbstractString,
    case::NamedTuple,
    input_npz::AbstractString,
)
    build_script = if case.model == :monomer
        joinpath(repo_root, "scripts", "end_to_end", "build_monomer_input_jl.jl")
    elseif case.model == :multimer
        joinpath(repo_root, "scripts", "end_to_end", "build_multimer_input_jl.jl")
    else
        error("Unsupported regression case model $(case.model) for $(case.name)")
    end

    build_args = if case.model == :monomer
        args = String[
            case.sequence_arg,
            input_npz,
            string(case.num_recycle),
        ]
        has_msa = !isempty(case.msa_files)
        has_templates = !isempty(case.template_pdbs)
        if has_msa || has_templates
            push!(args, has_msa ? case.msa_files[1] : "")
            if has_templates
                push!(args, _csv_or_empty(case.template_pdbs))
                push!(args, _csv_or_empty(case.template_chains))
            end
        end
        args
    else
        args = String[
            case.sequence_arg,
            input_npz,
            string(case.num_recycle),
        ]
        has_msa = !isempty(case.msa_files)
        has_templates = !isempty(case.template_pdbs)
        if has_msa || has_templates
            push!(args, _csv_or_empty(case.msa_files))
            if has_templates
                push!(args, _csv_or_empty(case.template_pdbs))
                push!(args, _csv_or_empty(case.template_chains))
            end
        end
        args
    end

    run(_julia_cmd(build_script, build_args; project=repo_root))
    return input_npz
end

# Legacy subprocess-based regression (runs builder + runner as separate processes)
function run_pure_julia_regression_case(
    repo_root::AbstractString,
    case::NamedTuple,
    params_path::AbstractString,
    workdir::AbstractString,
)
    mkpath(workdir)
    input_npz = joinpath(workdir, string(case.name, "_input.npz"))
    out_npz = joinpath(workdir, string(case.name, "_out.npz"))

    _build_features_subprocess(repo_root, case, input_npz)

    run_script = joinpath(repo_root, "scripts", "end_to_end", "run_af2_template_hybrid_jl.jl")
    run(_julia_cmd(run_script, [params_path, input_npz, out_npz]; project=repo_root))

    out_pdb = _regression_pdb_path_from_npz(out_npz)
    return (input_npz=input_npz, out_npz=out_npz, out_pdb=out_pdb)
end

# In-process regression using pre-built AF2Model (fast: no layer reconstruction)
function run_inprocess_regression_case(
    model,  # AF2Model
    case::NamedTuple,
    workdir::AbstractString;
    repo_root::AbstractString=normpath(joinpath(@__DIR__, "..", "..")),
)
    mkpath(workdir)
    input_npz = joinpath(workdir, string(case.name, "_input.npz"))
    out_npz = joinpath(workdir, string(case.name, "_out.npz"))

    _build_features_subprocess(repo_root, case, input_npz)

    features = NPZ.npzread(input_npz)
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
