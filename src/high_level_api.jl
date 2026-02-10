const DEFAULT_MONOMER_WEIGHTS = "alphafold2_model_1_ptm_dm_2022-12-06.safetensors"
const DEFAULT_MULTIMER_WEIGHTS = "alphafold2_model_1_multimer_v3_dm_2022-12-06.safetensors"

# AF2Config and AF2Model are defined in model_builder.jl.
# AF2InferenceResult, _infer, _write_fold_pdb, _write_fold_npz are in inference.jl.

Base.@kwdef struct FoldResult
    out_npz::String
    out_pdb::String
    mean_plddt::Float32
    min_plddt::Float32
    max_plddt::Float32
    mean_pae::Union{Nothing,Float32}=nothing
    ptm::Union{Nothing,Float32}=nothing
end

const _DEFAULT_MODEL = Ref{Union{Nothing,AF2Model}}(nothing)

@inline function _repo_root()
    return normpath(joinpath(@__DIR__, ".."))
end

@inline function _scalar_f32(x)
    return x isa AbstractArray ? Float32(x[]) : Float32(x)
end

function _as_string_vector(x)
    if x === nothing
        return String[]
    elseif x isa AbstractString
        s = String(strip(x))
        isempty(s) && return String[]
        return [String(strip(v)) for v in split(s, ",")]
    elseif x isa AbstractVector
        return [String(strip(String(v))) for v in x]
    else
        error("Expected String, Vector{String}, or nothing; got $(typeof(x))")
    end
end

function _load_model(kind::Symbol, filename::AbstractString; device=identity, repo_id::AbstractString=AF2_HF_REPO_ID, revision::AbstractString=AF2_HF_REVISION, cache::Bool=true, local_files_only::Bool=false, set_default::Bool=true)
    params_path = resolve_af2_params_path(
        filename;
        repo_id=repo_id,
        revision=revision,
        cache=cache,
        local_files_only=local_files_only,
    )
    arrs = af2_params_read(params_path)
    model = _build_af2_model(arrs)
    if model.config.kind != kind
        error("Model checkpoint is $(model.config.kind) but was loaded via load_$(kind)(). Use load_$(model.config.kind)() instead.")
    end
    if device !== identity
        if device === Flux.gpu
            CUDA.math_mode!(CUDA.FAST_MATH)
        end
        model = device(model)
    end
    if set_default
        _DEFAULT_MODEL[] = model
    end
    return model
end

function load_monomer(; filename::AbstractString=DEFAULT_MONOMER_WEIGHTS, device=identity, repo_id::AbstractString=AF2_HF_REPO_ID, revision::AbstractString=AF2_HF_REVISION, cache::Bool=true, local_files_only::Bool=false, set_default::Bool=true)
    return _load_model(
        :monomer,
        filename;
        device=device,
        repo_id=repo_id,
        revision=revision,
        cache=cache,
        local_files_only=local_files_only,
        set_default=set_default,
    )
end

function load_multimer(; filename::AbstractString=DEFAULT_MULTIMER_WEIGHTS, device=identity, repo_id::AbstractString=AF2_HF_REPO_ID, revision::AbstractString=AF2_HF_REVISION, cache::Bool=true, local_files_only::Bool=false, set_default::Bool=true)
    return _load_model(
        :multimer,
        filename;
        device=device,
        repo_id=repo_id,
        revision=revision,
        cache=cache,
        local_files_only=local_files_only,
        set_default=set_default,
    )
end

function _require_default_model()
    model = _DEFAULT_MODEL[]
    model === nothing && error("No default model is loaded. Call load_monomer() or load_multimer() first, or use fold(model, ...).")
    return model
end

function _compute_out_paths(out_prefix, kind::Symbol)
    if out_prefix === nothing
        run_dir = mktempdir(; prefix=kind == :monomer ? "af2_mono_" : "af2_multi_")
        base = joinpath(run_dir, "fold")
    else
        base = abspath(String(out_prefix))
        mkpath(dirname(base))
    end
    return string(base, "_input.npz"), string(base, "_out.npz")
end

function _run_builder_script(script_path::AbstractString, args::Vector{String})
    proj = _repo_root()
    cmd = `$(Base.julia_cmd()) --startup-file=no --history-file=no --project=$proj $script_path $args`
    run(cmd)
    return nothing
end

function fold(
    model::AF2Model,
    sequence::AbstractString;
    msas=nothing,
    templates=nothing,
    template_chains=nothing,
    num_recycle::Integer=(model.config.kind == :multimer ? 5 : 3),
    pairing_mode::AbstractString="block diagonal",
    pairing_seed::Integer=0,
    out_prefix=nothing,
)
    if model.config.kind == :multimer
        seqs = [strip(s) for s in split(sequence, ",") if !isempty(strip(s))]
        length(seqs) >= 2 || error("Multimer fold expected comma-separated sequences with at least 2 chains.")
        return fold(
            model,
            seqs;
            msas=msas,
            templates=templates,
            template_chains=template_chains,
            num_recycle=num_recycle,
            pairing_mode=pairing_mode,
            pairing_seed=pairing_seed,
            out_prefix=out_prefix,
        )
    end

    occursin(",", sequence) && error("Monomer fold received comma-separated sequence. Use a monomer sequence without commas.")
    msa_vec = _as_string_vector(msas)
    length(msa_vec) <= 1 || error("Monomer fold accepts at most one MSA file.")
    template_vec = _as_string_vector(templates)
    chain_vec = _as_string_vector(template_chains)
    isempty(chain_vec) || length(chain_vec) == length(template_vec) || error("template_chains count must match templates count for monomer fold.")

    input_npz, out_npz = _compute_out_paths(out_prefix, :monomer)
    builder = joinpath(_repo_root(), "scripts", "end_to_end", "build_monomer_input_jl.jl")
    args = String[sequence, input_npz, string(Int(num_recycle))]
    has_msa = !isempty(msa_vec)
    has_templates = !isempty(template_vec)
    if has_msa || has_templates
        push!(args, has_msa ? msa_vec[1] : "")
        if has_templates
            push!(args, join(template_vec, ","))
            push!(args, isempty(chain_vec) ? "A" : join(chain_vec, ","))
        end
    end
    _run_builder_script(builder, args)

    features = NPZ.npzread(input_npz)
    result = _infer(model, features; num_recycle=Int(num_recycle))

    out_pdb = _infer_pdb_path_from_npz(out_npz)
    _write_fold_pdb(out_pdb, result)
    _write_fold_npz(out_npz, result)

    mean_pae = result.pae !== nothing ? Float32(mean(result.pae)) : nothing

    return FoldResult(
        out_npz=String(out_npz),
        out_pdb=out_pdb,
        mean_plddt=Float32(mean(result.plddt)),
        min_plddt=Float32(minimum(result.plddt)),
        max_plddt=Float32(maximum(result.plddt)),
        mean_pae=mean_pae,
        ptm=result.ptm,
    )
end

function fold(
    model::AF2Model,
    sequences::AbstractVector{<:AbstractString};
    msas=nothing,
    templates=nothing,
    template_chains=nothing,
    num_recycle::Integer=(model.config.kind == :multimer ? 5 : 3),
    pairing_mode::AbstractString="block diagonal",
    pairing_seed::Integer=0,
    out_prefix=nothing,
)
    model.config.kind == :multimer || error("Vector-of-sequences fold is for multimer models. Load with load_multimer().")
    seqs = [uppercase(strip(s)) for s in sequences if !isempty(strip(s))]
    length(seqs) >= 2 || error("Multimer fold requires at least 2 chain sequences.")

    msa_vec = _as_string_vector(msas)
    isempty(msa_vec) || length(msa_vec) == length(seqs) || error("msas count must match chain count for multimer fold.")
    template_vec = _as_string_vector(templates)
    chain_vec = _as_string_vector(template_chains)
    isempty(chain_vec) || length(chain_vec) == length(seqs) || error("template_chains count must match chain count for multimer fold.")
    isempty(template_vec) || length(template_vec) == length(seqs) || error("templates count must match chain count for multimer fold.")

    input_npz, out_npz = _compute_out_paths(out_prefix, :multimer)
    builder = joinpath(_repo_root(), "scripts", "end_to_end", "build_multimer_input_jl.jl")
    args = String[join(String.(seqs), ","), input_npz, string(Int(num_recycle))]
    has_msa = !isempty(msa_vec)
    has_templates = !isempty(template_vec)
    if has_msa || has_templates
        push!(args, has_msa ? join(msa_vec, ",") : "")
        if has_templates
            push!(args, join(template_vec, ","))
            push!(args, join(chain_vec, ","))
        end
    end
    if pairing_mode != "block diagonal" || Int(pairing_seed) != 0
        push!(args, pairing_mode)
        push!(args, string(Int(pairing_seed)))
    end
    _run_builder_script(builder, args)

    features = NPZ.npzread(input_npz)
    result = _infer(model, features; num_recycle=Int(num_recycle))

    out_pdb = _infer_pdb_path_from_npz(out_npz)
    _write_fold_pdb(out_pdb, result)
    _write_fold_npz(out_npz, result)

    mean_pae = result.pae !== nothing ? Float32(mean(result.pae)) : nothing

    return FoldResult(
        out_npz=String(out_npz),
        out_pdb=out_pdb,
        mean_plddt=Float32(mean(result.plddt)),
        min_plddt=Float32(minimum(result.plddt)),
        max_plddt=Float32(maximum(result.plddt)),
        mean_pae=mean_pae,
        ptm=result.ptm,
    )
end

function fold(sequence_or_sequences; kwargs...)
    return fold(_require_default_model(), sequence_or_sequences; kwargs...)
end
