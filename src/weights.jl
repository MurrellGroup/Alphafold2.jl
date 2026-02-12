const AF2_HF_REPO_ID = "MurrellLab/AlphaFold2.jl"
const AF2_HF_REVISION = "main"

function _looks_like_local_path(spec::AbstractString)
    return startswith(spec, ".") || startswith(spec, "/") || startswith(spec, "~") || occursin('\\', spec) || occursin('/', spec)
end

function _is_safetensors(path::AbstractString)
    return endswith(lowercase(path), ".safetensors")
end

function _env_bool(name::AbstractString, default::Bool)
    raw = lowercase(strip(get(ENV, name, default ? "true" : "false")))
    return raw in ("1", "true", "yes", "on")
end

function af2_params_read(path::AbstractString)
    isfile(path) || error("AF2 params file not found: $(path)")
    try
        reader = ProtSafeTensors.Reader(path)
        out = Dict{String,Any}()
        for key in keys(reader.header)
            t = ProtSafeTensors.read_tensor(reader, key)
            out[key] = t isa Number ? t : Array(t)
        end
        return out
    catch err
        error("Failed to load AF2 params as safetensors from $(path): $(err)")
    end
end

function af2_params_read(arrs::AbstractDict)
    return arrs
end

function _get_arr(arrs::AbstractDict, key::AbstractString)
    haskey(arrs, key) && return arrs[key]
    alt = replace(key, "//" => "/")
    haskey(arrs, alt) && return arrs[alt]
    error("Missing key: $(key)")
end

function _has_arr_key(arrs::AbstractDict, key::AbstractString)
    return haskey(arrs, key) || haskey(arrs, replace(key, "//" => "/"))
end

function resolve_af2_params_path(
    spec::AbstractString;
    repo_id::AbstractString = AF2_HF_REPO_ID,
    revision::AbstractString = AF2_HF_REVISION,
    cache::Bool = true,
    local_files_only::Bool = _env_bool("AF2_HF_LOCAL_FILES_ONLY", false),
)
    cleaned = strip(spec)
    isempty(cleaned) && error("Empty AF2 params spec. Provide a local .safetensors path or HF safetensors filename.")

    if isfile(cleaned)
        # HF cache artifact paths are extensionless; enforce safetensors at read time.
        return abspath(cleaned)
    end
    if _looks_like_local_path(cleaned)
        error("AF2 params local path not found: $(cleaned)")
    end

    _is_safetensors(cleaned) ||
        error("AF2 params HF filename must end with .safetensors, got: $(cleaned)")

    path = hf_hub_download(
        repo_id,
        cleaned;
        revision = revision,
        cache = cache,
        local_files_only = local_files_only,
    )
    isfile(path) || error("Resolved HF params path is not a file: $(path)")
    return path
end
