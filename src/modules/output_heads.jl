@concrete struct MaskedMsaHead <: Onion.Layer
    logits
end

@layer MaskedMsaHead

function MaskedMsaHead(c_m::Int; num_output::Int=23)
    return MaskedMsaHead(
        LinearFirst(c_m, num_output),
    )
end

function (m::MaskedMsaHead)(msa_repr::AbstractArray)
    # msa_repr: (C_m, N_seq, N_res, B)
    logits = m.logits(msa_repr) # (num_output, N_seq, N_res, B)
    return Dict{Symbol,Any}(:logits => logits)
end

@concrete struct DistogramHead <: Onion.Layer
    half_logits
    num_bins::Int
    first_break::Float32
    last_break::Float32
end

@layer DistogramHead

function DistogramHead(c_z::Int; num_bins::Int=64, first_break::Real=2.3125f0, last_break::Real=21.6875f0)
    return DistogramHead(
        LinearFirst(c_z, num_bins),
        num_bins,
        Float32(first_break),
        Float32(last_break),
    )
end

function (m::DistogramHead)(pair_repr::AbstractArray)
    # pair_repr: (C_z, N_res, N_res, B)
    half_logits = m.half_logits(pair_repr) # (num_bins, N_res, N_res, B)
    logits = half_logits .+ permutedims(half_logits, (1, 3, 2, 4))
    breaks = collect(range(m.first_break, m.last_break; length=m.num_bins - 1))
    breaks = to_device(breaks, logits, Float32)
    return Dict{Symbol,Any}(:logits => logits, :bin_edges => breaks)
end

@concrete struct ExperimentallyResolvedHead <: Onion.Layer
    logits
end

@layer ExperimentallyResolvedHead

function ExperimentallyResolvedHead(c_s::Int)
    return ExperimentallyResolvedHead(
        LinearFirst(c_s, 37),
    )
end

function (m::ExperimentallyResolvedHead)(single_repr::AbstractArray)
    # single_repr: (C_s, N_res, B)
    logits = m.logits(single_repr) # (37, N_res, B)
    return Dict{Symbol,Any}(:logits => logits)
end

@inline function _output_head_get_arr(arrs::AbstractDict, key::AbstractString)
    if haskey(arrs, key)
        return arrs[key]
    end
    alt = replace(key, "//" => "/")
    if haskey(arrs, alt)
        return arrs[alt]
    end
    error("Missing key in NPZ: $(key)")
end

function load_masked_msa_head_npz!(
    m::MaskedMsaHead,
    params_source;
    prefix::AbstractString="alphafold/alphafold_iteration/masked_msa_head",
)
    arrs = af2_params_read(params_source)
    m.logits.weight .= permutedims(_output_head_get_arr(arrs, string(prefix, "/logits//weights")), (2, 1))
    m.logits.bias .= _output_head_get_arr(arrs, string(prefix, "/logits//bias"))
    return m
end

function load_distogram_head_npz!(
    m::DistogramHead,
    params_source;
    prefix::AbstractString="alphafold/alphafold_iteration/distogram_head",
)
    arrs = af2_params_read(params_source)
    m.half_logits.weight .= permutedims(_output_head_get_arr(arrs, string(prefix, "/half_logits//weights")), (2, 1))
    m.half_logits.bias .= _output_head_get_arr(arrs, string(prefix, "/half_logits//bias"))
    return m
end

function load_experimentally_resolved_head_npz!(
    m::ExperimentallyResolvedHead,
    params_source;
    prefix::AbstractString="alphafold/alphafold_iteration/experimentally_resolved_head",
)
    arrs = af2_params_read(params_source)
    m.logits.weight .= permutedims(_output_head_get_arr(arrs, string(prefix, "/logits//weights")), (2, 1))
    m.logits.bias .= _output_head_get_arr(arrs, string(prefix, "/logits//bias"))
    return m
end
