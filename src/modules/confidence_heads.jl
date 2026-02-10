@concrete struct PredictedLDDTHead <: Onion.Layer
    input_layer_norm
    act_0
    act_1
    logits
end

@layer PredictedLDDTHead

function PredictedLDDTHead(c_s::Int; num_channels::Int=128, num_bins::Int=50)
    return PredictedLDDTHead(
        LayerNormFirst(c_s),
        LinearFirst(c_s, num_channels),
        LinearFirst(num_channels, num_channels),
        LinearFirst(num_channels, num_bins),
    )
end

function (m::PredictedLDDTHead)(structure_module_repr::AbstractArray)
    # structure_module_repr: (C_s, L, B)
    act = m.input_layer_norm(structure_module_repr)
    act = max.(m.act_0(act), 0f0)
    act = max.(m.act_1(act), 0f0)
    logits = m.logits(act)
    return Dict{Symbol,Any}(:logits => logits)
end

@concrete struct PredictedAlignedErrorHead <: Onion.Layer
    logits
    num_bins::Int
    max_error_bin::Float32
end

@layer PredictedAlignedErrorHead

function PredictedAlignedErrorHead(c_z::Int; num_bins::Int=64, max_error_bin::Real=31.0f0)
    return PredictedAlignedErrorHead(
        LinearFirst(c_z, num_bins),
        num_bins,
        Float32(max_error_bin),
    )
end

function (m::PredictedAlignedErrorHead)(pair_repr::AbstractArray)
    # pair_repr: (C_z, L, L, B)
    logits = m.logits(pair_repr) # (num_bins, L, L, B)
    breaks = collect(range(0f0, m.max_error_bin; length=m.num_bins - 1))
    breaks = to_device(breaks, logits, Float32)
    return Dict{Symbol,Any}(:logits => logits, :breaks => breaks)
end


function load_predicted_lddt_head_npz!(
    m::PredictedLDDTHead,
    params_source;
    prefix::AbstractString="alphafold/alphafold_iteration/predicted_lddt_head",
)
    arrs = af2_params_read(params_source)

    m.input_layer_norm.w .= _get_arr(arrs, string(prefix, "/input_layer_norm//scale"))
    m.input_layer_norm.b .= _get_arr(arrs, string(prefix, "/input_layer_norm//offset"))

    m.act_0.weight .= permutedims(_get_arr(arrs, string(prefix, "/act_0//weights")), (2, 1))
    m.act_0.bias .= _get_arr(arrs, string(prefix, "/act_0//bias"))
    m.act_1.weight .= permutedims(_get_arr(arrs, string(prefix, "/act_1//weights")), (2, 1))
    m.act_1.bias .= _get_arr(arrs, string(prefix, "/act_1//bias"))
    m.logits.weight .= permutedims(_get_arr(arrs, string(prefix, "/logits//weights")), (2, 1))
    m.logits.bias .= _get_arr(arrs, string(prefix, "/logits//bias"))

    return m
end

function load_predicted_aligned_error_head_npz!(
    m::PredictedAlignedErrorHead,
    params_source;
    prefix::AbstractString="alphafold/alphafold_iteration/predicted_aligned_error_head",
)
    arrs = af2_params_read(params_source)
    m.logits.weight .= permutedims(_get_arr(arrs, string(prefix, "/logits//weights")), (2, 1))
    m.logits.bias .= _get_arr(arrs, string(prefix, "/logits//bias"))
    return m
end
