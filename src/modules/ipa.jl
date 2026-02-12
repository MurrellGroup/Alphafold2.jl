using NNlib

# IPA layer types are defined in Onion.jl (shared with ESMFold).
# Import them here — GPU overrides (flash IPA) are in Onion.jl/gpu_layers.jl.
using Onion: PointProjection, PointProjectionMultimer
using Onion: InvariantPointAttention, MultimerInvariantPointAttention

# ============================================================================
# Weight loading (AF2-specific key conventions)
# ============================================================================

function _copy_point_projection_af2!(m::PointProjection, arrs::AbstractDict, prefix::String)
    _copy_linear_af2!(m.linear, arrs, prefix)
    return m
end

function _copy_point_projection_af2!(m::PointProjectionMultimer, arrs::AbstractDict, prefix::String)
    _copy_linear_af2!(m.linear, arrs, prefix)
    return m
end

"""
    load_invariant_point_attention_npz!(m, npz_path)

Load AF2 InvariantPointAttention parameters saved by
`scripts/parity/dump_ipa_py.py`.
"""
function load_invariant_point_attention_npz!(m::InvariantPointAttention, params_source)
    arrs = af2_params_read(params_source)

    _copy_linear_af2!(m.linear_q, arrs, "q_scalar")
    _copy_point_projection_af2!(m.linear_q_points, arrs, "q_point_local")
    _copy_linear_af2!(m.linear_kv, arrs, "kv_scalar")
    _copy_point_projection_af2!(m.linear_kv_points, arrs, "kv_point_local")
    _copy_linear_af2!(m.linear_b, arrs, "attention_2d")

    m.head_weights .= arrs["trainable_point_weights"]

    _copy_linear_af2!(m.linear_out, arrs, "output_projection")

    return m
end

function load_invariant_point_attention_npz!(m::MultimerInvariantPointAttention, params_source)
    arrs = af2_params_read(params_source)

    _copy_linear_af2!(m.linear_q, arrs, "q_scalar_projection")
    _copy_linear_af2!(m.linear_k, arrs, "k_scalar_projection")
    _copy_linear_af2!(m.linear_v, arrs, "v_scalar_projection")
    _copy_point_projection_af2!(m.linear_q_points, arrs, "q_point_projection/point_projection")
    _copy_point_projection_af2!(m.linear_k_points, arrs, "k_point_projection/point_projection")
    _copy_point_projection_af2!(m.linear_v_points, arrs, "v_point_projection/point_projection")
    _copy_linear_af2!(m.linear_b, arrs, "attention_2d")

    m.head_weights .= arrs["trainable_point_weights"]

    _copy_linear_af2!(m.linear_out, arrs, "output_projection")

    return m
end
