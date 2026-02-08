@concrete struct FoldIterationCore <: Onion.Layer
    ipa
    attention_layer_norm
    transition_layers
    transition_layer_norm
    affine_update
end

@layer FoldIterationCore

function FoldIterationCore(
    c_s::Int,
    c_z::Int,
    c_hidden::Int,
    no_heads::Int,
    no_qk_points::Int,
    no_v_points::Int,
    num_transition_layers::Int,
    ;
    multimer_ipa::Bool=false,
)
    ipa = multimer_ipa ?
        MultimerInvariantPointAttention(c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points) :
        InvariantPointAttention(c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points)
    attention_layer_norm = LayerNormFirst(c_s)
    transition_layers = [LinearFirst(c_s, c_s) for _ in 1:num_transition_layers]
    transition_layer_norm = LayerNormFirst(c_s)
    affine_update = LinearFirst(c_s, 6)

    return FoldIterationCore(
        ipa,
        attention_layer_norm,
        transition_layers,
        transition_layer_norm,
        affine_update,
    )
end

function (m::FoldIterationCore)(act::AbstractArray, z::AbstractArray, mask::AbstractArray, rigids::Rigid)
    # act: (C_s, L, B)
    # z: (C_z, L, L, B)
    # mask: (L, B)
    act = act .+ m.ipa(act, z, rigids, mask)
    act = m.attention_layer_norm(act)

    input_act = act
    for i in eachindex(m.transition_layers)
        act = m.transition_layers[i](act)
        if i < length(m.transition_layers)
            act = max.(act, 0f0)
        end
    end
    act = act .+ input_act
    act = m.transition_layer_norm(act)

    affine_update = m.affine_update(act)
    new_rigids = compose_q_update_vec(rigids, affine_update)

    return act, new_rigids, affine_update
end

function (m::FoldIterationCore)(act::AbstractArray, z::AbstractArray, mask::AbstractArray)
    rigids = rigid_identity((size(act, 2), size(act, 3)), act; fmt=:quat)
    return m(act, z, mask, rigids)
end

"""
    load_fold_iteration_core_npz!(m, npz_path)

Load AF2 FoldIteration-core parameters saved by
`scripts/parity/dump_fold_iteration_core_from_params_py.py`.
"""
function load_fold_iteration_core_npz!(m::FoldIterationCore, params_source)
    arrs = af2_params_read(params_source)

    if m.ipa isa MultimerInvariantPointAttention
        _copy_linear_af2!(m.ipa.linear_q, arrs, "q_scalar_projection")
        _copy_linear_af2!(m.ipa.linear_k, arrs, "k_scalar_projection")
        _copy_linear_af2!(m.ipa.linear_v, arrs, "v_scalar_projection")
        _copy_linear_af2!(m.ipa.linear_q_points.linear, arrs, "q_point_projection/point_projection")
        _copy_linear_af2!(m.ipa.linear_k_points.linear, arrs, "k_point_projection/point_projection")
        _copy_linear_af2!(m.ipa.linear_v_points.linear, arrs, "v_point_projection/point_projection")
        _copy_linear_af2!(m.ipa.linear_b, arrs, "attention_2d")
        m.ipa.head_weights .= arrs["trainable_point_weights"]
        _copy_linear_af2!(m.ipa.linear_out, arrs, "output_projection")
    else
        _copy_linear_af2!(m.ipa.linear_q, arrs, "q_scalar")
        _copy_linear_af2!(m.ipa.linear_q_points.linear, arrs, "q_point_local")
        _copy_linear_af2!(m.ipa.linear_kv, arrs, "kv_scalar")
        _copy_linear_af2!(m.ipa.linear_kv_points.linear, arrs, "kv_point_local")
        _copy_linear_af2!(m.ipa.linear_b, arrs, "attention_2d")
        m.ipa.head_weights .= arrs["trainable_point_weights"]
        _copy_linear_af2!(m.ipa.linear_out, arrs, "output_projection")
    end

    _copy_ln_af2!(m.attention_layer_norm, arrs, "attention_layer_norm")
    _copy_ln_af2!(m.transition_layer_norm, arrs, "transition_layer_norm")
    _copy_linear_af2!(m.affine_update, arrs, "affine_update")

    for i in eachindex(m.transition_layers)
        prefix = i == 1 ? "transition" : "transition_$(i-1)"
        _copy_linear_af2!(m.transition_layers[i], arrs, prefix)
    end

    return m
end
