@concrete struct TemplatePairStackBlock <: Onion.Layer
    triangle_attention_starting_node
    triangle_attention_ending_node
    triangle_multiplication_outgoing
    triangle_multiplication_incoming
    pair_transition
end

@layer TemplatePairStackBlock

function TemplatePairStackBlock(c_t::Int, num_head_pair::Int, pair_head_dim::Int, c_tri_mul::Int, pair_transition_factor::Real)
    return TemplatePairStackBlock(
        TriangleAttention(c_t, num_head_pair, pair_head_dim; orientation=:per_row),
        TriangleAttention(c_t, num_head_pair, pair_head_dim; orientation=:per_column),
        TriangleMultiplication(c_t, c_tri_mul; outgoing=true),
        TriangleMultiplication(c_t, c_tri_mul; outgoing=false),
        Transition(c_t, pair_transition_factor),
    )
end

function (m::TemplatePairStackBlock)(pair_act::AbstractArray, pair_mask::AbstractArray)
    pair_act = pair_act .+ m.triangle_attention_starting_node(pair_act, pair_mask)
    pair_act = pair_act .+ m.triangle_attention_ending_node(pair_act, pair_mask)
    pair_act = pair_act .+ m.triangle_multiplication_outgoing(pair_act, pair_mask)
    pair_act = pair_act .+ m.triangle_multiplication_incoming(pair_act, pair_mask)
    pair_act = pair_act .+ m.pair_transition(pair_act, pair_mask)
    return pair_act
end

@concrete struct TemplatePairStack <: Onion.Layer
    blocks
end

@layer TemplatePairStack

function TemplatePairStack(
    c_t::Int,
    num_block::Int;
    num_head_pair::Int,
    pair_head_dim::Int,
    c_tri_mul::Int,
    pair_transition_factor::Real=2.0,
)
    blocks = [TemplatePairStackBlock(c_t, num_head_pair, pair_head_dim, c_tri_mul, pair_transition_factor) for _ in 1:num_block]
    return TemplatePairStack(blocks)
end

function (m::TemplatePairStack)(pair_act::AbstractArray, pair_mask::AbstractArray)
    for blk in m.blocks
        pair_act = blk(pair_act, pair_mask)
    end
    return pair_act
end

@inline function _slice_leading_block(arr::AbstractArray, block_idx::Int, num_blocks::Int)
    if ndims(arr) > 0 && size(arr, 1) == num_blocks
        return dropdims(view(arr, block_idx:block_idx, ntuple(_ -> Colon(), ndims(arr) - 1)...); dims=1)
    end
    return arr
end

@inline function _get_arr(arrs::AbstractDict, key::AbstractString)
    if haskey(arrs, key)
        return arrs[key]
    end
    alt = replace(key, "//" => "/")
    if haskey(arrs, alt)
        return arrs[alt]
    end
    error("Missing key in NPZ: $(key)")
end

function _load_linear_raw_block!(lin::LinearFirst, arrs::AbstractDict, base::AbstractString, block_idx::Int, num_blocks::Int)
    w = _slice_leading_block(_get_arr(arrs, string(base, "//weights")), block_idx, num_blocks)
    lin.weight .= permutedims(w, (2, 1))
    if lin.use_bias
        b = _slice_leading_block(_get_arr(arrs, string(base, "//bias")), block_idx, num_blocks)
        lin.bias .= b
    end
    return lin
end

function _load_ln_raw_block!(ln::LayerNormFirst, arrs::AbstractDict, base::AbstractString, block_idx::Int, num_blocks::Int)
    ln.w .= _slice_leading_block(_get_arr(arrs, string(base, "//scale")), block_idx, num_blocks)
    ln.b .= _slice_leading_block(_get_arr(arrs, string(base, "//offset")), block_idx, num_blocks)
    return ln
end

function _load_attention_raw_block!(att::AF2Attention, arrs::AbstractDict, base::AbstractString, block_idx::Int, num_blocks::Int)
    att.query_w .= _slice_leading_block(_get_arr(arrs, string(base, "//query_w")), block_idx, num_blocks)
    att.key_w .= _slice_leading_block(_get_arr(arrs, string(base, "//key_w")), block_idx, num_blocks)
    att.value_w .= _slice_leading_block(_get_arr(arrs, string(base, "//value_w")), block_idx, num_blocks)
    att.output_w .= _slice_leading_block(_get_arr(arrs, string(base, "//output_w")), block_idx, num_blocks)
    att.output_b .= _slice_leading_block(_get_arr(arrs, string(base, "//output_b")), block_idx, num_blocks)
    if att.gating
        att.gating_w .= _slice_leading_block(_get_arr(arrs, string(base, "//gating_w")), block_idx, num_blocks)
        att.gating_b .= _slice_leading_block(_get_arr(arrs, string(base, "//gating_b")), block_idx, num_blocks)
    end
    return att
end

"""
    load_template_pair_stack_npz!(m, npz_path; prefix=...)

Load AF2 template pair stack parameters from a real checkpoint NPZ.
By default this reads:
`alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state`.
"""
function load_template_pair_stack_npz!(
    m::TemplatePairStack,
    npz_path::AbstractString;
    prefix::AbstractString="alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state",
)
    arrs = NPZ.npzread(npz_path)
    num_blocks = length(m.blocks)

    for (i, blk) in enumerate(m.blocks)
        _load_ln_raw_block!(blk.triangle_attention_starting_node.query_norm, arrs, string(prefix, "/triangle_attention_starting_node/query_norm"), i, num_blocks)
        blk.triangle_attention_starting_node.feat_2d_weights .= _slice_leading_block(_get_arr(arrs, string(prefix, "/triangle_attention_starting_node//feat_2d_weights")), i, num_blocks)
        _load_attention_raw_block!(blk.triangle_attention_starting_node.attention, arrs, string(prefix, "/triangle_attention_starting_node/attention"), i, num_blocks)

        _load_ln_raw_block!(blk.triangle_attention_ending_node.query_norm, arrs, string(prefix, "/triangle_attention_ending_node/query_norm"), i, num_blocks)
        blk.triangle_attention_ending_node.feat_2d_weights .= _slice_leading_block(_get_arr(arrs, string(prefix, "/triangle_attention_ending_node//feat_2d_weights")), i, num_blocks)
        _load_attention_raw_block!(blk.triangle_attention_ending_node.attention, arrs, string(prefix, "/triangle_attention_ending_node/attention"), i, num_blocks)

        _load_ln_raw_block!(blk.triangle_multiplication_outgoing.layer_norm_input, arrs, string(prefix, "/triangle_multiplication_outgoing/layer_norm_input"), i, num_blocks)
        _load_linear_raw_block!(blk.triangle_multiplication_outgoing.left_projection, arrs, string(prefix, "/triangle_multiplication_outgoing/left_projection"), i, num_blocks)
        _load_linear_raw_block!(blk.triangle_multiplication_outgoing.right_projection, arrs, string(prefix, "/triangle_multiplication_outgoing/right_projection"), i, num_blocks)
        _load_linear_raw_block!(blk.triangle_multiplication_outgoing.left_gate, arrs, string(prefix, "/triangle_multiplication_outgoing/left_gate"), i, num_blocks)
        _load_linear_raw_block!(blk.triangle_multiplication_outgoing.right_gate, arrs, string(prefix, "/triangle_multiplication_outgoing/right_gate"), i, num_blocks)
        _load_ln_raw_block!(blk.triangle_multiplication_outgoing.center_layer_norm, arrs, string(prefix, "/triangle_multiplication_outgoing/center_layer_norm"), i, num_blocks)
        _load_linear_raw_block!(blk.triangle_multiplication_outgoing.output_projection, arrs, string(prefix, "/triangle_multiplication_outgoing/output_projection"), i, num_blocks)
        _load_linear_raw_block!(blk.triangle_multiplication_outgoing.gating_linear, arrs, string(prefix, "/triangle_multiplication_outgoing/gating_linear"), i, num_blocks)

        _load_ln_raw_block!(blk.triangle_multiplication_incoming.layer_norm_input, arrs, string(prefix, "/triangle_multiplication_incoming/layer_norm_input"), i, num_blocks)
        _load_linear_raw_block!(blk.triangle_multiplication_incoming.left_projection, arrs, string(prefix, "/triangle_multiplication_incoming/left_projection"), i, num_blocks)
        _load_linear_raw_block!(blk.triangle_multiplication_incoming.right_projection, arrs, string(prefix, "/triangle_multiplication_incoming/right_projection"), i, num_blocks)
        _load_linear_raw_block!(blk.triangle_multiplication_incoming.left_gate, arrs, string(prefix, "/triangle_multiplication_incoming/left_gate"), i, num_blocks)
        _load_linear_raw_block!(blk.triangle_multiplication_incoming.right_gate, arrs, string(prefix, "/triangle_multiplication_incoming/right_gate"), i, num_blocks)
        _load_ln_raw_block!(blk.triangle_multiplication_incoming.center_layer_norm, arrs, string(prefix, "/triangle_multiplication_incoming/center_layer_norm"), i, num_blocks)
        _load_linear_raw_block!(blk.triangle_multiplication_incoming.output_projection, arrs, string(prefix, "/triangle_multiplication_incoming/output_projection"), i, num_blocks)
        _load_linear_raw_block!(blk.triangle_multiplication_incoming.gating_linear, arrs, string(prefix, "/triangle_multiplication_incoming/gating_linear"), i, num_blocks)

        _load_ln_raw_block!(blk.pair_transition.input_layer_norm, arrs, string(prefix, "/pair_transition/input_layer_norm"), i, num_blocks)
        _load_linear_raw_block!(blk.pair_transition.transition1, arrs, string(prefix, "/pair_transition/transition1"), i, num_blocks)
        _load_linear_raw_block!(blk.pair_transition.transition2, arrs, string(prefix, "/pair_transition/transition2"), i, num_blocks)
    end

    return m
end
