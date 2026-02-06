using NPZ
using Printf
using Statistics
using NNlib

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

const EVO_PREFIX = "alphafold/alphafold_iteration/evoformer"
const EVO_BLOCK_PREFIX = "alphafold/alphafold_iteration/evoformer/evoformer_iteration"
const EXTRA_BLOCK_PREFIX = "alphafold/alphafold_iteration/evoformer/extra_msa_stack"
const SM_PREFIX = "alphafold/alphafold_iteration/structure_module"

@inline function _has_key(arrs::AbstractDict, key::AbstractString)
    return haskey(arrs, key)
end

@inline function _arr_get(arrs::AbstractDict, key::AbstractString)
    if haskey(arrs, key)
        return arrs[key]
    end
    alt = replace(key, "//" => "/")
    if haskey(arrs, alt)
        return arrs[alt]
    end
    error("Missing key: $(key)")
end

@inline function _slice_block(arr::AbstractArray, block_idx::Int)
    return dropdims(view(arr, block_idx:block_idx, ntuple(_ -> Colon(), ndims(arr) - 1)...); dims=1)
end

function _load_linear_raw!(
    lin::LinearFirst,
    arrs::AbstractDict,
    base::AbstractString;
    block_idx::Union{Nothing,Int}=nothing,
    split::Symbol=:full,
)
    wkey = string(base, "//weights")
    bkey = string(base, "//bias")

    wfull = block_idx === nothing ? _arr_get(arrs, wkey) : _slice_block(_arr_get(arrs, wkey), block_idx)
    w = if split === :full
        wfull
    elseif split === :first
        @view wfull[:, 1:Int(div(size(wfull, 2), 2))]
    elseif split === :second
        @view wfull[:, Int(div(size(wfull, 2), 2)) + 1:size(wfull, 2)]
    else
        error("Unsupported split mode: $(split)")
    end
    lin.weight .= permutedims(w, (2, 1))
    if lin.use_bias
        bfull = block_idx === nothing ? _arr_get(arrs, bkey) : _slice_block(_arr_get(arrs, bkey), block_idx)
        b = if split === :full
            bfull
        elseif split === :first
            @view bfull[1:Int(div(length(bfull), 2))]
        elseif split === :second
            @view bfull[Int(div(length(bfull), 2)) + 1:length(bfull)]
        else
            error("Unsupported split mode: $(split)")
        end
        lin.bias .= b
    end
    return lin
end

function _load_ln_raw!(ln::LayerNormFirst, arrs::AbstractDict, base::AbstractString; block_idx::Union{Nothing,Int}=nothing)
    skey = string(base, "//scale")
    okey = string(base, "//offset")
    s = block_idx === nothing ? _arr_get(arrs, skey) : _slice_block(_arr_get(arrs, skey), block_idx)
    o = block_idx === nothing ? _arr_get(arrs, okey) : _slice_block(_arr_get(arrs, okey), block_idx)
    ln.w .= s
    ln.b .= o
    return ln
end

function _load_attention_raw!(att::AF2Attention, arrs::AbstractDict, base::AbstractString; block_idx::Union{Nothing,Int}=nothing)
    g = key -> (block_idx === nothing ? _arr_get(arrs, key) : _slice_block(_arr_get(arrs, key), block_idx))
    att.query_w .= g(string(base, "//query_w"))
    att.key_w .= g(string(base, "//key_w"))
    att.value_w .= g(string(base, "//value_w"))
    att.output_w .= g(string(base, "//output_w"))
    att.output_b .= g(string(base, "//output_b"))
    if att.gating
        att.gating_w .= g(string(base, "//gating_w"))
        att.gating_b .= g(string(base, "//gating_b"))
    end
    return att
end

function _load_global_attention_raw!(att::AF2GlobalAttention, arrs::AbstractDict, base::AbstractString; block_idx::Union{Nothing,Int}=nothing)
    g = key -> (block_idx === nothing ? _arr_get(arrs, key) : _slice_block(_arr_get(arrs, key), block_idx))
    att.query_w .= g(string(base, "//query_w"))
    att.key_w .= g(string(base, "//key_w"))
    att.value_w .= g(string(base, "//value_w"))
    att.output_w .= g(string(base, "//output_w"))
    att.output_b .= g(string(base, "//output_b"))
    if att.gating
        att.gating_w .= g(string(base, "//gating_w"))
        att.gating_b .= g(string(base, "//gating_b"))
    end
    return att
end

function _load_evo_block_raw!(blk::EvoformerIteration, arrs::AbstractDict, bi::Int)
    p = EVO_BLOCK_PREFIX
    _load_ln_raw!(blk.outer_product_mean.layer_norm_input, arrs, string(p, "/outer_product_mean/layer_norm_input"); block_idx=bi)
    _load_linear_raw!(blk.outer_product_mean.left_projection, arrs, string(p, "/outer_product_mean/left_projection"); block_idx=bi)
    _load_linear_raw!(blk.outer_product_mean.right_projection, arrs, string(p, "/outer_product_mean/right_projection"); block_idx=bi)
    blk.outer_product_mean.output_w .= _slice_block(arrs[string(p, "/outer_product_mean//output_w")], bi)
    blk.outer_product_mean.output_b .= _slice_block(arrs[string(p, "/outer_product_mean//output_b")], bi)

    _load_ln_raw!(blk.msa_row_attention_with_pair_bias.query_norm, arrs, string(p, "/msa_row_attention_with_pair_bias/query_norm"); block_idx=bi)
    _load_ln_raw!(blk.msa_row_attention_with_pair_bias.feat_2d_norm, arrs, string(p, "/msa_row_attention_with_pair_bias/feat_2d_norm"); block_idx=bi)
    blk.msa_row_attention_with_pair_bias.feat_2d_weights .= _slice_block(arrs[string(p, "/msa_row_attention_with_pair_bias//feat_2d_weights")], bi)
    _load_attention_raw!(blk.msa_row_attention_with_pair_bias.attention, arrs, string(p, "/msa_row_attention_with_pair_bias/attention"); block_idx=bi)
    _load_ln_raw!(blk.msa_column_attention.query_norm, arrs, string(p, "/msa_column_attention/query_norm"); block_idx=bi)
    _load_attention_raw!(blk.msa_column_attention.attention, arrs, string(p, "/msa_column_attention/attention"); block_idx=bi)
    _load_ln_raw!(blk.msa_transition.input_layer_norm, arrs, string(p, "/msa_transition/input_layer_norm"); block_idx=bi)
    _load_linear_raw!(blk.msa_transition.transition1, arrs, string(p, "/msa_transition/transition1"); block_idx=bi)
    _load_linear_raw!(blk.msa_transition.transition2, arrs, string(p, "/msa_transition/transition2"); block_idx=bi)

    tri_out_base = string(p, "/triangle_multiplication_outgoing")
    if _has_key(arrs, string(tri_out_base, "/left_projection//weights"))
        _load_ln_raw!(blk.triangle_multiplication_outgoing.layer_norm_input, arrs, string(tri_out_base, "/layer_norm_input"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_outgoing.left_projection, arrs, string(tri_out_base, "/left_projection"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_outgoing.right_projection, arrs, string(tri_out_base, "/right_projection"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_outgoing.left_gate, arrs, string(tri_out_base, "/left_gate"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_outgoing.right_gate, arrs, string(tri_out_base, "/right_gate"); block_idx=bi)
        _load_ln_raw!(blk.triangle_multiplication_outgoing.center_layer_norm, arrs, string(tri_out_base, "/center_layer_norm"); block_idx=bi)
    else
        _load_ln_raw!(blk.triangle_multiplication_outgoing.layer_norm_input, arrs, string(tri_out_base, "/left_norm_input"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_outgoing.left_projection, arrs, string(tri_out_base, "/projection"); block_idx=bi, split=:first)
        _load_linear_raw!(blk.triangle_multiplication_outgoing.right_projection, arrs, string(tri_out_base, "/projection"); block_idx=bi, split=:second)
        _load_linear_raw!(blk.triangle_multiplication_outgoing.left_gate, arrs, string(tri_out_base, "/gate"); block_idx=bi, split=:first)
        _load_linear_raw!(blk.triangle_multiplication_outgoing.right_gate, arrs, string(tri_out_base, "/gate"); block_idx=bi, split=:second)
        _load_ln_raw!(blk.triangle_multiplication_outgoing.center_layer_norm, arrs, string(tri_out_base, "/center_norm"); block_idx=bi)
    end
    _load_linear_raw!(blk.triangle_multiplication_outgoing.output_projection, arrs, string(p, "/triangle_multiplication_outgoing/output_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.gating_linear, arrs, string(p, "/triangle_multiplication_outgoing/gating_linear"); block_idx=bi)

    tri_in_base = string(p, "/triangle_multiplication_incoming")
    if _has_key(arrs, string(tri_in_base, "/left_projection//weights"))
        _load_ln_raw!(blk.triangle_multiplication_incoming.layer_norm_input, arrs, string(tri_in_base, "/layer_norm_input"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_incoming.left_projection, arrs, string(tri_in_base, "/left_projection"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_incoming.right_projection, arrs, string(tri_in_base, "/right_projection"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_incoming.left_gate, arrs, string(tri_in_base, "/left_gate"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_incoming.right_gate, arrs, string(tri_in_base, "/right_gate"); block_idx=bi)
        _load_ln_raw!(blk.triangle_multiplication_incoming.center_layer_norm, arrs, string(tri_in_base, "/center_layer_norm"); block_idx=bi)
    else
        _load_ln_raw!(blk.triangle_multiplication_incoming.layer_norm_input, arrs, string(tri_in_base, "/left_norm_input"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_incoming.left_projection, arrs, string(tri_in_base, "/projection"); block_idx=bi, split=:first)
        _load_linear_raw!(blk.triangle_multiplication_incoming.right_projection, arrs, string(tri_in_base, "/projection"); block_idx=bi, split=:second)
        _load_linear_raw!(blk.triangle_multiplication_incoming.left_gate, arrs, string(tri_in_base, "/gate"); block_idx=bi, split=:first)
        _load_linear_raw!(blk.triangle_multiplication_incoming.right_gate, arrs, string(tri_in_base, "/gate"); block_idx=bi, split=:second)
        _load_ln_raw!(blk.triangle_multiplication_incoming.center_layer_norm, arrs, string(tri_in_base, "/center_norm"); block_idx=bi)
    end
    _load_linear_raw!(blk.triangle_multiplication_incoming.output_projection, arrs, string(p, "/triangle_multiplication_incoming/output_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_incoming.gating_linear, arrs, string(p, "/triangle_multiplication_incoming/gating_linear"); block_idx=bi)

    _load_ln_raw!(blk.triangle_attention_starting_node.query_norm, arrs, string(p, "/triangle_attention_starting_node/query_norm"); block_idx=bi)
    blk.triangle_attention_starting_node.feat_2d_weights .= _slice_block(arrs[string(p, "/triangle_attention_starting_node//feat_2d_weights")], bi)
    _load_attention_raw!(blk.triangle_attention_starting_node.attention, arrs, string(p, "/triangle_attention_starting_node/attention"); block_idx=bi)
    _load_ln_raw!(blk.triangle_attention_ending_node.query_norm, arrs, string(p, "/triangle_attention_ending_node/query_norm"); block_idx=bi)
    blk.triangle_attention_ending_node.feat_2d_weights .= _slice_block(arrs[string(p, "/triangle_attention_ending_node//feat_2d_weights")], bi)
    _load_attention_raw!(blk.triangle_attention_ending_node.attention, arrs, string(p, "/triangle_attention_ending_node/attention"); block_idx=bi)
    _load_ln_raw!(blk.pair_transition.input_layer_norm, arrs, string(p, "/pair_transition/input_layer_norm"); block_idx=bi)
    _load_linear_raw!(blk.pair_transition.transition1, arrs, string(p, "/pair_transition/transition1"); block_idx=bi)
    _load_linear_raw!(blk.pair_transition.transition2, arrs, string(p, "/pair_transition/transition2"); block_idx=bi)
    return blk
end

function _load_extra_block_raw!(blk::EvoformerIteration, arrs::AbstractDict, bi::Int)
    p = EXTRA_BLOCK_PREFIX
    _load_ln_raw!(blk.outer_product_mean.layer_norm_input, arrs, string(p, "/outer_product_mean/layer_norm_input"); block_idx=bi)
    _load_linear_raw!(blk.outer_product_mean.left_projection, arrs, string(p, "/outer_product_mean/left_projection"); block_idx=bi)
    _load_linear_raw!(blk.outer_product_mean.right_projection, arrs, string(p, "/outer_product_mean/right_projection"); block_idx=bi)
    blk.outer_product_mean.output_w .= _slice_block(arrs[string(p, "/outer_product_mean//output_w")], bi)
    blk.outer_product_mean.output_b .= _slice_block(arrs[string(p, "/outer_product_mean//output_b")], bi)
    _load_ln_raw!(blk.msa_row_attention_with_pair_bias.query_norm, arrs, string(p, "/msa_row_attention_with_pair_bias/query_norm"); block_idx=bi)
    _load_ln_raw!(blk.msa_row_attention_with_pair_bias.feat_2d_norm, arrs, string(p, "/msa_row_attention_with_pair_bias/feat_2d_norm"); block_idx=bi)
    blk.msa_row_attention_with_pair_bias.feat_2d_weights .= _slice_block(arrs[string(p, "/msa_row_attention_with_pair_bias//feat_2d_weights")], bi)
    _load_attention_raw!(blk.msa_row_attention_with_pair_bias.attention, arrs, string(p, "/msa_row_attention_with_pair_bias/attention"); block_idx=bi)
    _load_ln_raw!(blk.msa_column_attention.query_norm, arrs, string(p, "/msa_column_global_attention/query_norm"); block_idx=bi)
    _load_global_attention_raw!(blk.msa_column_attention.attention, arrs, string(p, "/msa_column_global_attention/attention"); block_idx=bi)
    _load_ln_raw!(blk.msa_transition.input_layer_norm, arrs, string(p, "/msa_transition/input_layer_norm"); block_idx=bi)
    _load_linear_raw!(blk.msa_transition.transition1, arrs, string(p, "/msa_transition/transition1"); block_idx=bi)
    _load_linear_raw!(blk.msa_transition.transition2, arrs, string(p, "/msa_transition/transition2"); block_idx=bi)
    tri_out_base = string(p, "/triangle_multiplication_outgoing")
    if _has_key(arrs, string(tri_out_base, "/left_projection//weights"))
        _load_ln_raw!(blk.triangle_multiplication_outgoing.layer_norm_input, arrs, string(tri_out_base, "/layer_norm_input"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_outgoing.left_projection, arrs, string(tri_out_base, "/left_projection"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_outgoing.right_projection, arrs, string(tri_out_base, "/right_projection"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_outgoing.left_gate, arrs, string(tri_out_base, "/left_gate"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_outgoing.right_gate, arrs, string(tri_out_base, "/right_gate"); block_idx=bi)
        _load_ln_raw!(blk.triangle_multiplication_outgoing.center_layer_norm, arrs, string(tri_out_base, "/center_layer_norm"); block_idx=bi)
    else
        _load_ln_raw!(blk.triangle_multiplication_outgoing.layer_norm_input, arrs, string(tri_out_base, "/left_norm_input"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_outgoing.left_projection, arrs, string(tri_out_base, "/projection"); block_idx=bi, split=:first)
        _load_linear_raw!(blk.triangle_multiplication_outgoing.right_projection, arrs, string(tri_out_base, "/projection"); block_idx=bi, split=:second)
        _load_linear_raw!(blk.triangle_multiplication_outgoing.left_gate, arrs, string(tri_out_base, "/gate"); block_idx=bi, split=:first)
        _load_linear_raw!(blk.triangle_multiplication_outgoing.right_gate, arrs, string(tri_out_base, "/gate"); block_idx=bi, split=:second)
        _load_ln_raw!(blk.triangle_multiplication_outgoing.center_layer_norm, arrs, string(tri_out_base, "/center_norm"); block_idx=bi)
    end
    _load_linear_raw!(blk.triangle_multiplication_outgoing.output_projection, arrs, string(p, "/triangle_multiplication_outgoing/output_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.gating_linear, arrs, string(p, "/triangle_multiplication_outgoing/gating_linear"); block_idx=bi)
    tri_in_base = string(p, "/triangle_multiplication_incoming")
    if _has_key(arrs, string(tri_in_base, "/left_projection//weights"))
        _load_ln_raw!(blk.triangle_multiplication_incoming.layer_norm_input, arrs, string(tri_in_base, "/layer_norm_input"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_incoming.left_projection, arrs, string(tri_in_base, "/left_projection"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_incoming.right_projection, arrs, string(tri_in_base, "/right_projection"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_incoming.left_gate, arrs, string(tri_in_base, "/left_gate"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_incoming.right_gate, arrs, string(tri_in_base, "/right_gate"); block_idx=bi)
        _load_ln_raw!(blk.triangle_multiplication_incoming.center_layer_norm, arrs, string(tri_in_base, "/center_layer_norm"); block_idx=bi)
    else
        _load_ln_raw!(blk.triangle_multiplication_incoming.layer_norm_input, arrs, string(tri_in_base, "/left_norm_input"); block_idx=bi)
        _load_linear_raw!(blk.triangle_multiplication_incoming.left_projection, arrs, string(tri_in_base, "/projection"); block_idx=bi, split=:first)
        _load_linear_raw!(blk.triangle_multiplication_incoming.right_projection, arrs, string(tri_in_base, "/projection"); block_idx=bi, split=:second)
        _load_linear_raw!(blk.triangle_multiplication_incoming.left_gate, arrs, string(tri_in_base, "/gate"); block_idx=bi, split=:first)
        _load_linear_raw!(blk.triangle_multiplication_incoming.right_gate, arrs, string(tri_in_base, "/gate"); block_idx=bi, split=:second)
        _load_ln_raw!(blk.triangle_multiplication_incoming.center_layer_norm, arrs, string(tri_in_base, "/center_norm"); block_idx=bi)
    end
    _load_linear_raw!(blk.triangle_multiplication_incoming.output_projection, arrs, string(p, "/triangle_multiplication_incoming/output_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_incoming.gating_linear, arrs, string(p, "/triangle_multiplication_incoming/gating_linear"); block_idx=bi)
    _load_ln_raw!(blk.triangle_attention_starting_node.query_norm, arrs, string(p, "/triangle_attention_starting_node/query_norm"); block_idx=bi)
    blk.triangle_attention_starting_node.feat_2d_weights .= _slice_block(arrs[string(p, "/triangle_attention_starting_node//feat_2d_weights")], bi)
    _load_attention_raw!(blk.triangle_attention_starting_node.attention, arrs, string(p, "/triangle_attention_starting_node/attention"); block_idx=bi)
    _load_ln_raw!(blk.triangle_attention_ending_node.query_norm, arrs, string(p, "/triangle_attention_ending_node/query_norm"); block_idx=bi)
    blk.triangle_attention_ending_node.feat_2d_weights .= _slice_block(arrs[string(p, "/triangle_attention_ending_node//feat_2d_weights")], bi)
    _load_attention_raw!(blk.triangle_attention_ending_node.attention, arrs, string(p, "/triangle_attention_ending_node/attention"); block_idx=bi)
    _load_ln_raw!(blk.pair_transition.input_layer_norm, arrs, string(p, "/pair_transition/input_layer_norm"); block_idx=bi)
    _load_linear_raw!(blk.pair_transition.transition1, arrs, string(p, "/pair_transition/transition1"); block_idx=bi)
    _load_linear_raw!(blk.pair_transition.transition2, arrs, string(p, "/pair_transition/transition2"); block_idx=bi)
    return blk
end

function _one_hot_aatype(aatype::AbstractVector{Int}, num_classes::Int)
    L = length(aatype)
    out = zeros(Float32, num_classes, L, 1)
    for i in 1:L
        idx = clamp(aatype[i], 0, num_classes - 1)
        out[idx + 1, i, 1] = 1f0
    end
    return out
end

function _build_target_and_msa_feat(aatype::AbstractVector{Int}; target_dim::Int=22)
    L = length(aatype)
    target_feat = if target_dim == 22
        cat(zeros(Float32, 1, L, 1), _one_hot_aatype(aatype, 21); dims=1) # (22,L,1)
    elseif target_dim == 21
        _one_hot_aatype(aatype, 21) # (21,L,1)
    else
        error("Unsupported target feature dim $(target_dim); expected 21 or 22")
    end

    msa_1hot = zeros(Float32, 23, 1, L, 1)
    for i in 1:L
        idx = clamp(aatype[i], 0, 22)
        msa_1hot[idx + 1, 1, i, 1] = 1f0
    end
    msa_feat = cat(msa_1hot, zeros(Float32, 1, 1, L, 1), zeros(Float32, 1, 1, L, 1), copy(msa_1hot), zeros(Float32, 1, 1, L, 1); dims=1) # (49,1,L,1)
    residue_index = collect(0:(L - 1))
    return target_feat, msa_feat, residue_index
end

function _deletion_value_transform(x::AbstractArray)
    return atan.(Float32.(x) ./ 3f0) .* (2f0 / Float32(π))
end

function _build_target_and_msa_feat(
    aatype::AbstractVector{Int},
    msa_int::AbstractArray,
    deletion_matrix::Union{Nothing,AbstractArray}=nothing,
    ;
    target_dim::Int=22,
)
    L = length(aatype)
    S = size(msa_int, 1)
    size(msa_int, 2) == L || error("msa length mismatch: expected L=$(L), got $(size(msa_int, 2))")

    target_feat = if target_dim == 22
        cat(zeros(Float32, 1, L, 1), _one_hot_aatype(aatype, 21); dims=1) # (22,L,1)
    elseif target_dim == 21
        _one_hot_aatype(aatype, 21) # (21,L,1)
    else
        error("Unsupported target feature dim $(target_dim); expected 21 or 22")
    end

    msa_1hot = zeros(Float32, 23, S, L, 1)
    for s in 1:S, i in 1:L
        idx = clamp(Int(msa_int[s, i]), 0, 22)
        msa_1hot[idx + 1, s, i, 1] = 1f0
    end

    del = deletion_matrix === nothing ? zeros(Float32, S, L) : Float32.(deletion_matrix)
    has_del = reshape(Float32.(del .> 0f0), 1, S, L, 1)
    del_val = reshape(_deletion_value_transform(del), 1, S, L, 1)
    del_mean = reshape(_deletion_value_transform(del), 1, S, L, 1)

    # Feature order matches AF2 create_msa_feature usage in embeddings:
    # [msa_1hot, has_deletion, deletion_value, cluster_profile, deletion_mean]
    msa_feat = cat(msa_1hot, has_del, del_val, copy(msa_1hot), del_mean; dims=1) # (49,S,L,1)
    residue_index = collect(0:(L - 1))
    return target_feat, msa_feat, residue_index
end

function _build_extra_msa_feat(aatype::AbstractVector{Int})
    L = length(aatype)
    msa_1hot = zeros(Float32, 23, 1, L, 1)
    for i in 1:L
        idx = clamp(aatype[i], 0, 22)
        msa_1hot[idx + 1, 1, i, 1] = 1f0
    end
    return cat(msa_1hot, zeros(Float32, 1, 1, L, 1), zeros(Float32, 1, 1, L, 1); dims=1) # (25,1,L,1)
end

function _build_extra_msa_feat(msa_int::AbstractArray, deletion_matrix::Union{Nothing,AbstractArray}=nothing)
    S = size(msa_int, 1)
    L = size(msa_int, 2)
    msa_1hot = zeros(Float32, 23, S, L, 1)
    for s in 1:S, i in 1:L
        idx = clamp(Int(msa_int[s, i]), 0, 22)
        msa_1hot[idx + 1, s, i, 1] = 1f0
    end
    del = deletion_matrix === nothing ? zeros(Float32, S, L) : Float32.(deletion_matrix)
    has_del = reshape(Float32.(del .> 0f0), 1, S, L, 1)
    del_val = reshape(_deletion_value_transform(del), 1, S, L, 1)
    return cat(msa_1hot, has_del, del_val; dims=1) # (25,S,L,1)
end

function _relpos_one_hot(residue_index::AbstractVector{Int}, max_relative_feature::Int)
    L = length(residue_index)
    out = zeros(Float32, 2 * max_relative_feature + 1, L, L, 1)
    for i in 1:L, j in 1:L
        idx = clamp((residue_index[i] - residue_index[j]) + max_relative_feature, 0, 2 * max_relative_feature)
        out[idx + 1, i, j, 1] = 1f0
    end
    return out
end

function _multimer_relpos_features(
    residue_index::AbstractVector{Int},
    asym_id::AbstractVector{Int},
    entity_id::AbstractVector{Int},
    sym_id::AbstractVector{Int},
    max_relative_idx::Int,
    max_relative_chain::Int,
)
    L = length(residue_index)
    rel_pos_dim = 2 * max_relative_idx + 2
    rel_chain_dim = 2 * max_relative_chain + 2
    out_dim = rel_pos_dim + 1 + rel_chain_dim
    out = zeros(Float32, out_dim, L, L, 1)
    for i in 1:L, j in 1:L
        asym_same = asym_id[i] == asym_id[j]
        entity_same = entity_id[i] == entity_id[j]

        off = residue_index[i] - residue_index[j]
        clipped = clamp(off + max_relative_idx, 0, 2 * max_relative_idx)
        relpos_idx = asym_same ? clipped : (2 * max_relative_idx + 1)
        out[relpos_idx + 1, i, j, 1] = 1f0

        out[rel_pos_dim + 1, i, j, 1] = entity_same ? 1f0 : 0f0

        rel_sym = sym_id[i] - sym_id[j]
        clipped_chain = clamp(rel_sym + max_relative_chain, 0, 2 * max_relative_chain)
        relchain_idx = entity_same ? clipped_chain : (2 * max_relative_chain + 1)
        out[rel_pos_dim + 1 + relchain_idx + 1, i, j, 1] = 1f0
    end
    return out
end

function _pseudo_beta_from_atom37(aatype::AbstractVector{Int}, atom37_bllc::AbstractArray)
    ca_idx = Alphafold2.atom_order["CA"] + 1
    cb_idx = Alphafold2.atom_order["CB"] + 1
    gly_idx = Alphafold2.restype_order["G"]
    L = length(aatype)
    out = zeros(Float32, 3, L, 1)
    for i in 1:L
        atom_idx = aatype[i] == gly_idx ? ca_idx : cb_idx
        out[:, i, 1] .= atom37_bllc[1, i, atom_idx, :]
    end
    return out
end

function _dgram_from_positions(positions::AbstractArray; num_bins::Int=15, min_bin::Real=3.25f0, max_bin::Real=20.75f0)
    L = size(positions, 2)
    out = zeros(Float32, num_bins, L, L, 1)
    lower_breaks = collect(range(Float32(min_bin), Float32(max_bin); length=num_bins))
    lower2 = lower_breaks .^ 2
    upper2 = vcat(lower2[2:end], Float32[1f8])
    p = permutedims(view(positions, :, :, 1), (2, 1)) # (L,3)
    diff = reshape(p, L, 1, 3) .- reshape(p, 1, L, 3)
    d2 = dropdims(sum(diff .^ 2; dims=3); dims=3)
    for k in 1:num_bins
        out[k, :, :, 1] .= Float32.(d2 .> lower2[k]) .* Float32.(d2 .< upper2[k])
    end
    return out
end

function _infer_transition_depth(arrs::AbstractDict, base::AbstractString)
    n = 0
    haskey(arrs, string(base, "//weights")) && (n += 1)
    i = 1
    while haskey(arrs, string(base, "_", i, "//weights"))
        n += 1
        i += 1
    end
    return n
end

function _load_structure_core_raw!(m::StructureModuleCore, arrs::AbstractDict)
    p = SM_PREFIX
    _load_ln_raw!(m.single_layer_norm, arrs, string(p, "/single_layer_norm"))
    _load_linear_raw!(m.initial_projection, arrs, string(p, "/initial_projection"))
    _load_ln_raw!(m.pair_layer_norm, arrs, string(p, "/pair_layer_norm"))

    ipa_base = string(p, "/fold_iteration/invariant_point_attention")
    if _has_key(arrs, string(ipa_base, "/q_scalar//weights"))
        _load_linear_raw!(m.fold_iteration_core.ipa.linear_q, arrs, string(ipa_base, "/q_scalar"))
        _load_linear_raw!(m.fold_iteration_core.ipa.linear_q_points.linear, arrs, string(ipa_base, "/q_point_local"))
        _load_linear_raw!(m.fold_iteration_core.ipa.linear_kv, arrs, string(ipa_base, "/kv_scalar"))
        _load_linear_raw!(m.fold_iteration_core.ipa.linear_kv_points.linear, arrs, string(ipa_base, "/kv_point_local"))
    else
        m.fold_iteration_core.ipa isa MultimerInvariantPointAttention ||
            error("Multimer IPA weights found, but structure module was not built with multimer_ipa=true")

        # Multimer checkpoint variant: split q/k/v projections.
        q_w = _arr_get(arrs, string(ipa_base, "/q_scalar_projection//weights")) # (Cin,H,D)
        k_w = _arr_get(arrs, string(ipa_base, "/k_scalar_projection//weights"))
        v_w = _arr_get(arrs, string(ipa_base, "/v_scalar_projection//weights"))
        # Python flattens trailing dims in row-major order; adjust Julia flattening accordingly.
        q_w2 = reshape(permutedims(q_w, (1, 3, 2)), size(q_w, 1), :)
        k_w2 = reshape(permutedims(k_w, (1, 3, 2)), size(k_w, 1), :)
        v_w2 = reshape(permutedims(v_w, (1, 3, 2)), size(v_w, 1), :)
        m.fold_iteration_core.ipa.linear_q.weight .= permutedims(q_w2, (2, 1))
        m.fold_iteration_core.ipa.linear_k.weight .= permutedims(k_w2, (2, 1))
        m.fold_iteration_core.ipa.linear_v.weight .= permutedims(v_w2, (2, 1))

        qp_w = _arr_get(arrs, string(ipa_base, "/q_point_projection/point_projection//weights")) # (Cin,H,Dp)
        kp_w = _arr_get(arrs, string(ipa_base, "/k_point_projection/point_projection//weights"))
        vp_w = _arr_get(arrs, string(ipa_base, "/v_point_projection/point_projection//weights"))
        qp_b = _arr_get(arrs, string(ipa_base, "/q_point_projection/point_projection//bias"))
        kp_b = _arr_get(arrs, string(ipa_base, "/k_point_projection/point_projection//bias"))
        vp_b = _arr_get(arrs, string(ipa_base, "/v_point_projection/point_projection//bias"))

        qp_w2 = reshape(permutedims(qp_w, (1, 3, 2)), size(qp_w, 1), :)
        kp_w2 = reshape(permutedims(kp_w, (1, 3, 2)), size(kp_w, 1), :)
        vp_w2 = reshape(permutedims(vp_w, (1, 3, 2)), size(vp_w, 1), :)
        qp_b2 = vec(permutedims(qp_b, (2, 1)))
        kp_b2 = vec(permutedims(kp_b, (2, 1)))
        vp_b2 = vec(permutedims(vp_b, (2, 1)))

        m.fold_iteration_core.ipa.linear_q_points.linear.weight .= permutedims(qp_w2, (2, 1))
        m.fold_iteration_core.ipa.linear_q_points.linear.bias .= qp_b2
        m.fold_iteration_core.ipa.linear_k_points.linear.weight .= permutedims(kp_w2, (2, 1))
        m.fold_iteration_core.ipa.linear_k_points.linear.bias .= kp_b2
        m.fold_iteration_core.ipa.linear_v_points.linear.weight .= permutedims(vp_w2, (2, 1))
        m.fold_iteration_core.ipa.linear_v_points.linear.bias .= vp_b2
    end

    _load_linear_raw!(m.fold_iteration_core.ipa.linear_b, arrs, string(p, "/fold_iteration/invariant_point_attention/attention_2d"))
    m.fold_iteration_core.ipa.head_weights .= _arr_get(arrs, string(p, "/fold_iteration/invariant_point_attention//trainable_point_weights"))
    _load_linear_raw!(m.fold_iteration_core.ipa.linear_out, arrs, string(p, "/fold_iteration/invariant_point_attention/output_projection"))
    _load_ln_raw!(m.fold_iteration_core.attention_layer_norm, arrs, string(p, "/fold_iteration/attention_layer_norm"))
    _load_ln_raw!(m.fold_iteration_core.transition_layer_norm, arrs, string(p, "/fold_iteration/transition_layer_norm"))
    if _has_key(arrs, string(p, "/fold_iteration/affine_update//weights"))
        _load_linear_raw!(m.fold_iteration_core.affine_update, arrs, string(p, "/fold_iteration/affine_update"))
    else
        _load_linear_raw!(m.fold_iteration_core.affine_update, arrs, string(p, "/fold_iteration/quat_rigid/rigid"))
    end
    for i in eachindex(m.fold_iteration_core.transition_layers)
        key = i == 1 ? string(p, "/fold_iteration/transition") : string(p, "/fold_iteration/transition_", i - 1)
        _load_linear_raw!(m.fold_iteration_core.transition_layers[i], arrs, key)
    end
    for i in eachindex(m.sidechain.input_projections)
        key = i == 1 ? string(p, "/fold_iteration/rigid_sidechain/input_projection") : string(p, "/fold_iteration/rigid_sidechain/input_projection_", i - 1)
        _load_linear_raw!(m.sidechain.input_projections[i], arrs, key)
    end
    for i in eachindex(m.sidechain.resblock1)
        k1 = i == 1 ? string(p, "/fold_iteration/rigid_sidechain/resblock1") : string(p, "/fold_iteration/rigid_sidechain/resblock1_", i - 1)
        k2 = i == 1 ? string(p, "/fold_iteration/rigid_sidechain/resblock2") : string(p, "/fold_iteration/rigid_sidechain/resblock2_", i - 1)
        _load_linear_raw!(m.sidechain.resblock1[i], arrs, k1)
        _load_linear_raw!(m.sidechain.resblock2[i], arrs, k2)
    end
    _load_linear_raw!(m.sidechain.unnormalized_angles, arrs, string(p, "/fold_iteration/rigid_sidechain/unnormalized_angles"))
    return m
end

function _ca_distance_metrics(atom37::AbstractArray, atom37_mask::AbstractArray)
    ca_idx = Alphafold2.atom_order["CA"] + 1
    L = size(atom37, 1)
    valid = atom37_mask[:, ca_idx] .> 0.5f0
    d = Float32[]
    for i in 1:(L - 1)
        if valid[i] && valid[i + 1]
            v = atom37[i + 1, ca_idx, :] .- atom37[i, ca_idx, :]
            push!(d, sqrt(sum(v .^ 2)))
        end
    end
    if isempty(d)
        return Dict(
            :mean => Float32(NaN),
            :std => Float32(NaN),
            :min => Float32(NaN),
            :max => Float32(NaN),
            :outlier_fraction => Float32(NaN),
        )
    end
    return Dict(
        :mean => Float32(mean(d)),
        :std => Float32(std(d; corrected=false)),
        :min => Float32(minimum(d)),
        :max => Float32(maximum(d)),
        :outlier_fraction => Float32(sum((d .< 3.2f0) .| (d .> 4.4f0)) / length(d)),
    )
end

function _pdb_path_from_npz(npz_path::AbstractString)
    if endswith(lowercase(npz_path), ".npz")
        return string(first(npz_path, lastindex(npz_path) - 4), ".pdb")
    end
    return string(npz_path, ".pdb")
end

function _write_pdb(
    path::AbstractString,
    atom37::AbstractArray,
    atom37_mask::AbstractArray,
    aatype::AbstractArray;
    bfactor_by_res::Union{Nothing,AbstractVector}=nothing,
    asym_id::Union{Nothing,AbstractVector}=nothing,
    residue_index::Union{Nothing,AbstractVector}=nothing,
)
    chain_alphabet = collect("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
    chain_char_for(i::Int) = chain_alphabet[clamp(i, 1, length(chain_alphabet))]

    function _chain_char(res_i::Int)
        if asym_id === nothing
            return 'A'
        end
        ai = Int(asym_id[res_i])
        return chain_char_for(ai <= 0 ? 1 : ai)
    end

    function _resseq(res_i::Int)
        if residue_index === nothing
            return res_i
        end
        return Int(residue_index[res_i]) + 1
    end

    atom_serial = 1
    open(path, "w") do io
        for i in 1:size(atom37, 1)
            aa_idx0 = Int(aatype[i])
            resname = if 0 <= aa_idx0 < length(Alphafold2.restypes)
                Alphafold2.restype_1to3[Alphafold2.restypes[aa_idx0 + 1]]
            else
                "UNK"
            end
            chain_id = _chain_char(i)
            resseq = _resseq(i)
            for a_idx in 1:length(Alphafold2.atom_types)
                atom37_mask[i, a_idx] < 0.5f0 && continue
                atom_name = Alphafold2.atom_types[a_idx]
                x = atom37[i, a_idx, 1]
                y = atom37[i, a_idx, 2]
                z = atom37[i, a_idx, 3]
                bfactor = bfactor_by_res === nothing ? 0.0 : Float64(bfactor_by_res[i])
                element = uppercase(first(atom_name, 1))
                line = @sprintf("ATOM  %5d %4s %3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s",
                    atom_serial, atom_name, resname, chain_id, resseq, x, y, z, 1.00, bfactor, element)
                println(io, line)
                atom_serial += 1
            end
            if i < size(atom37, 1) && _chain_char(i + 1) != chain_id
                println(io, @sprintf("TER   %5d      %3s %c%4d", atom_serial, resname, chain_id, resseq))
                atom_serial += 1
            end
        end
        println(io, "END")
    end
    return atom_serial - 1
end

function main()
    if length(ARGS) < 3
        error("Usage: julia run_af2_template_hybrid_jl.jl <params_model_1.npz> <input_dump.npz> <out.npz>")
    end
    params_path, dump_path, out_path = ARGS[1], ARGS[2], ARGS[3]
    arrs = NPZ.npzread(params_path)
    dump = NPZ.npzread(dump_path)

    # Main evo dims
    c_m = length(_arr_get(arrs, string(EVO_PREFIX, "/preprocess_1d//bias")))
    c_z = length(_arr_get(arrs, string(EVO_PREFIX, "/left_single//bias")))
    c_s = length(_arr_get(arrs, string(EVO_PREFIX, "/single_activations//bias")))
    num_blocks = size(_arr_get(arrs, string(EVO_BLOCK_PREFIX, "/msa_column_attention/attention//output_b")), 1)
    msa_qw = _arr_get(arrs, string(EVO_BLOCK_PREFIX, "/msa_column_attention/attention//query_w"))
    num_head_msa = size(msa_qw, 3)
    msa_head_dim = size(msa_qw, 4)
    pair_qw = _arr_get(arrs, string(EVO_BLOCK_PREFIX, "/triangle_attention_starting_node/attention//query_w"))
    num_head_pair = size(pair_qw, 3)
    pair_head_dim = size(pair_qw, 4)
    c_outer = size(_arr_get(arrs, string(EVO_BLOCK_PREFIX, "/outer_product_mean/left_projection//bias")), 2)
    c_tri_mul = if _has_key(arrs, string(EVO_BLOCK_PREFIX, "/triangle_multiplication_outgoing/left_projection//bias"))
        size(_arr_get(arrs, string(EVO_BLOCK_PREFIX, "/triangle_multiplication_outgoing/left_projection//bias")), 2)
    else
        Int(div(size(_arr_get(arrs, string(EVO_BLOCK_PREFIX, "/triangle_multiplication_outgoing/projection//bias")), 2), 2))
    end
    msa_transition_factor = size(_arr_get(arrs, string(EVO_BLOCK_PREFIX, "/msa_transition/transition1//bias")), 2) / c_m
    pair_transition_factor = size(_arr_get(arrs, string(EVO_BLOCK_PREFIX, "/pair_transition/transition1//bias")), 2) / c_z

    # Extra stack dims
    extra_qw = _arr_get(arrs, string(EXTRA_BLOCK_PREFIX, "/msa_row_attention_with_pair_bias/attention//query_w"))
    c_m_extra = size(extra_qw, 2)
    num_head_msa_extra = size(extra_qw, 3)
    msa_head_dim_extra = size(extra_qw, 4)
    extra_pair_qw = _arr_get(arrs, string(EXTRA_BLOCK_PREFIX, "/triangle_attention_starting_node/attention//query_w"))
    num_head_pair_extra = size(extra_pair_qw, 3)
    pair_head_dim_extra = size(extra_pair_qw, 4)
    num_extra_blocks = size(_arr_get(arrs, string(EXTRA_BLOCK_PREFIX, "/msa_transition/transition1//bias")), 1)
    c_outer_extra = size(_arr_get(arrs, string(EXTRA_BLOCK_PREFIX, "/outer_product_mean/left_projection//bias")), 2)
    c_tri_mul_extra = if _has_key(arrs, string(EXTRA_BLOCK_PREFIX, "/triangle_multiplication_outgoing/left_projection//bias"))
        size(_arr_get(arrs, string(EXTRA_BLOCK_PREFIX, "/triangle_multiplication_outgoing/left_projection//bias")), 2)
    else
        Int(div(size(_arr_get(arrs, string(EXTRA_BLOCK_PREFIX, "/triangle_multiplication_outgoing/projection//bias")), 2), 2))
    end
    msa_transition_factor_extra = size(_arr_get(arrs, string(EXTRA_BLOCK_PREFIX, "/msa_transition/transition1//bias")), 2) / c_m_extra
    pair_transition_factor_extra = size(_arr_get(arrs, string(EXTRA_BLOCK_PREFIX, "/pair_transition/transition1//bias")), 2) / c_z

    # Template embedding dims
    TEMPLATE_PREFIX = string(EVO_PREFIX, "/template_embedding")
    TEMPLATE_STACK_PREFIX = string(TEMPLATE_PREFIX, "/single_template_embedding/template_pair_stack/__layer_stack_no_state")
    has_monomer_template_embed = _has_key(arrs, string(TEMPLATE_PREFIX, "/single_template_embedding/embedding2d//bias"))
    has_template_embedding = has_monomer_template_embed
    c_t = 0
    num_template_blocks = 0
    num_head_pair_template = 0
    pair_head_dim_template = 0
    c_tri_mul_template = 0
    pair_transition_factor_template = 0.0
    num_head_tpa = 0
    key_dim_tpa = 0
    value_dim_tpa = 0
    dgram_num_bins_template = 0
    if has_template_embedding
        c_t = length(_arr_get(arrs, string(TEMPLATE_PREFIX, "/single_template_embedding/embedding2d//bias")))
        num_template_blocks = size(_arr_get(arrs, string(TEMPLATE_STACK_PREFIX, "/pair_transition/transition2//bias")), 1)
        num_head_pair_template = size(_arr_get(arrs, string(TEMPLATE_STACK_PREFIX, "/triangle_attention_starting_node//feat_2d_weights")), 3)
        pair_head_dim_template = size(_arr_get(arrs, string(TEMPLATE_STACK_PREFIX, "/triangle_attention_starting_node/attention//query_w")), 4)
        c_tri_mul_template = size(_arr_get(arrs, string(TEMPLATE_STACK_PREFIX, "/triangle_multiplication_outgoing/left_projection//bias")), 2)
        pair_transition_factor_template = size(_arr_get(arrs, string(TEMPLATE_STACK_PREFIX, "/pair_transition/transition1//bias")), 2) / c_t
        num_head_tpa = size(_arr_get(arrs, string(TEMPLATE_PREFIX, "/attention//query_w")), 2)
        key_dim_tpa = size(_arr_get(arrs, string(TEMPLATE_PREFIX, "/attention//query_w")), 2) * size(_arr_get(arrs, string(TEMPLATE_PREFIX, "/attention//query_w")), 3)
        value_dim_tpa = size(_arr_get(arrs, string(TEMPLATE_PREFIX, "/attention//value_w")), 2) * size(_arr_get(arrs, string(TEMPLATE_PREFIX, "/attention//value_w")), 3)
        dgram_num_bins_template = size(_arr_get(arrs, string(TEMPLATE_PREFIX, "/single_template_embedding/embedding2d//weights")), 1) - 49
    end

    is_multimer_checkpoint = !_has_key(arrs, string(EVO_PREFIX, "/pair_activiations//weights"))
    outer_first_evo = is_multimer_checkpoint

    blocks = [EvoformerIteration(c_m, c_z; num_head_msa=num_head_msa, msa_head_dim=msa_head_dim, num_head_pair=num_head_pair, pair_head_dim=pair_head_dim, c_outer=c_outer, c_tri_mul=c_tri_mul, msa_transition_factor=msa_transition_factor, pair_transition_factor=pair_transition_factor, outer_first=outer_first_evo) for _ in 1:num_blocks]
    for i in 1:num_blocks
        _load_evo_block_raw!(blocks[i], arrs, i)
    end

    extra_blocks = [EvoformerIteration(c_m_extra, c_z; num_head_msa=num_head_msa_extra, msa_head_dim=msa_head_dim_extra, num_head_pair=num_head_pair_extra, pair_head_dim=pair_head_dim_extra, c_outer=c_outer_extra, c_tri_mul=c_tri_mul_extra, msa_transition_factor=msa_transition_factor_extra, pair_transition_factor=pair_transition_factor_extra, outer_first=outer_first_evo, is_extra_msa=true) for _ in 1:num_extra_blocks]
    for i in 1:num_extra_blocks
        _load_extra_block_raw!(extra_blocks[i], arrs, i)
    end

    preprocess_1d_in_dim = size(_arr_get(arrs, string(EVO_PREFIX, "/preprocess_1d//weights")), 1)
    preprocess_1d = LinearFirst(preprocess_1d_in_dim, c_m)
    preprocess_msa = LinearFirst(49, c_m)
    left_single_in_dim = size(_arr_get(arrs, string(EVO_PREFIX, "/left_single//weights")), 1)
    right_single_in_dim = size(_arr_get(arrs, string(EVO_PREFIX, "/right_single//weights")), 1)
    left_single = LinearFirst(left_single_in_dim, c_z)
    right_single = LinearFirst(right_single_in_dim, c_z)
    extra_msa_activations = LinearFirst(25, c_m_extra)
    prev_pos_linear = LinearFirst(15, c_z)
    prev_msa_first_row_norm = LayerNormFirst(c_m)
    prev_pair_norm = LayerNormFirst(c_z)
    pair_relpos_base = if _has_key(arrs, string(EVO_PREFIX, "/pair_activiations//weights"))
        string(EVO_PREFIX, "/pair_activiations")
    else
        string(EVO_PREFIX, "/~_relative_encoding/position_activations")
    end
    pair_relpos_in_dim = size(_arr_get(arrs, string(pair_relpos_base, "//weights")), 1)
    pair_relpos = LinearFirst(pair_relpos_in_dim, c_z)
    _load_linear_raw!(preprocess_1d, arrs, string(EVO_PREFIX, "/preprocess_1d"))
    _load_linear_raw!(preprocess_msa, arrs, string(EVO_PREFIX, "/preprocess_msa"))
    _load_linear_raw!(left_single, arrs, string(EVO_PREFIX, "/left_single"))
    _load_linear_raw!(right_single, arrs, string(EVO_PREFIX, "/right_single"))
    _load_linear_raw!(extra_msa_activations, arrs, string(EVO_PREFIX, "/extra_msa_activations"))
    _load_linear_raw!(prev_pos_linear, arrs, string(EVO_PREFIX, "/prev_pos_linear"))
    _load_ln_raw!(prev_msa_first_row_norm, arrs, string(EVO_PREFIX, "/prev_msa_first_row_norm"))
    _load_ln_raw!(prev_pair_norm, arrs, string(EVO_PREFIX, "/prev_pair_norm"))
    _load_linear_raw!(pair_relpos, arrs, pair_relpos_base)
    relpos_is_multimer = occursin("position_activations", pair_relpos_base)
    relpos_max_relative_idx = relpos_is_multimer ? 32 : Int(div(pair_relpos_in_dim - 1, 2))
    relpos_max_relative_chain = relpos_is_multimer ? Int(div(pair_relpos_in_dim - (2 * relpos_max_relative_idx + 5), 2)) : 0

    single_activations = LinearFirst(c_m, c_s)
    _load_linear_raw!(single_activations, arrs, string(EVO_PREFIX, "/single_activations"))

    template_embedding = if has_template_embedding
        te = TemplateEmbedding(
            c_z,
            c_t,
            num_template_blocks;
            num_head_pair=num_head_pair_template,
            pair_head_dim=pair_head_dim_template,
            c_tri_mul=c_tri_mul_template,
            pair_transition_factor=pair_transition_factor_template,
            num_head_tpa=num_head_tpa,
            key_dim_tpa=key_dim_tpa,
            value_dim_tpa=value_dim_tpa,
            use_template_unit_vector=false,
            dgram_num_bins=dgram_num_bins_template,
        )
        load_template_embedding_npz!(te, params_path; prefix=TEMPLATE_PREFIX)
        te
    else
        nothing
    end

    num_transition_layers = _infer_transition_depth(arrs, string(SM_PREFIX, "/fold_iteration/transition"))
    num_residual_block = _infer_transition_depth(arrs, string(SM_PREFIX, "/fold_iteration/rigid_sidechain/resblock1"))
    no_heads = length(_arr_get(arrs, string(SM_PREFIX, "/fold_iteration/invariant_point_attention/attention_2d//bias")))
    c_hidden_total = if _has_key(arrs, string(SM_PREFIX, "/fold_iteration/invariant_point_attention/q_scalar//weights"))
        size(_arr_get(arrs, string(SM_PREFIX, "/fold_iteration/invariant_point_attention/q_scalar//weights")), 2)
    else
        size(_arr_get(arrs, string(SM_PREFIX, "/fold_iteration/invariant_point_attention/q_scalar_projection//weights")), 2) *
        size(_arr_get(arrs, string(SM_PREFIX, "/fold_iteration/invariant_point_attention/q_scalar_projection//weights")), 3)
    end
    c_hidden = Int(div(c_hidden_total, no_heads))
    q_point_total = if _has_key(arrs, string(SM_PREFIX, "/fold_iteration/invariant_point_attention/q_point_local//bias"))
        size(_arr_get(arrs, string(SM_PREFIX, "/fold_iteration/invariant_point_attention/q_point_local//bias")), 1)
    else
        length(_arr_get(arrs, string(SM_PREFIX, "/fold_iteration/invariant_point_attention/q_point_projection/point_projection//bias")))
    end
    kv_point_total = if _has_key(arrs, string(SM_PREFIX, "/fold_iteration/invariant_point_attention/kv_point_local//bias"))
        size(_arr_get(arrs, string(SM_PREFIX, "/fold_iteration/invariant_point_attention/kv_point_local//bias")), 1)
    else
        length(_arr_get(arrs, string(SM_PREFIX, "/fold_iteration/invariant_point_attention/k_point_projection/point_projection//bias"))) +
        length(_arr_get(arrs, string(SM_PREFIX, "/fold_iteration/invariant_point_attention/v_point_projection/point_projection//bias")))
    end
    no_qk_points = Int(div(q_point_total, 3 * no_heads))
    no_v_points = Int(div(kv_point_total, 3 * no_heads) - no_qk_points)
    sidechain_num_channel = length(_arr_get(arrs, string(SM_PREFIX, "/fold_iteration/rigid_sidechain/input_projection//bias")))
    structure_position_scale = is_multimer_checkpoint ? 20f0 : 10f0
    structure = StructureModuleCore(
        c_s,
        c_z,
        c_hidden,
        no_heads,
        no_qk_points,
        no_v_points,
        num_transition_layers,
        8,
        structure_position_scale,
        sidechain_num_channel,
        num_residual_block;
        multimer_ipa=is_multimer_checkpoint,
    )
    _load_structure_core_raw!(structure, arrs)

    lddt_prefix = "alphafold/alphafold_iteration/predicted_lddt_head"
    lddt_num_channels = length(_arr_get(arrs, string(lddt_prefix, "/act_0//bias")))
    lddt_num_bins = length(_arr_get(arrs, string(lddt_prefix, "/logits//bias")))
    predicted_lddt_head = PredictedLDDTHead(c_s; num_channels=lddt_num_channels, num_bins=lddt_num_bins)
    load_predicted_lddt_head_npz!(predicted_lddt_head, params_path; prefix=lddt_prefix)

    masked_msa_num_output = length(_arr_get(arrs, "alphafold/alphafold_iteration/masked_msa_head/logits//bias"))
    masked_msa_head = MaskedMsaHead(c_m; num_output=masked_msa_num_output)
    load_masked_msa_head_npz!(masked_msa_head, params_path)

    distogram_num_bins = length(_arr_get(arrs, "alphafold/alphafold_iteration/distogram_head/half_logits//bias"))
    distogram_head = DistogramHead(c_z; num_bins=distogram_num_bins, first_break=2.3125f0, last_break=21.6875f0)
    load_distogram_head_npz!(distogram_head, params_path)

    experimentally_resolved_head = ExperimentallyResolvedHead(c_s)
    load_experimentally_resolved_head_npz!(experimentally_resolved_head, params_path)

    pae_prefix = "alphafold/alphafold_iteration/predicted_aligned_error_head"
    has_pae_head = _has_key(arrs, string(pae_prefix, "/logits//weights"))
    predicted_aligned_error_head = if has_pae_head
        pae_num_bins = length(_arr_get(arrs, string(pae_prefix, "/logits//bias")))
        h = PredictedAlignedErrorHead(c_z; num_bins=pae_num_bins, max_error_bin=31f0)
        load_predicted_aligned_error_head_npz!(h, params_path; prefix=pae_prefix)
        h
    else
        nothing
    end

    has_template_raw = haskey(dump, "template_aatype") &&
                       haskey(dump, "template_all_atom_positions") &&
                       haskey(dump, "template_all_atom_masks")
    if template_embedding !== nothing && !has_template_raw
        error("Pre-evo dump is missing template atom inputs. Regenerate with run_af2_template_case_py.py.")
    end

    parity_mode = haskey(dump, "pair_after_recycle_relpos")
    pair_after_recycle_relpos_ref = parity_mode ? Float32.(dump["pair_after_recycle_relpos"]) : nothing # (I,L,L,C)
    template_pair_representation = parity_mode ? Float32.(dump["template_pair_representation"]) : nothing # (I,L,L,C)
    pair_after_template_ref = parity_mode ? Float32.(dump["pair_after_template"]) : nothing # (I,L,L,C)
    pair_after_extra_ref = parity_mode ? Float32.(dump["pair_after_extra"]) : nothing # (I,L,L,C)
    extra_msa_feat_ref = parity_mode ? Float32.(dump["extra_msa_feat"]) : nothing # (I,S,L,25)
    template_single_rows_ref = parity_mode && haskey(dump, "template_single_rows") ? Float32.(dump["template_single_rows"]) : nothing # (I,T,L,C)
    template_aatype = has_template_raw ? Int.(dump["template_aatype"]) : zeros(Int, 0, length(vec(Int.(dump["aatype"]))))
    template_all_atom_positions = has_template_raw ? Float32.(dump["template_all_atom_positions"]) : zeros(Float32, 0, length(vec(Int.(dump["aatype"]))), 37, 3)
    template_all_atom_masks = has_template_raw ? Float32.(dump["template_all_atom_masks"]) : zeros(Float32, 0, length(vec(Int.(dump["aatype"]))), 37)
    template_placeholder_for_undefined = if haskey(dump, "template_placeholder_for_undefined")
        v = dump["template_placeholder_for_undefined"]
        Int(v isa AbstractArray ? v[] : v) != 0
    else
        false
    end
    pre_msa_ref = parity_mode ? Float32.(dump["pre_msa"]) : nothing # (I,Sfull,L,C)
    pre_msa_mask_ref = parity_mode ? Float32.(dump["pre_msa_mask"]) : nothing # (I,Sfull,L)
    py_out_single = parity_mode ? Float32.(dump["out_single"]) : nothing
    py_out_pair = parity_mode ? Float32.(dump["out_pair"]) : nothing
    py_out_masked_msa_logits = parity_mode && haskey(dump, "out_masked_msa_logits") ? Float32.(dump["out_masked_msa_logits"]) : nothing
    py_out_distogram_logits = parity_mode && haskey(dump, "out_distogram_logits") ? Float32.(dump["out_distogram_logits"]) : nothing
    py_out_distogram_bin_edges = parity_mode && haskey(dump, "out_distogram_bin_edges") ? Float32.(dump["out_distogram_bin_edges"]) : nothing
    py_out_experimentally_resolved_logits = parity_mode && haskey(dump, "out_experimentally_resolved_logits") ? Float32.(dump["out_experimentally_resolved_logits"]) : nothing
    py_out_atom14 = parity_mode && haskey(dump, "out_atom14") ? Float32.(dump["out_atom14"]) : nothing
    py_out_affine = parity_mode && haskey(dump, "out_affine") ? Float32.(dump["out_affine"]) : nothing
    py_out_angles = parity_mode && haskey(dump, "out_angles") ? Float32.(dump["out_angles"]) : nothing
    py_out_unnorm_angles = parity_mode && haskey(dump, "out_unnormalized_angles") ? Float32.(dump["out_unnormalized_angles"]) : nothing
    py_out_lddt_logits = parity_mode && haskey(dump, "out_predicted_lddt_logits") ? Float32.(dump["out_predicted_lddt_logits"]) : nothing
    py_out_plddt = parity_mode && haskey(dump, "out_plddt") ? Float32.(dump["out_plddt"]) : nothing
    py_out_pae_logits = parity_mode && haskey(dump, "out_predicted_aligned_error_logits") ? Float32.(dump["out_predicted_aligned_error_logits"]) : nothing
    py_out_pae_breaks = parity_mode && haskey(dump, "out_predicted_aligned_error_breaks") ? Float32.(dump["out_predicted_aligned_error_breaks"]) : nothing
    py_out_pae = parity_mode && haskey(dump, "out_predicted_aligned_error") ? Float32.(dump["out_predicted_aligned_error"]) : nothing
    py_out_ptm = parity_mode && haskey(dump, "out_predicted_tm_score") ? Float32.(dump["out_predicted_tm_score"]) : nothing
    aatype = vec(Int.(dump["aatype"]))
    seq_mask = Float32.(vec(dump["seq_mask"]))
    asym_id = haskey(dump, "asym_id") ? vec(Int.(dump["asym_id"])) : ones(Int, length(aatype))
    entity_id = haskey(dump, "entity_id") ? vec(Int.(dump["entity_id"])) : ones(Int, length(aatype))
    sym_id = haskey(dump, "sym_id") ? vec(Int.(dump["sym_id"])) : ones(Int, length(aatype))
    msa_mask_input = if haskey(dump, "msa_mask")
        x = Float32.(dump["msa_mask"])
        ndims(x) == 1 ? reshape(x, 1, :) : x
    else
        nothing
    end
    extra_msa_mask_input = if haskey(dump, "extra_msa_mask")
        x = Float32.(dump["extra_msa_mask"])
        ndims(x) == 1 ? reshape(x, 1, :) : x
    else
        msa_mask_input
    end

    msa_int = if haskey(dump, "msa")
        x = Int.(dump["msa"])
        ndims(x) == 1 ? reshape(x, 1, :) : x
    else
        nothing
    end
    deletion_matrix = if haskey(dump, "deletion_matrix")
        x = Float32.(dump["deletion_matrix"])
        ndims(x) == 1 ? reshape(x, 1, :) : x
    else
        nothing
    end

    target_feat_override = haskey(dump, "target_feat") ? Float32.(dump["target_feat"]) : nothing
    msa_feat_override = haskey(dump, "msa_feat") ? Float32.(dump["msa_feat"]) : nothing
    target_feat, msa_feat, residue_index = if target_feat_override !== nothing && msa_feat_override !== nothing
        tf = target_feat_override
        if ndims(tf) == 3 && size(tf, 1) == 1
            tf = dropdims(tf; dims=1)
        end
        ndims(tf) == 2 || error("target_feat override must be rank-2 (L,C) or rank-3 with leading singleton.")
        size(tf, 2) == preprocess_1d_in_dim || error("target_feat dim mismatch: expected $(preprocess_1d_in_dim), got $(size(tf, 2))")
        Ltf = size(tf, 1)
        tf_first = reshape(permutedims(tf, (2, 1)), preprocess_1d_in_dim, Ltf, 1)

        mf = msa_feat_override
        if ndims(mf) == 4 && size(mf, 1) == 1
            mf = dropdims(mf; dims=1)
        end
        ndims(mf) == 3 || error("msa_feat override must be rank-3 (S,L,49) or rank-4 with leading singleton.")
        mf_first = if size(mf, 3) == 49
            reshape(permutedims(mf, (3, 1, 2)), 49, size(mf, 1), size(mf, 2), 1) # (49,S,L,1)
        elseif size(mf, 1) == 49
            reshape(mf, 49, size(mf, 2), size(mf, 3), 1) # already (49,S,L)
        else
            error("msa_feat override must have 49 channels in last or first dimension.")
        end
        ridx = haskey(dump, "residue_index") ? vec(Int.(dump["residue_index"])) : collect(0:(Ltf - 1))
        tf_first, mf_first, ridx
    elseif msa_int === nothing
        _build_target_and_msa_feat(aatype; target_dim=preprocess_1d_in_dim)
    else
        _build_target_and_msa_feat(aatype, msa_int, deletion_matrix; target_dim=preprocess_1d_in_dim)
    end

    extra_msa_int = if haskey(dump, "extra_msa")
        x = Int.(dump["extra_msa"])
        ndims(x) == 1 ? reshape(x, 1, :) : x
    else
        msa_int
    end
    extra_deletion_matrix = if haskey(dump, "extra_deletion_matrix")
        x = Float32.(dump["extra_deletion_matrix"])
        ndims(x) == 1 ? reshape(x, 1, :) : x
    elseif haskey(dump, "extra_deletion_value")
        x = Float32.(dump["extra_deletion_value"])
        x = ndims(x) == 1 ? reshape(x, 1, :) : x
        tan.(x .* (Float32(π) / 2f0)) .* 3f0
    else
        deletion_matrix
    end
    extra_msa_feat = if extra_msa_int === nothing
        _build_extra_msa_feat(aatype) # (25,1,L,1)
    else
        _build_extra_msa_feat(extra_msa_int, extra_deletion_matrix)
    end
    if haskey(dump, "residue_index")
        residue_index = vec(Int.(dump["residue_index"]))
    end

    L = length(aatype)
    template_rows_first, template_row_mask, template_mask = if has_template_raw && template_embedding !== nothing
        template_single = TemplateSingleRows(c_m)
        load_template_single_rows_npz!(template_single, params_path)
        rows, row_mask = template_single(
            template_aatype,
            template_all_atom_positions,
            template_all_atom_masks;
            placeholder_for_undefined=template_placeholder_for_undefined,
        )
        rows, row_mask, ones(Float32, size(template_aatype, 1))
    else
        zeros(Float32, c_m, 0, L, 1), zeros(Float32, 0, L), zeros(Float32, 0)
    end

    Iters = parity_mode ? size(pair_after_recycle_relpos_ref, 1) : (haskey(dump, "num_recycle") ? Int(dump["num_recycle"]) + 1 : 2)
    pair_mask = reshape(seq_mask, L, 1, 1) .* reshape(seq_mask, 1, L, 1)
    final_atom37 = nothing
    final_mask = nothing
    final_masked_msa_logits = nothing
    final_distogram_logits = nothing
    final_distogram_bin_edges = nothing
    final_experimentally_resolved_logits = nothing
    final_plddt = nothing
    final_lddt_logits = nothing
    final_pae = nothing
    final_pae_max = nothing
    final_ptm = nothing
    prev_atom37 = zeros(Float32, 1, L, 37, 3)
    prev_msa_first_row = zeros(Float32, c_m, L, 1)
    prev_pair = zeros(Float32, c_z, L, L, 1)

    for i in 1:Iters
        p1 = preprocess_1d(target_feat)
        pmsa = preprocess_msa(msa_feat)
        msa_base = pmsa .+ reshape(p1, c_m, 1, L, 1)

        left = left_single(target_feat)
        right = right_single(target_feat)
        pair_recycle = reshape(left, c_z, L, 1, 1) .+ reshape(right, c_z, 1, L, 1)
        prev_pb = _pseudo_beta_from_atom37(aatype, prev_atom37)
        pair_recycle = pair_recycle .+ prev_pos_linear(_dgram_from_positions(prev_pb))
        msa_base[:, 1:1, :, :] .+= reshape(prev_msa_first_row_norm(prev_msa_first_row), c_m, 1, L, 1)
        pair_recycle .+= prev_pair_norm(prev_pair)
        relpos_feat = if relpos_is_multimer
            _multimer_relpos_features(
                residue_index,
                asym_id,
                entity_id,
                sym_id,
                relpos_max_relative_idx,
                relpos_max_relative_chain,
            )
        else
            _relpos_one_hot(residue_index, relpos_max_relative_idx)
        end
        pair_recycle .+= pair_relpos(relpos_feat)

        pair_rr_jl = dropdims(first_to_af2_2d(pair_recycle); dims=1)
        d_recycle = parity_mode ? maximum(abs.(pair_rr_jl .- view(pair_after_recycle_relpos_ref, i, :, :, :))) : NaN32

        tpair = if template_embedding === nothing || size(template_aatype, 1) == 0
            zeros(Float32, size(pair_recycle)...)
        else
            template_embedding(
                pair_recycle,
                template_aatype,
                template_all_atom_positions,
                template_all_atom_masks,
                pair_mask;
                template_mask=template_mask,
            )
        end
        tpair_jl = dropdims(first_to_af2_2d(tpair); dims=1)
        d_template_embed = parity_mode ? maximum(abs.(tpair_jl .- view(template_pair_representation, i, :, :, :))) : NaN32
        pair_template_jl = pair_rr_jl .+ tpair_jl
        d_template = parity_mode ? maximum(abs.(pair_template_jl .- view(pair_after_template_ref, i, :, :, :))) : NaN32
        pair_act = reshape(permutedims(pair_template_jl, (3, 1, 2)), c_z, L, L, 1)

        extra_feat_jl = dropdims(permutedims(extra_msa_feat, (2, 3, 1, 4)); dims=4)
        d_extra_feat = parity_mode ? maximum(abs.(extra_feat_jl .- view(extra_msa_feat_ref, i, :, :, :))) : NaN32
        msa_extra = extra_msa_activations(extra_msa_feat)
        msa_extra_mask = if extra_msa_mask_input === nothing
            ones(Float32, size(extra_msa_feat, 2), L, 1)
        else
            size(extra_msa_mask_input, 1) == size(extra_msa_feat, 2) || error("extra_msa_mask row count mismatch")
            reshape(Float32.(extra_msa_mask_input), size(extra_msa_mask_input, 1), size(extra_msa_mask_input, 2), 1)
        end
        for b in eachindex(extra_blocks)
            msa_extra, pair_act = extra_blocks[b](msa_extra, pair_act, msa_extra_mask, pair_mask)
        end
        pair_after_extra_jl = dropdims(first_to_af2_2d(pair_act); dims=1)
        d_extra = parity_mode ? maximum(abs.(pair_after_extra_jl .- view(pair_after_extra_ref, i, :, :, :))) : NaN32

        # Build full evoformer input MSA: native base row plus Julia-computed template rows.
        T = size(template_rows_first, 2)
        msa_act = cat(msa_base, template_rows_first; dims=2)
        tmpl_rows_jl = dropdims(permutedims(template_rows_first, (2, 3, 1, 4)); dims=4)
        d_template_rows = if parity_mode && template_single_rows_ref !== nothing
            maximum(abs.(tmpl_rows_jl .- view(template_single_rows_ref, i, :, :, :)))
        else
            NaN32
        end
        pre_msa_jl = dropdims(permutedims(msa_act, (2, 3, 1, 4)); dims=4)
        d_pre_msa = parity_mode ? maximum(abs.(pre_msa_jl .- view(pre_msa_ref, i, :, :, :))) : NaN32
        query_msa_mask = if msa_mask_input === nothing
            ones(Float32, size(msa_base, 2), L, 1)
        else
            size(msa_mask_input, 1) == size(msa_base, 2) || error("msa_mask row count mismatch")
            reshape(Float32.(msa_mask_input), size(msa_mask_input, 1), size(msa_mask_input, 2), 1)
        end
        msa_mask = cat(query_msa_mask, reshape(template_row_mask, T, L, 1); dims=1)
        d_pre_msa_mask = parity_mode ? maximum(abs.(dropdims(msa_mask; dims=3) .- view(pre_msa_mask_ref, i, :, :))) : NaN32

        for blk in blocks
            msa_act, pair_act = blk(msa_act, pair_act, msa_mask, pair_mask)
        end

        single = single_activations(view(msa_act, :, 1, :, :))
        masked_msa_logits = masked_msa_head(msa_act)[:logits]
        distogram_out = distogram_head(pair_act)
        distogram_logits = distogram_out[:logits]
        distogram_bin_edges = distogram_out[:bin_edges]
        experimentally_resolved_logits = experimentally_resolved_head(single)[:logits]
        struct_out = structure(single, pair_act, reshape(seq_mask, :, 1), reshape(aatype, :, 1))
        lddt_logits = predicted_lddt_head(struct_out[:act])[:logits] # (num_bins, L, 1)
        plddt = vec(compute_plddt(lddt_logits))

        pae_out = predicted_aligned_error_head === nothing ? nothing : predicted_aligned_error_head(pair_act)
        pae_metrics = if pae_out === nothing
            nothing
        else
            compute_predicted_aligned_error(
                pae_out[:logits];
                max_bin=Int(round(predicted_aligned_error_head.max_error_bin)),
                no_bins=predicted_aligned_error_head.num_bins,
            )
        end
        ptm_score = if pae_out === nothing
            nothing
        else
            compute_tm(
                pae_out[:logits];
                max_bin=Int(round(predicted_aligned_error_head.max_error_bin)),
                no_bins=predicted_aligned_error_head.num_bins,
            )
        end

        atom14 = dropdims(view(struct_out[:atom_pos], size(struct_out[:atom_pos], 1):size(struct_out[:atom_pos], 1), :, :, :, :); dims=1)
        affine = dropdims(view(struct_out[:affine], size(struct_out[:affine], 1):size(struct_out[:affine], 1), :, :, :); dims=1)
        angles = dropdims(view(struct_out[:angles_sin_cos], size(struct_out[:angles_sin_cos], 1):size(struct_out[:angles_sin_cos], 1), :, :, :, :); dims=1)
        unnorm_angles = dropdims(view(struct_out[:unnormalized_angles_sin_cos], size(struct_out[:unnormalized_angles_sin_cos], 1):size(struct_out[:unnormalized_angles_sin_cos], 1), :, :, :, :); dims=1)
        protein = Dict{Symbol,Any}(:aatype => reshape(aatype, :, 1), :s_s => single)
        make_atom14_masks!(protein)
        atom37 = atom14_to_atom37(atom14, protein)

        single_py = dropdims(first_to_af2_3d(single); dims=1)
        pair_py = dropdims(first_to_af2_2d(pair_act); dims=1)
        masked_msa_logits_py = dropdims(permutedims(masked_msa_logits, (4, 2, 3, 1)); dims=1)
        distogram_logits_py = dropdims(first_to_af2_2d(distogram_logits); dims=1)
        experimentally_resolved_logits_py = dropdims(first_to_af2_3d(experimentally_resolved_logits); dims=1)
        lddt_logits_py = dropdims(first_to_af2_3d(lddt_logits); dims=1)
        atom14_py = dropdims(permutedims(atom14, (3, 2, 1, 4)); dims=4)
        affine_py = dropdims(permutedims(affine, (2, 1, 3)); dims=3)
        angles_py = dropdims(permutedims(angles, (3, 2, 1, 4)); dims=4)
        unnorm_angles_py = dropdims(permutedims(unnorm_angles, (3, 2, 1, 4)); dims=4)
        rot_py = Alphafold2.quat_to_rot_first(view(affine, 1:4, :, :))
        trans_py = view(affine, 5:7, :, :)
        affine_traj = cat(rot_py, reshape(trans_py .* structure.position_scale, 3, 1, size(trans_py, 2), size(trans_py, 3)); dims=2)
        affine_traj_py = dropdims(permutedims(affine_traj, (3, 1, 2, 4)); dims=4)
        if parity_mode
            ds = maximum(abs.(single_py .- view(py_out_single, i, :, :)))
            dp = maximum(abs.(pair_py .- view(py_out_pair, i, :, :, :)))
            dmasked = py_out_masked_msa_logits === nothing ? NaN32 : maximum(abs.(masked_msa_logits_py .- view(py_out_masked_msa_logits, i, :, :, :)))
            ddist = py_out_distogram_logits === nothing ? NaN32 : maximum(abs.(distogram_logits_py .- view(py_out_distogram_logits, i, :, :, :)))
            ddist_bins = py_out_distogram_bin_edges === nothing ? NaN32 : maximum(abs.(distogram_bin_edges .- py_out_distogram_bin_edges))
            dexp = py_out_experimentally_resolved_logits === nothing ? NaN32 : maximum(abs.(experimentally_resolved_logits_py .- view(py_out_experimentally_resolved_logits, i, :, :)))
            dlddt = py_out_lddt_logits === nothing ? NaN32 : maximum(abs.(lddt_logits_py .- view(py_out_lddt_logits, i, :, :)))
            dplddt = py_out_plddt === nothing ? NaN32 : maximum(abs.(plddt .- view(py_out_plddt, i, :)))
            dpae_logits = if py_out_pae_logits === nothing || pae_out === nothing || size(view(py_out_pae_logits, i, :, :, :), 3) == 0
                NaN32
            else
                pae_logits_py = dropdims(first_to_af2_2d(pae_out[:logits]); dims=1)
                maximum(abs.(pae_logits_py .- view(py_out_pae_logits, i, :, :, :)))
            end
            dpae_breaks = if py_out_pae_breaks === nothing || pae_out === nothing || length(py_out_pae_breaks) == 0
                NaN32
            else
                maximum(abs.(pae_out[:breaks] .- py_out_pae_breaks))
            end
            dpae = if py_out_pae === nothing || pae_metrics === nothing || size(view(py_out_pae, i, :, :), 1) == 0
                NaN32
            else
                maximum(abs.(dropdims(pae_metrics[:predicted_aligned_error]; dims=3) .- view(py_out_pae, i, :, :)))
            end
            dptm = if py_out_ptm === nothing || ptm_score === nothing
                NaN32
            else
                abs(Float32(ptm_score) - Float32(py_out_ptm[i]))
            end
            da = py_out_atom14 === nothing ? NaN32 : maximum(abs.(atom14_py .- view(py_out_atom14, i, :, :, :)))
            df = if py_out_affine === nothing || ndims(py_out_affine) != 3
                NaN32
            else
                maximum(abs.(affine_py .- view(py_out_affine, i, :, :)))
            end
            dtraj = if py_out_affine === nothing || ndims(py_out_affine) != 5
                NaN32
            else
                maximum(abs.(affine_traj_py .- view(py_out_affine, i, size(py_out_affine, 2), :, :, :)))
            end
            dangles = if py_out_angles === nothing || ndims(py_out_angles) != 5
                NaN32
            else
                maximum(abs.(angles_py .- view(py_out_angles, i, size(py_out_angles, 2), :, :, :)))
            end
            dangles_unnorm = if py_out_unnorm_angles === nothing || ndims(py_out_unnorm_angles) != 5
                NaN32
            else
                maximum(abs.(unnorm_angles_py .- view(py_out_unnorm_angles, i, size(py_out_unnorm_angles, 2), :, :, :)))
            end
            @printf("Iter %d hybrid parity\n", i - 1)
            @printf("  recycle_pair_max_abs: %.8g\n", d_recycle)
            @printf("  template_embed_max_abs: %.8g\n", d_template_embed)
            @printf("  template_pair_max_abs: %.8g\n", d_template)
            if template_single_rows_ref !== nothing
                @printf("  template_rows_max_abs: %.8g\n", d_template_rows)
            end
            @printf("  extra_feat_max_abs: %.8g\n", d_extra_feat)
            @printf("  pre_msa_max_abs:    %.8g\n", d_pre_msa)
            @printf("  pre_msa_mask_max_abs: %.8g\n", d_pre_msa_mask)
            @printf("  extra_pair_max_abs: %.8g\n", d_extra)
            @printf("  single_max_abs:     %.8g\n", ds)
            @printf("  pair_max_abs:       %.8g\n", dp)
            if py_out_masked_msa_logits !== nothing
                @printf("  masked_msa_max_abs: %.8g\n", dmasked)
            end
            if py_out_distogram_logits !== nothing
                @printf("  distogram_max_abs:  %.8g\n", ddist)
            end
            if py_out_distogram_bin_edges !== nothing
                @printf("  dist_bin_edges_max_abs: %.8g\n", ddist_bins)
            end
            if py_out_experimentally_resolved_logits !== nothing
                @printf("  experimentally_resolved_max_abs: %.8g\n", dexp)
            end
            if py_out_lddt_logits !== nothing
                @printf("  lddt_logits_max_abs: %.8g\n", dlddt)
            end
            if py_out_plddt !== nothing
                @printf("  plddt_max_abs:      %.8g\n", dplddt)
            end
            if py_out_pae_logits !== nothing
                @printf("  pae_logits_max_abs: %.8g\n", dpae_logits)
            end
            if py_out_pae_breaks !== nothing
                @printf("  pae_breaks_max_abs: %.8g\n", dpae_breaks)
            end
            if py_out_pae !== nothing
                @printf("  pae_max_abs:        %.8g\n", dpae)
            end
            if py_out_ptm !== nothing
                @printf("  ptm_abs:            %.8g\n", dptm)
            end
            if py_out_atom14 !== nothing
                @printf("  atom14_max_abs:     %.8g\n", da)
            end
            if py_out_affine !== nothing && ndims(py_out_affine) == 3
                @printf("  affine_max_abs:     %.8g\n", df)
            end
            if py_out_affine !== nothing && ndims(py_out_affine) == 5
                @printf("  traj3x4_max_abs:    %.8g\n", dtraj)
            end
            if py_out_angles !== nothing
                @printf("  angles_max_abs:     %.8g\n", dangles)
            end
            if py_out_unnorm_angles !== nothing
                @printf("  unnorm_angles_max_abs: %.8g\n", dangles_unnorm)
            end
        else
            @printf("Iter %d native run complete\n", i - 1)
        end

        prev_atom37 = atom37
        prev_msa_first_row = view(msa_act, :, 1, :, :)
        prev_pair = pair_act

        if i == Iters
            final_atom37 = dropdims(atom37; dims=1)
            final_mask = dropdims(permutedims(protein[:atom37_atom_exists], (2, 3, 1)); dims=2)
            final_masked_msa_logits = masked_msa_logits_py
            final_distogram_logits = distogram_logits_py
            final_distogram_bin_edges = distogram_bin_edges
            final_experimentally_resolved_logits = experimentally_resolved_logits_py
            final_plddt = plddt
            final_lddt_logits = lddt_logits_py
            if pae_metrics !== nothing
                final_pae = dropdims(pae_metrics[:predicted_aligned_error]; dims=3)
                final_pae_max = Float32(pae_metrics[:max_predicted_aligned_error])
                final_ptm = Float32(ptm_score)
            end
        end
    end

    ca = _ca_distance_metrics(final_atom37, final_mask)
    @printf("Final geometry (%s)\n", parity_mode ? "hybrid parity" : "native")
    @printf("  mean: %.6f A\n", ca[:mean])
    @printf("  std:  %.6f A\n", ca[:std])
    @printf("  min:  %.6f A\n", ca[:min])
    @printf("  max:  %.6f A\n", ca[:max])
    @printf("  outlier_fraction: %.3f\n", ca[:outlier_fraction])

    @printf("Final confidence\n")
    @printf("  mean_pLDDT: %.4f\n", mean(final_plddt))
    @printf("  min_pLDDT: %.4f\n", minimum(final_plddt))
    @printf("  max_pLDDT: %.4f\n", maximum(final_plddt))
    if final_pae !== nothing
        @printf("  mean_PAE: %.4f\n", mean(final_pae))
        @printf("  max_PAE_cap: %.4f\n", final_pae_max)
        @printf("  pTM: %.6f\n", final_ptm)
    else
        println("  PAE/pTM unavailable (checkpoint has no predicted_aligned_error_head weights).")
    end

    pdb_path = _pdb_path_from_npz(out_path)
    atoms_written = _write_pdb(
        pdb_path,
        final_atom37,
        final_mask,
        aatype;
        bfactor_by_res=final_plddt,
        asym_id=asym_id,
        residue_index=residue_index,
    )

    out_npz = Dict{String,Any}(
        "out_atom37" => final_atom37,
        "atom37_mask" => final_mask,
        "out_masked_msa_logits" => final_masked_msa_logits,
        "out_distogram_logits" => final_distogram_logits,
        "out_distogram_bin_edges" => final_distogram_bin_edges,
        "out_experimentally_resolved_logits" => final_experimentally_resolved_logits,
        "out_predicted_lddt_logits" => final_lddt_logits,
        "out_plddt" => final_plddt,
        "mean_plddt" => Float32(mean(final_plddt)),
        "min_plddt" => Float32(minimum(final_plddt)),
        "max_plddt" => Float32(maximum(final_plddt)),
        "ca_distance_mean" => ca[:mean],
        "ca_distance_std" => ca[:std],
        "ca_distance_min" => ca[:min],
        "ca_distance_max" => ca[:max],
        "ca_distance_outlier_fraction" => ca[:outlier_fraction],
    )
    if final_pae !== nothing
        out_npz["predicted_aligned_error"] = final_pae
        out_npz["max_predicted_aligned_error"] = final_pae_max
        out_npz["predicted_tm_score"] = final_ptm
        out_npz["mean_predicted_aligned_error"] = Float32(mean(final_pae))
    end
    NPZ.npzwrite(out_path, out_npz)
    println("Saved Julia run to ", out_path)
    println("Saved Julia PDB to ", pdb_path, " (atoms=", atoms_written, ")")
end

main()
