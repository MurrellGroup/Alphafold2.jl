using NPZ
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 3
    error("Usage: julia run_af2_core_jl.jl <params_model_1.npz> <sequence> <out.npz> [py_ref.npz] [num_recycle]")
end

function _maybe_parse_int(s::AbstractString)
    v = tryparse(Int, s)
    return v === nothing ? nothing : v
end

params_path = ARGS[1]
sequence = uppercase(strip(ARGS[2]))
out_path = ARGS[3]

py_ref_path = nothing
num_recycle = 1
if length(ARGS) >= 4
    maybe_n = _maybe_parse_int(ARGS[4])
    if maybe_n === nothing
        py_ref_path = ARGS[4]
    else
        num_recycle = max(0, maybe_n)
    end
end
if length(ARGS) >= 5
    maybe_n = _maybe_parse_int(ARGS[5])
    maybe_n === nothing && error("num_recycle must be an integer, got: $(ARGS[5])")
    num_recycle = max(0, maybe_n)
end

const EVO_PREFIX = "alphafold/alphafold_iteration/evoformer"
const EVO_BLOCK_PREFIX = "alphafold/alphafold_iteration/evoformer/evoformer_iteration"
const EXTRA_BLOCK_PREFIX = "alphafold/alphafold_iteration/evoformer/extra_msa_stack"
const SM_PREFIX = "alphafold/alphafold_iteration/structure_module"
const USE_EXTRA_MSA = false

@inline function _slice_block(arr::AbstractArray, block_idx::Int)
    return dropdims(view(arr, block_idx:block_idx, ntuple(_ -> Colon(), ndims(arr) - 1)...); dims=1)
end

function _load_linear_raw!(lin::LinearFirst, arrs::AbstractDict, base::AbstractString; block_idx::Union{Nothing,Int}=nothing)
    wkey = string(base, "//weights")
    bkey = string(base, "//bias")
    w = block_idx === nothing ? arrs[wkey] : _slice_block(arrs[wkey], block_idx)
    lin.weight .= permutedims(w, (2, 1))
    if lin.use_bias
        b = block_idx === nothing ? arrs[bkey] : _slice_block(arrs[bkey], block_idx)
        lin.bias .= b
    end
    return lin
end

function _load_ln_raw!(ln::LayerNormFirst, arrs::AbstractDict, base::AbstractString; block_idx::Union{Nothing,Int}=nothing)
    skey = string(base, "//scale")
    okey = string(base, "//offset")
    s = block_idx === nothing ? arrs[skey] : _slice_block(arrs[skey], block_idx)
    o = block_idx === nothing ? arrs[okey] : _slice_block(arrs[okey], block_idx)
    ln.w .= s
    ln.b .= o
    return ln
end

function _load_attention_raw!(att::AF2Attention, arrs::AbstractDict, base::AbstractString; block_idx::Union{Nothing,Int}=nothing)
    g = key -> (block_idx === nothing ? arrs[key] : _slice_block(arrs[key], block_idx))
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

    _load_ln_raw!(blk.triangle_multiplication_outgoing.layer_norm_input, arrs, string(p, "/triangle_multiplication_outgoing/layer_norm_input"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.left_projection, arrs, string(p, "/triangle_multiplication_outgoing/left_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.right_projection, arrs, string(p, "/triangle_multiplication_outgoing/right_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.left_gate, arrs, string(p, "/triangle_multiplication_outgoing/left_gate"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.right_gate, arrs, string(p, "/triangle_multiplication_outgoing/right_gate"); block_idx=bi)
    _load_ln_raw!(blk.triangle_multiplication_outgoing.center_layer_norm, arrs, string(p, "/triangle_multiplication_outgoing/center_layer_norm"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.output_projection, arrs, string(p, "/triangle_multiplication_outgoing/output_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.gating_linear, arrs, string(p, "/triangle_multiplication_outgoing/gating_linear"); block_idx=bi)

    _load_ln_raw!(blk.triangle_multiplication_incoming.layer_norm_input, arrs, string(p, "/triangle_multiplication_incoming/layer_norm_input"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_incoming.left_projection, arrs, string(p, "/triangle_multiplication_incoming/left_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_incoming.right_projection, arrs, string(p, "/triangle_multiplication_incoming/right_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_incoming.left_gate, arrs, string(p, "/triangle_multiplication_incoming/left_gate"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_incoming.right_gate, arrs, string(p, "/triangle_multiplication_incoming/right_gate"); block_idx=bi)
    _load_ln_raw!(blk.triangle_multiplication_incoming.center_layer_norm, arrs, string(p, "/triangle_multiplication_incoming/center_layer_norm"); block_idx=bi)
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

    _load_ln_raw!(blk.msa_transition.input_layer_norm, arrs, string(p, "/msa_transition/input_layer_norm"); block_idx=bi)
    _load_linear_raw!(blk.msa_transition.transition1, arrs, string(p, "/msa_transition/transition1"); block_idx=bi)
    _load_linear_raw!(blk.msa_transition.transition2, arrs, string(p, "/msa_transition/transition2"); block_idx=bi)

    _load_ln_raw!(blk.triangle_multiplication_outgoing.layer_norm_input, arrs, string(p, "/triangle_multiplication_outgoing/layer_norm_input"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.left_projection, arrs, string(p, "/triangle_multiplication_outgoing/left_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.right_projection, arrs, string(p, "/triangle_multiplication_outgoing/right_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.left_gate, arrs, string(p, "/triangle_multiplication_outgoing/left_gate"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.right_gate, arrs, string(p, "/triangle_multiplication_outgoing/right_gate"); block_idx=bi)
    _load_ln_raw!(blk.triangle_multiplication_outgoing.center_layer_norm, arrs, string(p, "/triangle_multiplication_outgoing/center_layer_norm"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.output_projection, arrs, string(p, "/triangle_multiplication_outgoing/output_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.gating_linear, arrs, string(p, "/triangle_multiplication_outgoing/gating_linear"); block_idx=bi)

    _load_ln_raw!(blk.triangle_multiplication_incoming.layer_norm_input, arrs, string(p, "/triangle_multiplication_incoming/layer_norm_input"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_incoming.left_projection, arrs, string(p, "/triangle_multiplication_incoming/left_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_incoming.right_projection, arrs, string(p, "/triangle_multiplication_incoming/right_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_incoming.left_gate, arrs, string(p, "/triangle_multiplication_incoming/left_gate"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_incoming.right_gate, arrs, string(p, "/triangle_multiplication_incoming/right_gate"); block_idx=bi)
    _load_ln_raw!(blk.triangle_multiplication_incoming.center_layer_norm, arrs, string(p, "/triangle_multiplication_incoming/center_layer_norm"); block_idx=bi)
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

struct ExtraMSAColumnGlobalAttention
    query_norm
    query_w
    key_w
    value_w
    output_w
    output_b
    gating_w
    gating_b
end

function _load_extra_column_global_attention(arrs::AbstractDict, bi::Int, c_m::Int)
    p = string(EXTRA_BLOCK_PREFIX, "/msa_column_global_attention")
    g = key -> _slice_block(arrs[key], bi)
    qn = LayerNormFirst(c_m)
    _load_ln_raw!(qn, arrs, string(p, "/query_norm"); block_idx=bi)
    return ExtraMSAColumnGlobalAttention(
        qn,
        g(string(p, "/attention//query_w")),
        g(string(p, "/attention//key_w")),
        g(string(p, "/attention//value_w")),
        g(string(p, "/attention//output_w")),
        g(string(p, "/attention//output_b")),
        g(string(p, "/attention//gating_w")),
        g(string(p, "/attention//gating_b")),
    )
end

function _softmax_last(x::AbstractArray)
    x0 = x .- maximum(x; dims=ndims(x))
    ex = exp.(x0)
    ex ./ sum(ex; dims=ndims(ex))
end

function _extra_column_global_attention(
    m::ExtraMSAColumnGlobalAttention,
    msa_act::AbstractArray,
    msa_mask::AbstractArray,
)
    # msa_act: (C, S, L, B), msa_mask: (S, L, B)
    C = size(msa_act, 1)
    S = size(msa_act, 2)
    L = size(msa_act, 3)
    B = size(msa_act, 4)
    H = size(m.query_w, 2)
    K = size(m.query_w, 3)
    V = size(m.value_w, 2)

    x = permutedims(msa_act, (1, 3, 2, 4))  # (C, L, S, B)
    mask = permutedims(msa_mask, (2, 1, 3)) # (L, S, B)
    x = m.query_norm(x)

    out = similar(x)
    for b in 1:B, l in 1:L
        q_data = permutedims(view(x, :, l, :, b), (2, 1)) # (S, C)
        q_mask = vec(view(mask, l, :, b))                 # (S)

        maskw = reshape(q_mask, S, 1)
        denom = max(sum(q_mask), 1f-6)
        q_avg = sum(q_data .* maskw; dims=1) ./ denom     # (1, C)
        q_avg_vec = vec(q_avg)                             # (C)

        q = reshape(q_avg_vec' * reshape(m.query_w, C, H * K), H, K) .* Float32(K)^-0.5f0 # (H, K)
        k = q_data * m.key_w                               # (S, K)
        v = q_data * m.value_w                             # (S, V)

        logits = q * permutedims(k, (2, 1))               # (H, S)
        for s in 1:S
            if q_mask[s] <= 0f0
                logits[:, s] .= -1f9
            end
        end
        weights = _softmax_last(reshape(logits, H, S, 1))
        weights = dropdims(weights; dims=3)               # (H, S)

        weighted_avg = weights * v                         # (H, V)

        gate = reshape(q_data * reshape(m.gating_w, C, H * V), S, H, V)
        gate .+= reshape(m.gating_b, 1, H, V)
        gate = NNlib.sigmoid.(gate)

        weighted = reshape(weighted_avg, 1, H, V) .* gate # (S, H, V)
        out_seq = reshape(weighted, S * H, V)'            # (V, S*H)
        out_seq = permutedims(weighted, (1, 2, 3))        # (S, H, V)

        y = zeros(Float32, S, C)
        for s in 1:S
            y[s, :] .= vec(reshape(weighted[s, :, :], 1, H * V) * reshape(m.output_w, H * V, C)) .+ m.output_b
        end

        out[:, l, :, b] .= permutedims(y, (2, 1))         # (C, S)
    end

    return permutedims(out, (1, 3, 2, 4)) # (C, S, L, B)
end

function _extra_block_forward!(
    blk::EvoformerIteration,
    col_attn::ExtraMSAColumnGlobalAttention,
    msa_act::AbstractArray,
    pair_act::AbstractArray,
    msa_mask::AbstractArray,
    pair_mask::AbstractArray,
)
    if blk.outer_first
        pair_act = pair_act .+ blk.outer_product_mean(msa_act, msa_mask)
    end

    msa_act = msa_act .+ blk.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act)
    msa_act = msa_act .+ _extra_column_global_attention(col_attn, msa_act, msa_mask)
    msa_act = msa_act .+ blk.msa_transition(msa_act, msa_mask)

    if !blk.outer_first
        pair_act = pair_act .+ blk.outer_product_mean(msa_act, msa_mask)
    end

    pair_act = pair_act .+ blk.triangle_multiplication_outgoing(pair_act, pair_mask)
    pair_act = pair_act .+ blk.triangle_multiplication_incoming(pair_act, pair_mask)
    pair_act = pair_act .+ blk.triangle_attention_starting_node(pair_act, pair_mask)
    pair_act = pair_act .+ blk.triangle_attention_ending_node(pair_act, pair_mask)
    pair_act = pair_act .+ blk.pair_transition(pair_act, pair_mask)

    return msa_act, pair_act
end

function _load_structure_core_raw!(m::StructureModuleCore, arrs::AbstractDict)
    p = SM_PREFIX

    _load_ln_raw!(m.single_layer_norm, arrs, string(p, "/single_layer_norm"))
    _load_linear_raw!(m.initial_projection, arrs, string(p, "/initial_projection"))
    _load_ln_raw!(m.pair_layer_norm, arrs, string(p, "/pair_layer_norm"))

    _load_linear_raw!(m.fold_iteration_core.ipa.linear_q, arrs, string(p, "/fold_iteration/invariant_point_attention/q_scalar"))
    _load_linear_raw!(m.fold_iteration_core.ipa.linear_q_points.linear, arrs, string(p, "/fold_iteration/invariant_point_attention/q_point_local"))
    _load_linear_raw!(m.fold_iteration_core.ipa.linear_kv, arrs, string(p, "/fold_iteration/invariant_point_attention/kv_scalar"))
    _load_linear_raw!(m.fold_iteration_core.ipa.linear_kv_points.linear, arrs, string(p, "/fold_iteration/invariant_point_attention/kv_point_local"))
    _load_linear_raw!(m.fold_iteration_core.ipa.linear_b, arrs, string(p, "/fold_iteration/invariant_point_attention/attention_2d"))
    m.fold_iteration_core.ipa.head_weights .= arrs[string(p, "/fold_iteration/invariant_point_attention//trainable_point_weights")]
    _load_linear_raw!(m.fold_iteration_core.ipa.linear_out, arrs, string(p, "/fold_iteration/invariant_point_attention/output_projection"))

    _load_ln_raw!(m.fold_iteration_core.attention_layer_norm, arrs, string(p, "/fold_iteration/attention_layer_norm"))
    _load_ln_raw!(m.fold_iteration_core.transition_layer_norm, arrs, string(p, "/fold_iteration/transition_layer_norm"))
    _load_linear_raw!(m.fold_iteration_core.affine_update, arrs, string(p, "/fold_iteration/affine_update"))

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

function _sequence_to_aatype(seq::AbstractString)
    L = length(seq)
    out = Array{Int}(undef, L, 1)
    for (i, ch) in enumerate(seq)
        out[i, 1] = get(Alphafold2.restype_order, string(ch), 20)
    end
    return out
end

function _one_hot_aatype(aatype::AbstractMatrix{Int}, num_classes::Int)
    L, B = size(aatype)
    out = zeros(Float32, num_classes, L, B)
    for b in 1:B, i in 1:L
        k = clamp(aatype[i, b] + 1, 1, num_classes)
        out[k, i, b] = 1f0
    end
    return out
end

function _msa_one_hot(msa::Array{Int,3}, num_classes::Int)
    S, L, B = size(msa)
    out = zeros(Float32, num_classes, S, L, B)
    for b in 1:B, i in 1:L, s in 1:S
        k = clamp(msa[s, i, b] + 1, 1, num_classes)
        out[k, s, i, b] = 1f0
    end
    return out
end

function _build_minimal_features(aatype::AbstractMatrix{Int})
    L, B = size(aatype)
    S = 1

    seq_mask = ones(Float32, L, B)
    msa_mask = ones(Float32, S, L, B)
    residue_index = repeat(reshape(collect(0:(L - 1)), L, 1), 1, B)

    has_break = zeros(Float32, 1, L, B)
    aatype_1hot = _one_hot_aatype(aatype, 21)
    target_feat = cat(has_break, aatype_1hot; dims=1) # (22, L, B)

    msa = zeros(Int, S, L, B)
    for b in 1:B, i in 1:L
        msa[1, i, b] = aatype[i, b]
    end

    msa_1hot = _msa_one_hot(msa, 23)
    has_del = zeros(Float32, 1, S, L, B)
    del_val = zeros(Float32, 1, S, L, B)
    cluster_profile = copy(msa_1hot)
    deletion_mean = zeros(Float32, 1, S, L, B)

    msa_feat = cat(msa_1hot, has_del, del_val, cluster_profile, deletion_mean; dims=1) # (49, S, L, B)

    return Dict{Symbol,Any}(
        :target_feat => target_feat,
        :msa_feat => msa_feat,
        :seq_mask => seq_mask,
        :msa_mask => msa_mask,
        :residue_index => residue_index,
        :aatype => aatype,
    )
end

function _relpos_one_hot(residue_index::AbstractMatrix{Int}, max_relative_feature::Int)
    L, B = size(residue_index)
    offset = reshape(residue_index, L, 1, B) .- reshape(residue_index, 1, L, B)
    idx = clamp.(offset .+ max_relative_feature, 0, 2 * max_relative_feature)
    oh = Alphafold2.one_hot_last(idx, 2 * max_relative_feature + 1) # (L, L, B, C)
    return Float32.(permutedims(oh, (4, 1, 2, 3))) # (C, L, L, B)
end

function _pseudo_beta_from_atom37(aatype::AbstractMatrix{Int}, atom37_bllc::AbstractArray)
    # aatype: (L, B), atom37_bllc: (B, L, 37, 3), return: (3, L, B)
    ca_idx = Alphafold2.atom_order["CA"] + 1
    cb_idx = Alphafold2.atom_order["CB"] + 1
    gly_idx = Alphafold2.restype_order["G"]

    ca = permutedims(view(atom37_bllc, :, :, ca_idx, :), (3, 2, 1)) # (3, L, B)
    cb = permutedims(view(atom37_bllc, :, :, cb_idx, :), (3, 2, 1)) # (3, L, B)

    is_gly = reshape(aatype .== gly_idx, 1, size(aatype, 1), size(aatype, 2)) # (1, L, B)
    return ifelse.(is_gly, ca, cb)
end

function _dgram_from_positions(positions::AbstractArray; num_bins::Int=15, min_bin::Real=3.25f0, max_bin::Real=20.75f0)
    # positions: (3, L, B), return: (num_bins, L, L, B)
    L = size(positions, 2)
    B = size(positions, 3)
    out = zeros(Float32, num_bins, L, L, B)

    lower_breaks = collect(range(Float32(min_bin), Float32(max_bin); length=num_bins))
    lower2 = lower_breaks .^ 2
    upper2 = vcat(lower2[2:end], Float32[1f8])

    for b in 1:B
        p = permutedims(view(positions, :, :, b), (2, 1)) # (L, 3)
        diff = reshape(p, L, 1, 3) .- reshape(p, 1, L, 3)
        dist2 = sum(diff .^ 2; dims=3) # (L, L, 1)
        d2 = dropdims(dist2; dims=3)
        for k in 1:num_bins
            out[k, :, :, b] .= Float32.(d2 .> lower2[k]) .* Float32.(d2 .< upper2[k])
        end
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

function _ca_distance_metrics(atom37::AbstractArray, atom37_mask::AbstractArray)
    ca_idx = Alphafold2.atom_order["CA"] + 1
    L = size(atom37, 1)
    valid = atom37_mask[:, ca_idx] .> 0.5f0

    dists = Float32[]
    for i in 1:(L - 1)
        if valid[i] && valid[i + 1]
            dx = atom37[i + 1, ca_idx, 1] - atom37[i, ca_idx, 1]
            dy = atom37[i + 1, ca_idx, 2] - atom37[i, ca_idx, 2]
            dz = atom37[i + 1, ca_idx, 3] - atom37[i, ca_idx, 3]
            push!(dists, sqrt(dx * dx + dy * dy + dz * dz))
        end
    end

    if isempty(dists)
        return Dict{Symbol,Any}(
            :distances => Float32[],
            :count => 0,
            :mean => NaN32,
            :std => NaN32,
            :min => NaN32,
            :max => NaN32,
            :outlier_count => 0,
            :outlier_fraction => NaN32,
            :ok => false,
        )
    end

    d = collect(dists)
    outliers = sum((d .< 3.2f0) .| (d .> 4.4f0))
    outlier_frac = outliers / length(d)
    mean_d = mean(d)
    std_d = std(d)
    min_d = minimum(d)
    max_d = maximum(d)

    # Conservative plausibility rule for consecutive Cα distances.
    ok = (min_d > 3.0f0) && (max_d < 4.8f0) && (outlier_frac < 0.25)

    return Dict{Symbol,Any}(
        :distances => d,
        :count => length(d),
        :mean => Float32(mean_d),
        :std => Float32(std_d),
        :min => Float32(min_d),
        :max => Float32(max_d),
        :outlier_count => Int(outliers),
        :outlier_fraction => Float32(outlier_frac),
        :ok => ok,
    )
end

function _pdb_path_from_npz(npz_path::AbstractString)
    if endswith(lowercase(npz_path), ".npz")
        return string(first(npz_path, lastindex(npz_path) - 4), ".pdb")
    end
    return string(npz_path, ".pdb")
end

function _write_pdb(path::AbstractString, atom37::AbstractArray, atom37_mask::AbstractArray, aatype::AbstractArray)
    L = size(atom37, 1)
    atom_serial = 1
    chain_id = 'A'

    open(path, "w") do io
        for i in 1:L
            aa_idx0 = Int(aatype[i, 1])
            resname = if 0 <= aa_idx0 < length(Alphafold2.restypes)
                Alphafold2.restype_1to3[Alphafold2.restypes[aa_idx0 + 1]]
            else
                "UNK"
            end

            for a_idx in 1:length(Alphafold2.atom_types)
                atom37_mask[i, a_idx] < 0.5f0 && continue
                atom_name = Alphafold2.atom_types[a_idx]
                x = atom37[i, a_idx, 1]
                y = atom37[i, a_idx, 2]
                z = atom37[i, a_idx, 3]
                element = uppercase(first(atom_name, 1))

                line = @sprintf(
                    "ATOM  %5d %4s %3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s",
                    atom_serial,
                    atom_name,
                    resname,
                    chain_id,
                    i,
                    x,
                    y,
                    z,
                    1.00,
                    0.00,
                    element,
                )
                println(io, line)
                atom_serial += 1
            end
        end
        println(io, "END")
    end

    return atom_serial - 1
end

arrs = NPZ.npzread(params_path)

c_m = length(arrs[string(EVO_PREFIX, "/preprocess_1d//bias")])
c_z = length(arrs[string(EVO_PREFIX, "/left_single//bias")])
c_s = length(arrs[string(EVO_PREFIX, "/single_activations//bias")])
c_m_extra = length(arrs[string(EVO_PREFIX, "/extra_msa_activations//bias")])

num_blocks = size(arrs[string(EVO_BLOCK_PREFIX, "/msa_column_attention/attention//output_b")], 1)
num_extra_blocks = size(arrs[string(EXTRA_BLOCK_PREFIX, "/msa_transition/transition1//bias")], 1)

msa_qw = arrs[string(EVO_BLOCK_PREFIX, "/msa_column_attention/attention//query_w")]
num_head_msa = size(msa_qw, 3)
msa_head_dim = size(msa_qw, 4)

pair_qw = arrs[string(EVO_BLOCK_PREFIX, "/triangle_attention_starting_node/attention//query_w")]
num_head_pair = size(pair_qw, 3)
pair_head_dim = size(pair_qw, 4)

c_outer = size(arrs[string(EVO_BLOCK_PREFIX, "/outer_product_mean/left_projection//bias")], 2)
c_tri_mul = size(arrs[string(EVO_BLOCK_PREFIX, "/triangle_multiplication_outgoing/left_projection//bias")], 2)

msa_transition_factor = size(arrs[string(EVO_BLOCK_PREFIX, "/msa_transition/transition1//bias")], 2) / c_m
pair_transition_factor = size(arrs[string(EVO_BLOCK_PREFIX, "/pair_transition/transition1//bias")], 2) / c_z

extra_msa_qw = arrs[string(EXTRA_BLOCK_PREFIX, "/msa_row_attention_with_pair_bias/attention//query_w")]
num_head_msa_extra = size(extra_msa_qw, 3)
msa_head_dim_extra = size(extra_msa_qw, 4)

extra_pair_qw = arrs[string(EXTRA_BLOCK_PREFIX, "/triangle_attention_starting_node/attention//query_w")]
num_head_pair_extra = size(extra_pair_qw, 3)
pair_head_dim_extra = size(extra_pair_qw, 4)

c_outer_extra = size(arrs[string(EXTRA_BLOCK_PREFIX, "/outer_product_mean/left_projection//bias")], 2)
c_tri_mul_extra = size(arrs[string(EXTRA_BLOCK_PREFIX, "/triangle_multiplication_outgoing/left_projection//bias")], 2)
msa_transition_factor_extra = size(arrs[string(EXTRA_BLOCK_PREFIX, "/msa_transition/transition1//bias")], 2) / c_m_extra
pair_transition_factor_extra = size(arrs[string(EXTRA_BLOCK_PREFIX, "/pair_transition/transition1//bias")], 2) / c_z

preprocess_1d = LinearFirst(22, c_m)
preprocess_msa = LinearFirst(49, c_m)
left_single = LinearFirst(22, c_z)
right_single = LinearFirst(22, c_z)
extra_msa_activations = LinearFirst(25, c_m_extra)
prev_pos_linear = LinearFirst(15, c_z)
prev_msa_first_row_norm = LayerNormFirst(c_m)
prev_pair_norm = LayerNormFirst(c_z)
pair_relpos = LinearFirst(65, c_z)
single_activations = LinearFirst(c_m, c_s)

_load_linear_raw!(preprocess_1d, arrs, string(EVO_PREFIX, "/preprocess_1d"))
_load_linear_raw!(preprocess_msa, arrs, string(EVO_PREFIX, "/preprocess_msa"))
_load_linear_raw!(left_single, arrs, string(EVO_PREFIX, "/left_single"))
_load_linear_raw!(right_single, arrs, string(EVO_PREFIX, "/right_single"))
_load_linear_raw!(extra_msa_activations, arrs, string(EVO_PREFIX, "/extra_msa_activations"))
_load_linear_raw!(prev_pos_linear, arrs, string(EVO_PREFIX, "/prev_pos_linear"))
_load_ln_raw!(prev_msa_first_row_norm, arrs, string(EVO_PREFIX, "/prev_msa_first_row_norm"))
_load_ln_raw!(prev_pair_norm, arrs, string(EVO_PREFIX, "/prev_pair_norm"))
_load_linear_raw!(pair_relpos, arrs, string(EVO_PREFIX, "/pair_activiations"))
_load_linear_raw!(single_activations, arrs, string(EVO_PREFIX, "/single_activations"))

blocks = [
    EvoformerIteration(
        c_m,
        c_z;
        num_head_msa=num_head_msa,
        msa_head_dim=msa_head_dim,
        num_head_pair=num_head_pair,
        pair_head_dim=pair_head_dim,
        c_outer=c_outer,
        c_tri_mul=c_tri_mul,
        msa_transition_factor=msa_transition_factor,
        pair_transition_factor=pair_transition_factor,
        outer_first=false,
    )
    for _ in 1:num_blocks
]

for i in 1:num_blocks
    _load_evo_block_raw!(blocks[i], arrs, i)
end

extra_blocks = [
    EvoformerIteration(
        c_m_extra,
        c_z;
        num_head_msa=num_head_msa_extra,
        msa_head_dim=msa_head_dim_extra,
        num_head_pair=num_head_pair_extra,
        pair_head_dim=pair_head_dim_extra,
        c_outer=c_outer_extra,
        c_tri_mul=c_tri_mul_extra,
        msa_transition_factor=msa_transition_factor_extra,
        pair_transition_factor=pair_transition_factor_extra,
        outer_first=false,
    )
    for _ in 1:num_extra_blocks
]

for i in 1:num_extra_blocks
    _load_extra_block_raw!(extra_blocks[i], arrs, i)
end
extra_col_blocks = [_load_extra_column_global_attention(arrs, i, c_m_extra) for i in 1:num_extra_blocks]

num_transition_layers = _infer_transition_depth(arrs, string(SM_PREFIX, "/fold_iteration/transition"))
num_residual_block = _infer_transition_depth(arrs, string(SM_PREFIX, "/fold_iteration/rigid_sidechain/resblock1"))

no_heads = length(arrs[string(SM_PREFIX, "/fold_iteration/invariant_point_attention/attention_2d//bias")])
c_hidden_total = size(arrs[string(SM_PREFIX, "/fold_iteration/invariant_point_attention/q_scalar//weights")], 2)
c_hidden = Int(div(c_hidden_total, no_heads))
q_point_total = size(arrs[string(SM_PREFIX, "/fold_iteration/invariant_point_attention/q_point_local//bias")], 1)
kv_point_total = size(arrs[string(SM_PREFIX, "/fold_iteration/invariant_point_attention/kv_point_local//bias")], 1)
no_qk_points = Int(div(q_point_total, 3 * no_heads))
no_v_points = Int(div(kv_point_total, 3 * no_heads) - no_qk_points)

sidechain_num_channel = length(arrs[string(SM_PREFIX, "/fold_iteration/rigid_sidechain/input_projection//bias")])
num_structure_layers = 8
position_scale = 10f0

structure = StructureModuleCore(
    c_s,
    c_z,
    c_hidden,
    no_heads,
    no_qk_points,
    no_v_points,
    num_transition_layers,
    num_structure_layers,
    position_scale,
    sidechain_num_channel,
    num_residual_block,
)
_load_structure_core_raw!(structure, arrs)

features = _build_minimal_features(_sequence_to_aatype(sequence))

target_feat = features[:target_feat]
msa_feat = features[:msa_feat]
extra_msa_feat = view(msa_feat, 1:25, :, :, :)
seq_mask = features[:seq_mask]
msa_mask = features[:msa_mask]
residue_index = features[:residue_index]
aatype = features[:aatype]

pair_mask = reshape(seq_mask, size(seq_mask, 1), 1, size(seq_mask, 2)) .* reshape(seq_mask, 1, size(seq_mask, 1), size(seq_mask, 2))

L = size(aatype, 1)
B = size(aatype, 2)

function _run_recycling(
    num_recycle::Int,
    use_extra_msa::Bool,
    preprocess_1d,
    preprocess_msa,
    left_single,
    right_single,
    extra_msa_activations,
    prev_pos_linear,
    prev_msa_first_row_norm,
    prev_pair_norm,
    pair_relpos,
    extra_blocks,
    extra_col_blocks,
    blocks,
    single_activations,
    structure,
    target_feat,
    msa_feat,
    extra_msa_feat,
    seq_mask,
    msa_mask,
    residue_index,
    aatype,
    pair_mask,
    c_m::Int,
    c_z::Int,
    L::Int,
    B::Int,
)
    prev_atom37 = zeros(Float32, B, L, 37, 3)
    prev_msa_first_row = zeros(Float32, c_m, L, B)
    prev_pair = zeros(Float32, c_z, L, L, B)

    msa_act = nothing
    pair_act = nothing
    single = nothing
    struct_out = nothing
    final_atom14 = nothing
    final_affine = nothing
    final_atom37 = nothing
    protein = nothing

    for recycle_idx in 0:num_recycle
        p1 = preprocess_1d(target_feat)
        pmsa = preprocess_msa(msa_feat)
        msa_act = pmsa .+ reshape(p1, size(p1, 1), 1, size(p1, 2), size(p1, 3))

        left = left_single(target_feat)
        right = right_single(target_feat)
        pair_act = reshape(left, size(left, 1), size(left, 2), 1, size(left, 3)) .+
                   reshape(right, size(right, 1), 1, size(right, 2), size(right, 3))

        prev_pb = _pseudo_beta_from_atom37(aatype, prev_atom37)
        pair_act = pair_act .+ prev_pos_linear(_dgram_from_positions(prev_pb))

        msa_act[:, 1:1, :, :] .+= reshape(prev_msa_first_row_norm(prev_msa_first_row), c_m, 1, L, B)
        pair_act .+= prev_pair_norm(prev_pair)

        pair_act = pair_act .+ pair_relpos(_relpos_one_hot(residue_index, 32))

        if use_extra_msa
            extra_msa_act = extra_msa_activations(extra_msa_feat)
            for i in eachindex(extra_blocks)
                extra_msa_act, pair_act = _extra_block_forward!(
                    extra_blocks[i],
                    extra_col_blocks[i],
                    extra_msa_act,
                    pair_act,
                    msa_mask,
                    pair_mask,
                )
            end
        end

        for block in blocks
            msa_act, pair_act = block(msa_act, pair_act, msa_mask, pair_mask)
        end

        single = single_activations(view(msa_act, :, 1, :, :))
        struct_out = structure(single, pair_act, seq_mask, aatype)

        final_atom14 = dropdims(view(struct_out[:atom_pos], size(struct_out[:atom_pos], 1):size(struct_out[:atom_pos], 1), :, :, :, :); dims=1)
        final_affine = dropdims(view(struct_out[:affine], size(struct_out[:affine], 1):size(struct_out[:affine], 1), :, :, :); dims=1)

        protein = Dict{Symbol,Any}(:aatype => aatype, :s_s => single)
        make_atom14_masks!(protein)
        final_atom37 = atom14_to_atom37(final_atom14, protein)

        prev_atom37 = final_atom37
        prev_msa_first_row = view(msa_act, :, 1, :, :)
        prev_pair = pair_act

        @printf("Recycle %d/%d complete\n", recycle_idx, num_recycle)
    end

    return msa_act, pair_act, single, struct_out, final_atom14, final_affine, final_atom37, protein
end

msa_act, pair_act, single, struct_out, final_atom14, final_affine, final_atom37, protein = _run_recycling(
    num_recycle,
    USE_EXTRA_MSA,
    preprocess_1d,
    preprocess_msa,
    left_single,
    right_single,
    extra_msa_activations,
    prev_pos_linear,
    prev_msa_first_row_norm,
    prev_pair_norm,
    pair_relpos,
    extra_blocks,
    extra_col_blocks,
    blocks,
    single_activations,
    structure,
    target_feat,
    msa_feat,
    extra_msa_feat,
    seq_mask,
    msa_mask,
    residue_index,
    aatype,
    pair_mask,
    c_m,
    c_z,
    L,
    B,
)

single_py = dropdims(first_to_af2_3d(single); dims=1)
pair_py = dropdims(first_to_af2_2d(pair_act); dims=1)
atom14_py = dropdims(permutedims(final_atom14, (3, 2, 1, 4)); dims=4)
affine_py = dropdims(permutedims(final_affine, (2, 1, 3)); dims=3)
atom37_py = dropdims(final_atom37; dims=1)
atom37_mask_py = dropdims(permutedims(protein[:atom37_atom_exists], (2, 3, 1)); dims=2)

ca_metrics = _ca_distance_metrics(atom37_py, atom37_mask_py)
@printf("Geometry check (consecutive C-alpha distances)\n")
@printf("  count: %d\n", ca_metrics[:count])
@printf("  mean: %.6f A\n", ca_metrics[:mean])
@printf("  std:  %.6f A\n", ca_metrics[:std])
@printf("  min:  %.6f A\n", ca_metrics[:min])
@printf("  max:  %.6f A\n", ca_metrics[:max])
@printf("  outliers (<3.2 or >4.4 A): %d (%.3f)\n", ca_metrics[:outlier_count], ca_metrics[:outlier_fraction])
println("  status: ", ca_metrics[:ok] ? "PASS" : "WARN")

pdb_path = _pdb_path_from_npz(out_path)
pdb_atoms = _write_pdb(pdb_path, atom37_py, atom37_mask_py, aatype)
println("Wrote PDB: ", pdb_path, " (atoms=", pdb_atoms, ")")

if py_ref_path !== nothing
    ref = NPZ.npzread(py_ref_path)

    ds = maximum(abs.(single_py .- ref["out_single"]))
    dp = maximum(abs.(pair_py .- ref["out_pair"]))
    da = maximum(abs.(atom14_py .- ref["out_atom14"]))
    df = maximum(abs.(affine_py .- ref["out_affine"]))

    @printf("AF2 core parity vs Python ref\n")
    @printf("  single_max_abs: %.8g\n", ds)
    @printf("  pair_max_abs:   %.8g\n", dp)
    @printf("  atom14_max_abs: %.8g\n", da)
    @printf("  affine_max_abs: %.8g\n", df)
end

out = Dict{String,Any}(
    "aatype" => aatype,
    "target_feat" => dropdims(permutedims(target_feat, (2, 1, 3)); dims=3),
    "msa_feat" => dropdims(permutedims(msa_feat, (2, 3, 1, 4)); dims=4),
    "seq_mask" => dropdims(seq_mask; dims=2),
    "msa_mask" => dropdims(permutedims(msa_mask, (1, 2, 3)); dims=3),
    "residue_index" => dropdims(residue_index; dims=2),
    "out_single" => single_py,
    "out_pair" => pair_py,
    "out_atom14" => atom14_py,
    "out_affine" => affine_py,
    "out_atom37" => atom37_py,
    "atom37_mask" => atom37_mask_py,
    "ca_consecutive_distances" => ca_metrics[:distances],
    "ca_distance_mean" => ca_metrics[:mean],
    "ca_distance_std" => ca_metrics[:std],
    "ca_distance_min" => ca_metrics[:min],
    "ca_distance_max" => ca_metrics[:max],
    "ca_distance_outlier_fraction" => ca_metrics[:outlier_fraction],
    "num_recycle" => Int32(num_recycle),
)

mkpath(dirname(out_path))
NPZ.npzwrite(out_path, out)
println("Saved AF2 core Julia run to ", out_path)
