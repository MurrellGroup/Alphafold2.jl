using NPZ
using NNlib
using Printf
using Statistics
using Zygote

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 2
    error("Usage: julia check_full_model_zygote.jl <params_model(.npz)> <sequence_or_sequences_csv> [num_evo_blocks]")
end

params_path = ARGS[1]
sequence_arg = ARGS[2]
num_evo_blocks_override = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : -1

const EVO_PREFIX = "alphafold/alphafold_iteration/evoformer"
const EVO_BLOCK_PREFIX = "alphafold/alphafold_iteration/evoformer/evoformer_iteration"
const SM_PREFIX = "alphafold/alphafold_iteration/structure_module"
const LDDT_PREFIX = "alphafold/alphafold_iteration/predicted_lddt_head"

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

function _load_evo_block_raw!(blk::EvoformerIteration, arrs::AbstractDict, bi::Int)
    p = EVO_BLOCK_PREFIX

    _load_ln_raw!(blk.outer_product_mean.layer_norm_input, arrs, string(p, "/outer_product_mean/layer_norm_input"); block_idx=bi)
    _load_linear_raw!(blk.outer_product_mean.left_projection, arrs, string(p, "/outer_product_mean/left_projection"); block_idx=bi)
    _load_linear_raw!(blk.outer_product_mean.right_projection, arrs, string(p, "/outer_product_mean/right_projection"); block_idx=bi)
    blk.outer_product_mean.output_w .= _slice_block(_arr_get(arrs, string(p, "/outer_product_mean//output_w")), bi)
    blk.outer_product_mean.output_b .= _slice_block(_arr_get(arrs, string(p, "/outer_product_mean//output_b")), bi)

    _load_ln_raw!(blk.msa_row_attention_with_pair_bias.query_norm, arrs, string(p, "/msa_row_attention_with_pair_bias/query_norm"); block_idx=bi)
    _load_ln_raw!(blk.msa_row_attention_with_pair_bias.feat_2d_norm, arrs, string(p, "/msa_row_attention_with_pair_bias/feat_2d_norm"); block_idx=bi)
    blk.msa_row_attention_with_pair_bias.feat_2d_weights .= _slice_block(_arr_get(arrs, string(p, "/msa_row_attention_with_pair_bias//feat_2d_weights")), bi)
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
    blk.triangle_attention_starting_node.feat_2d_weights .= _slice_block(_arr_get(arrs, string(p, "/triangle_attention_starting_node//feat_2d_weights")), bi)
    _load_attention_raw!(blk.triangle_attention_starting_node.attention, arrs, string(p, "/triangle_attention_starting_node/attention"); block_idx=bi)

    _load_ln_raw!(blk.triangle_attention_ending_node.query_norm, arrs, string(p, "/triangle_attention_ending_node/query_norm"); block_idx=bi)
    blk.triangle_attention_ending_node.feat_2d_weights .= _slice_block(_arr_get(arrs, string(p, "/triangle_attention_ending_node//feat_2d_weights")), bi)
    _load_attention_raw!(blk.triangle_attention_ending_node.attention, arrs, string(p, "/triangle_attention_ending_node/attention"); block_idx=bi)

    _load_ln_raw!(blk.pair_transition.input_layer_norm, arrs, string(p, "/pair_transition/input_layer_norm"); block_idx=bi)
    _load_linear_raw!(blk.pair_transition.transition1, arrs, string(p, "/pair_transition/transition1"); block_idx=bi)
    _load_linear_raw!(blk.pair_transition.transition2, arrs, string(p, "/pair_transition/transition2"); block_idx=bi)
    return blk
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
        q_w = _arr_get(arrs, string(ipa_base, "/q_scalar_projection//weights"))
        k_w = _arr_get(arrs, string(ipa_base, "/k_scalar_projection//weights"))
        v_w = _arr_get(arrs, string(ipa_base, "/v_scalar_projection//weights"))
        q_w2 = reshape(permutedims(q_w, (1, 3, 2)), size(q_w, 1), :)
        k_w2 = reshape(permutedims(k_w, (1, 3, 2)), size(k_w, 1), :)
        v_w2 = reshape(permutedims(v_w, (1, 3, 2)), size(v_w, 1), :)
        m.fold_iteration_core.ipa.linear_q.weight .= permutedims(q_w2, (2, 1))
        m.fold_iteration_core.ipa.linear_k.weight .= permutedims(k_w2, (2, 1))
        m.fold_iteration_core.ipa.linear_v.weight .= permutedims(v_w2, (2, 1))

        qp_w = _arr_get(arrs, string(ipa_base, "/q_point_projection/point_projection//weights"))
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

    _load_linear_raw!(m.fold_iteration_core.ipa.linear_b, arrs, string(ipa_base, "/attention_2d"))
    m.fold_iteration_core.ipa.head_weights .= _arr_get(arrs, string(ipa_base, "//trainable_point_weights"))
    _load_linear_raw!(m.fold_iteration_core.ipa.linear_out, arrs, string(ipa_base, "/output_projection"))

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

function _infer_transition_depth(arrs::AbstractDict, base::AbstractString)
    n = 0
    _has_key(arrs, string(base, "//weights")) && (n += 1)
    i = 1
    while _has_key(arrs, string(base, "_", i, "//weights"))
        n += 1
        i += 1
    end
    return n
end

function _parse_sequences_csv(arg::AbstractString)
    seqs = [uppercase(strip(s)) for s in split(arg, ",") if !isempty(strip(s))]
    isempty(seqs) && error("No sequence provided.")
    return seqs
end

function _chain_metadata(seqs::Vector{String})
    chain_aatype = [Int.(get.(Ref(Alphafold2.restype_order), string.(collect(s)), 20)) for s in seqs]
    aatype = vcat(chain_aatype...)
    chain_lens = [length(s) for s in seqs]
    starts = cumsum(vcat(1, chain_lens[1:end-1]))
    L = sum(chain_lens)

    residue_index = zeros(Int, L)
    asym_id = zeros(Int, L)
    entity_id = zeros(Int, L)
    sym_id = zeros(Int, L)

    seq_to_entity = Dict{String,Int}()
    next_entity = 1
    entity_by_chain = Vector{Int}(undef, length(seqs))
    for i in eachindex(seqs)
        s = seqs[i]
        if !haskey(seq_to_entity, s)
            seq_to_entity[s] = next_entity
            next_entity += 1
        end
        entity_by_chain[i] = seq_to_entity[s]
    end
    entity_counts = Dict{Int,Int}()

    for ci in eachindex(seqs)
        st = starts[ci]
        en = st + chain_lens[ci] - 1
        residue_index[st:en] .= collect(0:(chain_lens[ci] - 1))
        asym_id[st:en] .= ci
        ent = entity_by_chain[ci]
        entity_id[st:en] .= ent
        sym = get(entity_counts, ent, 0) + 1
        entity_counts[ent] = sym
        sym_id[st:en] .= sym
    end

    return aatype, residue_index, asym_id, entity_id, sym_id, chain_lens
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

function _dgram_from_positions(positions::AbstractArray; num_bins::Int=15, min_bin::Real=3.25f0, max_bin::Real=20.75f0)
    L = size(positions, 2)
    B = size(positions, 3)
    out = zeros(Float32, num_bins, L, L, B)
    lower_breaks = collect(range(Float32(min_bin), Float32(max_bin); length=num_bins))
    lower2 = lower_breaks .^ 2
    upper2 = vcat(lower2[2:end], Float32[1f8])
    for b in 1:B
        p = permutedims(view(positions, :, :, b), (2, 1))
        diff = reshape(p, L, 1, 3) .- reshape(p, 1, L, 3)
        d2 = dropdims(sum(diff .^ 2; dims=3); dims=3)
        for k in 1:num_bins
            out[k, :, :, b] .= Float32.(d2 .> lower2[k]) .* Float32.(d2 .< upper2[k])
        end
    end
    return out
end

struct AF2GradModel
    preprocess_1d
    preprocess_msa
    left_single
    right_single
    prev_pos_linear
    prev_msa_first_row_norm
    prev_pair_norm
    pair_relpos
    blocks
    single_activations
    structure
    predicted_lddt
    seq_mask
    msa_mask
    pair_mask
    residue_index
    aatype
    prev_pos_dgram
    target_feat_dim
    relpos_feat
end

function (m::AF2GradModel)(seq_logits::AbstractArray)
    seq_feats = build_soft_sequence_features(seq_logits)
    target_feat = m.target_feat_dim == 22 ? seq_feats[:target_feat] : seq_feats[:seq_probs]
    msa_feat = seq_feats[:msa_feat]

    L = size(target_feat, 2)
    B = size(target_feat, 3)
    c_m = length(m.prev_msa_first_row_norm.w)

    msa_act = m.preprocess_msa(msa_feat) .+ reshape(m.preprocess_1d(target_feat), c_m, 1, L, B)
    left = m.left_single(target_feat)
    right = m.right_single(target_feat)
    pair_act = reshape(left, size(left, 1), size(left, 2), 1, size(left, 3)) .+
               reshape(right, size(right, 1), 1, size(right, 2), size(right, 3))
    pair_act = pair_act .+ m.prev_pos_linear(m.prev_pos_dgram)

    prev_msa_first_row = zeros(Float32, c_m, L, B)
    prev_pair = zeros(Float32, size(pair_act, 1), L, L, B)

    msa_act = msa_act .+ reshape(m.prev_msa_first_row_norm(prev_msa_first_row), c_m, 1, L, B)
    pair_act = pair_act .+ m.prev_pair_norm(prev_pair)
    pair_act = pair_act .+ m.pair_relpos(m.relpos_feat)

    for block in m.blocks
        msa_act, pair_act = block(msa_act, pair_act, m.msa_mask, m.pair_mask)
    end

    single = m.single_activations(view(msa_act, :, 1, :, :))
    struct_out = m.structure(single, pair_act, m.seq_mask, m.aatype)
    lddt_logits = m.predicted_lddt(struct_out[:act])[:logits]
    return mean_plddt_loss(lddt_logits)
end

function _sum_abs_grad(x)
    x === nothing && return 0.0
    if x isa AbstractArray
        return sum(_sum_abs_grad(v) for v in x)
    elseif x isa NamedTuple
        return sum(_sum_abs_grad(v) for v in values(x))
    elseif x isa Tuple
        return sum(_sum_abs_grad(v) for v in x)
    elseif x isa Number
        return float(abs(x))
    else
        return 0.0
    end
end

function _max_abs_grad(x)
    x === nothing && return 0.0
    if x isa AbstractArray
        return maximum((_max_abs_grad(v) for v in x); init=0.0)
    elseif x isa NamedTuple
        return maximum((_max_abs_grad(v) for v in values(x)); init=0.0)
    elseif x isa Tuple
        return maximum((_max_abs_grad(v) for v in x); init=0.0)
    elseif x isa Number
        return float(abs(x))
    else
        return 0.0
    end
end

arrs = NPZ.npzread(params_path)

c_m = length(_arr_get(arrs, string(EVO_PREFIX, "/preprocess_1d//bias")))
c_z = length(_arr_get(arrs, string(EVO_PREFIX, "/left_single//bias")))
c_s = length(_arr_get(arrs, string(EVO_PREFIX, "/single_activations//bias")))

num_blocks_total = size(_arr_get(arrs, string(EVO_BLOCK_PREFIX, "/msa_column_attention/attention//output_b")), 1)
num_blocks = num_evo_blocks_override > 0 ? min(num_evo_blocks_override, num_blocks_total) : num_blocks_total

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

preprocess_1d_in_dim = size(_arr_get(arrs, string(EVO_PREFIX, "/preprocess_1d//weights")), 1)
left_single_in_dim = size(_arr_get(arrs, string(EVO_PREFIX, "/left_single//weights")), 1)
right_single_in_dim = size(_arr_get(arrs, string(EVO_PREFIX, "/right_single//weights")), 1)

preprocess_1d = LinearFirst(preprocess_1d_in_dim, c_m)
preprocess_msa = LinearFirst(49, c_m)
left_single = LinearFirst(left_single_in_dim, c_z)
right_single = LinearFirst(right_single_in_dim, c_z)
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
single_activations = LinearFirst(c_m, c_s)

_load_linear_raw!(preprocess_1d, arrs, string(EVO_PREFIX, "/preprocess_1d"))
_load_linear_raw!(preprocess_msa, arrs, string(EVO_PREFIX, "/preprocess_msa"))
_load_linear_raw!(left_single, arrs, string(EVO_PREFIX, "/left_single"))
_load_linear_raw!(right_single, arrs, string(EVO_PREFIX, "/right_single"))
_load_linear_raw!(prev_pos_linear, arrs, string(EVO_PREFIX, "/prev_pos_linear"))
_load_ln_raw!(prev_msa_first_row_norm, arrs, string(EVO_PREFIX, "/prev_msa_first_row_norm"))
_load_ln_raw!(prev_pair_norm, arrs, string(EVO_PREFIX, "/prev_pair_norm"))
_load_linear_raw!(pair_relpos, arrs, pair_relpos_base)
_load_linear_raw!(single_activations, arrs, string(EVO_PREFIX, "/single_activations"))

is_multimer_checkpoint = !_has_key(arrs, string(EVO_PREFIX, "/pair_activiations//weights"))
outer_first_evo = is_multimer_checkpoint
relpos_is_multimer = occursin("position_activations", pair_relpos_base)
relpos_max_relative_idx = relpos_is_multimer ? 32 : Int(div(pair_relpos_in_dim - 1, 2))
relpos_max_relative_chain = relpos_is_multimer ? Int(div(pair_relpos_in_dim - (2 * relpos_max_relative_idx + 5), 2)) : 0

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
        outer_first=outer_first_evo,
    )
    for _ in 1:num_blocks
]
for i in 1:num_blocks
    _load_evo_block_raw!(blocks[i], arrs, i)
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
num_structure_layers = 8
position_scale = is_multimer_checkpoint ? 20f0 : 10f0

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
    num_residual_block;
    multimer_ipa=is_multimer_checkpoint,
)
_load_structure_core_raw!(structure, arrs)

lddt_num_channels = length(_arr_get(arrs, string(LDDT_PREFIX, "/act_0//bias")))
lddt_num_bins = length(_arr_get(arrs, string(LDDT_PREFIX, "/logits//bias")))
predicted_lddt = PredictedLDDTHead(c_s; num_channels=lddt_num_channels, num_bins=lddt_num_bins)
load_predicted_lddt_head_npz!(predicted_lddt, params_path; prefix=LDDT_PREFIX)

seqs = _parse_sequences_csv(sequence_arg)
aatype_vec, residue_index_vec, asym_id_vec, entity_id_vec, sym_id_vec, chain_lens = _chain_metadata(seqs)
L = length(aatype_vec)
B = 1
aatype = reshape(aatype_vec, L, B)
seq_mask = ones(Float32, L, B)
msa_mask = ones(Float32, 1, L, B)
pair_mask = reshape(seq_mask, L, 1, B) .* reshape(seq_mask, 1, L, B)
prev_pos_dgram = _dgram_from_positions(zeros(Float32, 3, L, B))
relpos_feat = if relpos_is_multimer
    _multimer_relpos_features(
        residue_index_vec,
        asym_id_vec,
        entity_id_vec,
        sym_id_vec,
        relpos_max_relative_idx,
        relpos_max_relative_chain,
    )
else
    _relpos_one_hot(residue_index_vec, relpos_max_relative_idx)
end

model = AF2GradModel(
    preprocess_1d,
    preprocess_msa,
    left_single,
    right_single,
    prev_pos_linear,
    prev_msa_first_row_norm,
    prev_pair_norm,
    pair_relpos,
    blocks,
    single_activations,
    structure,
    predicted_lddt,
    seq_mask,
    msa_mask,
    pair_mask,
    residue_index_vec,
    aatype,
    prev_pos_dgram,
    preprocess_1d_in_dim,
    relpos_feat,
)

seq_logits = fill(-6f0, 21, L, B)
for i in 1:L
    k = clamp(aatype[i, 1] + 1, 1, 21)
    seq_logits[k, i, 1] = 6f0
end
seq_logits .+= 0.05f0 .* randn(Float32, size(seq_logits)...)

loss = model(seq_logits)
g_model, g_seq = Zygote.gradient((m, s) -> m(s), model, seq_logits)

seq_grad_l1 = _sum_abs_grad(g_seq)
seq_grad_max = _max_abs_grad(g_seq)
param_grad_l1 = _sum_abs_grad(g_model)
param_grad_max = _max_abs_grad(g_model)

@printf("Zygote full-model gradient check\n")
@printf("  checkpoint_mode: %s\n", is_multimer_checkpoint ? "multimer" : "monomer-family")
@printf("  chains: %d\n", length(seqs))
@printf("  chain_lengths: %s\n", string(chain_lens))
@printf("  sequence length: %d\n", L)
@printf("  evoformer blocks used: %d / %d\n", num_blocks, num_blocks_total)
@printf("  structure layers: %d\n", num_structure_layers)
@printf("  loss (mean pLDDT): %.6f\n", loss)
@printf("  seq_grad_l1: %.8g\n", seq_grad_l1)
@printf("  seq_grad_max_abs: %.8g\n", seq_grad_max)
@printf("  param_grad_l1: %.8g\n", param_grad_l1)
@printf("  param_grad_max_abs: %.8g\n", param_grad_max)

@printf("  sample_grad preprocess_1d.weight L1: %.8g\n", _sum_abs_grad(g_model.preprocess_1d.weight))
@printf("  sample_grad block1.msa_transition.transition1.weight L1: %.8g\n", _sum_abs_grad(g_model.blocks[1].msa_transition.transition1.weight))
@printf("  sample_grad structure.fold_iteration_core.ipa.linear_q.weight L1: %.8g\n", _sum_abs_grad(g_model.structure.fold_iteration_core.ipa.linear_q.weight))
@printf("  sample_grad predicted_lddt.logits.weight L1: %.8g\n", _sum_abs_grad(g_model.predicted_lddt.logits.weight))

if !(isfinite(seq_grad_l1) && seq_grad_l1 > 0 && isfinite(param_grad_l1) && param_grad_l1 > 0)
    error("Gradient check failed: gradients are zero or non-finite.")
end

println("PASS")
