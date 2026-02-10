# Model construction: extract dimensions from params dict, build all layers, load weights.
# Extracted from scripts/end_to_end/run_af2_template_hybrid_jl.jl lines 14-897.

using Printf

# ── Weight key prefixes ──────────────────────────────────────────────────────

const _EVO_PREFIX = "alphafold/alphafold_iteration/evoformer"
const _EVO_BLOCK_PREFIX = "alphafold/alphafold_iteration/evoformer/evoformer_iteration"
const _EXTRA_BLOCK_PREFIX = "alphafold/alphafold_iteration/evoformer/extra_msa_stack"
const _SM_PREFIX = "alphafold/alphafold_iteration/structure_module"

# ── Helpers for reading param arrays ─────────────────────────────────────────

@inline function _builder_has_key(arrs::AbstractDict, key::AbstractString)
    return haskey(arrs, key)
end

@inline function _builder_arr_get(arrs::AbstractDict, key::AbstractString)
    if haskey(arrs, key)
        return arrs[key]
    end
    alt = replace(key, "//" => "/")
    if haskey(arrs, alt)
        return arrs[alt]
    end
    error("Missing key: $(key)")
end

@inline function _builder_slice_block(arr::AbstractArray, block_idx::Int)
    return dropdims(view(arr, block_idx:block_idx, ntuple(_ -> Colon(), ndims(arr) - 1)...); dims=1)
end

# ── Weight loading functions ─────────────────────────────────────────────────

function _load_linear_raw!(
    lin::LinearFirst,
    arrs::AbstractDict,
    base::AbstractString;
    block_idx::Union{Nothing,Int}=nothing,
    split::Symbol=:full,
)
    wkey = string(base, "//weights")
    bkey = string(base, "//bias")

    wfull = block_idx === nothing ? _builder_arr_get(arrs, wkey) : _builder_slice_block(_builder_arr_get(arrs, wkey), block_idx)
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
        bfull = block_idx === nothing ? _builder_arr_get(arrs, bkey) : _builder_slice_block(_builder_arr_get(arrs, bkey), block_idx)
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
    s = block_idx === nothing ? _builder_arr_get(arrs, skey) : _builder_slice_block(_builder_arr_get(arrs, skey), block_idx)
    o = block_idx === nothing ? _builder_arr_get(arrs, okey) : _builder_slice_block(_builder_arr_get(arrs, okey), block_idx)
    ln.w .= s
    ln.b .= o
    return ln
end

function _load_attention_raw!(att::AF2Attention, arrs::AbstractDict, base::AbstractString; block_idx::Union{Nothing,Int}=nothing)
    g = key -> (block_idx === nothing ? _builder_arr_get(arrs, key) : _builder_slice_block(_builder_arr_get(arrs, key), block_idx))
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
    g = key -> (block_idx === nothing ? _builder_arr_get(arrs, key) : _builder_slice_block(_builder_arr_get(arrs, key), block_idx))
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
    p = _EVO_BLOCK_PREFIX
    _load_ln_raw!(blk.outer_product_mean.layer_norm_input, arrs, string(p, "/outer_product_mean/layer_norm_input"); block_idx=bi)
    _load_linear_raw!(blk.outer_product_mean.left_projection, arrs, string(p, "/outer_product_mean/left_projection"); block_idx=bi)
    _load_linear_raw!(blk.outer_product_mean.right_projection, arrs, string(p, "/outer_product_mean/right_projection"); block_idx=bi)
    blk.outer_product_mean.output_w .= _builder_slice_block(arrs[string(p, "/outer_product_mean//output_w")], bi)
    blk.outer_product_mean.output_b .= _builder_slice_block(arrs[string(p, "/outer_product_mean//output_b")], bi)

    _load_ln_raw!(blk.msa_row_attention_with_pair_bias.query_norm, arrs, string(p, "/msa_row_attention_with_pair_bias/query_norm"); block_idx=bi)
    _load_ln_raw!(blk.msa_row_attention_with_pair_bias.feat_2d_norm, arrs, string(p, "/msa_row_attention_with_pair_bias/feat_2d_norm"); block_idx=bi)
    blk.msa_row_attention_with_pair_bias.feat_2d_weights .= _builder_slice_block(arrs[string(p, "/msa_row_attention_with_pair_bias//feat_2d_weights")], bi)
    _load_attention_raw!(blk.msa_row_attention_with_pair_bias.attention, arrs, string(p, "/msa_row_attention_with_pair_bias/attention"); block_idx=bi)
    _load_ln_raw!(blk.msa_column_attention.query_norm, arrs, string(p, "/msa_column_attention/query_norm"); block_idx=bi)
    _load_attention_raw!(blk.msa_column_attention.attention, arrs, string(p, "/msa_column_attention/attention"); block_idx=bi)
    _load_ln_raw!(blk.msa_transition.input_layer_norm, arrs, string(p, "/msa_transition/input_layer_norm"); block_idx=bi)
    _load_linear_raw!(blk.msa_transition.transition1, arrs, string(p, "/msa_transition/transition1"); block_idx=bi)
    _load_linear_raw!(blk.msa_transition.transition2, arrs, string(p, "/msa_transition/transition2"); block_idx=bi)

    tri_out_base = string(p, "/triangle_multiplication_outgoing")
    if _builder_has_key(arrs, string(tri_out_base, "/left_projection//weights"))
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
    if _builder_has_key(arrs, string(tri_in_base, "/left_projection//weights"))
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
    blk.triangle_attention_starting_node.feat_2d_weights .= _builder_slice_block(arrs[string(p, "/triangle_attention_starting_node//feat_2d_weights")], bi)
    _load_attention_raw!(blk.triangle_attention_starting_node.attention, arrs, string(p, "/triangle_attention_starting_node/attention"); block_idx=bi)
    _load_ln_raw!(blk.triangle_attention_ending_node.query_norm, arrs, string(p, "/triangle_attention_ending_node/query_norm"); block_idx=bi)
    blk.triangle_attention_ending_node.feat_2d_weights .= _builder_slice_block(arrs[string(p, "/triangle_attention_ending_node//feat_2d_weights")], bi)
    _load_attention_raw!(blk.triangle_attention_ending_node.attention, arrs, string(p, "/triangle_attention_ending_node/attention"); block_idx=bi)
    _load_ln_raw!(blk.pair_transition.input_layer_norm, arrs, string(p, "/pair_transition/input_layer_norm"); block_idx=bi)
    _load_linear_raw!(blk.pair_transition.transition1, arrs, string(p, "/pair_transition/transition1"); block_idx=bi)
    _load_linear_raw!(blk.pair_transition.transition2, arrs, string(p, "/pair_transition/transition2"); block_idx=bi)
    return blk
end

function _load_extra_block_raw!(blk::EvoformerIteration, arrs::AbstractDict, bi::Int)
    p = _EXTRA_BLOCK_PREFIX
    _load_ln_raw!(blk.outer_product_mean.layer_norm_input, arrs, string(p, "/outer_product_mean/layer_norm_input"); block_idx=bi)
    _load_linear_raw!(blk.outer_product_mean.left_projection, arrs, string(p, "/outer_product_mean/left_projection"); block_idx=bi)
    _load_linear_raw!(blk.outer_product_mean.right_projection, arrs, string(p, "/outer_product_mean/right_projection"); block_idx=bi)
    blk.outer_product_mean.output_w .= _builder_slice_block(arrs[string(p, "/outer_product_mean//output_w")], bi)
    blk.outer_product_mean.output_b .= _builder_slice_block(arrs[string(p, "/outer_product_mean//output_b")], bi)
    _load_ln_raw!(blk.msa_row_attention_with_pair_bias.query_norm, arrs, string(p, "/msa_row_attention_with_pair_bias/query_norm"); block_idx=bi)
    _load_ln_raw!(blk.msa_row_attention_with_pair_bias.feat_2d_norm, arrs, string(p, "/msa_row_attention_with_pair_bias/feat_2d_norm"); block_idx=bi)
    blk.msa_row_attention_with_pair_bias.feat_2d_weights .= _builder_slice_block(arrs[string(p, "/msa_row_attention_with_pair_bias//feat_2d_weights")], bi)
    _load_attention_raw!(blk.msa_row_attention_with_pair_bias.attention, arrs, string(p, "/msa_row_attention_with_pair_bias/attention"); block_idx=bi)
    _load_ln_raw!(blk.msa_column_attention.query_norm, arrs, string(p, "/msa_column_global_attention/query_norm"); block_idx=bi)
    _load_global_attention_raw!(blk.msa_column_attention.attention, arrs, string(p, "/msa_column_global_attention/attention"); block_idx=bi)
    _load_ln_raw!(blk.msa_transition.input_layer_norm, arrs, string(p, "/msa_transition/input_layer_norm"); block_idx=bi)
    _load_linear_raw!(blk.msa_transition.transition1, arrs, string(p, "/msa_transition/transition1"); block_idx=bi)
    _load_linear_raw!(blk.msa_transition.transition2, arrs, string(p, "/msa_transition/transition2"); block_idx=bi)
    tri_out_base = string(p, "/triangle_multiplication_outgoing")
    if _builder_has_key(arrs, string(tri_out_base, "/left_projection//weights"))
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
    if _builder_has_key(arrs, string(tri_in_base, "/left_projection//weights"))
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
    blk.triangle_attention_starting_node.feat_2d_weights .= _builder_slice_block(arrs[string(p, "/triangle_attention_starting_node//feat_2d_weights")], bi)
    _load_attention_raw!(blk.triangle_attention_starting_node.attention, arrs, string(p, "/triangle_attention_starting_node/attention"); block_idx=bi)
    _load_ln_raw!(blk.triangle_attention_ending_node.query_norm, arrs, string(p, "/triangle_attention_ending_node/query_norm"); block_idx=bi)
    blk.triangle_attention_ending_node.feat_2d_weights .= _builder_slice_block(arrs[string(p, "/triangle_attention_ending_node//feat_2d_weights")], bi)
    _load_attention_raw!(blk.triangle_attention_ending_node.attention, arrs, string(p, "/triangle_attention_ending_node/attention"); block_idx=bi)
    _load_ln_raw!(blk.pair_transition.input_layer_norm, arrs, string(p, "/pair_transition/input_layer_norm"); block_idx=bi)
    _load_linear_raw!(blk.pair_transition.transition1, arrs, string(p, "/pair_transition/transition1"); block_idx=bi)
    _load_linear_raw!(blk.pair_transition.transition2, arrs, string(p, "/pair_transition/transition2"); block_idx=bi)
    return blk
end

function _builder_infer_transition_depth(arrs::AbstractDict, base::AbstractString)
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
    p = _SM_PREFIX
    _load_ln_raw!(m.single_layer_norm, arrs, string(p, "/single_layer_norm"))
    _load_linear_raw!(m.initial_projection, arrs, string(p, "/initial_projection"))
    _load_ln_raw!(m.pair_layer_norm, arrs, string(p, "/pair_layer_norm"))

    ipa_base = string(p, "/fold_iteration/invariant_point_attention")
    if _builder_has_key(arrs, string(ipa_base, "/q_scalar//weights"))
        _load_linear_raw!(m.fold_iteration_core.ipa.linear_q, arrs, string(ipa_base, "/q_scalar"))
        _load_linear_raw!(m.fold_iteration_core.ipa.linear_q_points.linear, arrs, string(ipa_base, "/q_point_local"))
        _load_linear_raw!(m.fold_iteration_core.ipa.linear_kv, arrs, string(ipa_base, "/kv_scalar"))
        _load_linear_raw!(m.fold_iteration_core.ipa.linear_kv_points.linear, arrs, string(ipa_base, "/kv_point_local"))
    else
        m.fold_iteration_core.ipa isa MultimerInvariantPointAttention ||
            error("Multimer IPA weights found, but structure module was not built with multimer_ipa=true")

        q_w = _builder_arr_get(arrs, string(ipa_base, "/q_scalar_projection//weights"))
        k_w = _builder_arr_get(arrs, string(ipa_base, "/k_scalar_projection//weights"))
        v_w = _builder_arr_get(arrs, string(ipa_base, "/v_scalar_projection//weights"))
        q_w2 = reshape(permutedims(q_w, (1, 3, 2)), size(q_w, 1), :)
        k_w2 = reshape(permutedims(k_w, (1, 3, 2)), size(k_w, 1), :)
        v_w2 = reshape(permutedims(v_w, (1, 3, 2)), size(v_w, 1), :)
        m.fold_iteration_core.ipa.linear_q.weight .= permutedims(q_w2, (2, 1))
        m.fold_iteration_core.ipa.linear_k.weight .= permutedims(k_w2, (2, 1))
        m.fold_iteration_core.ipa.linear_v.weight .= permutedims(v_w2, (2, 1))

        qp_w = _builder_arr_get(arrs, string(ipa_base, "/q_point_projection/point_projection//weights"))
        kp_w = _builder_arr_get(arrs, string(ipa_base, "/k_point_projection/point_projection//weights"))
        vp_w = _builder_arr_get(arrs, string(ipa_base, "/v_point_projection/point_projection//weights"))
        qp_b = _builder_arr_get(arrs, string(ipa_base, "/q_point_projection/point_projection//bias"))
        kp_b = _builder_arr_get(arrs, string(ipa_base, "/k_point_projection/point_projection//bias"))
        vp_b = _builder_arr_get(arrs, string(ipa_base, "/v_point_projection/point_projection//bias"))

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
    m.fold_iteration_core.ipa.head_weights .= _builder_arr_get(arrs, string(p, "/fold_iteration/invariant_point_attention//trainable_point_weights"))
    _load_linear_raw!(m.fold_iteration_core.ipa.linear_out, arrs, string(p, "/fold_iteration/invariant_point_attention/output_projection"))
    _load_ln_raw!(m.fold_iteration_core.attention_layer_norm, arrs, string(p, "/fold_iteration/attention_layer_norm"))
    _load_ln_raw!(m.fold_iteration_core.transition_layer_norm, arrs, string(p, "/fold_iteration/transition_layer_norm"))
    if _builder_has_key(arrs, string(p, "/fold_iteration/affine_update//weights"))
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

# ── AF2Config: scalar configuration extracted from params ────────────────────

struct AF2Config
    kind::Symbol
    c_m::Int
    c_z::Int
    c_s::Int
    is_multimer_checkpoint::Bool
    relpos_is_multimer::Bool
    relpos_max_relative_idx::Int
    relpos_max_relative_chain::Int
    preprocess_1d_in_dim::Int
    template_single_feature_dim::Int
    has_template_embedding::Bool
    use_multimer_template_embedding::Bool
    structure_position_scale::Float32
end

# ── AF2Model: holds all constructed layers ───────────────────────────────────

@concrete struct AF2Model <: Onion.Layer
    config                                # AF2Config

    # Evoformer blocks
    blocks                                # Vector{EvoformerIteration}
    extra_blocks                          # Vector{EvoformerIteration}

    # Preprocessing
    preprocess_1d                         # LinearFirst
    preprocess_msa                        # LinearFirst
    left_single                           # LinearFirst
    right_single                          # LinearFirst
    extra_msa_activations                 # LinearFirst
    prev_pos_linear                       # LinearFirst
    prev_msa_first_row_norm               # LayerNormFirst
    prev_pair_norm                        # LayerNormFirst
    pair_relpos                           # LinearFirst
    single_activations                    # LinearFirst

    # Template
    template_embedding                    # Union{TemplateEmbedding, TemplateEmbeddingMultimer, Nothing}
    template_single                       # Union{TemplateSingleRows, Nothing}

    # Structure module
    structure                             # StructureModuleCore

    # Output heads
    predicted_lddt_head                   # PredictedLDDTHead
    masked_msa_head                       # MaskedMsaHead
    distogram_head                        # DistogramHead
    experimentally_resolved_head          # ExperimentallyResolvedHead
    predicted_aligned_error_head          # Union{PredictedAlignedErrorHead, Nothing}
end

@layer AF2Model

_model_on_gpu(m::AF2Model) = m.preprocess_1d.weight isa CUDA.CuArray

# ── Main model builder ───────────────────────────────────────────────────────

function _build_af2_model(arrs::AbstractDict)::AF2Model
    # ── Main evoformer dims ──────────────────────────────────────────────
    c_m = length(_builder_arr_get(arrs, string(_EVO_PREFIX, "/preprocess_1d//bias")))
    c_z = length(_builder_arr_get(arrs, string(_EVO_PREFIX, "/left_single//bias")))
    c_s = length(_builder_arr_get(arrs, string(_EVO_PREFIX, "/single_activations//bias")))
    num_blocks = size(_builder_arr_get(arrs, string(_EVO_BLOCK_PREFIX, "/msa_column_attention/attention//output_b")), 1)
    msa_qw = _builder_arr_get(arrs, string(_EVO_BLOCK_PREFIX, "/msa_column_attention/attention//query_w"))
    num_head_msa = size(msa_qw, 3)
    msa_head_dim = size(msa_qw, 4)
    pair_qw = _builder_arr_get(arrs, string(_EVO_BLOCK_PREFIX, "/triangle_attention_starting_node/attention//query_w"))
    num_head_pair = size(pair_qw, 3)
    pair_head_dim = size(pair_qw, 4)
    c_outer = size(_builder_arr_get(arrs, string(_EVO_BLOCK_PREFIX, "/outer_product_mean/left_projection//bias")), 2)
    c_tri_mul = if _builder_has_key(arrs, string(_EVO_BLOCK_PREFIX, "/triangle_multiplication_outgoing/left_projection//bias"))
        size(_builder_arr_get(arrs, string(_EVO_BLOCK_PREFIX, "/triangle_multiplication_outgoing/left_projection//bias")), 2)
    else
        Int(div(size(_builder_arr_get(arrs, string(_EVO_BLOCK_PREFIX, "/triangle_multiplication_outgoing/projection//bias")), 2), 2))
    end
    msa_transition_factor = size(_builder_arr_get(arrs, string(_EVO_BLOCK_PREFIX, "/msa_transition/transition1//bias")), 2) / c_m
    pair_transition_factor = size(_builder_arr_get(arrs, string(_EVO_BLOCK_PREFIX, "/pair_transition/transition1//bias")), 2) / c_z

    # ── Extra stack dims ─────────────────────────────────────────────────
    extra_qw = _builder_arr_get(arrs, string(_EXTRA_BLOCK_PREFIX, "/msa_row_attention_with_pair_bias/attention//query_w"))
    c_m_extra = size(extra_qw, 2)
    num_head_msa_extra = size(extra_qw, 3)
    msa_head_dim_extra = size(extra_qw, 4)
    extra_pair_qw = _builder_arr_get(arrs, string(_EXTRA_BLOCK_PREFIX, "/triangle_attention_starting_node/attention//query_w"))
    num_head_pair_extra = size(extra_pair_qw, 3)
    pair_head_dim_extra = size(extra_pair_qw, 4)
    num_extra_blocks = size(_builder_arr_get(arrs, string(_EXTRA_BLOCK_PREFIX, "/msa_transition/transition1//bias")), 1)
    c_outer_extra = size(_builder_arr_get(arrs, string(_EXTRA_BLOCK_PREFIX, "/outer_product_mean/left_projection//bias")), 2)
    c_tri_mul_extra = if _builder_has_key(arrs, string(_EXTRA_BLOCK_PREFIX, "/triangle_multiplication_outgoing/left_projection//bias"))
        size(_builder_arr_get(arrs, string(_EXTRA_BLOCK_PREFIX, "/triangle_multiplication_outgoing/left_projection//bias")), 2)
    else
        Int(div(size(_builder_arr_get(arrs, string(_EXTRA_BLOCK_PREFIX, "/triangle_multiplication_outgoing/projection//bias")), 2), 2))
    end
    msa_transition_factor_extra = size(_builder_arr_get(arrs, string(_EXTRA_BLOCK_PREFIX, "/msa_transition/transition1//bias")), 2) / c_m_extra
    pair_transition_factor_extra = size(_builder_arr_get(arrs, string(_EXTRA_BLOCK_PREFIX, "/pair_transition/transition1//bias")), 2) / c_z

    # ── Template embedding dims ──────────────────────────────────────────
    TEMPLATE_PREFIX = string(_EVO_PREFIX, "/template_embedding")
    TEMPLATE_STACK_PREFIX_MONOMER = string(TEMPLATE_PREFIX, "/single_template_embedding/template_pair_stack/__layer_stack_no_state")
    TEMPLATE_STACK_PREFIX_MULTIMER = string(TEMPLATE_PREFIX, "/single_template_embedding/template_embedding_iteration")
    has_monomer_template_embed = _builder_has_key(arrs, string(TEMPLATE_PREFIX, "/single_template_embedding/embedding2d//bias"))
    has_multimer_template_embed = _builder_has_key(arrs, string(TEMPLATE_PREFIX, "/single_template_embedding/template_pair_embedding_0//bias"))
    has_template_embedding = has_monomer_template_embed || has_multimer_template_embed
    use_multimer_template_embedding = has_multimer_template_embed && !has_monomer_template_embed
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
    template_single_feature_dim = _builder_has_key(arrs, string(_EVO_PREFIX, "/template_single_embedding//weights")) ?
        size(_builder_arr_get(arrs, string(_EVO_PREFIX, "/template_single_embedding//weights")), 1) : 57
    if has_template_embedding
        if use_multimer_template_embedding
            c_t = length(_builder_arr_get(arrs, string(TEMPLATE_PREFIX, "/single_template_embedding/template_pair_embedding_0//bias")))
            num_template_blocks = size(_builder_arr_get(arrs, string(TEMPLATE_STACK_PREFIX_MULTIMER, "/pair_transition/transition2//bias")), 1)
            num_head_pair_template = size(_builder_arr_get(arrs, string(TEMPLATE_STACK_PREFIX_MULTIMER, "/triangle_attention_starting_node//feat_2d_weights")), 3)
            pair_head_dim_template = size(_builder_arr_get(arrs, string(TEMPLATE_STACK_PREFIX_MULTIMER, "/triangle_attention_starting_node/attention//query_w")), 4)
            tri_mul_base = string(TEMPLATE_STACK_PREFIX_MULTIMER, "/triangle_multiplication_outgoing")
            if _builder_has_key(arrs, string(tri_mul_base, "/left_projection//bias"))
                c_tri_mul_template = size(_builder_arr_get(arrs, string(tri_mul_base, "/left_projection//bias")), 2)
            else
                c_tri_mul_template = Int(div(size(_builder_arr_get(arrs, string(tri_mul_base, "/projection//bias")), 2), 2))
            end
            pair_transition_factor_template = size(_builder_arr_get(arrs, string(TEMPLATE_STACK_PREFIX_MULTIMER, "/pair_transition/transition1//bias")), 2) / c_t
            dgram_num_bins_template = size(_builder_arr_get(arrs, string(TEMPLATE_PREFIX, "/single_template_embedding/template_pair_embedding_0//weights")), 1)
        else
            c_t = length(_builder_arr_get(arrs, string(TEMPLATE_PREFIX, "/single_template_embedding/embedding2d//bias")))
            num_template_blocks = size(_builder_arr_get(arrs, string(TEMPLATE_STACK_PREFIX_MONOMER, "/pair_transition/transition2//bias")), 1)
            num_head_pair_template = size(_builder_arr_get(arrs, string(TEMPLATE_STACK_PREFIX_MONOMER, "/triangle_attention_starting_node//feat_2d_weights")), 3)
            pair_head_dim_template = size(_builder_arr_get(arrs, string(TEMPLATE_STACK_PREFIX_MONOMER, "/triangle_attention_starting_node/attention//query_w")), 4)
            c_tri_mul_template = size(_builder_arr_get(arrs, string(TEMPLATE_STACK_PREFIX_MONOMER, "/triangle_multiplication_outgoing/left_projection//bias")), 2)
            pair_transition_factor_template = size(_builder_arr_get(arrs, string(TEMPLATE_STACK_PREFIX_MONOMER, "/pair_transition/transition1//bias")), 2) / c_t
            num_head_tpa = size(_builder_arr_get(arrs, string(TEMPLATE_PREFIX, "/attention//query_w")), 2)
            key_dim_tpa = size(_builder_arr_get(arrs, string(TEMPLATE_PREFIX, "/attention//query_w")), 2) * size(_builder_arr_get(arrs, string(TEMPLATE_PREFIX, "/attention//query_w")), 3)
            value_dim_tpa = size(_builder_arr_get(arrs, string(TEMPLATE_PREFIX, "/attention//value_w")), 2) * size(_builder_arr_get(arrs, string(TEMPLATE_PREFIX, "/attention//value_w")), 3)
            dgram_num_bins_template = size(_builder_arr_get(arrs, string(TEMPLATE_PREFIX, "/single_template_embedding/embedding2d//weights")), 1) - 49
        end
    end

    is_multimer_checkpoint = !_builder_has_key(arrs, string(_EVO_PREFIX, "/pair_activiations//weights"))
    outer_first_evo = is_multimer_checkpoint

    # ── Construct evoformer blocks ───────────────────────────────────────
    blocks = [EvoformerIteration(c_m, c_z; num_head_msa=num_head_msa, msa_head_dim=msa_head_dim, num_head_pair=num_head_pair, pair_head_dim=pair_head_dim, c_outer=c_outer, c_tri_mul=c_tri_mul, msa_transition_factor=msa_transition_factor, pair_transition_factor=pair_transition_factor, outer_first=outer_first_evo) for _ in 1:num_blocks]
    for i in 1:num_blocks
        _load_evo_block_raw!(blocks[i], arrs, i)
    end

    extra_blocks = [EvoformerIteration(c_m_extra, c_z; num_head_msa=num_head_msa_extra, msa_head_dim=msa_head_dim_extra, num_head_pair=num_head_pair_extra, pair_head_dim=pair_head_dim_extra, c_outer=c_outer_extra, c_tri_mul=c_tri_mul_extra, msa_transition_factor=msa_transition_factor_extra, pair_transition_factor=pair_transition_factor_extra, outer_first=outer_first_evo, is_extra_msa=true) for _ in 1:num_extra_blocks]
    for i in 1:num_extra_blocks
        _load_extra_block_raw!(extra_blocks[i], arrs, i)
    end

    # ── Preprocessing layers ─────────────────────────────────────────────
    preprocess_1d_in_dim = size(_builder_arr_get(arrs, string(_EVO_PREFIX, "/preprocess_1d//weights")), 1)
    preprocess_1d = LinearFirst(preprocess_1d_in_dim, c_m)
    preprocess_msa = LinearFirst(49, c_m)
    left_single_in_dim = size(_builder_arr_get(arrs, string(_EVO_PREFIX, "/left_single//weights")), 1)
    right_single_in_dim = size(_builder_arr_get(arrs, string(_EVO_PREFIX, "/right_single//weights")), 1)
    left_single = LinearFirst(left_single_in_dim, c_z)
    right_single = LinearFirst(right_single_in_dim, c_z)
    extra_msa_activations = LinearFirst(25, c_m_extra)
    prev_pos_linear = LinearFirst(15, c_z)
    prev_msa_first_row_norm = LayerNormFirst(c_m)
    prev_pair_norm = LayerNormFirst(c_z)
    pair_relpos_base = if _builder_has_key(arrs, string(_EVO_PREFIX, "/pair_activiations//weights"))
        string(_EVO_PREFIX, "/pair_activiations")
    else
        string(_EVO_PREFIX, "/~_relative_encoding/position_activations")
    end
    pair_relpos_in_dim = size(_builder_arr_get(arrs, string(pair_relpos_base, "//weights")), 1)
    pair_relpos = LinearFirst(pair_relpos_in_dim, c_z)
    _load_linear_raw!(preprocess_1d, arrs, string(_EVO_PREFIX, "/preprocess_1d"))
    _load_linear_raw!(preprocess_msa, arrs, string(_EVO_PREFIX, "/preprocess_msa"))
    _load_linear_raw!(left_single, arrs, string(_EVO_PREFIX, "/left_single"))
    _load_linear_raw!(right_single, arrs, string(_EVO_PREFIX, "/right_single"))
    _load_linear_raw!(extra_msa_activations, arrs, string(_EVO_PREFIX, "/extra_msa_activations"))
    _load_linear_raw!(prev_pos_linear, arrs, string(_EVO_PREFIX, "/prev_pos_linear"))
    _load_ln_raw!(prev_msa_first_row_norm, arrs, string(_EVO_PREFIX, "/prev_msa_first_row_norm"))
    _load_ln_raw!(prev_pair_norm, arrs, string(_EVO_PREFIX, "/prev_pair_norm"))
    _load_linear_raw!(pair_relpos, arrs, pair_relpos_base)
    relpos_is_multimer = occursin("position_activations", pair_relpos_base)
    relpos_max_relative_idx = relpos_is_multimer ? 32 : Int(div(pair_relpos_in_dim - 1, 2))
    relpos_max_relative_chain = relpos_is_multimer ? Int(div(pair_relpos_in_dim - (2 * relpos_max_relative_idx + 5), 2)) : 0

    single_activations = LinearFirst(c_m, c_s)
    _load_linear_raw!(single_activations, arrs, string(_EVO_PREFIX, "/single_activations"))

    # ── Template embedding ───────────────────────────────────────────────
    template_embedding = if has_template_embedding
        if use_multimer_template_embedding
            te = TemplateEmbeddingMultimer(
                c_z,
                c_t,
                num_template_blocks;
                num_head_pair=num_head_pair_template,
                pair_head_dim=pair_head_dim_template,
                c_tri_mul=c_tri_mul_template,
                pair_transition_factor=pair_transition_factor_template,
                use_template_unit_vector=true,
                dgram_num_bins=dgram_num_bins_template,
            )
            load_template_embedding_npz!(te, arrs; prefix=TEMPLATE_PREFIX)
            te
        else
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
            load_template_embedding_npz!(te, arrs; prefix=TEMPLATE_PREFIX)
            te
        end
    else
        nothing
    end

    # ── Template single rows (for monomer templates) ─────────────────────
    template_single = if has_template_embedding
        ts = TemplateSingleRows(c_m; feature_dim=template_single_feature_dim)
        load_template_single_rows_npz!(ts, arrs)
        ts
    else
        nothing
    end

    # ── Structure module ─────────────────────────────────────────────────
    num_transition_layers = _builder_infer_transition_depth(arrs, string(_SM_PREFIX, "/fold_iteration/transition"))
    num_residual_block = _builder_infer_transition_depth(arrs, string(_SM_PREFIX, "/fold_iteration/rigid_sidechain/resblock1"))
    no_heads = length(_builder_arr_get(arrs, string(_SM_PREFIX, "/fold_iteration/invariant_point_attention/attention_2d//bias")))
    c_hidden_total = if _builder_has_key(arrs, string(_SM_PREFIX, "/fold_iteration/invariant_point_attention/q_scalar//weights"))
        size(_builder_arr_get(arrs, string(_SM_PREFIX, "/fold_iteration/invariant_point_attention/q_scalar//weights")), 2)
    else
        size(_builder_arr_get(arrs, string(_SM_PREFIX, "/fold_iteration/invariant_point_attention/q_scalar_projection//weights")), 2) *
        size(_builder_arr_get(arrs, string(_SM_PREFIX, "/fold_iteration/invariant_point_attention/q_scalar_projection//weights")), 3)
    end
    c_hidden = Int(div(c_hidden_total, no_heads))
    q_point_total = if _builder_has_key(arrs, string(_SM_PREFIX, "/fold_iteration/invariant_point_attention/q_point_local//bias"))
        size(_builder_arr_get(arrs, string(_SM_PREFIX, "/fold_iteration/invariant_point_attention/q_point_local//bias")), 1)
    else
        length(_builder_arr_get(arrs, string(_SM_PREFIX, "/fold_iteration/invariant_point_attention/q_point_projection/point_projection//bias")))
    end
    kv_point_total = if _builder_has_key(arrs, string(_SM_PREFIX, "/fold_iteration/invariant_point_attention/kv_point_local//bias"))
        size(_builder_arr_get(arrs, string(_SM_PREFIX, "/fold_iteration/invariant_point_attention/kv_point_local//bias")), 1)
    else
        length(_builder_arr_get(arrs, string(_SM_PREFIX, "/fold_iteration/invariant_point_attention/k_point_projection/point_projection//bias"))) +
        length(_builder_arr_get(arrs, string(_SM_PREFIX, "/fold_iteration/invariant_point_attention/v_point_projection/point_projection//bias")))
    end
    no_qk_points = Int(div(q_point_total, 3 * no_heads))
    no_v_points = Int(div(kv_point_total, 3 * no_heads) - no_qk_points)
    sidechain_num_channel = length(_builder_arr_get(arrs, string(_SM_PREFIX, "/fold_iteration/rigid_sidechain/input_projection//bias")))
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

    # ── Output heads ─────────────────────────────────────────────────────
    lddt_prefix = "alphafold/alphafold_iteration/predicted_lddt_head"
    lddt_num_channels = length(_builder_arr_get(arrs, string(lddt_prefix, "/act_0//bias")))
    lddt_num_bins = length(_builder_arr_get(arrs, string(lddt_prefix, "/logits//bias")))
    predicted_lddt_head = PredictedLDDTHead(c_s; num_channels=lddt_num_channels, num_bins=lddt_num_bins)
    load_predicted_lddt_head_npz!(predicted_lddt_head, arrs; prefix=lddt_prefix)

    masked_msa_num_output = length(_builder_arr_get(arrs, "alphafold/alphafold_iteration/masked_msa_head/logits//bias"))
    masked_msa_head = MaskedMsaHead(c_m; num_output=masked_msa_num_output)
    load_masked_msa_head_npz!(masked_msa_head, arrs)

    distogram_num_bins = length(_builder_arr_get(arrs, "alphafold/alphafold_iteration/distogram_head/half_logits//bias"))
    distogram_head = DistogramHead(c_z; num_bins=distogram_num_bins, first_break=2.3125f0, last_break=21.6875f0)
    load_distogram_head_npz!(distogram_head, arrs)

    experimentally_resolved_head = ExperimentallyResolvedHead(c_s)
    load_experimentally_resolved_head_npz!(experimentally_resolved_head, arrs)

    pae_prefix = "alphafold/alphafold_iteration/predicted_aligned_error_head"
    has_pae_head = _builder_has_key(arrs, string(pae_prefix, "/logits//weights"))
    predicted_aligned_error_head = if has_pae_head
        pae_num_bins = length(_builder_arr_get(arrs, string(pae_prefix, "/logits//bias")))
        h = PredictedAlignedErrorHead(c_z; num_bins=pae_num_bins, max_error_bin=31f0)
        load_predicted_aligned_error_head_npz!(h, arrs; prefix=pae_prefix)
        h
    else
        nothing
    end

    # ── Build config ─────────────────────────────────────────────────────
    kind = is_multimer_checkpoint ? :multimer : :monomer
    config = AF2Config(
        kind,
        c_m,
        c_z,
        c_s,
        is_multimer_checkpoint,
        relpos_is_multimer,
        relpos_max_relative_idx,
        relpos_max_relative_chain,
        preprocess_1d_in_dim,
        template_single_feature_dim,
        has_template_embedding,
        use_multimer_template_embedding,
        Float32(structure_position_scale),
    )

    return AF2Model(
        config,
        blocks,
        extra_blocks,
        preprocess_1d,
        preprocess_msa,
        left_single,
        right_single,
        extra_msa_activations,
        prev_pos_linear,
        prev_msa_first_row_norm,
        prev_pair_norm,
        pair_relpos,
        single_activations,
        template_embedding,
        template_single,
        structure,
        predicted_lddt_head,
        masked_msa_head,
        distogram_head,
        experimentally_resolved_head,
        predicted_aligned_error_head,
    )
end
