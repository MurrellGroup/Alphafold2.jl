using NPZ
using Printf
using NNlib

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 2
    error("Usage: julia check_extra_stack_from_dump_jl.jl <params_path_or_hf_filename> <pre_evo_dump.npz>")
end

params_spec = ARGS[1]
dump_path = ARGS[2]

const EXTRA_BLOCK_PREFIX = "alphafold/alphafold_iteration/evoformer/extra_msa_stack"
const EVO_PREFIX = "alphafold/alphafold_iteration/evoformer"

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

function _extra_column_global_attention(m::ExtraMSAColumnGlobalAttention, msa_act::AbstractArray, msa_mask::AbstractArray)
    # msa_act: (C, S, L, B), msa_mask: (S, L, B)
    C, S, L, B = size(msa_act)
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
        denom = max(sum(q_mask), 1f-6)
        q_avg = vec(sum(q_data .* reshape(q_mask, S, 1); dims=1) ./ denom)  # (C)

        q = reshape(q_avg' * reshape(m.query_w, C, H * K), H, K) .* Float32(K)^-0.5f0
        k = q_data * m.key_w
        v = q_data * m.value_w

        logits = q * permutedims(k, (2, 1))
        for s in 1:S
            q_mask[s] <= 0f0 && (logits[:, s] .= -1f9)
        end
        weights = dropdims(_softmax_last(reshape(logits, H, S, 1)); dims=3) # (H,S)
        weighted_avg = weights * v # (H,V)

        gate = reshape(q_data * reshape(m.gating_w, C, H * V), S, H, V)
        gate .+= reshape(m.gating_b, 1, H, V)
        gate = NNlib.sigmoid.(gate)
        weighted = reshape(weighted_avg, 1, H, V) .* gate

        y = zeros(Float32, S, C)
        ow = reshape(m.output_w, H * V, C)
        for s in 1:S
            y[s, :] .= vec(reshape(weighted[s, :, :], 1, H * V) * ow) .+ m.output_b
        end
        out[:, l, :, b] .= permutedims(y, (2, 1))
    end
    return permutedims(out, (1, 3, 2, 4))
end

function _extra_block_forward!(blk::EvoformerIteration, col_attn::ExtraMSAColumnGlobalAttention, msa_act, pair_act, msa_mask, pair_mask)
    blk.outer_first && (pair_act = pair_act .+ blk.outer_product_mean(msa_act, msa_mask))
    msa_act = msa_act .+ blk.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act)
    msa_act = msa_act .+ _extra_column_global_attention(col_attn, msa_act, msa_mask)
    msa_act = msa_act .+ blk.msa_transition(msa_act, msa_mask)
    !blk.outer_first && (pair_act = pair_act .+ blk.outer_product_mean(msa_act, msa_mask))
    pair_act = pair_act .+ blk.triangle_multiplication_outgoing(pair_act, pair_mask)
    pair_act = pair_act .+ blk.triangle_multiplication_incoming(pair_act, pair_mask)
    pair_act = pair_act .+ blk.triangle_attention_starting_node(pair_act, pair_mask)
    pair_act = pair_act .+ blk.triangle_attention_ending_node(pair_act, pair_mask)
    pair_act = pair_act .+ blk.pair_transition(pair_act, pair_mask)
    return msa_act, pair_act
end

params_path = Alphafold2.resolve_af2_params_path(
    params_spec;
    repo_id=get(ENV, "AF2_HF_REPO_ID", Alphafold2.AF2_HF_REPO_ID),
    revision=get(ENV, "AF2_HF_REVISION", Alphafold2.AF2_HF_REVISION),
)
println("Using params file: ", params_path)
arrs = Alphafold2.af2_params_read(params_path)
dump = NPZ.npzread(dump_path)

extra_qw = arrs[string(EXTRA_BLOCK_PREFIX, "/msa_row_attention_with_pair_bias/attention//query_w")]
c_m_extra = size(extra_qw, 2)
num_head_msa_extra = size(extra_qw, 3)
msa_head_dim_extra = size(extra_qw, 4)

extra_pair_qw = arrs[string(EXTRA_BLOCK_PREFIX, "/triangle_attention_starting_node/attention//query_w")]
num_head_pair_extra = size(extra_pair_qw, 3)
pair_head_dim_extra = size(extra_pair_qw, 4)
c_z = size(extra_pair_qw, 2)

num_extra_blocks = size(arrs[string(EXTRA_BLOCK_PREFIX, "/msa_transition/transition1//bias")], 1)
c_outer_extra = size(arrs[string(EXTRA_BLOCK_PREFIX, "/outer_product_mean/left_projection//bias")], 2)
c_tri_mul_extra = size(arrs[string(EXTRA_BLOCK_PREFIX, "/triangle_multiplication_outgoing/left_projection//bias")], 2)
msa_transition_factor_extra = size(arrs[string(EXTRA_BLOCK_PREFIX, "/msa_transition/transition1//bias")], 2) / c_m_extra
pair_transition_factor_extra = size(arrs[string(EXTRA_BLOCK_PREFIX, "/pair_transition/transition1//bias")], 2) / c_z

extra_msa_activations = LinearFirst(25, c_m_extra)
_load_linear_raw!(extra_msa_activations, arrs, string(EVO_PREFIX, "/extra_msa_activations"))

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
extra_cols = [_load_extra_column_global_attention(arrs, i, c_m_extra) for i in 1:num_extra_blocks]

pair_after_template = Float32.(dump["pair_after_template"]) # (I, L, L, C)
pair_after_extra_ref = Float32.(dump["pair_after_extra"])   # (I, L, L, C)
extra_msa_feat = Float32.(dump["extra_msa_feat"])           # (I, S, L, 25)
I = size(pair_after_template, 1)
L = size(pair_after_template, 2)
S = size(extra_msa_feat, 2)

pair_mask = ones(Float32, L, L, 1)
msa_mask = ones(Float32, S, L, 1)

for i in 1:I
    pair_i = view(pair_after_template, i, :, :, :)
    feat_i = view(extra_msa_feat, i, :, :, :)

    pair_act = reshape(permutedims(pair_i, (3, 1, 2)), size(pair_i, 3), size(pair_i, 1), size(pair_i, 2), 1)
    feat_first = reshape(permutedims(feat_i, (3, 1, 2)), size(feat_i, 3), size(feat_i, 1), size(feat_i, 2), 1)
    msa_act = extra_msa_activations(feat_first)

    for b in eachindex(extra_blocks)
        msa_act, pair_act = _extra_block_forward!(extra_blocks[b], extra_cols[b], msa_act, pair_act, msa_mask, pair_mask)
    end

    pair_py = dropdims(first_to_af2_2d(pair_act); dims=1) # (L,L,C)
    ref = view(pair_after_extra_ref, i, :, :, :)
    d = maximum(abs.(pair_py .- ref))
    @printf("Iter %d extra-stack pair_max_abs: %.8g\n", i - 1, d)
end
