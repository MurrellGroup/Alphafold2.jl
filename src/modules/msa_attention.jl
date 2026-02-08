@concrete struct MSARowAttentionWithPairBias <: Onion.Layer
    query_norm
    feat_2d_norm
    feat_2d_weights
    attention
end

@layer MSARowAttentionWithPairBias

function MSARowAttentionWithPairBias(c_m::Int, c_z::Int, num_head::Int, head_dim::Int)
    query_norm = LayerNormFirst(c_m)
    feat_2d_norm = LayerNormFirst(c_z)
    feat_2d_weights = randn(Float32, c_z, num_head) .* Float32(1 / sqrt(Float32(c_z)))
    attention = AF2Attention(c_m, c_m, num_head * head_dim, num_head * head_dim, num_head, c_m; gating=true)
    return MSARowAttentionWithPairBias(query_norm, feat_2d_norm, feat_2d_weights, attention)
end

function _msa_row_attention_single(m::MSARowAttentionWithPairBias, msa_act::AbstractArray, msa_mask::AbstractArray, pair_act::AbstractArray)
    # msa_act: (C_m, N_seq, N_res), msa_mask: (N_seq, N_res), pair_act: (C_z, N_res, N_res)
    x = m.query_norm(msa_act)
    z = m.feat_2d_norm(pair_act)

    # Attention batch is N_seq, query/key is N_res.
    x_att = permutedims(x, (1, 3, 2)) # (C_m, N_res, N_seq)
    mask_att = reshape(msa_mask, size(msa_mask, 1), 1, 1, size(msa_mask, 2))

    z_bt = permutedims(z, (2, 3, 1)) # (N_res, N_res, C_z)
    bias2 = reshape(z_bt, size(z_bt, 1) * size(z_bt, 2), size(z_bt, 3)) * m.feat_2d_weights
    nonbatched_bias = reshape(bias2, size(z_bt, 1), size(z_bt, 2), size(m.feat_2d_weights, 2))
    nonbatched_bias = permutedims(nonbatched_bias, (3, 1, 2)) # (H, N_res, N_res)

    out_att = m.attention(x_att, x_att, mask_att; nonbatched_bias=nonbatched_bias)
    return permutedims(out_att, (1, 3, 2)) # (C_m, N_seq, N_res)
end

function (m::MSARowAttentionWithPairBias)(msa_act::AbstractArray, msa_mask::AbstractArray, pair_act::AbstractArray)
    # msa_act: (C_m, N_seq, N_res, B), msa_mask: (N_seq, N_res, B), pair_act: (C_z, N_res, N_res, B)
    B = size(msa_act, 4)
    outs = ntuple(
        b -> _msa_row_attention_single(
            m,
            view(msa_act, :, :, :, b),
            view(msa_mask, :, :, b),
            view(pair_act, :, :, :, b),
        ),
        B,
    )
    reshaped = map(x -> reshape(x, size(x)..., 1), outs)
    return cat(reshaped...; dims=4)
end

"""
    load_msa_row_attention_npz!(m, npz_path)

Load AF2 MSA row attention with pair bias parameters saved by
`scripts/parity/dump_msa_row_attention_py.py`.
"""
function load_msa_row_attention_npz!(m::MSARowAttentionWithPairBias, params_source)
    arrs = af2_params_read(params_source)

    _copy_ln_af2!(m.query_norm, arrs, "query_norm")
    _copy_ln_af2!(m.feat_2d_norm, arrs, "feat_2d_norm")
    m.feat_2d_weights .= arrs["feat_2d_weights"]
    _copy_attention_af2!(m.attention, arrs, "attention")

    return m
end

@concrete struct MSAColumnAttention <: Onion.Layer
    query_norm
    attention
end

@layer MSAColumnAttention

function MSAColumnAttention(c_m::Int, num_head::Int, head_dim::Int)
    query_norm = LayerNormFirst(c_m)
    attention = AF2Attention(c_m, c_m, num_head * head_dim, num_head * head_dim, num_head, c_m; gating=true)
    return MSAColumnAttention(query_norm, attention)
end

function _msa_column_attention_single(m::MSAColumnAttention, msa_act::AbstractArray, msa_mask::AbstractArray)
    # msa_act: (C_m, N_seq, N_res), msa_mask: (N_seq, N_res)
    x = permutedims(msa_act, (1, 3, 2))    # (C_m, N_res, N_seq)
    mask = permutedims(msa_mask, (2, 1))   # (N_res, N_seq)

    x = m.query_norm(x)

    x_att = permutedims(x, (1, 3, 2)) # (C_m, N_seq, N_res) as query/key with batch=N_res
    mask_att = reshape(mask, size(mask, 1), 1, 1, size(mask, 2))
    out_att = m.attention(x_att, x_att, mask_att)
    out = permutedims(out_att, (1, 3, 2)) # (C_m, N_res, N_seq)

    return permutedims(out, (1, 3, 2)) # (C_m, N_seq, N_res)
end

function (m::MSAColumnAttention)(msa_act::AbstractArray, msa_mask::AbstractArray)
    # msa_act: (C_m, N_seq, N_res, B), msa_mask: (N_seq, N_res, B)
    B = size(msa_act, 4)
    outs = ntuple(
        b -> _msa_column_attention_single(
            m,
            view(msa_act, :, :, :, b),
            view(msa_mask, :, :, b),
        ),
        B,
    )
    reshaped = map(x -> reshape(x, size(x)..., 1), outs)
    return cat(reshaped...; dims=4)
end

"""
    load_msa_column_attention_npz!(m, npz_path)

Load AF2 MSA column attention parameters saved by
`scripts/parity/dump_msa_column_attention_py.py`.
"""
function load_msa_column_attention_npz!(m::MSAColumnAttention, params_source)
    arrs = af2_params_read(params_source)

    _copy_ln_af2!(m.query_norm, arrs, "query_norm")
    _copy_attention_af2!(m.attention, arrs, "attention")

    return m
end

@concrete struct MSAColumnGlobalAttention <: Onion.Layer
    query_norm
    attention
end

@layer MSAColumnGlobalAttention

function MSAColumnGlobalAttention(c_m::Int, num_head::Int, head_dim::Int)
    query_norm = LayerNormFirst(c_m)
    attention = AF2GlobalAttention(c_m, c_m, num_head * head_dim, num_head * head_dim, num_head, c_m; gating=true)
    return MSAColumnGlobalAttention(query_norm, attention)
end

function _msa_column_global_attention_single(m::MSAColumnGlobalAttention, msa_act::AbstractArray, msa_mask::AbstractArray)
    # msa_act: (C_m, N_seq, N_res), msa_mask: (N_seq, N_res)
    x = permutedims(msa_act, (1, 3, 2))    # (C_m, N_res, N_seq)
    mask = permutedims(msa_mask, (2, 1))   # (N_res, N_seq)

    x = m.query_norm(x)
    x_att = permutedims(x, (1, 3, 2)) # (C_m, N_seq, N_res) where batch is N_res
    q_mask = reshape(permutedims(mask, (2, 1)), size(mask, 2), 1, size(mask, 1)) # (N_seq, 1, N_res)

    out_att = m.attention(x_att, x_att, q_mask) # (C_m, N_seq, N_res)
    out = permutedims(out_att, (1, 3, 2)) # (C_m, N_res, N_seq)

    return permutedims(out, (1, 3, 2)) # (C_m, N_seq, N_res)
end

function (m::MSAColumnGlobalAttention)(msa_act::AbstractArray, msa_mask::AbstractArray)
    # msa_act: (C_m, N_seq, N_res, B), msa_mask: (N_seq, N_res, B)
    B = size(msa_act, 4)
    outs = ntuple(
        b -> _msa_column_global_attention_single(
            m,
            view(msa_act, :, :, :, b),
            view(msa_mask, :, :, b),
        ),
        B,
    )
    reshaped = map(x -> reshape(x, size(x)..., 1), outs)
    return cat(reshaped...; dims=4)
end

"""
    load_msa_column_global_attention_npz!(m, npz_path)

Load AF2 MSA column global attention parameters saved by
`scripts/parity/dump_msa_column_global_attention_py.py`.
"""
function load_msa_column_global_attention_npz!(m::MSAColumnGlobalAttention, params_source)
    arrs = af2_params_read(params_source)

    _copy_ln_af2!(m.query_norm, arrs, "query_norm")
    _copy_global_attention_af2!(m.attention, arrs, "attention")

    return m
end
