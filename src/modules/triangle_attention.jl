@concrete struct TriangleAttention <: Onion.Layer
    query_norm
    feat_2d_weights
    attention
    orientation::Symbol
end

@layer TriangleAttention

function TriangleAttention(c_z::Int, no_heads::Int, head_dim::Int; orientation::Symbol=:per_row)
    @assert orientation == :per_row || orientation == :per_column

    query_norm = LayerNormFirst(c_z)
    feat_2d_weights = randn(Float32, c_z, no_heads) .* Float32(1 / sqrt(Float32(c_z)))
    attention = AF2Attention(c_z, c_z, no_heads * head_dim, no_heads * head_dim, no_heads, c_z; gating=true)

    return TriangleAttention(query_norm, feat_2d_weights, attention, orientation)
end

function _triangle_attention_single(m::TriangleAttention, pair_act::AbstractArray, pair_mask::AbstractArray)
    # pair_act: (C, L, L), pair_mask: (L, L)
    x = pair_act
    mask = pair_mask

    if m.orientation == :per_column
        x = permutedims(x, (1, 3, 2))
        mask = permutedims(mask, (2, 1))
    end

    L = size(x, 2)
    x_att = permutedims(x, (1, 3, 2)) # (C, Q=L, Batch=L)
    x_att = m.query_norm(x_att)

    mask_att = reshape(mask, L, 1, 1, L)

    # AF2: nonbatched_bias = einsum('qkc,ch->hqk', pair_act, weights)
    pair_bt = permutedims(x_att, (3, 2, 1)) # (L, L, C)
    bias2 = reshape(pair_bt, L * L, size(pair_bt, 3)) * m.feat_2d_weights # (L*L, H)
    nonbatched_bias = reshape(bias2, L, L, size(m.feat_2d_weights, 2))
    nonbatched_bias = permutedims(nonbatched_bias, (3, 1, 2)) # (H, L, L)

    out_att = m.attention(x_att, x_att, mask_att; nonbatched_bias=nonbatched_bias)
    out = permutedims(out_att, (1, 3, 2)) # (C, L, L)

    if m.orientation == :per_column
        out = permutedims(out, (1, 3, 2))
    end

    return out
end

function (m::TriangleAttention)(pair_act::AbstractArray, pair_mask::AbstractArray)
    # pair_act: (C, L, L, B), pair_mask: (L, L, B)
    B = size(pair_act, 4)
    outs = ntuple(
        b -> _triangle_attention_single(
            m,
            view(pair_act, :, :, :, b),
            view(pair_mask, :, :, b),
        ),
        B,
    )
    reshaped = map(x -> reshape(x, size(x)..., 1), outs)
    return cat(reshaped...; dims=4)
end

function (m::TriangleAttention)(pair_act::AbstractArray; pair_mask=nothing)
    if pair_mask === nothing
        pair_mask = ones_like(pair_act, size(pair_act, 2), size(pair_act, 3), size(pair_act, 4))
    end
    return m(pair_act, pair_mask)
end

"""
    load_triangle_attention_npz!(m, npz_path)

Load AF2 TriangleAttention parameters saved by
`scripts/parity/dump_triangle_attention_py.py`.
"""
function load_triangle_attention_npz!(m::TriangleAttention, npz_path::AbstractString)
    arrs = NPZ.npzread(npz_path)

    _copy_ln_af2!(m.query_norm, arrs, "query_norm")
    m.feat_2d_weights .= arrs["feat_2d_weights"]

    _copy_attention_af2!(m.attention, arrs, "attention")

    return m
end
