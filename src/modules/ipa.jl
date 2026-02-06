using NNlib

@concrete struct PointProjection <: Onion.Layer
    linear
    num_points::Int
    no_heads::Int
end

@layer PointProjection

function PointProjection(c_hidden::Int, num_points::Int, no_heads::Int)
    linear = LinearFirst(c_hidden, no_heads * 3 * num_points)
    return PointProjection(linear, num_points, no_heads)
end

function (m::PointProjection)(activations::AbstractArray, rigids::Rigid)
    raw = m.linear(activations) # (3*H*P, L, B)
    H = m.no_heads
    P = m.num_points
    raw = reshape(raw, P, H, 3, size(raw, 2), size(raw, 3))
    points_local = permutedims(raw, (3, 1, 2, 4, 5)) # (3, P, H, L, B)
    points_global = apply_rigid(rigids, points_local)
    return points_global
end

@concrete struct InvariantPointAttention <: Onion.Layer
    c_s::Int
    c_z::Int
    c_hidden::Int
    no_heads::Int
    no_qk_points::Int
    no_v_points::Int
    linear_q
    linear_q_points
    linear_kv
    linear_kv_points
    linear_b
    head_weights
    linear_out
    eps::Float32
    inf::Float32
end

@layer InvariantPointAttention

function InvariantPointAttention(
    c_s::Int,
    c_z::Int,
    c_hidden::Int,
    no_heads::Int,
    no_qk_points::Int,
    no_v_points::Int;
    inf::Real=1e5,
    eps::Real=1e-8,
)
    linear_q = LinearFirst(c_s, c_hidden * no_heads)
    linear_q_points = PointProjection(c_s, no_qk_points, no_heads)
    linear_kv = LinearFirst(c_s, 2 * c_hidden * no_heads)
    linear_kv_points = PointProjection(c_s, no_qk_points + no_v_points, no_heads)
    linear_b = LinearFirst(c_z, no_heads)
    head_weights = fill(0.54132485f0, no_heads)
    linear_out = LinearFirst(no_heads * (c_z + c_hidden + no_v_points * 4), c_s)

    return InvariantPointAttention(
        c_s,
        c_z,
        c_hidden,
        no_heads,
        no_qk_points,
        no_v_points,
        linear_q,
        linear_q_points,
        linear_kv,
        linear_kv_points,
        linear_b,
        head_weights,
        linear_out,
        Float32(eps),
        Float32(inf),
    )
end

function (m::InvariantPointAttention)(s::AbstractArray, z::AbstractArray, r::Rigid, mask::AbstractArray)
    # s: (C_s, L, B), z: (C_z, L, L, B), mask: (L, B)
    q = m.linear_q(s)
    q = reshape(q, m.c_hidden, m.no_heads, size(q, 2), size(q, 3)) # (C, H, L, B)

    kv = m.linear_kv(s)
    kv = reshape(kv, 2 * m.c_hidden, m.no_heads, size(kv, 2), size(kv, 3))
    k = view(kv, 1:m.c_hidden, :, :, :)
    v = view(kv, (m.c_hidden + 1):(2 * m.c_hidden), :, :, :)

    q_bhlc = permutedims(q, (4, 2, 3, 1))
    k_bhlc = permutedims(k, (4, 2, 3, 1))
    k_bhcl = permutedims(k_bhlc, (1, 2, 4, 3))

    B = size(q_bhlc, 1)
    H = size(q_bhlc, 2)
    L = size(q_bhlc, 3)
    C = size(q_bhlc, 4)

    q3 = permutedims(reshape(q_bhlc, B * H, L, C), (2, 3, 1))
    k3 = permutedims(reshape(k_bhcl, B * H, C, L), (2, 3, 1))
    a3 = NNlib.batched_mul(q3, k3)
    a = reshape(a3, L, L, B, H)
    a = permutedims(a, (3, 4, 1, 2)) # (B, H, L, L)

    a = a .* sqrt(1f0 / (3f0 * m.c_hidden))

    b = m.linear_b(z) # (H, L, L, B)
    b_perm = permutedims(b, (4, 1, 2, 3))
    a = a .+ sqrt(1f0 / 3f0) .* b_perm

    q_pts = m.linear_q_points(s, r)   # (3, Pq, H, L, B)
    kv_pts = m.linear_kv_points(s, r) # (3, Pq+Pv, H, L, B)
    k_pts = view(kv_pts, :, 1:m.no_qk_points, :, :, :)
    v_pts = view(kv_pts, :, (m.no_qk_points + 1):(m.no_qk_points + m.no_v_points), :, :, :)

    q_pts = permutedims(q_pts, (5, 4, 3, 2, 1)) # (B, L, H, Pq, 3)
    k_pts = permutedims(k_pts, (5, 4, 3, 2, 1)) # (B, L, H, Pq, 3)
    v_pts = permutedims(v_pts, (5, 4, 3, 2, 1)) # (B, L, H, Pv, 3)

    q_exp = reshape(q_pts, size(q_pts, 1), size(q_pts, 2), 1, size(q_pts, 3), size(q_pts, 4), size(q_pts, 5))
    k_exp = reshape(k_pts, size(k_pts, 1), 1, size(k_pts, 2), size(k_pts, 3), size(k_pts, 4), size(k_pts, 5))
    pt_att = q_exp .- k_exp
    pt_att = sum(pt_att .^ 2; dims=6)

    head_weights = NNlib.softplus.(m.head_weights)
    head_weights = head_weights .* sqrt(1f0 / (3f0 * (m.no_qk_points * 9f0 / 2f0)))
    hw = reshape(head_weights, 1, 1, 1, m.no_heads, 1, 1)
    pt_att = sum(pt_att .* hw; dims=5) .* (-0.5f0)
    pt_att = dropdims(pt_att; dims=(5, 6))
    pt_att = permutedims(pt_att, (1, 4, 2, 3)) # (B, H, L, L)

    square_mask = reshape(mask, size(mask, 1), 1, size(mask, 2)) .* reshape(mask, 1, size(mask, 1), size(mask, 2))
    square_mask = permutedims(square_mask, (3, 1, 2)) # (B, L, L)
    square_mask = m.inf .* (square_mask .- 1)

    a = a .+ pt_att
    a = a .+ reshape(square_mask, size(square_mask, 1), 1, size(square_mask, 2), size(square_mask, 3))
    a = NNlib.softmax(a; dims=4)

    v_bhlc = permutedims(v, (4, 2, 3, 1))
    a3 = reshape(permutedims(a, (3, 4, 1, 2)), L, L, :)
    v3 = permutedims(reshape(v_bhlc, B * H, L, C), (2, 3, 1))
    o3 = NNlib.batched_mul(a3, v3)
    o = reshape(o3, L, C, B, H)
    o = permutedims(o, (3, 1, 4, 2)) # (B, L, H, C)
    o = permutedims(o, (1, 2, 4, 3)) # (B, L, C, H)
    o = reshape(o, B, L, H * C)

    v_pts_x = _view_last1(v_pts, 1)
    v_pts_y = _view_last1(v_pts, 2)
    v_pts_z = _view_last1(v_pts, 3)

    vpx = permutedims(v_pts_x, (2, 4, 1, 3))
    vpy = permutedims(v_pts_y, (2, 4, 1, 3))
    vpz = permutedims(v_pts_z, (2, 4, 1, 3))

    vpx3 = reshape(vpx, L, m.no_v_points, :)
    vpy3 = reshape(vpy, L, m.no_v_points, :)
    vpz3 = reshape(vpz, L, m.no_v_points, :)

    o_px = NNlib.batched_mul(a3, vpx3)
    o_py = NNlib.batched_mul(a3, vpy3)
    o_pz = NNlib.batched_mul(a3, vpz3)

    o_px = reshape(o_px, L, m.no_v_points, B, m.no_heads)
    o_py = reshape(o_py, L, m.no_v_points, B, m.no_heads)
    o_pz = reshape(o_pz, L, m.no_v_points, B, m.no_heads)

    o_px = permutedims(o_px, (3, 1, 4, 2))
    o_py = permutedims(o_py, (3, 1, 4, 2))
    o_pz = permutedims(o_pz, (3, 1, 4, 2))

    o_pt = cat(o_px, o_py, o_pz; dims=5) # (B, L, H, P, 3)
    o_pt = invert_apply_rigid(r, permutedims(o_pt, (5, 4, 3, 2, 1))) # (3, P, H, L, B)
    o_pt = permutedims(o_pt, (5, 4, 3, 2, 1)) # (B, L, H, P, 3)

    o_pt_norm = sqrt.(sum(o_pt .^ 2; dims=5) .+ m.eps)
    o_pt_norm = dropdims(o_pt_norm; dims=5)
    o_pt_norm = permutedims(o_pt_norm, (1, 2, 4, 3)) # (B, L, P, H)
    o_pt_norm = reshape(o_pt_norm, size(o_pt_norm, 1), size(o_pt_norm, 2), m.no_heads * m.no_v_points)

    o_pt = permutedims(o_pt, (1, 2, 4, 3, 5))
    o_pt = reshape(o_pt, size(o_pt)[1:end-3]..., m.no_heads * m.no_v_points, 3)
    o_px = _view_last1(o_pt, 1)
    o_py = _view_last1(o_pt, 2)
    o_pz = _view_last1(o_pt, 3)

    a_t = permutedims(a, (1, 2, 4, 3))
    a_exp = reshape(a_t, size(a_t, 1), size(a_t, 2), size(a_t, 3), size(a_t, 4), 1)
    z_swap = permutedims(z, (4, 3, 2, 1))
    z_exp = reshape(z_swap, size(z_swap, 1), 1, size(z_swap, 2), size(z_swap, 3), size(z_swap, 4))
    o_pair = sum(a_exp .* z_exp; dims=3)
    o_pair = dropdims(o_pair; dims=3)
    o_pair = permutedims(o_pair, (1, 3, 2, 4)) # (B, L, H, C_z)
    o_pair = permutedims(o_pair, (1, 2, 4, 3)) # (B, L, C_z, H)
    o_pair = reshape(o_pair, size(o_pair, 1), size(o_pair, 2), m.no_heads * m.c_z)

    o_feat = permutedims(o, (3, 2, 1))
    o_px = permutedims(o_px, (3, 2, 1))
    o_py = permutedims(o_py, (3, 2, 1))
    o_pz = permutedims(o_pz, (3, 2, 1))
    o_pt_norm = permutedims(o_pt_norm, (3, 2, 1))
    o_pair = permutedims(o_pair, (3, 2, 1))

    concat = cat(o_feat, o_px, o_py, o_pz, o_pt_norm, o_pair; dims=1)
    return m.linear_out(concat)
end

function _copy_point_projection_af2!(m::PointProjection, arrs::AbstractDict, prefix::String)
    _copy_linear_af2!(m.linear, arrs, prefix)
    return m
end

"""
    load_invariant_point_attention_npz!(m, npz_path)

Load AF2 InvariantPointAttention parameters saved by
`scripts/parity/dump_ipa_py.py`.
"""
function load_invariant_point_attention_npz!(m::InvariantPointAttention, npz_path::AbstractString)
    arrs = NPZ.npzread(npz_path)

    _copy_linear_af2!(m.linear_q, arrs, "q_scalar")
    _copy_point_projection_af2!(m.linear_q_points, arrs, "q_point_local")
    _copy_linear_af2!(m.linear_kv, arrs, "kv_scalar")
    _copy_point_projection_af2!(m.linear_kv_points, arrs, "kv_point_local")
    _copy_linear_af2!(m.linear_b, arrs, "attention_2d")

    m.head_weights .= arrs["trainable_point_weights"]

    _copy_linear_af2!(m.linear_out, arrs, "output_projection")

    return m
end
