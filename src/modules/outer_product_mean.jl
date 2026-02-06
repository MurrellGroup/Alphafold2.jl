@concrete struct OuterProductMean <: Onion.Layer
    layer_norm_input
    left_projection
    right_projection
    output_w
    output_b
    epsilon::Float32
end

@layer OuterProductMean

function OuterProductMean(c_m::Int, c_outer::Int, c_z::Int; epsilon::Real=1f-3)
    ln = LayerNormFirst(c_m)
    left = LinearFirst(c_m, c_outer)
    right = LinearFirst(c_m, c_outer)

    wscale = Float32(1 / sqrt(Float32(c_outer)))
    output_w = randn(Float32, c_outer, c_outer, c_z) .* wscale
    output_b = zeros(Float32, c_z)

    return OuterProductMean(ln, left, right, output_w, output_b, Float32(epsilon))
end

function _outer_product_mean_single(m::OuterProductMean, act::AbstractArray, mask::AbstractArray)
    # act: (C_m, N_seq, N_res), mask: (N_seq, N_res)
    x = m.layer_norm_input(act)

    mask3 = reshape(mask, 1, size(mask, 1), size(mask, 2))
    left = mask3 .* m.left_projection(x)  # (C_o, N_seq, N_res)
    right = mask3 .* m.right_projection(x) # (C_o, N_seq, N_res)

    C_o = size(left, 1)
    N_seq = size(left, 2)
    N_res = size(left, 3)
    C_z = size(m.output_b, 1)

    # Build all (i, j) sequence outer-products using batched matmul.
    lrep = repeat(reshape(left, C_o, N_seq, N_res, 1), 1, 1, 1, N_res)
    lrep = reshape(lrep, C_o, N_seq, N_res * N_res)

    r = permutedims(right, (2, 1, 3)) # (N_seq, C_o, N_res)
    rrep = repeat(reshape(r, N_seq, C_o, 1, N_res), 1, 1, N_res, 1)
    rrep = reshape(rrep, N_seq, C_o, N_res * N_res)

    outer3 = NNlib.batched_mul(lrep, rrep) # (C_o, C_o, N_res^2)
    outer = reshape(outer3, C_o, C_o, N_res, N_res)

    outer_flat = reshape(permutedims(outer, (3, 4, 1, 2)), N_res * N_res, C_o * C_o)
    out_flat = outer_flat * reshape(m.output_w, C_o * C_o, C_z)
    out_flat = out_flat .+ reshape(m.output_b, 1, C_z)
    out = reshape(out_flat, N_res, N_res, C_z)

    norm = transpose(mask) * mask # (N_res, N_res)
    out = out ./ (m.epsilon .+ reshape(norm, size(norm, 1), size(norm, 2), 1))

    return permutedims(out, (3, 1, 2)) # (C_z, N_res, N_res)
end

function (m::OuterProductMean)(act::AbstractArray, mask::AbstractArray)
    # act: (C_m, N_seq, N_res, B), mask: (N_seq, N_res, B)
    B = size(act, 4)
    outs = ntuple(
        b -> _outer_product_mean_single(
            m,
            view(act, :, :, :, b),
            view(mask, :, :, b),
        ),
        B,
    )
    reshaped = map(x -> reshape(x, size(x)..., 1), outs)
    return cat(reshaped...; dims=4)
end

function (m::OuterProductMean)(act::AbstractArray; mask=nothing)
    if mask === nothing
        mask = ones_like(act, size(act, 2), size(act, 3), size(act, 4))
    end
    return m(act, mask)
end

"""
    load_outer_product_mean_npz!(m, npz_path)

Load AF2 OuterProductMean parameters saved by
`scripts/parity/dump_outer_product_mean_py.py`.
"""
function load_outer_product_mean_npz!(m::OuterProductMean, npz_path::AbstractString)
    arrs = NPZ.npzread(npz_path)

    _copy_ln_af2!(m.layer_norm_input, arrs, "layer_norm_input")
    _copy_linear_af2!(m.left_projection, arrs, "left_projection")
    _copy_linear_af2!(m.right_projection, arrs, "right_projection")

    m.output_w .= arrs["output_w"]
    m.output_b .= arrs["output_b"]

    return m
end
