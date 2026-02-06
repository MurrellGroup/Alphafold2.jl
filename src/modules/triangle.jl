using NNlib

@concrete struct TriangleMultiplication <: Onion.Layer
    layer_norm_input
    left_projection
    right_projection
    left_gate
    right_gate
    center_layer_norm
    output_projection
    gating_linear
    outgoing::Bool
end

@layer TriangleMultiplication

function TriangleMultiplication(c_z::Int, c_hidden::Int; outgoing::Bool=true)
    layer_norm_input = LayerNormFirst(c_z)
    left_projection = LinearFirst(c_z, c_hidden)
    right_projection = LinearFirst(c_z, c_hidden)
    left_gate = LinearFirst(c_z, c_hidden)
    right_gate = LinearFirst(c_z, c_hidden)
    center_layer_norm = LayerNormFirst(c_hidden)
    output_projection = LinearFirst(c_hidden, c_z)
    gating_linear = LinearFirst(c_z, c_z)

    return TriangleMultiplication(
        layer_norm_input,
        left_projection,
        right_projection,
        left_gate,
        right_gate,
        center_layer_norm,
        output_projection,
        gating_linear,
        outgoing,
    )
end

function _triangle_contract(left::AbstractArray, right::AbstractArray, outgoing::Bool)
    # left/right: (C, L, L, B)
    L = size(left, 2)
    C = size(left, 1)
    B = size(left, 4)

    a = permutedims(left, (2, 3, 1, 4))
    b = permutedims(right, (2, 3, 1, 4))

    a3 = reshape(a, L, L, C * B)
    b3 = reshape(b, L, L, C * B)

    out3 = if outgoing
        # 'ikc,jkc->ijc'
        NNlib.batched_mul(a3, permutedims(b3, (2, 1, 3)))
    else
        # 'kjc,kic->ijc'
        NNlib.batched_mul(permutedims(b3, (2, 1, 3)), a3)
    end

    out = reshape(out3, L, L, C, B)
    return permutedims(out, (3, 1, 2, 4))
end

function (m::TriangleMultiplication)(left_act::AbstractArray, left_mask::AbstractArray)
    # left_act: (Cz, L, L, B)
    # left_mask: (L, L, B)
    mask = reshape(left_mask, 1, size(left_mask, 1), size(left_mask, 2), size(left_mask, 3))

    act = m.layer_norm_input(left_act)
    input_act = act

    left_proj_act = mask .* m.left_projection(act)
    right_proj_act = mask .* m.right_projection(act)

    left_proj_act = left_proj_act .* NNlib.sigmoid.(m.left_gate(act))
    right_proj_act = right_proj_act .* NNlib.sigmoid.(m.right_gate(act))

    act = _triangle_contract(left_proj_act, right_proj_act, m.outgoing)

    act = m.center_layer_norm(act)
    act = m.output_projection(act)
    act = act .* NNlib.sigmoid.(m.gating_linear(input_act))

    return act
end

function (m::TriangleMultiplication)(left_act::AbstractArray; left_mask=nothing)
    if left_mask === nothing
        left_mask = ones_like(left_act, size(left_act, 2), size(left_act, 3), size(left_act, 4))
    end
    return m(left_act, left_mask)
end

function _copy_linear_af2!(lin::LinearFirst, arrs::AbstractDict, prefix::String)
    lin.weight .= permutedims(arrs[string(prefix, "_weights")], (2, 1))
    lin.use_bias && (lin.bias .= arrs[string(prefix, "_bias")])
    return lin
end

function _copy_ln_af2!(ln::LayerNormFirst, arrs::AbstractDict, prefix::String)
    ln.w .= arrs[string(prefix, "_scale")]
    ln.b .= arrs[string(prefix, "_offset")]
    return ln
end

"""
    load_triangle_multiplication_npz!(m, npz_path)

Load AF2 TriangleMultiplication parameters saved by
`scripts/parity/dump_triangle_py.py` into a Julia `TriangleMultiplication` layer.
"""
function load_triangle_multiplication_npz!(m::TriangleMultiplication, npz_path::AbstractString)
    arrs = NPZ.npzread(npz_path)

    _copy_ln_af2!(m.layer_norm_input, arrs, "layer_norm_input")
    _copy_linear_af2!(m.left_projection, arrs, "left_projection")
    _copy_linear_af2!(m.right_projection, arrs, "right_projection")
    _copy_linear_af2!(m.left_gate, arrs, "left_gate")
    _copy_linear_af2!(m.right_gate, arrs, "right_gate")
    _copy_ln_af2!(m.center_layer_norm, arrs, "center_layer_norm")
    _copy_linear_af2!(m.output_projection, arrs, "output_projection")
    _copy_linear_af2!(m.gating_linear, arrs, "gating_linear")

    return m
end
