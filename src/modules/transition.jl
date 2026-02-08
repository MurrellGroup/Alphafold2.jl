@concrete struct Transition <: Onion.Layer
    input_layer_norm
    transition1
    transition2
end

@layer Transition

function Transition(c::Int, num_intermediate_factor::Real)
    num_intermediate = Int(floor(c * num_intermediate_factor))
    ln = LayerNormFirst(c)
    t1 = LinearFirst(c, num_intermediate)
    t2 = LinearFirst(num_intermediate, c)
    return Transition(ln, t1, t2)
end

function (m::Transition)(act::AbstractArray, mask::AbstractArray)
    # act: (C, N, B), mask: (N, B)
    _ = mask
    x = m.input_layer_norm(act)
    x = max.(m.transition1(x), 0f0)
    x = m.transition2(x)
    return x
end

function (m::Transition)(act::AbstractArray; mask=nothing)
    if mask === nothing
        mask = ones_like(act, size(act, 2), size(act, 3))
    end
    return m(act, mask)
end

"""
    load_transition_npz!(m, npz_path)

Load AF2 Transition parameters saved by `scripts/parity/dump_transition_py.py`.
"""
function load_transition_npz!(m::Transition, params_source)
    arrs = af2_params_read(params_source)

    _copy_ln_af2!(m.input_layer_norm, arrs, "input_layer_norm")
    _copy_linear_af2!(m.transition1, arrs, "transition1")
    _copy_linear_af2!(m.transition2, arrs, "transition2")

    return m
end
