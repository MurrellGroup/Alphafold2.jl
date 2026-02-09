using Statistics

@concrete struct LayerNormFirst <: Onion.Layer
    w
    b
    eps::Float32
end

@layer LayerNormFirst

function LayerNormFirst(dim::Int; eps=1f-5)
    w = ones(Float32, dim)
    b = zeros(Float32, dim)
    return LayerNormFirst(w, b, Float32(eps))
end

function (ln::LayerNormFirst)(x::AbstractArray)
    return Onion.layernorm_first_forward(x, ln.w, ln.b; eps=ln.eps)
end

@concrete struct LinearFirst <: Onion.Layer
    weight
    bias
    use_bias::Bool
end

@layer LinearFirst

function LinearFirst(in_dim::Int, out_dim::Int; bias::Bool=true)
    scale = Float32(1 / sqrt(Float32(in_dim)))
    weight = randn(Float32, out_dim, in_dim) .* scale
    b = bias ? zeros(Float32, out_dim) : zeros(Float32, 0)
    return LinearFirst(weight, b, bias)
end

function (m::LinearFirst)(x::AbstractArray)
    in_dim = size(m.weight, 2)
    out_dim = size(m.weight, 1)
    # Contiguous copy needed for GPU views (SubArray + reshape triggers scalar indexing)
    xc = x isa SubArray ? copy(x) : x
    x2 = reshape(xc, in_dim, :)
    y2 = m.weight * x2
    if m.use_bias
        y2 = y2 .+ m.bias
    end
    out_shape = (out_dim, size(x)[2:end]...)
    return reshape(y2, out_shape)
end
