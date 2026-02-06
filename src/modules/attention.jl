using NNlib

const SOFTMAX_MASK = -1f9

@concrete struct AF2Attention <: Onion.Layer
    query_w
    key_w
    value_w
    output_w
    output_b
    gating_w
    gating_b
    num_head::Int
    key_head_dim::Int
    value_head_dim::Int
    gating::Bool
end

@layer AF2Attention

function AF2Attention(
    q_channels::Int,
    m_channels::Int,
    key_dim::Int,
    value_dim::Int,
    num_head::Int,
    output_dim::Int;
    gating::Bool=true,
)
    @assert key_dim % num_head == 0 "key_dim must be divisible by num_head"
    @assert value_dim % num_head == 0 "value_dim must be divisible by num_head"

    key_head_dim = div(key_dim, num_head)
    value_head_dim = div(value_dim, num_head)

    scale_q = Float32(1 / sqrt(Float32(q_channels)))
    scale_m = Float32(1 / sqrt(Float32(m_channels)))
    scale_o = Float32(1 / sqrt(Float32(num_head * value_head_dim)))

    query_w = randn(Float32, q_channels, num_head, key_head_dim) .* scale_q
    key_w = randn(Float32, m_channels, num_head, key_head_dim) .* scale_m
    value_w = randn(Float32, m_channels, num_head, value_head_dim) .* scale_m

    output_w = randn(Float32, num_head, value_head_dim, output_dim) .* scale_o
    output_b = zeros(Float32, output_dim)

    if gating
        gating_w = zeros(Float32, q_channels, num_head, value_head_dim)
        gating_b = ones(Float32, num_head, value_head_dim)
    else
        gating_w = zeros(Float32, 0, 0, 0)
        gating_b = zeros(Float32, 0, 0)
    end

    return AF2Attention(
        query_w,
        key_w,
        value_w,
        output_w,
        output_b,
        gating_w,
        gating_b,
        num_head,
        key_head_dim,
        value_head_dim,
        gating,
    )
end

function _linear_bqhc(x_bqc::AbstractArray, w::AbstractArray)
    # x_bqc: (B, Q, C_in), w: (C_in, H, C_head) -> (B, Q, H, C_head)
    B, Q, C_in = size(x_bqc)
    H = size(w, 2)
    C_head = size(w, 3)

    y2 = reshape(x_bqc, B * Q, C_in) * reshape(w, C_in, H * C_head)
    return reshape(y2, B, Q, H, C_head)
end

function _apply_attention(
    q::AbstractArray,
    k::AbstractArray,
    v::AbstractArray,
    mask,
    nonbatched_bias,
)
    # q: (B, Q, H, Ck), k: (B, K, H, Ck), v: (B, K, H, Cv)
    B, Q, H, Ck = size(q)
    K = size(k, 2)
    Cv = size(v, 4)

    q_bhqc = permutedims(q, (1, 3, 2, 4))
    k_bhkc = permutedims(k, (1, 3, 2, 4))

    q3 = permutedims(reshape(q_bhqc, B * H, Q, Ck), (2, 3, 1))
    k3 = permutedims(reshape(k_bhkc, B * H, K, Ck), (2, 3, 1))
    logits3 = NNlib.batched_mul(q3, permutedims(k3, (2, 1, 3)))

    logits = reshape(logits3, Q, K, B, H)
    logits = permutedims(logits, (3, 4, 1, 2)) # (B, H, Q, K)

    if nonbatched_bias !== nothing
        logits = logits .+ reshape(nonbatched_bias, 1, size(nonbatched_bias)...)
    end

    if mask !== nothing
        mask_bool = mask .!= 0
        logits = ifelse.(mask_bool, logits, SOFTMAX_MASK)
    end

    weights = NNlib.softmax(logits; dims=4)

    v_bhkc = permutedims(v, (1, 3, 2, 4))
    a3 = reshape(permutedims(weights, (3, 4, 1, 2)), Q, K, B * H)
    v3 = permutedims(reshape(v_bhkc, B * H, K, Cv), (2, 3, 1))

    out3 = NNlib.batched_mul(a3, v3) # (Q, Cv, B*H)
    out = reshape(out3, Q, Cv, B, H)
    out = permutedims(out, (3, 1, 4, 2)) # (B, Q, H, Cv)

    return out
end

function (m::AF2Attention)(q_data::AbstractArray, m_data::AbstractArray, mask; nonbatched_bias=nothing)
    # q_data: (Cq, Q, B), m_data: (Cm, K, B)
    q_bt = permutedims(q_data, (3, 2, 1))
    m_bt = permutedims(m_data, (3, 2, 1))

    q = _linear_bqhc(q_bt, m.query_w)
    k = _linear_bqhc(m_bt, m.key_w)
    v = _linear_bqhc(m_bt, m.value_w)

    q = q .* (Float32(m.key_head_dim)^-0.5f0)

    out = _apply_attention(q, k, v, mask, nonbatched_bias) # (B, Q, H, Cv)

    if m.gating
        gate = _linear_bqhc(q_bt, m.gating_w)
        gate = NNlib.sigmoid.(gate .+ reshape(m.gating_b, 1, 1, size(m.gating_b, 1), size(m.gating_b, 2)))
        out = out .* gate
    end

    B, Q, H, Cv = size(out)
    out2 = reshape(out, B * Q, H * Cv) * reshape(m.output_w, H * Cv, size(m.output_w, 3))
    out2 = out2 .+ reshape(m.output_b, 1, :)
    out_bt = reshape(out2, B, Q, size(m.output_w, 3))

    return permutedims(out_bt, (3, 2, 1))
end

function _copy_attention_af2!(m::AF2Attention, arrs::AbstractDict, prefix::String)
    m.query_w .= arrs[string(prefix, "_query_w")]
    m.key_w .= arrs[string(prefix, "_key_w")]
    m.value_w .= arrs[string(prefix, "_value_w")]
    m.output_w .= arrs[string(prefix, "_output_w")]
    m.output_b .= arrs[string(prefix, "_output_b")]

    if m.gating
        m.gating_w .= arrs[string(prefix, "_gating_w")]
        m.gating_b .= arrs[string(prefix, "_gating_b")]
    end

    return m
end

@concrete struct AF2GlobalAttention <: Onion.Layer
    query_w
    key_w
    value_w
    output_w
    output_b
    gating_w
    gating_b
    num_head::Int
    key_head_dim::Int
    value_head_dim::Int
    gating::Bool
end

@layer AF2GlobalAttention

function AF2GlobalAttention(
    q_channels::Int,
    m_channels::Int,
    key_dim::Int,
    value_dim::Int,
    num_head::Int,
    output_dim::Int;
    gating::Bool=true,
)
    @assert key_dim % num_head == 0 "key_dim must be divisible by num_head"
    @assert value_dim % num_head == 0 "value_dim must be divisible by num_head"

    key_head_dim = div(key_dim, num_head)
    value_head_dim = div(value_dim, num_head)

    scale_q = Float32(1 / sqrt(Float32(q_channels)))
    scale_m = Float32(1 / sqrt(Float32(m_channels)))
    scale_o = Float32(1 / sqrt(Float32(num_head * value_head_dim)))

    query_w = randn(Float32, q_channels, num_head, key_head_dim) .* scale_q
    key_w = randn(Float32, m_channels, key_head_dim) .* scale_m
    value_w = randn(Float32, m_channels, value_head_dim) .* scale_m

    output_w = randn(Float32, num_head, value_head_dim, output_dim) .* scale_o
    output_b = zeros(Float32, output_dim)

    if gating
        gating_w = zeros(Float32, q_channels, num_head, value_head_dim)
        gating_b = ones(Float32, num_head, value_head_dim)
    else
        gating_w = zeros(Float32, 0, 0, 0)
        gating_b = zeros(Float32, 0, 0)
    end

    return AF2GlobalAttention(
        query_w,
        key_w,
        value_w,
        output_w,
        output_b,
        gating_w,
        gating_b,
        num_head,
        key_head_dim,
        value_head_dim,
        gating,
    )
end

function (m::AF2GlobalAttention)(q_data::AbstractArray, m_data::AbstractArray, q_mask::AbstractArray)
    # q_data: (Cq, Q, B), m_data: (Cm, K, B), q_mask: (Q, B) or (Q, 1, B)
    q_bt = permutedims(q_data, (3, 2, 1)) # (B, Q, Cq)
    m_bt = permutedims(m_data, (3, 2, 1)) # (B, K, Cm)

    B = size(q_bt, 1)
    Q = size(q_bt, 2)
    K = size(m_bt, 2)
    Cq = size(q_bt, 3)
    Cm = size(m_bt, 3)
    H = m.num_head
    Ck = m.key_head_dim
    Cv = m.value_head_dim

    qmask = ndims(q_mask) == 2 ? reshape(q_mask, Q, 1, B) : q_mask
    qmask_bt = permutedims(qmask, (3, 1, 2)) # (B, Q, 1)

    denom = max.(sum(qmask_bt; dims=2), 1f-6)
    q_avg = sum(q_bt .* qmask_bt; dims=2) ./ denom # (B, 1, Cq)
    q_avg = dropdims(q_avg; dims=2) # (B, Cq)

    q = reshape(q_avg, B, Cq) * reshape(m.query_w, Cq, H * Ck)
    q = reshape(q, B, H, Ck) .* Float32(Ck)^-0.5f0

    k = reshape(m_bt, B * K, Cm) * m.key_w
    k = reshape(k, B, K, Ck)
    v = reshape(m_bt, B * K, Cm) * m.value_w
    v = reshape(v, B, K, Cv)

    q3 = permutedims(q, (2, 3, 1)) # (H, Ck, B)
    k3 = permutedims(k, (3, 2, 1)) # (Ck, K, B)
    logits3 = NNlib.batched_mul(q3, k3) # (H, K, B)
    logits = permutedims(logits3, (3, 1, 2)) # (B, H, K)

    @assert K == Q "AF2GlobalAttention expects K == Q for masked logits"
    key_mask = qmask_bt[:, :, 1]
    logits = ifelse.(reshape(key_mask .> 0, B, 1, K), logits, SOFTMAX_MASK)
    weights = NNlib.softmax(logits; dims=3) # (B, H, K)

    w3 = permutedims(weights, (2, 3, 1)) # (H, K, B)
    v3 = permutedims(v, (2, 3, 1)) # (K, Cv, B)
    weighted_avg3 = NNlib.batched_mul(w3, v3) # (H, Cv, B)
    weighted_avg = permutedims(weighted_avg3, (3, 1, 2)) # (B, H, Cv)

    if m.gating
        gate = reshape(q_bt, B * Q, Cq) * reshape(m.gating_w, Cq, H * Cv)
        gate = reshape(gate, B, Q, H, Cv)
        gate = NNlib.sigmoid.(gate .+ reshape(m.gating_b, 1, 1, H, Cv))

        out_pre = reshape(weighted_avg, B, 1, H, Cv) .* gate # (B, Q, H, Cv)
        out2 = reshape(out_pre, B * Q, H * Cv) * reshape(m.output_w, H * Cv, size(m.output_w, 3))
        out2 .+= reshape(m.output_b, 1, :)
        out_bt = reshape(out2, B, Q, size(m.output_w, 3))
    else
        out2 = reshape(weighted_avg, B, H * Cv) * reshape(m.output_w, H * Cv, size(m.output_w, 3))
        out2 .+= reshape(m.output_b, 1, :)
        out_bt = reshape(out2, B, 1, size(m.output_w, 3))
    end

    return permutedims(out_bt, (3, 2, 1)) # (Cout, Q, B)
end

function _copy_global_attention_af2!(m::AF2GlobalAttention, arrs::AbstractDict, prefix::String)
    m.query_w .= arrs[string(prefix, "_query_w")]
    m.key_w .= arrs[string(prefix, "_key_w")]
    m.value_w .= arrs[string(prefix, "_value_w")]
    m.output_w .= arrs[string(prefix, "_output_w")]
    m.output_b .= arrs[string(prefix, "_output_b")]

    if m.gating
        m.gating_w .= arrs[string(prefix, "_gating_w")]
        m.gating_b .= arrs[string(prefix, "_gating_b")]
    end

    return m
end
