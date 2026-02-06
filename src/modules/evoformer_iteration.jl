@concrete struct EvoformerIteration <: Onion.Layer
    outer_product_mean
    msa_row_attention_with_pair_bias
    msa_column_attention
    msa_transition
    triangle_multiplication_outgoing
    triangle_multiplication_incoming
    triangle_attention_starting_node
    triangle_attention_ending_node
    pair_transition
    outer_first::Bool
end

@layer EvoformerIteration

function EvoformerIteration(
    c_m::Int,
    c_z::Int;
    num_head_msa::Int,
    msa_head_dim::Int,
    num_head_pair::Int,
    pair_head_dim::Int,
    c_outer::Int,
    c_tri_mul::Int,
    msa_transition_factor::Real=2.0,
    pair_transition_factor::Real=2.0,
    outer_first::Bool=true,
    is_extra_msa::Bool=false,
)
    outer_product_mean = OuterProductMean(c_m, c_outer, c_z)
    msa_row_attention_with_pair_bias = MSARowAttentionWithPairBias(c_m, c_z, num_head_msa, msa_head_dim)
    msa_column_attention = is_extra_msa ?
        MSAColumnGlobalAttention(c_m, num_head_msa, msa_head_dim) :
        MSAColumnAttention(c_m, num_head_msa, msa_head_dim)
    msa_transition = Transition(c_m, msa_transition_factor)

    triangle_multiplication_outgoing = TriangleMultiplication(c_z, c_tri_mul; outgoing=true)
    triangle_multiplication_incoming = TriangleMultiplication(c_z, c_tri_mul; outgoing=false)
    triangle_attention_starting_node = TriangleAttention(c_z, num_head_pair, pair_head_dim; orientation=:per_row)
    triangle_attention_ending_node = TriangleAttention(c_z, num_head_pair, pair_head_dim; orientation=:per_column)
    pair_transition = Transition(c_z, pair_transition_factor)

    return EvoformerIteration(
        outer_product_mean,
        msa_row_attention_with_pair_bias,
        msa_column_attention,
        msa_transition,
        triangle_multiplication_outgoing,
        triangle_multiplication_incoming,
        triangle_attention_starting_node,
        triangle_attention_ending_node,
        pair_transition,
        outer_first,
    )
end

function (m::EvoformerIteration)(msa_act::AbstractArray, pair_act::AbstractArray, msa_mask::AbstractArray, pair_mask::AbstractArray)
    # msa_act: (C_m, N_seq, N_res, B)
    # pair_act: (C_z, N_res, N_res, B)
    if m.outer_first
        pair_act = pair_act .+ m.outer_product_mean(msa_act, msa_mask)
    end

    msa_act = msa_act .+ m.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act)
    msa_act = msa_act .+ m.msa_column_attention(msa_act, msa_mask)
    msa_act = msa_act .+ m.msa_transition(msa_act, msa_mask)

    if !m.outer_first
        pair_act = pair_act .+ m.outer_product_mean(msa_act, msa_mask)
    end

    pair_act = pair_act .+ m.triangle_multiplication_outgoing(pair_act, pair_mask)
    pair_act = pair_act .+ m.triangle_multiplication_incoming(pair_act, pair_mask)
    pair_act = pair_act .+ m.triangle_attention_starting_node(pair_act, pair_mask)
    pair_act = pair_act .+ m.triangle_attention_ending_node(pair_act, pair_mask)
    pair_act = pair_act .+ m.pair_transition(pair_act, pair_mask)

    return msa_act, pair_act
end
