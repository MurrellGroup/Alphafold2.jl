@concrete struct SingleTemplateEmbedding <: Onion.Layer
    embedding2d
    template_pair_stack
    output_layer_norm
    use_template_unit_vector::Bool
    dgram_num_bins::Int
    dgram_min_bin::Float32
    dgram_max_bin::Float32
end

@layer SingleTemplateEmbedding

function SingleTemplateEmbedding(
    c_t::Int,
    num_block::Int;
    num_head_pair::Int,
    pair_head_dim::Int,
    c_tri_mul::Int,
    pair_transition_factor::Real=2.0,
    use_template_unit_vector::Bool=false,
    dgram_num_bins::Int=39,
    dgram_min_bin::Real=3.25f0,
    dgram_max_bin::Real=50.75f0,
)
    return SingleTemplateEmbedding(
        LinearFirst(dgram_num_bins + 49, c_t),
        TemplatePairStack(
            c_t,
            num_block;
            num_head_pair=num_head_pair,
            pair_head_dim=pair_head_dim,
            c_tri_mul=c_tri_mul,
            pair_transition_factor=pair_transition_factor,
        ),
        LayerNormFirst(c_t),
        use_template_unit_vector,
        dgram_num_bins,
        Float32(dgram_min_bin),
        Float32(dgram_max_bin),
    )
end

@concrete struct TemplateEmbedding <: Onion.Layer
    single_template_embedding
    attention
end

@layer TemplateEmbedding

function TemplateEmbedding(
    c_z::Int,
    c_t::Int,
    num_block::Int;
    num_head_pair::Int,
    pair_head_dim::Int,
    c_tri_mul::Int,
    pair_transition_factor::Real=2.0,
    num_head_tpa::Int=4,
    key_dim_tpa::Int=64,
    value_dim_tpa::Int=64,
    gating_tpa::Bool=false,
    use_template_unit_vector::Bool=false,
    dgram_num_bins::Int=39,
    dgram_min_bin::Real=3.25f0,
    dgram_max_bin::Real=50.75f0,
)
    st = SingleTemplateEmbedding(
        c_t,
        num_block;
        num_head_pair=num_head_pair,
        pair_head_dim=pair_head_dim,
        c_tri_mul=c_tri_mul,
        pair_transition_factor=pair_transition_factor,
        use_template_unit_vector=use_template_unit_vector,
        dgram_num_bins=dgram_num_bins,
        dgram_min_bin=dgram_min_bin,
        dgram_max_bin=dgram_max_bin,
    )
    att = AF2Attention(c_z, c_t, key_dim_tpa, value_dim_tpa, num_head_tpa, c_z; gating=gating_tpa)
    return TemplateEmbedding(st, att)
end

function _dgram_from_positions_template(
    positions::AbstractArray;
    num_bins::Int,
    min_bin::Float32,
    max_bin::Float32,
)
    # positions: (3, L)
    L = size(positions, 2)
    out = zeros(Float32, num_bins, L, L)
    lower_breaks = collect(range(min_bin, max_bin; length=num_bins))
    lower2 = lower_breaks .^ 2
    upper2 = vcat(lower2[2:end], Float32[1f8])

    p = permutedims(positions, (2, 1)) # (L,3)
    diff = reshape(p, L, 1, 3) .- reshape(p, 1, L, 3)
    d2 = dropdims(sum(diff .^ 2; dims=3); dims=3)
    for k in 1:num_bins
        out[k, :, :] .= Float32.(d2 .> lower2[k]) .* Float32.(d2 .< upper2[k])
    end
    return out
end

function _make_transform_from_reference(n_xyz::AbstractArray, ca_xyz::AbstractArray, c_xyz::AbstractArray)
    # all args: (3, L). Returns rotation (3,3,L) and translation (3,L).
    L = size(n_xyz, 2)
    rot = zeros(Float32, 3, 3, L)
    trans = zeros(Float32, 3, L)

    for i in 1:L
        nx, ny, nz = n_xyz[1, i], n_xyz[2, i], n_xyz[3, i]
        cax, cay, caz = ca_xyz[1, i], ca_xyz[2, i], ca_xyz[3, i]
        cx, cy, cz = c_xyz[1, i], c_xyz[2, i], c_xyz[3, i]

        tx, ty, tz = -cax, -cay, -caz
        nx += tx; ny += ty; nz += tz
        cx += tx; cy += ty; cz += tz

        denom_xy = sqrt(1f-20 + cx * cx + cy * cy)
        sin_c1 = -cy / denom_xy
        cos_c1 = cx / denom_xy

        denom_xyz = sqrt(1f-20 + cx * cx + cy * cy + cz * cz)
        sin_c2 = cz / denom_xyz
        cos_c2 = sqrt(cx * cx + cy * cy) / denom_xyz

        c1_11 = cos_c1; c1_12 = -sin_c1; c1_13 = 0f0
        c1_21 = sin_c1; c1_22 = cos_c1; c1_23 = 0f0
        c1_31 = 0f0; c1_32 = 0f0; c1_33 = 1f0

        c2_11 = cos_c2; c2_12 = 0f0; c2_13 = sin_c2
        c2_21 = 0f0; c2_22 = 1f0; c2_23 = 0f0
        c2_31 = -sin_c2; c2_32 = 0f0; c2_33 = cos_c2

        c_11 = c2_11 * c1_11 + c2_12 * c1_21 + c2_13 * c1_31
        c_12 = c2_11 * c1_12 + c2_12 * c1_22 + c2_13 * c1_32
        c_13 = c2_11 * c1_13 + c2_12 * c1_23 + c2_13 * c1_33
        c_21 = c2_21 * c1_11 + c2_22 * c1_21 + c2_23 * c1_31
        c_22 = c2_21 * c1_12 + c2_22 * c1_22 + c2_23 * c1_32
        c_23 = c2_21 * c1_13 + c2_22 * c1_23 + c2_23 * c1_33
        c_31 = c2_31 * c1_11 + c2_32 * c1_21 + c2_33 * c1_31
        c_32 = c2_31 * c1_12 + c2_32 * c1_22 + c2_33 * c1_32
        c_33 = c2_31 * c1_13 + c2_32 * c1_23 + c2_33 * c1_33

        n2x = c_11 * nx + c_12 * ny + c_13 * nz
        n2y = c_21 * nx + c_22 * ny + c_23 * nz
        n2z = c_31 * nx + c_32 * ny + c_33 * nz

        denom_n = sqrt(1f-20 + n2y * n2y + n2z * n2z)
        sin_n = -n2z / denom_n
        cos_n = n2y / denom_n

        nrot_11 = 1f0; nrot_12 = 0f0; nrot_13 = 0f0
        nrot_21 = 0f0; nrot_22 = cos_n; nrot_23 = -sin_n
        nrot_31 = 0f0; nrot_32 = sin_n; nrot_33 = cos_n

        r_11 = nrot_11 * c_11 + nrot_12 * c_21 + nrot_13 * c_31
        r_12 = nrot_11 * c_12 + nrot_12 * c_22 + nrot_13 * c_32
        r_13 = nrot_11 * c_13 + nrot_12 * c_23 + nrot_13 * c_33
        r_21 = nrot_21 * c_11 + nrot_22 * c_21 + nrot_23 * c_31
        r_22 = nrot_21 * c_12 + nrot_22 * c_22 + nrot_23 * c_32
        r_23 = nrot_21 * c_13 + nrot_22 * c_23 + nrot_23 * c_33
        r_31 = nrot_31 * c_11 + nrot_32 * c_21 + nrot_33 * c_31
        r_32 = nrot_31 * c_12 + nrot_32 * c_22 + nrot_33 * c_32
        r_33 = nrot_31 * c_13 + nrot_32 * c_23 + nrot_33 * c_33

        # make_transform_from_reference returns transpose(rotation), -translation.
        rot[:, :, i] .= Float32[
            r_11 r_21 r_31
            r_12 r_22 r_32
            r_13 r_23 r_33
        ]
        trans[:, i] .= Float32[-tx, -ty, -tz]
    end

    return rot, trans
end

function _single_template_embedding(
    m::SingleTemplateEmbedding,
    template_aatype::AbstractVector{Int},
    template_all_atom_positions::AbstractArray,
    template_all_atom_masks::AbstractArray,
    pair_mask::AbstractArray,
)
    # template_aatype: (L), template_all_atom_positions: (L,37,3), template_all_atom_masks: (L,37)
    L = length(template_aatype)
    dtype = Float32

    n_idx = atom_order["N"] + 1
    ca_idx = atom_order["CA"] + 1
    c_idx = atom_order["C"] + 1
    cb_idx = atom_order["CB"] + 1
    gly_idx = restype_order["G"]

    pseudo_beta = zeros(dtype, 3, L)
    pseudo_beta_mask = zeros(dtype, L)
    for r in 1:L
        aa = clamp(template_aatype[r], 0, 20)
        is_gly = aa == gly_idx
        atom_idx = is_gly ? ca_idx : cb_idx
        pseudo_beta[:, r] .= template_all_atom_positions[r, atom_idx, :]
        pseudo_beta_mask[r] = template_all_atom_masks[r, atom_idx]
    end

    template_dgram = _dgram_from_positions_template(
        pseudo_beta;
        num_bins=m.dgram_num_bins,
        min_bin=m.dgram_min_bin,
        max_bin=m.dgram_max_bin,
    )

    pseudo_mask_2d = pseudo_beta_mask * transpose(pseudo_beta_mask)

    aatype_1hot = zeros(dtype, 22, L)
    for r in 1:L
        aa = clamp(template_aatype[r], 0, 21)
        aatype_1hot[aa + 1, r] = 1f0
    end
    aatype_col = repeat(reshape(aatype_1hot, 22, L, 1), 1, 1, L)
    aatype_row = repeat(reshape(aatype_1hot, 22, 1, L), 1, L, 1)

    n_xyz = permutedims(view(template_all_atom_positions, :, n_idx, :), (2, 1))
    ca_xyz = permutedims(view(template_all_atom_positions, :, ca_idx, :), (2, 1))
    c_xyz = permutedims(view(template_all_atom_positions, :, c_idx, :), (2, 1))
    rot, trans = _make_transform_from_reference(n_xyz, ca_xyz, c_xyz)

    affine_vec = zeros(dtype, 3, L, L)
    for i in 1:L
        diff = trans .- reshape(trans[:, i], 3, 1)
        affine_vec[:, i, :] .= transpose(rot[:, :, i]) * diff
    end

    inv_distance_scalar = 1f0 ./ sqrt.(1f-6 .+ dropdims(sum(affine_vec .^ 2; dims=1); dims=1))

    backbone_mask = view(template_all_atom_masks, :, n_idx) .* view(template_all_atom_masks, :, ca_idx) .* view(template_all_atom_masks, :, c_idx)
    backbone_mask_2d = backbone_mask * transpose(backbone_mask)
    inv_distance_scalar .*= backbone_mask_2d

    if !m.use_template_unit_vector
        unit_x = zeros(dtype, L, L)
        unit_y = zeros(dtype, L, L)
        unit_z = zeros(dtype, L, L)
    else
        unit_x = view(affine_vec, 1, :, :) .* inv_distance_scalar
        unit_y = view(affine_vec, 2, :, :) .* inv_distance_scalar
        unit_z = view(affine_vec, 3, :, :) .* inv_distance_scalar
    end

    to_concat = (
        template_dgram,
        reshape(pseudo_mask_2d, 1, L, L),
        aatype_row,
        aatype_col,
        reshape(unit_x, 1, L, L),
        reshape(unit_y, 1, L, L),
        reshape(unit_z, 1, L, L),
        reshape(backbone_mask_2d, 1, L, L),
    )

    act = cat(to_concat...; dims=1)
    act .*= reshape(backbone_mask_2d, 1, L, L)
    act = reshape(act, size(act, 1), L, L, 1)

    act = m.embedding2d(act)
    act = m.template_pair_stack(act, pair_mask)
    act = m.output_layer_norm(act)
    return act
end

function (m::SingleTemplateEmbedding)(
    template_aatype::AbstractVector{Int},
    template_all_atom_positions::AbstractArray,
    template_all_atom_masks::AbstractArray,
    pair_mask::AbstractArray,
)
    return _single_template_embedding(
        m,
        template_aatype,
        template_all_atom_positions,
        template_all_atom_masks,
        pair_mask,
    )
end

function (m::TemplateEmbedding)(
    query_embedding::AbstractArray,
    template_aatype::AbstractMatrix{Int},
    template_all_atom_positions::AbstractArray,
    template_all_atom_masks::AbstractArray,
    pair_mask::AbstractArray;
    template_mask::AbstractVector=ones(Float32, size(template_aatype, 1)),
)
    # query_embedding: (C_z, L, L, 1)
    # template_aatype: (T, L), template_all_atom_positions: (T, L, 37, 3), template_all_atom_masks: (T, L, 37)
    # pair_mask: (L, L, 1)
    T, L = size(template_aatype)
    c_t = size(m.single_template_embedding.embedding2d.weight, 1)
    c_z = size(query_embedding, 1)

    template_pair = zeros(Float32, c_t, L, L, T)
    for t in 1:T
        t_act = m.single_template_embedding(
            vec(view(template_aatype, t, :)),
            view(template_all_atom_positions, t, :, :, :),
            view(template_all_atom_masks, t, :, :),
            pair_mask,
        )
        template_pair[:, :, :, t] .= dropdims(t_act; dims=4)
    end

    Bflat = L * L
    flat_query = reshape(dropdims(query_embedding; dims=4), c_z, 1, Bflat)
    flat_templates = reshape(permutedims(template_pair, (1, 4, 2, 3)), c_t, T, Bflat)

    tmask = Float32.(template_mask)
    mask = reshape(tmask, 1, 1, 1, T)
    embedding_flat = m.attention(flat_query, flat_templates, mask)
    embedding = reshape(dropdims(embedding_flat; dims=2), c_z, L, L, 1)
    embedding .*= (sum(tmask) > 0f0 ? 1f0 : 0f0)
    return embedding
end

function load_template_embedding_npz!(
    m::TemplateEmbedding,
    params_source;
    prefix::AbstractString="alphafold/alphafold_iteration/evoformer/template_embedding",
)
    arrs = af2_params_read(params_source)

    m.single_template_embedding.embedding2d.weight .= permutedims(
        _get_arr(arrs, string(prefix, "/single_template_embedding/embedding2d//weights")),
        (2, 1),
    )
    m.single_template_embedding.embedding2d.bias .= _get_arr(arrs, string(prefix, "/single_template_embedding/embedding2d//bias"))

    load_template_pair_stack_npz!(
        m.single_template_embedding.template_pair_stack,
        params_source;
        prefix=string(prefix, "/single_template_embedding/template_pair_stack/__layer_stack_no_state"),
    )
    m.single_template_embedding.output_layer_norm.w .= _get_arr(arrs, string(prefix, "/single_template_embedding/output_layer_norm//scale"))
    m.single_template_embedding.output_layer_norm.b .= _get_arr(arrs, string(prefix, "/single_template_embedding/output_layer_norm//offset"))

    m.attention.query_w .= _get_arr(arrs, string(prefix, "/attention//query_w"))
    m.attention.key_w .= _get_arr(arrs, string(prefix, "/attention//key_w"))
    m.attention.value_w .= _get_arr(arrs, string(prefix, "/attention//value_w"))
    m.attention.output_w .= _get_arr(arrs, string(prefix, "/attention//output_w"))
    m.attention.output_b .= _get_arr(arrs, string(prefix, "/attention//output_b"))
    if m.attention.gating
        m.attention.gating_w .= _get_arr(arrs, string(prefix, "/attention//gating_w"))
        m.attention.gating_b .= _get_arr(arrs, string(prefix, "/attention//gating_b"))
    end

    return m
end

@concrete struct SingleTemplateEmbeddingMultimer <: Onion.Layer
    template_pair_embedding_0
    template_pair_embedding_1
    template_pair_embedding_2
    template_pair_embedding_3
    template_pair_embedding_4
    template_pair_embedding_5
    template_pair_embedding_6
    template_pair_embedding_7
    template_pair_embedding_8
    query_embedding_norm
    template_pair_stack
    output_layer_norm
    use_template_unit_vector::Bool
    dgram_num_bins::Int
    dgram_min_bin::Float32
    dgram_max_bin::Float32
end

@layer SingleTemplateEmbeddingMultimer

function SingleTemplateEmbeddingMultimer(
    c_z::Int,
    c_t::Int,
    num_block::Int;
    num_head_pair::Int,
    pair_head_dim::Int,
    c_tri_mul::Int,
    pair_transition_factor::Real=2.0,
    use_template_unit_vector::Bool=true,
    dgram_num_bins::Int=39,
    dgram_min_bin::Real=3.25f0,
    dgram_max_bin::Real=50.75f0,
)
    return SingleTemplateEmbeddingMultimer(
        LinearFirst(dgram_num_bins, c_t),
        LinearFirst(1, c_t),
        LinearFirst(22, c_t),
        LinearFirst(22, c_t),
        LinearFirst(1, c_t),
        LinearFirst(1, c_t),
        LinearFirst(1, c_t),
        LinearFirst(1, c_t),
        LinearFirst(c_z, c_t),
        LayerNormFirst(c_z),
        TemplatePairStack(
            c_t,
            num_block;
            num_head_pair=num_head_pair,
            pair_head_dim=pair_head_dim,
            c_tri_mul=c_tri_mul,
            pair_transition_factor=pair_transition_factor,
        ),
        LayerNormFirst(c_t),
        use_template_unit_vector,
        dgram_num_bins,
        Float32(dgram_min_bin),
        Float32(dgram_max_bin),
    )
end

@concrete struct TemplateEmbeddingMultimer <: Onion.Layer
    single_template_embedding
    output_linear
end

@layer TemplateEmbeddingMultimer

function TemplateEmbeddingMultimer(
    c_z::Int,
    c_t::Int,
    num_block::Int;
    num_head_pair::Int,
    pair_head_dim::Int,
    c_tri_mul::Int,
    pair_transition_factor::Real=2.0,
    use_template_unit_vector::Bool=true,
    dgram_num_bins::Int=39,
    dgram_min_bin::Real=3.25f0,
    dgram_max_bin::Real=50.75f0,
)
    st = SingleTemplateEmbeddingMultimer(
        c_z,
        c_t,
        num_block;
        num_head_pair=num_head_pair,
        pair_head_dim=pair_head_dim,
        c_tri_mul=c_tri_mul,
        pair_transition_factor=pair_transition_factor,
        use_template_unit_vector=use_template_unit_vector,
        dgram_num_bins=dgram_num_bins,
        dgram_min_bin=dgram_min_bin,
        dgram_max_bin=dgram_max_bin,
    )
    return TemplateEmbeddingMultimer(st, LinearFirst(c_t, c_z))
end

@inline function _pair_mask_2d(mask::AbstractArray, L::Int)
    if ndims(mask) == 2
        return mask
    elseif ndims(mask) == 3
        return dropdims(mask; dims=3)
    end
    error("Expected pair mask with rank 2 or 3 for length $(L), got rank $(ndims(mask))")
end

function _single_template_embedding_multimer(
    m::SingleTemplateEmbeddingMultimer,
    query_embedding::AbstractArray,
    template_aatype::AbstractVector{Int},
    template_all_atom_positions::AbstractArray,
    template_all_atom_masks::AbstractArray,
    pair_mask::AbstractArray,
    multichain_mask::AbstractArray,
)
    L = length(template_aatype)
    dtype = Float32
    multichain_mask_2d = Float32.(_pair_mask_2d(multichain_mask, L))

    n_idx = atom_order["N"] + 1
    ca_idx = atom_order["CA"] + 1
    c_idx = atom_order["C"] + 1
    cb_idx = atom_order["CB"] + 1
    gly_idx = restype_order["G"]

    pseudo_beta = zeros(dtype, 3, L)
    pseudo_beta_mask = zeros(dtype, L)
    for r in 1:L
        aa = clamp(template_aatype[r], 0, 20)
        atom_idx = aa == gly_idx ? ca_idx : cb_idx
        pseudo_beta[:, r] .= template_all_atom_positions[r, atom_idx, :]
        pseudo_beta_mask[r] = template_all_atom_masks[r, atom_idx]
    end

    template_dgram = _dgram_from_positions_template(
        pseudo_beta;
        num_bins=m.dgram_num_bins,
        min_bin=m.dgram_min_bin,
        max_bin=m.dgram_max_bin,
    )

    pseudo_mask_2d = (pseudo_beta_mask * transpose(pseudo_beta_mask)) .* multichain_mask_2d
    template_dgram .*= reshape(pseudo_mask_2d, 1, L, L)

    aatype_1hot = zeros(dtype, 22, L)
    for r in 1:L
        aa = clamp(template_aatype[r], 0, 21)
        aatype_1hot[aa + 1, r] = 1f0
    end
    aatype_col = repeat(reshape(aatype_1hot, 22, L, 1), 1, 1, L)
    aatype_row = repeat(reshape(aatype_1hot, 22, 1, L), 1, L, 1)

    n_xyz = permutedims(view(template_all_atom_positions, :, n_idx, :), (2, 1))
    ca_xyz = permutedims(view(template_all_atom_positions, :, ca_idx, :), (2, 1))
    c_xyz = permutedims(view(template_all_atom_positions, :, c_idx, :), (2, 1))
    rot, trans = _make_transform_from_reference(n_xyz, ca_xyz, c_xyz)

    affine_vec = zeros(dtype, 3, L, L)
    for i in 1:L
        diff = trans .- reshape(trans[:, i], 3, 1)
        affine_vec[:, i, :] .= transpose(rot[:, :, i]) * diff
    end

    inv_distance_scalar = 1f0 ./ sqrt.(1f-6 .+ dropdims(sum(affine_vec .^ 2; dims=1); dims=1))
    backbone_mask = view(template_all_atom_masks, :, n_idx) .* view(template_all_atom_masks, :, ca_idx) .* view(template_all_atom_masks, :, c_idx)
    backbone_mask_2d = (backbone_mask * transpose(backbone_mask)) .* multichain_mask_2d
    inv_distance_scalar .*= backbone_mask_2d

    if !m.use_template_unit_vector
        unit_x = zeros(dtype, L, L)
        unit_y = zeros(dtype, L, L)
        unit_z = zeros(dtype, L, L)
    else
        unit_x = view(affine_vec, 1, :, :) .* inv_distance_scalar
        unit_y = view(affine_vec, 2, :, :) .* inv_distance_scalar
        unit_z = view(affine_vec, 3, :, :) .* inv_distance_scalar
    end

    query_norm = m.query_embedding_norm(query_embedding)

    act = m.template_pair_embedding_0(reshape(template_dgram, m.dgram_num_bins, L, L, 1))
    act .+= m.template_pair_embedding_1(reshape(pseudo_mask_2d, 1, L, L, 1))
    act .+= m.template_pair_embedding_2(reshape(aatype_row, 22, L, L, 1))
    act .+= m.template_pair_embedding_3(reshape(aatype_col, 22, L, L, 1))
    act .+= m.template_pair_embedding_4(reshape(unit_x, 1, L, L, 1))
    act .+= m.template_pair_embedding_5(reshape(unit_y, 1, L, L, 1))
    act .+= m.template_pair_embedding_6(reshape(unit_z, 1, L, L, 1))
    act .+= m.template_pair_embedding_7(reshape(backbone_mask_2d, 1, L, L, 1))
    act .+= m.template_pair_embedding_8(query_norm)

    for blk in m.template_pair_stack.blocks
        act = act .+ blk.triangle_multiplication_outgoing(act, pair_mask)
        act = act .+ blk.triangle_multiplication_incoming(act, pair_mask)
        act = act .+ blk.triangle_attention_starting_node(act, pair_mask)
        act = act .+ blk.triangle_attention_ending_node(act, pair_mask)
        act = act .+ blk.pair_transition(act, pair_mask)
    end

    return m.output_layer_norm(act)
end

function (m::SingleTemplateEmbeddingMultimer)(
    query_embedding::AbstractArray,
    template_aatype::AbstractVector{Int},
    template_all_atom_positions::AbstractArray,
    template_all_atom_masks::AbstractArray,
    pair_mask::AbstractArray,
    multichain_mask::AbstractArray,
)
    return _single_template_embedding_multimer(
        m,
        query_embedding,
        template_aatype,
        template_all_atom_positions,
        template_all_atom_masks,
        pair_mask,
        multichain_mask,
    )
end

function (m::TemplateEmbeddingMultimer)(
    query_embedding::AbstractArray,
    template_aatype::AbstractMatrix{Int},
    template_all_atom_positions::AbstractArray,
    template_all_atom_masks::AbstractArray,
    pair_mask::AbstractArray;
    template_mask=nothing,
    multichain_mask::AbstractArray=ones(Float32, size(template_aatype, 2), size(template_aatype, 2), 1),
)
    T, L = size(template_aatype)
    if T == 0
        return zeros(Float32, size(query_embedding))
    end

    weights = template_mask === nothing ? ones(Float32, T) : Float32.(template_mask)
    sumw = sum(weights)
    sumw > 0f0 || return zeros(Float32, size(query_embedding))

    c_t = size(m.output_linear.weight, 2)
    accum = zeros(Float32, c_t, L, L, 1)
    for t in 1:T
        w = weights[t]
        w == 0f0 && continue
        t_act = m.single_template_embedding(
            query_embedding,
            vec(view(template_aatype, t, :)),
            view(template_all_atom_positions, t, :, :, :),
            view(template_all_atom_masks, t, :, :),
            pair_mask,
            multichain_mask,
        )
        accum .+= w .* t_act
    end

    embedding = accum ./ sumw
    embedding = max.(embedding, 0f0)
    return m.output_linear(embedding)
end


function _load_linear_raw_block_split!(
    lin::LinearFirst,
    arrs::AbstractDict,
    base::AbstractString,
    block_idx::Int,
    num_blocks::Int;
    split::Symbol=:full,
)
    wfull = _slice_leading_block(_get_arr(arrs, string(base, "//weights")), block_idx, num_blocks)
    w = if split === :full
        wfull
    elseif split === :first
        @view wfull[:, 1:Int(div(size(wfull, 2), 2))]
    elseif split === :second
        @view wfull[:, Int(div(size(wfull, 2), 2)) + 1:size(wfull, 2)]
    else
        error("Unsupported split mode: $(split)")
    end
    lin.weight .= permutedims(w, (2, 1))
    if lin.use_bias
        bfull = _slice_leading_block(_get_arr(arrs, string(base, "//bias")), block_idx, num_blocks)
        b = if split === :full
            bfull
        elseif split === :first
            @view bfull[1:Int(div(length(bfull), 2))]
        elseif split === :second
            @view bfull[Int(div(length(bfull), 2)) + 1:length(bfull)]
        else
            error("Unsupported split mode: $(split)")
        end
        lin.bias .= b
    end
    return lin
end

function _load_template_pair_stack_multimer_arrs!(
    m::TemplatePairStack,
    arrs::AbstractDict,
    prefix::AbstractString,
)
    num_blocks = length(m.blocks)

    for (i, blk) in enumerate(m.blocks)
        _load_ln_raw_block!(blk.triangle_attention_starting_node.query_norm, arrs, string(prefix, "/triangle_attention_starting_node/query_norm"), i, num_blocks)
        blk.triangle_attention_starting_node.feat_2d_weights .= _slice_leading_block(_get_arr(arrs, string(prefix, "/triangle_attention_starting_node//feat_2d_weights")), i, num_blocks)
        _load_attention_raw_block!(blk.triangle_attention_starting_node.attention, arrs, string(prefix, "/triangle_attention_starting_node/attention"), i, num_blocks)

        _load_ln_raw_block!(blk.triangle_attention_ending_node.query_norm, arrs, string(prefix, "/triangle_attention_ending_node/query_norm"), i, num_blocks)
        blk.triangle_attention_ending_node.feat_2d_weights .= _slice_leading_block(_get_arr(arrs, string(prefix, "/triangle_attention_ending_node//feat_2d_weights")), i, num_blocks)
        _load_attention_raw_block!(blk.triangle_attention_ending_node.attention, arrs, string(prefix, "/triangle_attention_ending_node/attention"), i, num_blocks)

        tri_out = string(prefix, "/triangle_multiplication_outgoing")
        if _has_arr_key(arrs, string(tri_out, "/left_projection//weights"))
            _load_ln_raw_block!(blk.triangle_multiplication_outgoing.layer_norm_input, arrs, string(tri_out, "/layer_norm_input"), i, num_blocks)
            _load_linear_raw_block!(blk.triangle_multiplication_outgoing.left_projection, arrs, string(tri_out, "/left_projection"), i, num_blocks)
            _load_linear_raw_block!(blk.triangle_multiplication_outgoing.right_projection, arrs, string(tri_out, "/right_projection"), i, num_blocks)
            _load_linear_raw_block!(blk.triangle_multiplication_outgoing.left_gate, arrs, string(tri_out, "/left_gate"), i, num_blocks)
            _load_linear_raw_block!(blk.triangle_multiplication_outgoing.right_gate, arrs, string(tri_out, "/right_gate"), i, num_blocks)
            _load_ln_raw_block!(blk.triangle_multiplication_outgoing.center_layer_norm, arrs, string(tri_out, "/center_layer_norm"), i, num_blocks)
        else
            _load_ln_raw_block!(blk.triangle_multiplication_outgoing.layer_norm_input, arrs, string(tri_out, "/left_norm_input"), i, num_blocks)
            _load_linear_raw_block_split!(blk.triangle_multiplication_outgoing.left_projection, arrs, string(tri_out, "/projection"), i, num_blocks; split=:first)
            _load_linear_raw_block_split!(blk.triangle_multiplication_outgoing.right_projection, arrs, string(tri_out, "/projection"), i, num_blocks; split=:second)
            _load_linear_raw_block_split!(blk.triangle_multiplication_outgoing.left_gate, arrs, string(tri_out, "/gate"), i, num_blocks; split=:first)
            _load_linear_raw_block_split!(blk.triangle_multiplication_outgoing.right_gate, arrs, string(tri_out, "/gate"), i, num_blocks; split=:second)
            _load_ln_raw_block!(blk.triangle_multiplication_outgoing.center_layer_norm, arrs, string(tri_out, "/center_norm"), i, num_blocks)
        end
        _load_linear_raw_block!(blk.triangle_multiplication_outgoing.output_projection, arrs, string(tri_out, "/output_projection"), i, num_blocks)
        _load_linear_raw_block!(blk.triangle_multiplication_outgoing.gating_linear, arrs, string(tri_out, "/gating_linear"), i, num_blocks)

        tri_in = string(prefix, "/triangle_multiplication_incoming")
        if _has_arr_key(arrs, string(tri_in, "/left_projection//weights"))
            _load_ln_raw_block!(blk.triangle_multiplication_incoming.layer_norm_input, arrs, string(tri_in, "/layer_norm_input"), i, num_blocks)
            _load_linear_raw_block!(blk.triangle_multiplication_incoming.left_projection, arrs, string(tri_in, "/left_projection"), i, num_blocks)
            _load_linear_raw_block!(blk.triangle_multiplication_incoming.right_projection, arrs, string(tri_in, "/right_projection"), i, num_blocks)
            _load_linear_raw_block!(blk.triangle_multiplication_incoming.left_gate, arrs, string(tri_in, "/left_gate"), i, num_blocks)
            _load_linear_raw_block!(blk.triangle_multiplication_incoming.right_gate, arrs, string(tri_in, "/right_gate"), i, num_blocks)
            _load_ln_raw_block!(blk.triangle_multiplication_incoming.center_layer_norm, arrs, string(tri_in, "/center_layer_norm"), i, num_blocks)
        else
            _load_ln_raw_block!(blk.triangle_multiplication_incoming.layer_norm_input, arrs, string(tri_in, "/left_norm_input"), i, num_blocks)
            _load_linear_raw_block_split!(blk.triangle_multiplication_incoming.left_projection, arrs, string(tri_in, "/projection"), i, num_blocks; split=:first)
            _load_linear_raw_block_split!(blk.triangle_multiplication_incoming.right_projection, arrs, string(tri_in, "/projection"), i, num_blocks; split=:second)
            _load_linear_raw_block_split!(blk.triangle_multiplication_incoming.left_gate, arrs, string(tri_in, "/gate"), i, num_blocks; split=:first)
            _load_linear_raw_block_split!(blk.triangle_multiplication_incoming.right_gate, arrs, string(tri_in, "/gate"), i, num_blocks; split=:second)
            _load_ln_raw_block!(blk.triangle_multiplication_incoming.center_layer_norm, arrs, string(tri_in, "/center_norm"), i, num_blocks)
        end
        _load_linear_raw_block!(blk.triangle_multiplication_incoming.output_projection, arrs, string(tri_in, "/output_projection"), i, num_blocks)
        _load_linear_raw_block!(blk.triangle_multiplication_incoming.gating_linear, arrs, string(tri_in, "/gating_linear"), i, num_blocks)

        _load_ln_raw_block!(blk.pair_transition.input_layer_norm, arrs, string(prefix, "/pair_transition/input_layer_norm"), i, num_blocks)
        _load_linear_raw_block!(blk.pair_transition.transition1, arrs, string(prefix, "/pair_transition/transition1"), i, num_blocks)
        _load_linear_raw_block!(blk.pair_transition.transition2, arrs, string(prefix, "/pair_transition/transition2"), i, num_blocks)
    end

    return m
end

function _load_multimer_pair_embedding_linear!(lin::LinearFirst, arrs::AbstractDict, base::AbstractString)
    w = _get_arr(arrs, string(base, "//weights"))
    if ndims(w) == 1
        size(lin.weight, 2) == 1 || error("Expected scalar-input linear for $(base), got in-dim $(size(lin.weight, 2))")
        lin.weight[:, 1] .= vec(w)
    elseif ndims(w) == 2
        lin.weight .= permutedims(w, (2, 1))
    else
        error("Unexpected weight rank $(ndims(w)) for $(base)")
    end
    lin.bias .= _get_arr(arrs, string(base, "//bias"))
    return lin
end

function load_template_embedding_npz!(
    m::TemplateEmbeddingMultimer,
    params_source;
    prefix::AbstractString="alphafold/alphafold_iteration/evoformer/template_embedding",
)
    arrs = af2_params_read(params_source)
    st = m.single_template_embedding

    _load_multimer_pair_embedding_linear!(st.template_pair_embedding_0, arrs, string(prefix, "/single_template_embedding/template_pair_embedding_0"))
    _load_multimer_pair_embedding_linear!(st.template_pair_embedding_1, arrs, string(prefix, "/single_template_embedding/template_pair_embedding_1"))
    _load_multimer_pair_embedding_linear!(st.template_pair_embedding_2, arrs, string(prefix, "/single_template_embedding/template_pair_embedding_2"))
    _load_multimer_pair_embedding_linear!(st.template_pair_embedding_3, arrs, string(prefix, "/single_template_embedding/template_pair_embedding_3"))
    _load_multimer_pair_embedding_linear!(st.template_pair_embedding_4, arrs, string(prefix, "/single_template_embedding/template_pair_embedding_4"))
    _load_multimer_pair_embedding_linear!(st.template_pair_embedding_5, arrs, string(prefix, "/single_template_embedding/template_pair_embedding_5"))
    _load_multimer_pair_embedding_linear!(st.template_pair_embedding_6, arrs, string(prefix, "/single_template_embedding/template_pair_embedding_6"))
    _load_multimer_pair_embedding_linear!(st.template_pair_embedding_7, arrs, string(prefix, "/single_template_embedding/template_pair_embedding_7"))
    _load_multimer_pair_embedding_linear!(st.template_pair_embedding_8, arrs, string(prefix, "/single_template_embedding/template_pair_embedding_8"))

    st.query_embedding_norm.w .= _get_arr(arrs, string(prefix, "/single_template_embedding/query_embedding_norm//scale"))
    st.query_embedding_norm.b .= _get_arr(arrs, string(prefix, "/single_template_embedding/query_embedding_norm//offset"))

    _load_template_pair_stack_multimer_arrs!(
        st.template_pair_stack,
        arrs,
        string(prefix, "/single_template_embedding/template_embedding_iteration"),
    )
    st.output_layer_norm.w .= _get_arr(arrs, string(prefix, "/single_template_embedding/output_layer_norm//scale"))
    st.output_layer_norm.b .= _get_arr(arrs, string(prefix, "/single_template_embedding/output_layer_norm//offset"))

    m.output_linear.weight .= permutedims(_get_arr(arrs, string(prefix, "/output_linear//weights")), (2, 1))
    m.output_linear.bias .= _get_arr(arrs, string(prefix, "/output_linear//bias"))
    return m
end
