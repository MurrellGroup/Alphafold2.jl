const _chi_atom_indices = permutedims(Int32[
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;;;
    0 1 3 5;
    1 3 5 11;
    3 5 11 23;
    5 11 23 32;;;
    0 1 3 5;
    1 3 5 16;
    0 0 0 0;
    0 0 0 0;;;
    0 1 3 5;
    1 3 5 16;
    0 0 0 0;
    0 0 0 0;;;
    0 1 3 10;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;;;
    0 1 3 5;
    1 3 5 11;
    3 5 11 26;
    0 0 0 0;;;
    0 1 3 5;
    1 3 5 11;
    3 5 11 26;
    0 0 0 0;;;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;;;
    0 1 3 5;
    1 3 5 14;
    0 0 0 0;
    0 0 0 0;;;
    0 1 3 6;
    1 3 6 12;
    0 0 0 0;
    0 0 0 0;;;
    0 1 3 5;
    1 3 5 12;
    0 0 0 0;
    0 0 0 0;;;
    0 1 3 5;
    1 3 5 11;
    3 5 11 19;
    5 11 19 35;;;
    0 1 3 5;
    1 3 5 18;
    3 5 18 19;
    0 0 0 0;;;
    0 1 3 5;
    1 3 5 12;
    0 0 0 0;
    0 0 0 0;;;
    0 1 3 5;
    1 3 5 11;
    0 0 0 0;
    0 0 0 0;;;
    0 1 3 8;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;;;
    0 1 3 9;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;;;
    0 1 3 5;
    1 3 5 12;
    0 0 0 0;
    0 0 0 0;;;
    0 1 3 5;
    1 3 5 12;
    0 0 0 0;
    0 0 0 0;;;
    0 1 3 6;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;;;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0
], (3, 1, 2))

const _chi_angles_mask = Float32[
    0 0 0 0;
    1 1 1 1;
    1 1 0 0;
    1 1 0 0;
    1 0 0 0;
    1 1 1 0;
    1 1 1 0;
    0 0 0 0;
    1 1 0 0;
    1 1 0 0;
    1 1 0 0;
    1 1 1 1;
    1 1 1 0;
    1 1 0 0;
    1 1 0 0;
    1 0 0 0;
    1 0 0 0;
    1 1 0 0;
    1 1 0 0;
    1 0 0 0;
    0 0 0 0
]

const _chi_pi_periodic = Float32[
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 1 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 1 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 1 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 1 0 0;
    0 0 0 0;
    0 0 0 0
]

@inline function _torsion_sin_cos(p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z, p4x, p4y, p4z)
    e0x = p3x - p2x
    e0y = p3y - p2y
    e0z = p3z - p2z
    n0 = sqrt(e0x * e0x + e0y * e0y + e0z * e0z + 1f-8)
    e0x /= n0
    e0y /= n0
    e0z /= n0

    v1x = p1x - p3x
    v1y = p1y - p3y
    v1z = p1z - p3z
    c = v1x * e0x + v1y * e0y + v1z * e0z
    e1x = v1x - c * e0x
    e1y = v1y - c * e0y
    e1z = v1z - c * e0z
    n1 = sqrt(e1x * e1x + e1y * e1y + e1z * e1z + 1f-8)
    e1x /= n1
    e1y /= n1
    e1z /= n1

    e2x = e0y * e1z - e0z * e1y
    e2y = e0z * e1x - e0x * e1z
    e2z = e0x * e1y - e0y * e1x

    vx = p4x - p3x
    vy = p4y - p3y
    vz = p4z - p3z

    y = vx * e1x + vy * e1y + vz * e1z
    z = vx * e2x + vy * e2y + vz * e2z

    n = sqrt(y * y + z * z + 1f-8)
    return z / n, y / n
end

function atom37_to_torsion_angles(
    aatype::AbstractMatrix{Int},
    all_atom_pos::AbstractArray,
    all_atom_mask::AbstractArray;
    placeholder_for_undefined::Bool=false,
)
    # aatype: (T, L), all_atom_pos: (T, L, 37, 3), all_atom_mask: (T, L, 37)
    T, L = size(aatype)
    torsion = zeros(Float32, T, L, 7, 2)
    torsion_mask = zeros(Float32, T, L, 7)

    n_idx = atom_order["N"] + 1
    ca_idx = atom_order["CA"] + 1
    c_idx = atom_order["C"] + 1
    o_idx = atom_order["O"] + 1

    for t in 1:T
        for r in 1:L
            aa = clamp(aatype[t, r], 0, 20) + 1

            pre_ca_x = 0f0
            pre_ca_y = 0f0
            pre_ca_z = 0f0
            pre_c_x = 0f0
            pre_c_y = 0f0
            pre_c_z = 0f0
            pre_ca_m = 0f0
            pre_c_m = 0f0
            if r > 1
                pre_ca_x = all_atom_pos[t, r - 1, ca_idx, 1]
                pre_ca_y = all_atom_pos[t, r - 1, ca_idx, 2]
                pre_ca_z = all_atom_pos[t, r - 1, ca_idx, 3]
                pre_c_x = all_atom_pos[t, r - 1, c_idx, 1]
                pre_c_y = all_atom_pos[t, r - 1, c_idx, 2]
                pre_c_z = all_atom_pos[t, r - 1, c_idx, 3]
                pre_ca_m = all_atom_mask[t, r - 1, ca_idx]
                pre_c_m = all_atom_mask[t, r - 1, c_idx]
            end

            pre_mask = pre_ca_m * pre_c_m * all_atom_mask[t, r, n_idx] * all_atom_mask[t, r, ca_idx]
            torsion_mask[t, r, 1] = pre_mask
            s, c = _torsion_sin_cos(
                pre_ca_x, pre_ca_y, pre_ca_z,
                pre_c_x, pre_c_y, pre_c_z,
                all_atom_pos[t, r, n_idx, 1], all_atom_pos[t, r, n_idx, 2], all_atom_pos[t, r, n_idx, 3],
                all_atom_pos[t, r, ca_idx, 1], all_atom_pos[t, r, ca_idx, 2], all_atom_pos[t, r, ca_idx, 3],
            )
            torsion[t, r, 1, 1] = s
            torsion[t, r, 1, 2] = c

            phi_mask = pre_c_m * all_atom_mask[t, r, n_idx] * all_atom_mask[t, r, ca_idx] * all_atom_mask[t, r, c_idx]
            torsion_mask[t, r, 2] = phi_mask
            s, c = _torsion_sin_cos(
                pre_c_x, pre_c_y, pre_c_z,
                all_atom_pos[t, r, n_idx, 1], all_atom_pos[t, r, n_idx, 2], all_atom_pos[t, r, n_idx, 3],
                all_atom_pos[t, r, ca_idx, 1], all_atom_pos[t, r, ca_idx, 2], all_atom_pos[t, r, ca_idx, 3],
                all_atom_pos[t, r, c_idx, 1], all_atom_pos[t, r, c_idx, 2], all_atom_pos[t, r, c_idx, 3],
            )
            torsion[t, r, 2, 1] = s
            torsion[t, r, 2, 2] = c

            psi_mask = all_atom_mask[t, r, n_idx] * all_atom_mask[t, r, ca_idx] *
                       all_atom_mask[t, r, c_idx] * all_atom_mask[t, r, o_idx]
            torsion_mask[t, r, 3] = psi_mask
            s, c = _torsion_sin_cos(
                all_atom_pos[t, r, n_idx, 1], all_atom_pos[t, r, n_idx, 2], all_atom_pos[t, r, n_idx, 3],
                all_atom_pos[t, r, ca_idx, 1], all_atom_pos[t, r, ca_idx, 2], all_atom_pos[t, r, ca_idx, 3],
                all_atom_pos[t, r, c_idx, 1], all_atom_pos[t, r, c_idx, 2], all_atom_pos[t, r, c_idx, 3],
                all_atom_pos[t, r, o_idx, 1], all_atom_pos[t, r, o_idx, 2], all_atom_pos[t, r, o_idx, 3],
            )
            torsion[t, r, 3, 1] = -s
            torsion[t, r, 3, 2] = -c

            for chi in 1:4
                exists = _chi_angles_mask[aa, chi]
                i1 = Int(_chi_atom_indices[aa, chi, 1]) + 1
                i2 = Int(_chi_atom_indices[aa, chi, 2]) + 1
                i3 = Int(_chi_atom_indices[aa, chi, 3]) + 1
                i4 = Int(_chi_atom_indices[aa, chi, 4]) + 1
                m = exists *
                    all_atom_mask[t, r, i1] *
                    all_atom_mask[t, r, i2] *
                    all_atom_mask[t, r, i3] *
                    all_atom_mask[t, r, i4]
                torsion_mask[t, r, 3 + chi] = m
                s, c = _torsion_sin_cos(
                    all_atom_pos[t, r, i1, 1], all_atom_pos[t, r, i1, 2], all_atom_pos[t, r, i1, 3],
                    all_atom_pos[t, r, i2, 1], all_atom_pos[t, r, i2, 2], all_atom_pos[t, r, i2, 3],
                    all_atom_pos[t, r, i3, 1], all_atom_pos[t, r, i3, 2], all_atom_pos[t, r, i3, 3],
                    all_atom_pos[t, r, i4, 1], all_atom_pos[t, r, i4, 2], all_atom_pos[t, r, i4, 3],
                )
                torsion[t, r, 3 + chi, 1] = s
                torsion[t, r, 3 + chi, 2] = c
            end
        end
    end

    alt = similar(torsion)
    for t in 1:T, r in 1:L
        aa = clamp(aatype[t, r], 0, 20) + 1
        alt[t, r, 1:3, :] .= torsion[t, r, 1:3, :]
        for chi in 1:4
            mult = 1f0 - 2f0 * _chi_pi_periodic[aa, chi]
            alt[t, r, 3 + chi, 1] = torsion[t, r, 3 + chi, 1] * mult
            alt[t, r, 3 + chi, 2] = torsion[t, r, 3 + chi, 2] * mult
        end
    end

    if placeholder_for_undefined
        for t in 1:T, r in 1:L, k in 1:7
            if torsion_mask[t, r, k] < 0.5f0
                torsion[t, r, k, 1] = 1f0
                torsion[t, r, k, 2] = 0f0
                alt[t, r, k, 1] = 1f0
                alt[t, r, k, 2] = 0f0
            end
        end
    end

    return Dict(
        :torsion_angles_sin_cos => torsion,
        :alt_torsion_angles_sin_cos => alt,
        :torsion_angles_mask => torsion_mask,
    )
end

@concrete struct TemplateSingleRows <: Onion.Layer
    template_single_embedding
    template_projection
end

@layer TemplateSingleRows

function TemplateSingleRows(c_m::Int; feature_dim::Int=57)
    feature_dim in (34, 57) || error("TemplateSingleRows supports feature_dim 34 (multimer) or 57 (monomer), got $(feature_dim)")
    return TemplateSingleRows(
        LinearFirst(feature_dim, c_m),
        LinearFirst(c_m, c_m),
    )
end

function (m::TemplateSingleRows)(
    template_aatype::AbstractMatrix{Int},
    template_all_atom_positions::AbstractArray,
    template_all_atom_masks::AbstractArray;
    placeholder_for_undefined::Bool=false,
)
    # template_aatype: (T, L)
    # template_all_atom_positions: (T, L, 37, 3)
    # template_all_atom_masks: (T, L, 37)
    T, L = size(template_aatype)
    ret = atom37_to_torsion_angles(
        template_aatype,
        template_all_atom_positions,
        template_all_atom_masks;
        placeholder_for_undefined=placeholder_for_undefined,
    )

    feature_dim = size(m.template_single_embedding.weight, 2)
    features = zeros(Float32, feature_dim, T, L, 1)
    for t in 1:T, r in 1:L
        aa = clamp(template_aatype[t, r], 0, 21)
        features[aa + 1, t, r, 1] = 1f0
    end

    tors = ret[:torsion_angles_sin_cos]
    mask = ret[:torsion_angles_mask]

    torsion_angle_mask = if feature_dim == 57
        alt_tors = ret[:alt_torsion_angles_sin_cos]
        for t in 1:T, r in 1:L, k in 1:7
            features[22 + (k - 1) * 2 + 1, t, r, 1] = tors[t, r, k, 1]
            features[22 + (k - 1) * 2 + 2, t, r, 1] = tors[t, r, k, 2]
            features[36 + (k - 1) * 2 + 1, t, r, 1] = alt_tors[t, r, k, 1]
            features[36 + (k - 1) * 2 + 2, t, r, 1] = alt_tors[t, r, k, 2]
            features[50 + k, t, r, 1] = mask[t, r, k]
        end
        reshape(mask[:, :, 3], T, L, 1)
    elseif feature_dim == 34
        for t in 1:T, r in 1:L, chi in 1:4
            k = 3 + chi
            mχ = mask[t, r, k]
            features[22 + chi, t, r, 1] = tors[t, r, k, 1] * mχ
            features[26 + chi, t, r, 1] = tors[t, r, k, 2] * mχ
            features[30 + chi, t, r, 1] = mχ
        end
        reshape(mask[:, :, 4], T, L, 1)
    else
        error("Unsupported template_single feature dimension $(feature_dim)")
    end

    act = m.template_single_embedding(features)
    act = max.(act, 0f0)
    act = m.template_projection(act)
    return act, torsion_angle_mask
end

function load_template_single_rows_npz!(
    m::TemplateSingleRows,
    params_source;
    prefix::AbstractString="alphafold/alphafold_iteration/evoformer",
)
    arrs = af2_params_read(params_source)
    w1 = arrs[string(prefix, "/template_single_embedding//weights")]
    b1 = arrs[string(prefix, "/template_single_embedding//bias")]
    w2 = arrs[string(prefix, "/template_projection//weights")]
    b2 = arrs[string(prefix, "/template_projection//bias")]

    m.template_single_embedding.weight .= permutedims(w1, (2, 1))
    m.template_single_embedding.bias .= b1
    m.template_projection.weight .= permutedims(w2, (2, 1))
    m.template_projection.bias .= b2
    return m
end
