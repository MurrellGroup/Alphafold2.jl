using NNlib

@concrete struct MultiRigidSidechain <: Onion.Layer
    input_projections
    resblock1
    resblock2
    unnormalized_angles
    eps::Float32
end

@layer MultiRigidSidechain

function MultiRigidSidechain(
    c_in::Int,
    c_hidden::Int,
    num_residual_block::Int;
    num_representations::Int=2,
    eps::Real=1e-12,
)
    input_projections = [LinearFirst(c_in, c_hidden) for _ in 1:num_representations]
    resblock1 = [LinearFirst(c_hidden, c_hidden) for _ in 1:num_residual_block]
    resblock2 = [LinearFirst(c_hidden, c_hidden) for _ in 1:num_residual_block]
    unnormalized_angles = LinearFirst(c_hidden, 14)

    return MultiRigidSidechain(
        input_projections,
        resblock1,
        resblock2,
        unnormalized_angles,
        Float32(eps),
    )
end

@inline function _ensure_aatype_2d(aatype::AbstractArray)
    if ndims(aatype) == 1
        return reshape(aatype, size(aatype, 1), 1)
    elseif ndims(aatype) == 2
        return aatype
    else
        error("aatype must have shape (L) or (L, B)")
    end
end

@inline function _l2_normalize_first(x::AbstractArray, eps::Float32)
    n = sqrt.(max.(sum(x .^ 2; dims=1), eps))
    return x ./ n
end

function (m::MultiRigidSidechain)(rigids::Rigid, representations_list::AbstractVector{<:AbstractArray}, aatype::AbstractArray)
    @assert length(representations_list) == length(m.input_projections)

    act = m.input_projections[1](max.(representations_list[1], 0f0))
    for i in 2:length(representations_list)
        act = act .+ m.input_projections[i](max.(representations_list[i], 0f0))
    end

    for i in eachindex(m.resblock1)
        old_act = act
        act = m.resblock1[i](max.(act, 0f0))
        act = m.resblock2[i](max.(act, 0f0))
        act = act .+ old_act
    end

    unnormalized_angles_flat = m.unnormalized_angles(max.(act, 0f0)) # (14, L, B)

    unnormalized_angles = reshape(
        unnormalized_angles_flat,
        2,
        7,
        size(unnormalized_angles_flat, 2),
        size(unnormalized_angles_flat, 3),
    ) # (2, 7, L, B)

    angles = _l2_normalize_first(unnormalized_angles, m.eps)

    aatype2d = _ensure_aatype_2d(aatype)

    default_frames = to_device(restype_rigid_group_default_frame, angles, eltype(angles))
    group_idx = to_device(restype_atom14_to_rigid_group, angles, Int)
    atom_mask = to_device(restype_atom14_mask, angles, eltype(angles))
    lit_positions = to_device(restype_atom14_rigid_group_positions, angles, eltype(angles))

    all_frames_to_global = torsion_angles_to_frames(rigids, angles, aatype2d, default_frames)

    pred_positions = frames_and_literature_positions_to_atom14_pos(
        all_frames_to_global,
        aatype2d,
        default_frames,
        group_idx,
        atom_mask,
        lit_positions,
    )

    return Dict{Symbol,Any}(
        :angles_sin_cos => angles,
        :unnormalized_angles_sin_cos => unnormalized_angles,
        :atom_pos => pred_positions,
        :frames => all_frames_to_global,
    )
end

function _sidechain_key(base::String, i::Int)
    return i == 1 ? base : "$(base)_$(i-1)"
end

"""
    load_multi_rigid_sidechain_npz!(m, npz_path)

Load AF2 sidechain parameters saved by
`scripts/parity/dump_sidechain_from_params_py.py`.
"""
function load_multi_rigid_sidechain_npz!(m::MultiRigidSidechain, npz_path::AbstractString)
    arrs = NPZ.npzread(npz_path)

    for i in eachindex(m.input_projections)
        _copy_linear_af2!(m.input_projections[i], arrs, _sidechain_key("input_projection", i))
    end

    for i in eachindex(m.resblock1)
        _copy_linear_af2!(m.resblock1[i], arrs, _sidechain_key("resblock1", i))
        _copy_linear_af2!(m.resblock2[i], arrs, _sidechain_key("resblock2", i))
    end

    _copy_linear_af2!(m.unnormalized_angles, arrs, "unnormalized_angles")

    return m
end
