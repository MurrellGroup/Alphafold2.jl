@concrete struct StructureModuleCore <: Onion.Layer
    single_layer_norm
    initial_projection
    pair_layer_norm
    fold_iteration_core
    sidechain
    num_layer::Int
    position_scale::Float32
end

@layer StructureModuleCore

function StructureModuleCore(
    c_s::Int,
    c_z::Int,
    c_hidden::Int,
    no_heads::Int,
    no_qk_points::Int,
    no_v_points::Int,
    num_transition_layers::Int,
    num_layer::Int,
    position_scale::Real,
    sidechain_num_channel::Int,
    sidechain_num_residual_block::Int,
)
    single_layer_norm = LayerNormFirst(c_s)
    initial_projection = LinearFirst(c_s, c_s)
    pair_layer_norm = LayerNormFirst(c_z)

    fold_iteration_core = FoldIterationCore(
        c_s,
        c_z,
        c_hidden,
        no_heads,
        no_qk_points,
        no_v_points,
        num_transition_layers,
    )

    sidechain = MultiRigidSidechain(
        c_s,
        sidechain_num_channel,
        sidechain_num_residual_block;
        num_representations=2,
    )

    return StructureModuleCore(
        single_layer_norm,
        initial_projection,
        pair_layer_norm,
        fold_iteration_core,
        sidechain,
        num_layer,
        Float32(position_scale),
    )
end

function _stack_firstdim(arrs::AbstractVector)
    return cat((reshape(x, 1, size(x)...) for x in arrs)...; dims=1)
end

function (m::StructureModuleCore)(single::AbstractArray, pair::AbstractArray, seq_mask::AbstractArray, aatype::AbstractArray)
    # single: (C_s, L, B), pair: (C_z, L, L, B), seq_mask: (L, B), aatype: (L, B) or (L)
    act = m.single_layer_norm(single)
    initial_act = act
    act = m.initial_projection(act)
    act2d = m.pair_layer_norm(pair)

    rigids = rigid_identity((size(act, 2), size(act, 3)), act; fmt=:quat)

    affine_traj = Vector{Any}(undef, m.num_layer)
    angles_traj = Vector{Any}(undef, m.num_layer)
    unnormalized_angles_traj = Vector{Any}(undef, m.num_layer)
    atom_pos_traj = Vector{Any}(undef, m.num_layer)
    frames_traj = Vector{Any}(undef, m.num_layer)

    for i in 1:m.num_layer
        act, rigids, _ = m.fold_iteration_core(act, act2d, seq_mask, rigids)

        scaled_rigids = scale_translation(rigids, m.position_scale)
        sc = m.sidechain(scaled_rigids, [act, initial_act], aatype)

        affine_traj[i] = to_tensor_7(rigids)
        angles_traj[i] = sc[:angles_sin_cos]
        unnormalized_angles_traj[i] = sc[:unnormalized_angles_sin_cos]
        atom_pos_traj[i] = sc[:atom_pos]
        frames_traj[i] = to_tensor_4x4(sc[:frames])
    end

    return Dict{Symbol,Any}(
        :act => act,
        :affine => _stack_firstdim(affine_traj),
        :angles_sin_cos => _stack_firstdim(angles_traj),
        :unnormalized_angles_sin_cos => _stack_firstdim(unnormalized_angles_traj),
        :atom_pos => _stack_firstdim(atom_pos_traj),
        :frames => _stack_firstdim(frames_traj),
    )
end

"""
    load_structure_module_core_npz!(m, npz_path)

Load real-weight AF2 structure-module-core parameters saved by
`scripts/parity/dump_structure_module_core_from_params_py.py`.
"""
function load_structure_module_core_npz!(m::StructureModuleCore, npz_path::AbstractString)
    arrs = NPZ.npzread(npz_path)

    _copy_ln_af2!(m.single_layer_norm, arrs, "single_layer_norm")
    _copy_linear_af2!(m.initial_projection, arrs, "initial_projection")
    _copy_ln_af2!(m.pair_layer_norm, arrs, "pair_layer_norm")

    load_fold_iteration_core_npz!(m.fold_iteration_core, npz_path)
    load_multi_rigid_sidechain_npz!(m.sidechain, npz_path)

    return m
end
