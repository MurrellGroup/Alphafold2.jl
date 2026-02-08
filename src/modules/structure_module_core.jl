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
    ;
    multimer_ipa::Bool=false,
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
        ; multimer_ipa=multimer_ipa,
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

function _stack_firstdim(arrs)
    reshaped = map(x -> reshape(x, 1, size(x)...), arrs)
    return cat(reshaped...; dims=1)
end

function (m::StructureModuleCore)(single::AbstractArray, pair::AbstractArray, seq_mask::AbstractArray, aatype::AbstractArray)
    # single: (C_s, L, B), pair: (C_z, L, L, B), seq_mask: (L, B), aatype: (L, B) or (L)
    act = m.single_layer_norm(single)
    initial_act = act
    act = m.initial_projection(act)
    act2d = m.pair_layer_norm(pair)

    rigids = rigid_identity((size(act, 2), size(act, 3)), act; fmt=:quat)

    traj, act, _ = _run_structure_module_loop(
        m.fold_iteration_core,
        m.sidechain,
        m.position_scale,
        m.num_layer,
        act,
        act2d,
        seq_mask,
        rigids,
        initial_act,
        aatype,
    )

    return Dict{Symbol,Any}(
        :act => act,
        :affine => _stack_firstdim(getfield.(traj, :affine)),
        :angles_sin_cos => _stack_firstdim(getfield.(traj, :angles)),
        :unnormalized_angles_sin_cos => _stack_firstdim(getfield.(traj, :unnormalized_angles)),
        :atom_pos => _stack_firstdim(getfield.(traj, :atom_pos)),
        :frames => _stack_firstdim(getfield.(traj, :frames)),
    )
end

function _run_structure_module_loop(
    fold_iteration_core,
    sidechain,
    position_scale::Float32,
    num_layer::Int,
    act,
    act2d,
    seq_mask,
    rigids,
    initial_act,
    aatype,
)
    if num_layer <= 0
        return (), act, rigids
    end

    act_next, rigids_next, _ = fold_iteration_core(act, act2d, seq_mask, rigids)
    scaled_rigids = scale_translation(rigids_next, position_scale)
    sc = sidechain(scaled_rigids, [act_next, initial_act], aatype)

    head = (
        affine = to_tensor_7(rigids_next),
        angles = sc[:angles_sin_cos],
        unnormalized_angles = sc[:unnormalized_angles_sin_cos],
        atom_pos = sc[:atom_pos],
        frames = to_tensor_4x4(sc[:frames]),
    )

    tail, act_final, rigids_final = _run_structure_module_loop(
        fold_iteration_core,
        sidechain,
        position_scale,
        num_layer - 1,
        act_next,
        act2d,
        seq_mask,
        rigids_next,
        initial_act,
        aatype,
    )
    return (head, tail...), act_final, rigids_final
end

"""
    load_structure_module_core_npz!(m, npz_path)

Load real-weight AF2 structure-module-core parameters saved by
`scripts/parity/dump_structure_module_core_from_params_py.py`.
"""
function load_structure_module_core_npz!(m::StructureModuleCore, params_source)
    arrs = af2_params_read(params_source)

    _copy_ln_af2!(m.single_layer_norm, arrs, "single_layer_norm")
    _copy_linear_af2!(m.initial_projection, arrs, "initial_projection")
    _copy_ln_af2!(m.pair_layer_norm, arrs, "pair_layer_norm")

    load_fold_iteration_core_npz!(m.fold_iteration_core, params_source)
    load_multi_rigid_sidechain_npz!(m.sidechain, params_source)

    return m
end
