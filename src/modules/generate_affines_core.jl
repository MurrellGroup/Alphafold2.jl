@concrete struct GenerateAffinesCore <: Onion.Layer
    single_layer_norm
    initial_projection
    pair_layer_norm
    fold_iteration_core
    num_layer::Int
end

@layer GenerateAffinesCore

function GenerateAffinesCore(
    c_s::Int,
    c_z::Int,
    c_hidden::Int,
    no_heads::Int,
    no_qk_points::Int,
    no_v_points::Int,
    num_transition_layers::Int,
    num_layer::Int,
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

    return GenerateAffinesCore(
        single_layer_norm,
        initial_projection,
        pair_layer_norm,
        fold_iteration_core,
        num_layer,
    )
end

function (m::GenerateAffinesCore)(single::AbstractArray, pair::AbstractArray, seq_mask::AbstractArray)
    # single: (C_s, L, B), pair: (C_z, L, L, B), seq_mask: (L, B)
    act = m.single_layer_norm(single)
    act = m.initial_projection(act)
    act2d = m.pair_layer_norm(pair)

    rigids = rigid_identity((size(act, 2), size(act, 3)), act; fmt=:quat)

    traj, act, _ = _run_generate_affines_loop(
        m.fold_iteration_core,
        m.num_layer,
        act,
        act2d,
        seq_mask,
        rigids,
    )
    reshaped = map(x -> reshape(x, 1, size(x)...), traj)
    affine_traj = cat(reshaped...; dims=1)
    return act, affine_traj
end

function _run_generate_affines_loop(
    fold_iteration_core,
    num_layer::Int,
    act,
    act2d,
    seq_mask,
    rigids,
)
    if num_layer <= 0
        return (), act, rigids
    end
    act_next, rigids_next, _ = fold_iteration_core(act, act2d, seq_mask, rigids)
    head = to_tensor_7(rigids_next)
    tail, act_final, rigids_final = _run_generate_affines_loop(
        fold_iteration_core,
        num_layer - 1,
        act_next,
        act2d,
        seq_mask,
        rigids_next,
    )
    return (head, tail...), act_final, rigids_final
end

"""
    load_generate_affines_core_npz!(m, npz_path)

Load real-weight AF2 generate-affines-core parameters saved by
`scripts/parity/dump_generate_affines_core_from_params_py.py`.
"""
function load_generate_affines_core_npz!(m::GenerateAffinesCore, npz_path::AbstractString)
    arrs = NPZ.npzread(npz_path)

    _copy_ln_af2!(m.single_layer_norm, arrs, "single_layer_norm")
    _copy_linear_af2!(m.initial_projection, arrs, "initial_projection")
    _copy_ln_af2!(m.pair_layer_norm, arrs, "pair_layer_norm")

    load_fold_iteration_core_npz!(m.fold_iteration_core, npz_path)
    return m
end
