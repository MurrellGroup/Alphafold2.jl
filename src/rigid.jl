# Rigid body types and functions — imported from Onion.jl (shared with ESMFold).
# Only utility view helpers are defined locally.

using Onion: AbstractRotation, RotMatRotation, QuatRotation, Rigid
using Onion: rot_from_mat, rot_from_quat, rot_matmul_first, rot_vec_mul_first
using Onion: quat_multiply_first, quat_multiply_by_vec_first, quat_to_rot_first
using Onion: rotation_identity, get_rot_mats, get_quats
using Onion: compose_q_update_vec, rigid_identity, apply_rotation, apply_rigid
using Onion: invert_apply_rigid, compose, scale_translation, to_tensor_7, to_tensor_4x4
using Onion: rigid_index

# ── View helpers (used by openfold_feats.jl, structure_module_core.jl, etc.) ──

@inline function _view_last2(x, I, J)
    return view(x, ntuple(_ -> Colon(), ndims(x) - 2)..., I, J)
end

@inline function _view_last1(x, I)
    return view(x, ntuple(_ -> Colon(), ndims(x) - 1)..., I)
end

@inline function _view_first2(x, I, J)
    return view(x, I, J, ntuple(_ -> Colon(), ndims(x) - 2)...)
end

@inline function _view_first1(x, I)
    return view(x, I, ntuple(_ -> Colon(), ndims(x) - 1)...)
end
