using NPZ
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 1
    error("Usage: julia scripts/parity/check_fold_iteration_core_jl.jl <dump.npz> [tol]")
end

path = ARGS[1]
tol = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 2e-3

arrs = NPZ.npzread(path)

act_py = arrs["act"]
z_py = arrs["z"]
mask_py = arrs["mask"]
out_act_py = arrs["out_act"]
out_affine_update_py = arrs["out_affine_update"]
out_affine_tensor_py = arrs["out_affine_tensor"]

act = af2_to_first_3d(act_py)
z = af2_to_first_2d(z_py)
mask = reshape(mask_py, size(mask_py, 1), size(mask_py, 2))

num_head = Int(arrs["num_head"][1])
c_hidden = Int(arrs["c_hidden"][1])
num_point_qk = Int(arrs["num_point_qk"][1])
num_point_v = Int(arrs["num_point_v"][1])
num_transition_layers = Int(arrs["num_transition_layers"][1])

m = FoldIterationCore(
    size(act, 1),
    size(z, 1),
    c_hidden,
    num_head,
    num_point_qk,
    num_point_v,
    num_transition_layers,
)
load_fold_iteration_core_npz!(m, path)

rigid0 = rigid_identity((size(act, 2), size(act, 3)), act; fmt=:quat)
out_act, out_rigid, out_affine_update = m(act, z, mask, rigid0)
out_affine_tensor = Alphafold2.to_tensor_7(out_rigid)

out_act_jl = dropdims(first_to_af2_3d(out_act); dims=1)
out_affine_update_jl = dropdims(first_to_af2_3d(out_affine_update); dims=1)
out_affine_tensor_jl = dropdims(first_to_af2_3d(out_affine_tensor); dims=1)

act_diff = out_act_jl .- out_act_py
upd_diff = out_affine_update_jl .- out_affine_update_py
aff_diff = out_affine_tensor_jl .- out_affine_tensor_py

act_max = maximum(abs, act_diff)
upd_max = maximum(abs, upd_diff)
aff_max = maximum(abs, aff_diff)

max_abs = max(act_max, upd_max, aff_max)

@printf("FoldIterationCore parity\n")
@printf("  file: %s\n", path)
@printf("  act_max_abs: %.8g\n", act_max)
@printf("  affine_update_max_abs: %.8g\n", upd_max)
@printf("  affine_tensor_max_abs: %.8g\n", aff_max)
@printf("  tol: %.8g\n", tol)

if max_abs > tol
    error("Parity check failed: max_abs=$(max_abs) > tol=$(tol)")
end

println("PASS")
