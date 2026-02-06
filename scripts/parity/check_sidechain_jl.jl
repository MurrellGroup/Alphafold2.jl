include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2
using NPZ

npz_path = length(ARGS) >= 1 ? ARGS[1] : error("Usage: julia check_sidechain_jl.jl /path/to/sidechain_dump.npz [tol]")
tol = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 3e-3

arrs = NPZ.npzread(npz_path)

c_in = size(arrs["act"], 2)
c_hidden = length(arrs["input_projection_bias"])
num_residual_block = Int(arrs["num_residual_block"][])

m = MultiRigidSidechain(c_in, c_hidden, num_residual_block)
load_multi_rigid_sidechain_npz!(m, npz_path)

function _py_nc_to_first_ncb(x::AbstractArray)
    y = permutedims(x, (2, 1))
    return reshape(y, size(y, 1), size(y, 2), 1)
end

act = _py_nc_to_first_ncb(arrs["act"])
initial_act = _py_nc_to_first_ncb(arrs["initial_act"])

aff = arrs["affine_tensor"] # (N, 7)
q = permutedims(aff[:, 1:4], (2, 1))
t = permutedims(aff[:, 5:7], (2, 1))
q = reshape(q, 4, size(q, 2), 1)
t = reshape(t, 3, size(t, 2), 1)
rigids = Rigid(Alphafold2.rot_from_quat(q; normalize=true), t)

aatype = reshape(Int.(arrs["aatype"]), :, 1)

out = m(rigids, [act, initial_act], aatype)

angles_jl = permutedims(dropdims(out[:angles_sin_cos]; dims=4), (3, 2, 1))
unnorm_jl = permutedims(dropdims(out[:unnormalized_angles_sin_cos]; dims=4), (3, 2, 1))
atom_pos_jl = permutedims(dropdims(out[:atom_pos]; dims=4), (3, 2, 1))

angles_py = arrs["out_angles"]
unnorm_py = arrs["out_unnormalized_angles"]
atom_pos_py = arrs["out_atom_pos"]

max_abs_angles = maximum(abs.(angles_jl .- angles_py))
max_abs_unnorm = maximum(abs.(unnorm_jl .- unnorm_py))
max_abs_atom_pos = maximum(abs.(atom_pos_jl .- atom_pos_py))

println("Sidechain parity max abs diff (angles):      ", max_abs_angles)
println("Sidechain parity max abs diff (unnormalized): ", max_abs_unnorm)
println("Sidechain parity max abs diff (atom_pos):     ", max_abs_atom_pos)

@assert max_abs_angles <= tol "angles mismatch: $(max_abs_angles) > $(tol)"
@assert max_abs_unnorm <= tol "unnormalized mismatch: $(max_abs_unnorm) > $(tol)"
@assert max_abs_atom_pos <= tol "atom_pos mismatch: $(max_abs_atom_pos) > $(tol)"

println("Sidechain parity check passed at tol=$(tol).")
