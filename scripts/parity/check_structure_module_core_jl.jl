include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2
using NPZ

npz_path = length(ARGS) >= 1 ? ARGS[1] : error("Usage: julia check_structure_module_core_jl.jl /path/to/dump.npz [tol_main] [tol_angles] [tol_atom]")
tol_main = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 2e-2
tol_angles = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : max(tol_main, 3e-2)
tol_atom = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 2e-1

arrs = NPZ.npzread(npz_path)

c_s = size(arrs["single"], 2)
c_z = size(arrs["pair"], 3)
c_hidden = Int(arrs["c_hidden"][])
num_head = Int(arrs["num_head"][])
num_point_qk = Int(arrs["num_point_qk"][])
num_point_v = Int(arrs["num_point_v"][])
num_transition_layers = Int(arrs["num_transition_layers"][])
num_layer = Int(arrs["num_layer"][])
position_scale = Float32(arrs["position_scale"][])

sidechain_num_channel = length(arrs["input_projection_bias"])
num_residual_block = Int(arrs["num_residual_block"][])

m = StructureModuleCore(
    c_s,
    c_z,
    c_hidden,
    num_head,
    num_point_qk,
    num_point_v,
    num_transition_layers,
    num_layer,
    position_scale,
    sidechain_num_channel,
    num_residual_block,
)
load_structure_module_core_npz!(m, npz_path)

function _py_nc_to_first_ncb(x::AbstractArray)
    y = permutedims(x, (2, 1))
    return reshape(y, size(y, 1), size(y, 2), 1)
end

single = _py_nc_to_first_ncb(arrs["single"])
pair = af2_to_first_2d(arrs["pair"])
seq_mask = reshape(arrs["seq_mask"], :, 1)
aatype = reshape(Int.(arrs["aatype"]), :, 1)

out = m(single, pair, seq_mask, aatype)

out_act_jl = dropdims(first_to_af2_3d(out[:act]); dims=1)
out_affine_jl = permutedims(dropdims(out[:affine]; dims=4), (1, 3, 2))
out_angles_jl = permutedims(dropdims(out[:angles_sin_cos]; dims=5), (1, 4, 3, 2))
out_unnorm_jl = permutedims(dropdims(out[:unnormalized_angles_sin_cos]; dims=5), (1, 4, 3, 2))
out_atom_pos_jl = permutedims(dropdims(out[:atom_pos]; dims=5), (1, 4, 3, 2))

out_act_py = arrs["out_act"]
out_affine_py = arrs["out_affine"]
out_angles_py = arrs["out_angles"]
out_unnorm_py = arrs["out_unnormalized_angles"]
out_atom_pos_py = arrs["out_atom_pos"]

act_max_abs = maximum(abs.(out_act_jl .- out_act_py))
affine_max_abs = maximum(abs.(out_affine_jl .- out_affine_py))
angles_max_abs = maximum(abs.(out_angles_jl .- out_angles_py))
unnorm_max_abs = maximum(abs.(out_unnorm_jl .- out_unnorm_py))
atom_pos_max_abs = maximum(abs.(out_atom_pos_jl .- out_atom_pos_py))

println("StructureModuleCore parity")
println("  file: ", npz_path)
println("  act_max_abs: ", act_max_abs)
println("  affine_max_abs: ", affine_max_abs)
println("  angles_max_abs: ", angles_max_abs)
println("  unnormalized_angles_max_abs: ", unnorm_max_abs)
println("  atom_pos_max_abs: ", atom_pos_max_abs)
println("  tol_main: ", tol_main)
println("  tol_angles: ", tol_angles)
println("  tol_atom: ", tol_atom)

@assert act_max_abs <= tol_main "act mismatch: $(act_max_abs) > $(tol_main)"
@assert affine_max_abs <= tol_main "affine mismatch: $(affine_max_abs) > $(tol_main)"
@assert angles_max_abs <= tol_angles "angles mismatch: $(angles_max_abs) > $(tol_angles)"
@assert unnorm_max_abs <= tol_main "unnormalized angles mismatch: $(unnorm_max_abs) > $(tol_main)"
@assert atom_pos_max_abs <= tol_atom "atom_pos mismatch: $(atom_pos_max_abs) > $(tol_atom)"

println("PASS")
