using NPZ
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 1
    error("Usage: julia scripts/parity/check_generate_affines_core_jl.jl <dump.npz> [tol]")
end

path = ARGS[1]
tol = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 2e-2

arrs = NPZ.npzread(path)

single_py = arrs["single"]
pair_py = arrs["pair"]
seq_mask_py = arrs["seq_mask"]
out_act_py = arrs["out_act"]
out_affine_py = arrs["out_affine"]

single = af2_to_first_3d(single_py)
pair = af2_to_first_2d(pair_py)
seq_mask = reshape(seq_mask_py, size(seq_mask_py, 1), 1)

num_head = Int(arrs["num_head"][1])
c_hidden = Int(arrs["c_hidden"][1])
num_point_qk = Int(arrs["num_point_qk"][1])
num_point_v = Int(arrs["num_point_v"][1])
num_transition_layers = Int(arrs["num_transition_layers"][1])
num_layer = Int(arrs["num_layer"][1])

m = GenerateAffinesCore(
    size(single, 1),
    size(pair, 1),
    c_hidden,
    num_head,
    num_point_qk,
    num_point_v,
    num_transition_layers,
    num_layer,
)
load_generate_affines_core_npz!(m, path)

out_act, out_affine = m(single, pair, seq_mask)

out_act_jl = dropdims(first_to_af2_3d(out_act); dims=1)
# out_affine: (num_layer, 7, N, B) -> (num_layer, N, 7)
out_affine_jl = permutedims(out_affine, (1, 3, 2, 4))
out_affine_jl = dropdims(out_affine_jl; dims=4)

act_diff = out_act_jl .- out_act_py
aff_diff = out_affine_jl .- out_affine_py

act_max = maximum(abs, act_diff)
aff_max = maximum(abs, aff_diff)
max_abs = max(act_max, aff_max)

@printf("GenerateAffinesCore parity\n")
@printf("  file: %s\n", path)
@printf("  act_max_abs: %.8g\n", act_max)
@printf("  affine_max_abs: %.8g\n", aff_max)
@printf("  tol: %.8g\n", tol)

if max_abs > tol
    error("Parity check failed: max_abs=$(max_abs) > tol=$(tol)")
end

println("PASS")
