using NPZ
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 1
    error("Usage: julia scripts/parity/check_ipa_jl.jl <dump.npz> [tol]")
end

path = ARGS[1]
tol = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 2e-4

arrs = NPZ.npzread(path)

s_py = arrs["s"]
z_py = arrs["z"]
mask_py = arrs["mask"]
out_py = arrs["out"]

s = af2_to_first_3d(s_py)
z = af2_to_first_2d(z_py)
mask = reshape(mask_py, size(mask_py, 1), size(mask_py, 2))

num_head = Int(arrs["num_head"][1])
c_hidden = Int(arrs["c_hidden"][1])
num_point_qk = Int(arrs["num_point_qk"][1])
num_point_v = Int(arrs["num_point_v"][1])

ipa = InvariantPointAttention(size(s, 1), size(z, 1), c_hidden, num_head, num_point_qk, num_point_v)
load_invariant_point_attention_npz!(ipa, path)

r = rigid_identity((size(s, 2), size(s, 3)), s; fmt=:quat)
out_jl = ipa(s, z, r, mask)
out_jl_py = dropdims(first_to_af2_3d(out_jl); dims=1)

diff = out_jl_py .- out_py
max_abs = maximum(abs, diff)
mean_abs = mean(abs, diff)
rms = sqrt(mean(diff .^ 2))

@printf("InvariantPointAttention parity\n")
@printf("  file: %s\n", path)
@printf("  max_abs: %.8g\n", max_abs)
@printf("  mean_abs: %.8g\n", mean_abs)
@printf("  rms: %.8g\n", rms)
@printf("  tol: %.8g\n", tol)

if max_abs > tol
    error("Parity check failed: max_abs=$(max_abs) > tol=$(tol)")
end

println("PASS")
