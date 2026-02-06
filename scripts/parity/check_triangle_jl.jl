using NPZ
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 1
    error("Usage: julia scripts/parity/check_triangle_jl.jl <dump.npz> [tol]")
end

path = ARGS[1]
tol = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 5e-5

arrs = NPZ.npzread(path)

act_py = arrs["act"]
mask_py = arrs["mask"]
out_py = arrs["out"]
outgoing = arrs["outgoing"][1] == 1

act = af2_to_first_2d(act_py)
mask = reshape(mask_py, size(mask_py, 1), size(mask_py, 2), 1)

c_z = size(act, 1)
c_hidden = size(arrs["left_projection_bias"], 1)

tm = TriangleMultiplication(c_z, c_hidden; outgoing=outgoing)
load_triangle_multiplication_npz!(tm, path)

out_jl = tm(act, mask)
out_jl_af2 = dropdims(first_to_af2_2d(out_jl); dims=1)

diff = out_jl_af2 .- out_py
max_abs = maximum(abs, diff)
mean_abs = mean(abs, diff)
rms = sqrt(mean(diff .^ 2))

@printf("TriangleMultiplication parity\n")
@printf("  file: %s\n", path)
@printf("  outgoing: %s\n", string(outgoing))
@printf("  max_abs: %.8g\n", max_abs)
@printf("  mean_abs: %.8g\n", mean_abs)
@printf("  rms: %.8g\n", rms)
@printf("  tol: %.8g\n", tol)

if max_abs > tol
    error("Parity check failed: max_abs=$(max_abs) > tol=$(tol)")
end

println("PASS")
