using NPZ
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 1
    error("Usage: julia scripts/parity/check_outer_product_mean_jl.jl <dump.npz> [tol]")
end

path = ARGS[1]
tol = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1e-4

arrs = NPZ.npzread(path)

act_py = arrs["act"]
mask_py = arrs["mask"]
out_py = arrs["out"]

# [N_seq, N_res, C_m] -> [C_m, N_seq, N_res, 1]
act = reshape(permutedims(act_py, (3, 1, 2)), size(act_py, 3), size(act_py, 1), size(act_py, 2), 1)
mask = reshape(mask_py, size(mask_py, 1), size(mask_py, 2), 1)

c_m = size(act, 1)
c_outer = size(arrs["left_projection_bias"], 1)
c_z = size(arrs["output_b"], 1)

opm = OuterProductMean(c_m, c_outer, c_z)
load_outer_product_mean_npz!(opm, path)

out_jl = opm(act, mask)
out_jl_py = dropdims(first_to_af2_2d(out_jl); dims=1)

diff = out_jl_py .- out_py
max_abs = maximum(abs, diff)
mean_abs = mean(abs, diff)
rms = sqrt(mean(diff .^ 2))

@printf("OuterProductMean parity\n")
@printf("  file: %s\n", path)
@printf("  max_abs: %.8g\n", max_abs)
@printf("  mean_abs: %.8g\n", mean_abs)
@printf("  rms: %.8g\n", rms)
@printf("  tol: %.8g\n", tol)

if max_abs > tol
    error("Parity check failed: max_abs=$(max_abs) > tol=$(tol)")
end

println("PASS")
