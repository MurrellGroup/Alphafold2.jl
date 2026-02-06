using NPZ
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 1
    error("Usage: julia scripts/parity/check_transition_jl.jl <dump.npz> [tol]")
end

path = ARGS[1]
tol = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 8e-5

arrs = NPZ.npzread(path)

act_py = arrs["act"]
mask_py = arrs["mask"]
out_py = arrs["out"]

act = permutedims(act_py, (3, 2, 1))
mask = permutedims(mask_py, (2, 1))

c = size(act, 1)
num_intermediate = size(arrs["transition1_bias"], 1)
factor = num_intermediate / c

tr = Transition(c, factor)
load_transition_npz!(tr, path)
out_jl = tr(act, mask)
out_jl_py = permutedims(out_jl, (3, 2, 1))

diff = out_jl_py .- out_py
max_abs = maximum(abs, diff)
mean_abs = mean(abs, diff)
rms = sqrt(mean(diff .^ 2))

@printf("Transition parity\n")
@printf("  file: %s\n", path)
@printf("  max_abs: %.8g\n", max_abs)
@printf("  mean_abs: %.8g\n", mean_abs)
@printf("  rms: %.8g\n", rms)
@printf("  tol: %.8g\n", tol)

if max_abs > tol
    error("Parity check failed: max_abs=$(max_abs) > tol=$(tol)")
end

println("PASS")
