using NPZ
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 2
    error("Usage: julia check_experimentally_resolved_jl.jl <params_model_1.npz> <dump.npz> [tol]")
end

params_path = ARGS[1]
dump_path = ARGS[2]
tol = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 5e-4

arrs = NPZ.npzread(dump_path)

single_py = Float32.(arrs["single"]) # (N_res, C_s)
logits_py = Float32.(arrs["logits"]) # (N_res, 37)

c_s = size(single_py, 2)

m = ExperimentallyResolvedHead(c_s)
load_experimentally_resolved_head_npz!(m, params_path)

single = af2_to_first_3d(single_py)
out = m(single)
logits_jl_py = dropdims(first_to_af2_3d(out[:logits]); dims=1)

diff = logits_jl_py .- logits_py
max_abs = maximum(abs, diff)
mean_abs = mean(abs, diff)
rms = sqrt(mean(diff .^ 2))

@printf("ExperimentallyResolvedHead parity\n")
@printf("  max_abs: %.8g\n", max_abs)
@printf("  mean_abs: %.8g\n", mean_abs)
@printf("  rms: %.8g\n", rms)
@printf("  tol: %.8g\n", tol)

if max_abs > tol
    error("Parity check failed: max_abs=$(max_abs) > tol=$(tol)")
end

println("PASS")
