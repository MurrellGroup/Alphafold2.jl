using NPZ
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 2
    error("Usage: julia check_masked_msa_jl.jl <params_model_1.npz> <dump.npz> [tol]")
end

params_path = ARGS[1]
dump_path = ARGS[2]
tol = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 5e-4

arrs = NPZ.npzread(dump_path)

msa_py = Float32.(arrs["msa"]) # (N_seq, N_res, C_m)
logits_py = Float32.(arrs["logits"]) # (N_seq, N_res, num_output)

c_m = size(msa_py, 3)
num_output = size(logits_py, 3)

m = MaskedMsaHead(c_m; num_output=num_output)
load_masked_msa_head_npz!(m, params_path)

msa = reshape(permutedims(msa_py, (3, 1, 2)), c_m, size(msa_py, 1), size(msa_py, 2), 1)
out = m(msa)
logits_jl_py = dropdims(permutedims(out[:logits], (4, 2, 3, 1)); dims=1)

diff = logits_jl_py .- logits_py
max_abs = maximum(abs, diff)
mean_abs = mean(abs, diff)
rms = sqrt(mean(diff .^ 2))

@printf("MaskedMsaHead parity\n")
@printf("  max_abs: %.8g\n", max_abs)
@printf("  mean_abs: %.8g\n", mean_abs)
@printf("  rms: %.8g\n", rms)
@printf("  tol: %.8g\n", tol)

if max_abs > tol
    error("Parity check failed: max_abs=$(max_abs) > tol=$(tol)")
end

println("PASS")
