using NPZ
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 1
    error("Usage: julia scripts/parity/check_msa_row_attention_jl.jl <dump.npz> [tol]")
end

path = ARGS[1]
tol = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1e-4

arrs = NPZ.npzread(path)

msa_act_py = arrs["msa_act"]
msa_mask_py = arrs["msa_mask"]
pair_act_py = arrs["pair_act"]
out_py = arrs["out"]

msa_act = reshape(permutedims(msa_act_py, (3, 1, 2)), size(msa_act_py, 3), size(msa_act_py, 1), size(msa_act_py, 2), 1)
msa_mask = reshape(msa_mask_py, size(msa_mask_py, 1), size(msa_mask_py, 2), 1)
pair_act = af2_to_first_2d(pair_act_py)

num_head = Int(arrs["num_head"][1])
head_dim = Int(arrs["head_dim"][1])

m = MSARowAttentionWithPairBias(size(msa_act, 1), size(pair_act, 1), num_head, head_dim)
load_msa_row_attention_npz!(m, path)

out_jl = m(msa_act, msa_mask, pair_act)
out_jl_py = dropdims(permutedims(out_jl, (4, 2, 3, 1)); dims=1)

diff = out_jl_py .- out_py
max_abs = maximum(abs, diff)
mean_abs = mean(abs, diff)
rms = sqrt(mean(diff .^ 2))

@printf("MSARowAttentionWithPairBias parity\n")
@printf("  file: %s\n", path)
@printf("  max_abs: %.8g\n", max_abs)
@printf("  mean_abs: %.8g\n", mean_abs)
@printf("  rms: %.8g\n", rms)
@printf("  tol: %.8g\n", tol)

if max_abs > tol
    error("Parity check failed: max_abs=$(max_abs) > tol=$(tol)")
end

println("PASS")
