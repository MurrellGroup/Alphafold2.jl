using NPZ
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 1
    error("Usage: julia check_predicted_aligned_error_jl.jl <dump.npz> [tol_logits] [tol_helpers]")
end

dump_path = ARGS[1]
tol_logits = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 5e-4
tol_helpers = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 5e-4

arrs = NPZ.npzread(dump_path)

pair_repr_py = Float32.(arrs["pair_repr"]) # (N, N, C_z)
logits_py = Float32.(arrs["logits"]) # (N, N, num_bins)
breaks_py = Float32.(arrs["breaks"]) # (num_bins - 1)
weights_py = Float32.(arrs["logits_weights"]) # (C_z, num_bins)
bias_py = Float32.(arrs["logits_bias"]) # (num_bins)

num_bins = size(logits_py, 3)
c_z = size(pair_repr_py, 3)
max_error_bin = haskey(arrs, "max_error_bin") ? Float32(arrs["max_error_bin"][]) : 31f0

m = PredictedAlignedErrorHead(c_z; num_bins=num_bins, max_error_bin=max_error_bin)
m.logits.weight .= permutedims(weights_py, (2, 1))
m.logits.bias .= bias_py

pair_repr = af2_to_first_2d(pair_repr_py) # (C_z, N, N, 1)
out = m(pair_repr)

logits_jl_py = dropdims(first_to_af2_2d(out[:logits]); dims=1) # (N, N, num_bins)
breaks_jl = out[:breaks]

diff_logits = logits_jl_py .- logits_py
max_abs_logits = maximum(abs, diff_logits)
mean_abs_logits = mean(abs, diff_logits)
rms_logits = sqrt(mean(diff_logits .^ 2))
max_abs_breaks = maximum(abs.(breaks_jl .- breaks_py))

helper = compute_predicted_aligned_error(out[:logits]; max_bin=Int(round(max_error_bin)), no_bins=num_bins)
aligned_probs_jl_py = dropdims(first_to_af2_2d(helper[:aligned_confidence_probs]); dims=1)
pae_jl = dropdims(helper[:predicted_aligned_error]; dims=3)
max_pae_jl = Float32(helper[:max_predicted_aligned_error])
ptm_jl = Float32(compute_tm(out[:logits]; max_bin=Int(round(max_error_bin)), no_bins=num_bins))

max_abs_probs = haskey(arrs, "aligned_confidence_probs") ? maximum(abs.(aligned_probs_jl_py .- Float32.(arrs["aligned_confidence_probs"]))) : 0f0
max_abs_pae = haskey(arrs, "predicted_aligned_error") ? maximum(abs.(pae_jl .- Float32.(arrs["predicted_aligned_error"]))) : 0f0
abs_max_pae_cap = haskey(arrs, "max_predicted_aligned_error") ? abs(max_pae_jl - Float32(arrs["max_predicted_aligned_error"][])) : 0f0
abs_ptm = haskey(arrs, "predicted_tm_score") ? abs(ptm_jl - Float32(arrs["predicted_tm_score"][])) : 0f0

@printf("PredictedAlignedError parity\n")
@printf("  max_abs_logits: %.8g\n", max_abs_logits)
@printf("  mean_abs_logits: %.8g\n", mean_abs_logits)
@printf("  rms_logits: %.8g\n", rms_logits)
@printf("  max_abs_breaks: %.8g\n", max_abs_breaks)
@printf("  max_abs_aligned_probs: %.8g\n", max_abs_probs)
@printf("  max_abs_predicted_aligned_error: %.8g\n", max_abs_pae)
@printf("  abs_max_predicted_aligned_error: %.8g\n", abs_max_pae_cap)
@printf("  abs_predicted_tm: %.8g\n", abs_ptm)
@printf("  tol_logits: %.8g\n", tol_logits)
@printf("  tol_helpers: %.8g\n", tol_helpers)

if max(max_abs_logits, max_abs_breaks) > tol_logits
    error("Parity check failed: head mismatch exceeds tol_logits=$(tol_logits)")
end
if max(max(max_abs_probs, max_abs_pae), max(abs_max_pae_cap, abs_ptm)) > tol_helpers
    error("Parity check failed: helper mismatch exceeds tol_helpers=$(tol_helpers)")
end

println("PASS")
