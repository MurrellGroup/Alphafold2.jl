using NPZ
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 2
    error("Usage: julia check_distogram_jl.jl <params_model_1.npz> <dump.npz> [tol_logits] [tol_bins]")
end

params_path = ARGS[1]
dump_path = ARGS[2]
tol_logits = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 5e-4
tol_bins = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 5e-6

arrs = NPZ.npzread(dump_path)

pair_py = Float32.(arrs["pair"]) # (N, N, C_z)
logits_py = Float32.(arrs["logits"]) # (N, N, num_bins)
bin_edges_py = Float32.(arrs["bin_edges"]) # (num_bins - 1)

c_z = size(pair_py, 3)
num_bins = size(logits_py, 3)
first_break = haskey(arrs, "first_break") ? Float32(arrs["first_break"][]) : bin_edges_py[1]
last_break = haskey(arrs, "last_break") ? Float32(arrs["last_break"][]) : bin_edges_py[end]

m = DistogramHead(c_z; num_bins=num_bins, first_break=first_break, last_break=last_break)
load_distogram_head_npz!(m, params_path)

pair = af2_to_first_2d(pair_py)
out = m(pair)
logits_jl_py = dropdims(first_to_af2_2d(out[:logits]); dims=1)
bin_edges_jl = out[:bin_edges]

diff = logits_jl_py .- logits_py
max_abs = maximum(abs, diff)
mean_abs = mean(abs, diff)
rms = sqrt(mean(diff .^ 2))
max_abs_bins = maximum(abs.(bin_edges_jl .- bin_edges_py))

@printf("DistogramHead parity\n")
@printf("  max_abs_logits: %.8g\n", max_abs)
@printf("  mean_abs_logits: %.8g\n", mean_abs)
@printf("  rms_logits: %.8g\n", rms)
@printf("  max_abs_bin_edges: %.8g\n", max_abs_bins)
@printf("  tol_logits: %.8g\n", tol_logits)
@printf("  tol_bins: %.8g\n", tol_bins)

if max_abs > tol_logits
    error("Parity check failed: logits max_abs=$(max_abs) > tol_logits=$(tol_logits)")
end
if max_abs_bins > tol_bins
    error("Parity check failed: bin_edges max_abs=$(max_abs_bins) > tol_bins=$(tol_bins)")
end

println("PASS")
