using NPZ
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 2
    error("Usage: julia check_predicted_lddt_jl.jl <params_model_1.npz> <dump.npz> [tol_logits] [tol_plddt]")
end

params_path = ARGS[1]
dump_path = ARGS[2]
tol_logits = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 5e-4
tol_plddt = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 5e-4

arrs = NPZ.npzread(dump_path)

structure_module_py = Float32.(arrs["structure_module"]) # (N, C_s)
logits_py = Float32.(arrs["logits"]) # (N, num_bins)
plddt_py = haskey(arrs, "plddt") ? Float32.(arrs["plddt"]) : nothing # (N,)

c_s = size(structure_module_py, 2)
num_channels = haskey(arrs, "num_channels") ? Int(arrs["num_channels"][]) : 128
num_bins = size(logits_py, 2)

m = PredictedLDDTHead(c_s; num_channels=num_channels, num_bins=num_bins)
load_predicted_lddt_head_npz!(m, params_path)

structure_module = af2_to_first_3d(structure_module_py) # (C_s, N, 1)
logits_jl = m(structure_module)[:logits] # (num_bins, N, 1)
logits_jl_py = dropdims(first_to_af2_3d(logits_jl); dims=1) # (N, num_bins)

diff_logits = logits_jl_py .- logits_py
max_abs_logits = maximum(abs, diff_logits)
mean_abs_logits = mean(abs, diff_logits)
rms_logits = sqrt(mean(diff_logits .^ 2))

max_abs_plddt = 0f0
if plddt_py !== nothing
    plddt_jl = vec(compute_plddt(logits_jl))
    max_abs_plddt = maximum(abs.(plddt_jl .- plddt_py))
end

@printf("PredictedLDDT parity\n")
@printf("  max_abs_logits: %.8g\n", max_abs_logits)
@printf("  mean_abs_logits: %.8g\n", mean_abs_logits)
@printf("  rms_logits: %.8g\n", rms_logits)
if plddt_py !== nothing
    @printf("  max_abs_plddt: %.8g\n", max_abs_plddt)
end
@printf("  tol_logits: %.8g\n", tol_logits)
if plddt_py !== nothing
    @printf("  tol_plddt: %.8g\n", tol_plddt)
end

if max_abs_logits > tol_logits
    error("Parity check failed: logits max_abs=$(max_abs_logits) > tol_logits=$(tol_logits)")
end
if plddt_py !== nothing && max_abs_plddt > tol_plddt
    error("Parity check failed: plddt max_abs=$(max_abs_plddt) > tol_plddt=$(tol_plddt)")
end

println("PASS")
