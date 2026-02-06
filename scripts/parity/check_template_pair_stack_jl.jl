using NPZ
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 1
    error("Usage: julia scripts/parity/check_template_pair_stack_jl.jl <dump.npz> [tol]")
end

path = ARGS[1]
tol = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 3e-4

arrs = NPZ.npzread(path)

pair_py = arrs["pair"]
pair_mask_py = arrs["pair_mask"]
out_py = arrs["out"]

pair = af2_to_first_2d(pair_py)
pair_mask = reshape(pair_mask_py, size(pair_mask_py, 1), size(pair_mask_py, 2), 1)

c_t = size(pair, 1)
num_block = Int(arrs["num_block"][1])
num_head_pair = Int(arrs["num_head_pair"][1])
pair_head_dim = Int(arrs["pair_head_dim"][1])
c_tri_mul = Int(arrs["c_tri_mul"][1])

prefix = "alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state"
pair_transition_factor = Float64(arrs["pair_transition_factor"][1])

m = TemplatePairStack(
    c_t,
    num_block;
    num_head_pair=num_head_pair,
    pair_head_dim=pair_head_dim,
    c_tri_mul=c_tri_mul,
    pair_transition_factor=pair_transition_factor,
)

load_template_pair_stack_npz!(m, path; prefix=prefix)

out_jl = m(pair, pair_mask)
out_jl_py = dropdims(first_to_af2_2d(out_jl); dims=1)

diff = out_jl_py .- out_py
max_abs = maximum(abs, diff)
mean_abs = mean(abs, diff)
rms = sqrt(mean(diff .^ 2))

@printf("TemplatePairStack parity\n")
@printf("  file: %s\n", path)
@printf("  max_abs: %.8g\n", max_abs)
@printf("  mean_abs: %.8g\n", mean_abs)
@printf("  rms: %.8g\n", rms)
@printf("  tol: %.8g\n", tol)

if max_abs > tol
    error("Parity check failed: max_abs=$(max_abs) > tol=$(tol)")
end

println("PASS")
