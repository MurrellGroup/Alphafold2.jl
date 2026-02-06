using NPZ
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 1
    error("Usage: julia scripts/parity/check_triangle_attention_jl.jl <dump.npz> [tol]")
end

path = ARGS[1]
tol = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 8e-5

arrs = NPZ.npzread(path)

pair_act = af2_to_first_2d(arrs["pair_act"])
pair_mask_py = arrs["pair_mask"]
pair_mask = reshape(pair_mask_py, size(pair_mask_py, 1), size(pair_mask_py, 2), 1)
out_py = arrs["out"]

column = arrs["column"][1] == 1
num_head = Int(arrs["num_head"][1])
head_dim = Int(arrs["head_dim"][1])

orientation = column ? :per_column : :per_row
tri_att = TriangleAttention(size(pair_act, 1), num_head, head_dim; orientation=orientation)
load_triangle_attention_npz!(tri_att, path)

out_jl = tri_att(pair_act, pair_mask)
out_jl_af2 = dropdims(first_to_af2_2d(out_jl); dims=1)

diff = out_jl_af2 .- out_py
max_abs = maximum(abs, diff)
mean_abs = mean(abs, diff)
rms = sqrt(mean(diff .^ 2))

@printf("TriangleAttention parity\n")
@printf("  file: %s\n", path)
@printf("  orientation: %s\n", String(orientation))
@printf("  max_abs: %.8g\n", max_abs)
@printf("  mean_abs: %.8g\n", mean_abs)
@printf("  rms: %.8g\n", rms)
@printf("  tol: %.8g\n", tol)

if max_abs > tol
    error("Parity check failed: max_abs=$(max_abs) > tol=$(tol)")
end

println("PASS")
