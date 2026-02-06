using NPZ
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 1
    error("Usage: julia scripts/parity/check_evoformer_iteration_jl.jl <dump_dir> [tol]")
end

dump_dir = ARGS[1]
tol = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 2e-4

io = NPZ.npzread(joinpath(dump_dir, "evo_io.npz"))

msa_py = io["msa"]
pair_py = io["pair"]
msa_mask_py = io["msa_mask"]
pair_mask_py = io["pair_mask"]
out_msa_py = io["out_msa"]
out_pair_py = io["out_pair"]

msa = reshape(permutedims(msa_py, (3, 1, 2)), size(msa_py, 3), size(msa_py, 1), size(msa_py, 2), 1)
pair = af2_to_first_2d(pair_py)
msa_mask = reshape(msa_mask_py, size(msa_mask_py, 1), size(msa_mask_py, 2), 1)
pair_mask = reshape(pair_mask_py, size(pair_mask_py, 1), size(pair_mask_py, 2), 1)

num_head_msa = Int(io["num_head_msa"][1])
msa_head_dim = Int(io["msa_head_dim"][1])
num_head_pair = Int(io["num_head_pair"][1])
pair_head_dim = Int(io["pair_head_dim"][1])
c_outer = Int(io["c_outer"][1])
c_tri_mul = Int(io["c_tri_mul"][1])
outer_first = io["outer_first"][1] == 1

msa_transition_npz = NPZ.npzread(joinpath(dump_dir, "msa_transition.npz"))
pair_transition_npz = NPZ.npzread(joinpath(dump_dir, "pair_transition.npz"))
msa_transition_factor = size(msa_transition_npz["transition1_bias"], 1) / size(msa, 1)
pair_transition_factor = size(pair_transition_npz["transition1_bias"], 1) / size(pair, 1)

evo = EvoformerIteration(
    size(msa, 1),
    size(pair, 1);
    num_head_msa=num_head_msa,
    msa_head_dim=msa_head_dim,
    num_head_pair=num_head_pair,
    pair_head_dim=pair_head_dim,
    c_outer=c_outer,
    c_tri_mul=c_tri_mul,
    msa_transition_factor=msa_transition_factor,
    pair_transition_factor=pair_transition_factor,
    outer_first=outer_first,
)

load_outer_product_mean_npz!(evo.outer_product_mean, joinpath(dump_dir, "outer_product_mean.npz"))
load_msa_row_attention_npz!(evo.msa_row_attention_with_pair_bias, joinpath(dump_dir, "msa_row_attention.npz"))
load_msa_column_attention_npz!(evo.msa_column_attention, joinpath(dump_dir, "msa_column_attention.npz"))
load_transition_npz!(evo.msa_transition, joinpath(dump_dir, "msa_transition.npz"))
load_transition_npz!(evo.pair_transition, joinpath(dump_dir, "pair_transition.npz"))
load_triangle_multiplication_npz!(evo.triangle_multiplication_outgoing, joinpath(dump_dir, "tri_mul_out.npz"))
load_triangle_multiplication_npz!(evo.triangle_multiplication_incoming, joinpath(dump_dir, "tri_mul_in.npz"))
load_triangle_attention_npz!(evo.triangle_attention_starting_node, joinpath(dump_dir, "tri_att_start.npz"))
load_triangle_attention_npz!(evo.triangle_attention_ending_node, joinpath(dump_dir, "tri_att_end.npz"))

msa_out, pair_out = evo(msa, pair, msa_mask, pair_mask)

msa_out_py = dropdims(permutedims(msa_out, (4, 2, 3, 1)); dims=1)
pair_out_py = dropdims(first_to_af2_2d(pair_out); dims=1)

msa_diff = msa_out_py .- out_msa_py
pair_diff = pair_out_py .- out_pair_py

msa_max = maximum(abs, msa_diff)
msa_mean = mean(abs, msa_diff)
msa_rms = sqrt(mean(msa_diff .^ 2))

pair_max = maximum(abs, pair_diff)
pair_mean = mean(abs, pair_diff)
pair_rms = sqrt(mean(pair_diff .^ 2))

max_abs = max(msa_max, pair_max)

@printf("EvoformerIteration parity\n")
@printf("  dir: %s\n", dump_dir)
@printf("  msa_max_abs: %.8g\n", msa_max)
@printf("  msa_mean_abs: %.8g\n", msa_mean)
@printf("  msa_rms: %.8g\n", msa_rms)
@printf("  pair_max_abs: %.8g\n", pair_max)
@printf("  pair_mean_abs: %.8g\n", pair_mean)
@printf("  pair_rms: %.8g\n", pair_rms)
@printf("  tol: %.8g\n", tol)

if max_abs > tol
    error("Parity check failed: max_abs=$(max_abs) > tol=$(tol)")
end

println("PASS")
