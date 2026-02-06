using NPZ
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 2
    error("Usage: julia scripts/parity/check_template_embedding_jl.jl <params_model_1.npz> <dump.npz> [tol]")
end

params_path = ARGS[1]
dump_path = ARGS[2]
tol = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 5e-4

arrs = NPZ.npzread(dump_path)
par = NPZ.npzread(params_path)

query_embedding_py = Float32.(arrs["query_embedding"]) # (L,L,Cz)
mask_2d = Float32.(arrs["mask_2d"]) # (L,L)
template_mask = Float32.(vec(arrs["template_mask"])) # (T)
template_aatype = Int.(arrs["template_aatype"]) # (T,L)
template_all_atom_positions = Float32.(arrs["template_all_atom_positions"]) # (T,L,37,3)
template_all_atom_masks = Float32.(arrs["template_all_atom_masks"]) # (T,L,37)
out_py = Float32.(arrs["out"]) # (L,L,Cz)

prefix = "alphafold/alphafold_iteration/evoformer/template_embedding"
stack_prefix = string(prefix, "/single_template_embedding/template_pair_stack/__layer_stack_no_state")

c_z = size(query_embedding_py, 3)
c_t = length(par[string(prefix, "/single_template_embedding/embedding2d//bias")])
num_block = size(par[string(stack_prefix, "/pair_transition/transition2//bias")], 1)
num_head_pair = size(par[string(stack_prefix, "/triangle_attention_starting_node//feat_2d_weights")], 3)
pair_head_dim = size(par[string(stack_prefix, "/triangle_attention_starting_node/attention//query_w")], 4)
c_tri_mul = size(par[string(stack_prefix, "/triangle_multiplication_outgoing/left_projection//bias")], 2)
pair_transition_factor = size(par[string(stack_prefix, "/pair_transition/transition1//bias")], 2) / c_t
num_head_tpa = size(par[string(prefix, "/attention//query_w")], 2)
key_dim_tpa = size(par[string(prefix, "/attention//query_w")], 2) * size(par[string(prefix, "/attention//query_w")], 3)
value_dim_tpa = size(par[string(prefix, "/attention//value_w")], 2) * size(par[string(prefix, "/attention//value_w")], 3)
dgram_num_bins = size(par[string(prefix, "/single_template_embedding/embedding2d//weights")], 1) - 49

m = TemplateEmbedding(
    c_z,
    c_t,
    num_block;
    num_head_pair=num_head_pair,
    pair_head_dim=pair_head_dim,
    c_tri_mul=c_tri_mul,
    pair_transition_factor=pair_transition_factor,
    num_head_tpa=num_head_tpa,
    key_dim_tpa=key_dim_tpa,
    value_dim_tpa=value_dim_tpa,
    dgram_num_bins=dgram_num_bins,
)

load_template_embedding_npz!(m, params_path; prefix=prefix)

query_embedding = af2_to_first_2d(query_embedding_py) # (Cz,L,L,1)
pair_mask = reshape(mask_2d, size(mask_2d, 1), size(mask_2d, 2), 1)
out_jl = m(
    query_embedding,
    template_aatype,
    template_all_atom_positions,
    template_all_atom_masks,
    pair_mask;
    template_mask=template_mask,
)

out_jl_py = dropdims(first_to_af2_2d(out_jl); dims=1)
diff = out_jl_py .- out_py
max_abs = maximum(abs, diff)
mean_abs = mean(abs, diff)
rms = sqrt(mean(diff .^ 2))

single_max_abs = 0f0
if haskey(arrs, "single_out")
    single_py = Float32.(arrs["single_out"]) # (L,L,Ct)
    single_jl = m.single_template_embedding(
        vec(view(template_aatype, 1, :)),
        view(template_all_atom_positions, 1, :, :, :),
        view(template_all_atom_masks, 1, :, :),
        pair_mask,
    )
    single_jl_py = dropdims(first_to_af2_2d(single_jl); dims=1)
    single_max_abs = maximum(abs.(single_jl_py .- single_py))
end

@printf("TemplateEmbedding parity\n")
if haskey(arrs, "single_out")
    @printf("  single_max_abs: %.8g\n", single_max_abs)
end
@printf("  max_abs: %.8g\n", max_abs)
@printf("  mean_abs: %.8g\n", mean_abs)
@printf("  rms: %.8g\n", rms)
@printf("  tol: %.8g\n", tol)

if max_abs > tol
    error("Parity check failed: max_abs=$(max_abs) > tol=$(tol)")
end

println("PASS")
