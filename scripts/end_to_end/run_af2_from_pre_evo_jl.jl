using NPZ
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 3
    error("Usage: julia run_af2_from_pre_evo_jl.jl <params_model_1.npz> <pre_evo_dump.npz> <out.npz>")
end

params_path = ARGS[1]
dump_path = ARGS[2]
out_path = ARGS[3]

const EVO_PREFIX = "alphafold/alphafold_iteration/evoformer"
const EVO_BLOCK_PREFIX = "alphafold/alphafold_iteration/evoformer/evoformer_iteration"
const SM_PREFIX = "alphafold/alphafold_iteration/structure_module"

@inline function _slice_block(arr::AbstractArray, block_idx::Int)
    return dropdims(view(arr, block_idx:block_idx, ntuple(_ -> Colon(), ndims(arr) - 1)...); dims=1)
end

function _load_linear_raw!(lin::LinearFirst, arrs::AbstractDict, base::AbstractString; block_idx::Union{Nothing,Int}=nothing)
    wkey = string(base, "//weights")
    bkey = string(base, "//bias")
    w = block_idx === nothing ? arrs[wkey] : _slice_block(arrs[wkey], block_idx)
    lin.weight .= permutedims(w, (2, 1))
    if lin.use_bias
        b = block_idx === nothing ? arrs[bkey] : _slice_block(arrs[bkey], block_idx)
        lin.bias .= b
    end
    return lin
end

function _load_ln_raw!(ln::LayerNormFirst, arrs::AbstractDict, base::AbstractString; block_idx::Union{Nothing,Int}=nothing)
    skey = string(base, "//scale")
    okey = string(base, "//offset")
    s = block_idx === nothing ? arrs[skey] : _slice_block(arrs[skey], block_idx)
    o = block_idx === nothing ? arrs[okey] : _slice_block(arrs[okey], block_idx)
    ln.w .= s
    ln.b .= o
    return ln
end

function _load_attention_raw!(att::AF2Attention, arrs::AbstractDict, base::AbstractString; block_idx::Union{Nothing,Int}=nothing)
    g = key -> (block_idx === nothing ? arrs[key] : _slice_block(arrs[key], block_idx))
    att.query_w .= g(string(base, "//query_w"))
    att.key_w .= g(string(base, "//key_w"))
    att.value_w .= g(string(base, "//value_w"))
    att.output_w .= g(string(base, "//output_w"))
    att.output_b .= g(string(base, "//output_b"))
    if att.gating
        att.gating_w .= g(string(base, "//gating_w"))
        att.gating_b .= g(string(base, "//gating_b"))
    end
    return att
end

function _load_evo_block_raw!(blk::EvoformerIteration, arrs::AbstractDict, bi::Int)
    p = EVO_BLOCK_PREFIX

    _load_ln_raw!(blk.outer_product_mean.layer_norm_input, arrs, string(p, "/outer_product_mean/layer_norm_input"); block_idx=bi)
    _load_linear_raw!(blk.outer_product_mean.left_projection, arrs, string(p, "/outer_product_mean/left_projection"); block_idx=bi)
    _load_linear_raw!(blk.outer_product_mean.right_projection, arrs, string(p, "/outer_product_mean/right_projection"); block_idx=bi)
    blk.outer_product_mean.output_w .= _slice_block(arrs[string(p, "/outer_product_mean//output_w")], bi)
    blk.outer_product_mean.output_b .= _slice_block(arrs[string(p, "/outer_product_mean//output_b")], bi)

    _load_ln_raw!(blk.msa_row_attention_with_pair_bias.query_norm, arrs, string(p, "/msa_row_attention_with_pair_bias/query_norm"); block_idx=bi)
    _load_ln_raw!(blk.msa_row_attention_with_pair_bias.feat_2d_norm, arrs, string(p, "/msa_row_attention_with_pair_bias/feat_2d_norm"); block_idx=bi)
    blk.msa_row_attention_with_pair_bias.feat_2d_weights .= _slice_block(arrs[string(p, "/msa_row_attention_with_pair_bias//feat_2d_weights")], bi)
    _load_attention_raw!(blk.msa_row_attention_with_pair_bias.attention, arrs, string(p, "/msa_row_attention_with_pair_bias/attention"); block_idx=bi)

    _load_ln_raw!(blk.msa_column_attention.query_norm, arrs, string(p, "/msa_column_attention/query_norm"); block_idx=bi)
    _load_attention_raw!(blk.msa_column_attention.attention, arrs, string(p, "/msa_column_attention/attention"); block_idx=bi)

    _load_ln_raw!(blk.msa_transition.input_layer_norm, arrs, string(p, "/msa_transition/input_layer_norm"); block_idx=bi)
    _load_linear_raw!(blk.msa_transition.transition1, arrs, string(p, "/msa_transition/transition1"); block_idx=bi)
    _load_linear_raw!(blk.msa_transition.transition2, arrs, string(p, "/msa_transition/transition2"); block_idx=bi)

    _load_ln_raw!(blk.triangle_multiplication_outgoing.layer_norm_input, arrs, string(p, "/triangle_multiplication_outgoing/layer_norm_input"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.left_projection, arrs, string(p, "/triangle_multiplication_outgoing/left_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.right_projection, arrs, string(p, "/triangle_multiplication_outgoing/right_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.left_gate, arrs, string(p, "/triangle_multiplication_outgoing/left_gate"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.right_gate, arrs, string(p, "/triangle_multiplication_outgoing/right_gate"); block_idx=bi)
    _load_ln_raw!(blk.triangle_multiplication_outgoing.center_layer_norm, arrs, string(p, "/triangle_multiplication_outgoing/center_layer_norm"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.output_projection, arrs, string(p, "/triangle_multiplication_outgoing/output_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_outgoing.gating_linear, arrs, string(p, "/triangle_multiplication_outgoing/gating_linear"); block_idx=bi)

    _load_ln_raw!(blk.triangle_multiplication_incoming.layer_norm_input, arrs, string(p, "/triangle_multiplication_incoming/layer_norm_input"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_incoming.left_projection, arrs, string(p, "/triangle_multiplication_incoming/left_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_incoming.right_projection, arrs, string(p, "/triangle_multiplication_incoming/right_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_incoming.left_gate, arrs, string(p, "/triangle_multiplication_incoming/left_gate"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_incoming.right_gate, arrs, string(p, "/triangle_multiplication_incoming/right_gate"); block_idx=bi)
    _load_ln_raw!(blk.triangle_multiplication_incoming.center_layer_norm, arrs, string(p, "/triangle_multiplication_incoming/center_layer_norm"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_incoming.output_projection, arrs, string(p, "/triangle_multiplication_incoming/output_projection"); block_idx=bi)
    _load_linear_raw!(blk.triangle_multiplication_incoming.gating_linear, arrs, string(p, "/triangle_multiplication_incoming/gating_linear"); block_idx=bi)

    _load_ln_raw!(blk.triangle_attention_starting_node.query_norm, arrs, string(p, "/triangle_attention_starting_node/query_norm"); block_idx=bi)
    blk.triangle_attention_starting_node.feat_2d_weights .= _slice_block(arrs[string(p, "/triangle_attention_starting_node//feat_2d_weights")], bi)
    _load_attention_raw!(blk.triangle_attention_starting_node.attention, arrs, string(p, "/triangle_attention_starting_node/attention"); block_idx=bi)

    _load_ln_raw!(blk.triangle_attention_ending_node.query_norm, arrs, string(p, "/triangle_attention_ending_node/query_norm"); block_idx=bi)
    blk.triangle_attention_ending_node.feat_2d_weights .= _slice_block(arrs[string(p, "/triangle_attention_ending_node//feat_2d_weights")], bi)
    _load_attention_raw!(blk.triangle_attention_ending_node.attention, arrs, string(p, "/triangle_attention_ending_node/attention"); block_idx=bi)

    _load_ln_raw!(blk.pair_transition.input_layer_norm, arrs, string(p, "/pair_transition/input_layer_norm"); block_idx=bi)
    _load_linear_raw!(blk.pair_transition.transition1, arrs, string(p, "/pair_transition/transition1"); block_idx=bi)
    _load_linear_raw!(blk.pair_transition.transition2, arrs, string(p, "/pair_transition/transition2"); block_idx=bi)

    return blk
end

function _load_structure_core_raw!(m::StructureModuleCore, arrs::AbstractDict)
    p = SM_PREFIX
    _load_ln_raw!(m.single_layer_norm, arrs, string(p, "/single_layer_norm"))
    _load_linear_raw!(m.initial_projection, arrs, string(p, "/initial_projection"))
    _load_ln_raw!(m.pair_layer_norm, arrs, string(p, "/pair_layer_norm"))

    _load_linear_raw!(m.fold_iteration_core.ipa.linear_q, arrs, string(p, "/fold_iteration/invariant_point_attention/q_scalar"))
    _load_linear_raw!(m.fold_iteration_core.ipa.linear_q_points.linear, arrs, string(p, "/fold_iteration/invariant_point_attention/q_point_local"))
    _load_linear_raw!(m.fold_iteration_core.ipa.linear_kv, arrs, string(p, "/fold_iteration/invariant_point_attention/kv_scalar"))
    _load_linear_raw!(m.fold_iteration_core.ipa.linear_kv_points.linear, arrs, string(p, "/fold_iteration/invariant_point_attention/kv_point_local"))
    _load_linear_raw!(m.fold_iteration_core.ipa.linear_b, arrs, string(p, "/fold_iteration/invariant_point_attention/attention_2d"))
    m.fold_iteration_core.ipa.head_weights .= arrs[string(p, "/fold_iteration/invariant_point_attention//trainable_point_weights")]
    _load_linear_raw!(m.fold_iteration_core.ipa.linear_out, arrs, string(p, "/fold_iteration/invariant_point_attention/output_projection"))
    _load_ln_raw!(m.fold_iteration_core.attention_layer_norm, arrs, string(p, "/fold_iteration/attention_layer_norm"))
    _load_ln_raw!(m.fold_iteration_core.transition_layer_norm, arrs, string(p, "/fold_iteration/transition_layer_norm"))
    _load_linear_raw!(m.fold_iteration_core.affine_update, arrs, string(p, "/fold_iteration/affine_update"))

    for i in eachindex(m.fold_iteration_core.transition_layers)
        key = i == 1 ? string(p, "/fold_iteration/transition") : string(p, "/fold_iteration/transition_", i - 1)
        _load_linear_raw!(m.fold_iteration_core.transition_layers[i], arrs, key)
    end

    for i in eachindex(m.sidechain.input_projections)
        key = i == 1 ? string(p, "/fold_iteration/rigid_sidechain/input_projection") : string(p, "/fold_iteration/rigid_sidechain/input_projection_", i - 1)
        _load_linear_raw!(m.sidechain.input_projections[i], arrs, key)
    end
    for i in eachindex(m.sidechain.resblock1)
        k1 = i == 1 ? string(p, "/fold_iteration/rigid_sidechain/resblock1") : string(p, "/fold_iteration/rigid_sidechain/resblock1_", i - 1)
        k2 = i == 1 ? string(p, "/fold_iteration/rigid_sidechain/resblock2") : string(p, "/fold_iteration/rigid_sidechain/resblock2_", i - 1)
        _load_linear_raw!(m.sidechain.resblock1[i], arrs, k1)
        _load_linear_raw!(m.sidechain.resblock2[i], arrs, k2)
    end
    _load_linear_raw!(m.sidechain.unnormalized_angles, arrs, string(p, "/fold_iteration/rigid_sidechain/unnormalized_angles"))
    return m
end

function _infer_transition_depth(arrs::AbstractDict, base::AbstractString)
    n = 0
    haskey(arrs, string(base, "//weights")) && (n += 1)
    i = 1
    while haskey(arrs, string(base, "_", i, "//weights"))
        n += 1
        i += 1
    end
    return n
end

function _ca_distance_metrics(atom37::AbstractArray, atom37_mask::AbstractArray)
    ca_idx = Alphafold2.atom_order["CA"] + 1
    L = size(atom37, 1)
    valid = atom37_mask[:, ca_idx] .> 0.5f0
    dists = Float32[]
    for i in 1:(L - 1)
        if valid[i] && valid[i + 1]
            dx = atom37[i + 1, ca_idx, 1] - atom37[i, ca_idx, 1]
            dy = atom37[i + 1, ca_idx, 2] - atom37[i, ca_idx, 2]
            dz = atom37[i + 1, ca_idx, 3] - atom37[i, ca_idx, 3]
            push!(dists, sqrt(dx * dx + dy * dy + dz * dz))
        end
    end
    if isempty(dists)
        return Dict{Symbol,Any}(:distances => Float32[], :mean => NaN32, :std => NaN32, :min => NaN32, :max => NaN32, :outlier_fraction => NaN32)
    end
    d = collect(dists)
    outlier_frac = sum((d .< 3.2f0) .| (d .> 4.4f0)) / length(d)
    return Dict{Symbol,Any}(
        :distances => d,
        :mean => Float32(mean(d)),
        :std => Float32(std(d)),
        :min => Float32(minimum(d)),
        :max => Float32(maximum(d)),
        :outlier_fraction => Float32(outlier_frac),
    )
end

function _pdb_path_from_npz(npz_path::AbstractString)
    if endswith(lowercase(npz_path), ".npz")
        return string(first(npz_path, lastindex(npz_path) - 4), ".pdb")
    end
    return string(npz_path, ".pdb")
end

function _write_pdb(path::AbstractString, atom37::AbstractArray, atom37_mask::AbstractArray, aatype::AbstractArray)
    atom_serial = 1
    open(path, "w") do io
        for i in 1:size(atom37, 1)
            aa_idx0 = Int(aatype[i])
            resname = if 0 <= aa_idx0 < length(Alphafold2.restypes)
                Alphafold2.restype_1to3[Alphafold2.restypes[aa_idx0 + 1]]
            else
                "UNK"
            end
            for a_idx in 1:length(Alphafold2.atom_types)
                atom37_mask[i, a_idx] < 0.5f0 && continue
                atom_name = Alphafold2.atom_types[a_idx]
                x = atom37[i, a_idx, 1]
                y = atom37[i, a_idx, 2]
                z = atom37[i, a_idx, 3]
                element = uppercase(first(atom_name, 1))
                line = @sprintf("ATOM  %5d %4s %3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s",
                    atom_serial, atom_name, resname, 'A', i, x, y, z, 1.00, 0.00, element)
                println(io, line)
                atom_serial += 1
            end
        end
        println(io, "END")
    end
    return atom_serial - 1
end

arrs = NPZ.npzread(params_path)
dump = NPZ.npzread(dump_path)

c_m = length(arrs[string(EVO_PREFIX, "/preprocess_1d//bias")])
c_z = length(arrs[string(EVO_PREFIX, "/left_single//bias")])
c_s = length(arrs[string(EVO_PREFIX, "/single_activations//bias")])
num_blocks = size(arrs[string(EVO_BLOCK_PREFIX, "/msa_column_attention/attention//output_b")], 1)

msa_qw = arrs[string(EVO_BLOCK_PREFIX, "/msa_column_attention/attention//query_w")]
num_head_msa = size(msa_qw, 3)
msa_head_dim = size(msa_qw, 4)
pair_qw = arrs[string(EVO_BLOCK_PREFIX, "/triangle_attention_starting_node/attention//query_w")]
num_head_pair = size(pair_qw, 3)
pair_head_dim = size(pair_qw, 4)
c_outer = size(arrs[string(EVO_BLOCK_PREFIX, "/outer_product_mean/left_projection//bias")], 2)
c_tri_mul = size(arrs[string(EVO_BLOCK_PREFIX, "/triangle_multiplication_outgoing/left_projection//bias")], 2)
msa_transition_factor = size(arrs[string(EVO_BLOCK_PREFIX, "/msa_transition/transition1//bias")], 2) / c_m
pair_transition_factor = size(arrs[string(EVO_BLOCK_PREFIX, "/pair_transition/transition1//bias")], 2) / c_z

single_activations = LinearFirst(c_m, c_s)
_load_linear_raw!(single_activations, arrs, string(EVO_PREFIX, "/single_activations"))

blocks = [
    EvoformerIteration(
        c_m,
        c_z;
        num_head_msa=num_head_msa,
        msa_head_dim=msa_head_dim,
        num_head_pair=num_head_pair,
        pair_head_dim=pair_head_dim,
        c_outer=c_outer,
        c_tri_mul=c_tri_mul,
        msa_transition_factor=msa_transition_factor,
        pair_transition_factor=pair_transition_factor,
        outer_first=false,
    )
    for _ in 1:num_blocks
]
for i in 1:num_blocks
    _load_evo_block_raw!(blocks[i], arrs, i)
end

num_transition_layers = _infer_transition_depth(arrs, string(SM_PREFIX, "/fold_iteration/transition"))
num_residual_block = _infer_transition_depth(arrs, string(SM_PREFIX, "/fold_iteration/rigid_sidechain/resblock1"))
no_heads = length(arrs[string(SM_PREFIX, "/fold_iteration/invariant_point_attention/attention_2d//bias")])
c_hidden_total = size(arrs[string(SM_PREFIX, "/fold_iteration/invariant_point_attention/q_scalar//weights")], 2)
c_hidden = Int(div(c_hidden_total, no_heads))
q_point_total = size(arrs[string(SM_PREFIX, "/fold_iteration/invariant_point_attention/q_point_local//bias")], 1)
kv_point_total = size(arrs[string(SM_PREFIX, "/fold_iteration/invariant_point_attention/kv_point_local//bias")], 1)
no_qk_points = Int(div(q_point_total, 3 * no_heads))
no_v_points = Int(div(kv_point_total, 3 * no_heads) - no_qk_points)
sidechain_num_channel = length(arrs[string(SM_PREFIX, "/fold_iteration/rigid_sidechain/input_projection//bias")])

structure = StructureModuleCore(
    c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points,
    num_transition_layers, 8, 10f0, sidechain_num_channel, num_residual_block,
)
_load_structure_core_raw!(structure, arrs)

aatype = vec(Int.(dump["aatype"]))
seq_mask = Float32.(dump["seq_mask"])
pre_msa = Float32.(dump["pre_msa"])        # (I, S, L, C)
pre_pair = Float32.(dump["pre_pair"])      # (I, L, L, C)
pre_msa_mask = Float32.(dump["pre_msa_mask"]) # (I, S, L)
py_out_single = Float32.(dump["out_single"])  # (I, L, C_s)
py_out_pair = Float32.(dump["out_pair"])      # (I, L, L, C_z)
py_out_atom14 = Float32.(dump["out_atom14"])  # (I, L, 14, 3)
py_out_affine = Float32.(dump["out_affine"])  # (I, L, 7)
num_iter = size(pre_msa, 1)

pair_mask = reshape(seq_mask, :, 1, 1) .* reshape(seq_mask, 1, :, 1) # (L, L, 1)

single_max = zeros(Float32, num_iter)
pair_max = zeros(Float32, num_iter)
atom14_max = zeros(Float32, num_iter)
affine_max = zeros(Float32, num_iter)

final_atom37_py = nothing
final_mask_py = nothing

for i in 1:num_iter
    msa_i = view(pre_msa, i, :, :, :)         # (S, L, C)
    pair_i = view(pre_pair, i, :, :, :)       # (L, L, C)
    mask_i = view(pre_msa_mask, i, :, :)      # (S, L)

    msa_act = reshape(permutedims(msa_i, (3, 1, 2)), size(msa_i, 3), size(msa_i, 1), size(msa_i, 2), 1) # (C,S,L,1)
    pair_act = reshape(permutedims(pair_i, (3, 1, 2)), size(pair_i, 3), size(pair_i, 1), size(pair_i, 2), 1) # (C,L,L,1)
    msa_mask = reshape(mask_i, size(mask_i, 1), size(mask_i, 2), 1)

    for blk in blocks
        msa_act, pair_act = blk(msa_act, pair_act, msa_mask, pair_mask)
    end

    single = single_activations(view(msa_act, :, 1, :, :))
    struct_out = structure(single, pair_act, reshape(seq_mask, :, 1), reshape(aatype, :, 1))
    final_atom14 = dropdims(view(struct_out[:atom_pos], size(struct_out[:atom_pos], 1):size(struct_out[:atom_pos], 1), :, :, :, :); dims=1)
    final_affine = dropdims(view(struct_out[:affine], size(struct_out[:affine], 1):size(struct_out[:affine], 1), :, :, :); dims=1)

    protein = Dict{Symbol,Any}(:aatype => reshape(aatype, :, 1), :s_s => single)
    make_atom14_masks!(protein)
    final_atom37 = atom14_to_atom37(final_atom14, protein)

    single_py = dropdims(first_to_af2_3d(single); dims=1)
    pair_py = dropdims(first_to_af2_2d(pair_act); dims=1)
    atom14_py = dropdims(permutedims(final_atom14, (3, 2, 1, 4)); dims=4)
    affine_py = dropdims(permutedims(final_affine, (2, 1, 3)); dims=3)

    single_max[i] = maximum(abs.(single_py .- view(py_out_single, i, :, :)))
    pair_max[i] = maximum(abs.(pair_py .- view(py_out_pair, i, :, :, :)))
    atom14_max[i] = maximum(abs.(atom14_py .- view(py_out_atom14, i, :, :, :)))
    affine_max[i] = maximum(abs.(affine_py .- view(py_out_affine, i, :, :)))

    if i == num_iter
        global final_atom37_py = dropdims(final_atom37; dims=1)
        global final_mask_py = dropdims(permutedims(protein[:atom37_atom_exists], (2, 3, 1)); dims=2)
    end
end

for i in 1:num_iter
    @printf("Iter %d parity vs Python dump\n", i - 1)
    @printf("  single_max_abs: %.8g\n", single_max[i])
    @printf("  pair_max_abs:   %.8g\n", pair_max[i])
    @printf("  atom14_max_abs: %.8g\n", atom14_max[i])
    @printf("  affine_max_abs: %.8g\n", affine_max[i])
end

ca = _ca_distance_metrics(final_atom37_py, final_mask_py)
@printf("Geometry check (consecutive C-alpha distances)\n")
@printf("  mean: %.6f A\n", ca[:mean])
@printf("  std:  %.6f A\n", ca[:std])
@printf("  min:  %.6f A\n", ca[:min])
@printf("  max:  %.6f A\n", ca[:max])
@printf("  outlier_fraction: %.3f\n", ca[:outlier_fraction])

pdb_path = _pdb_path_from_npz(out_path)
pdb_atoms = _write_pdb(pdb_path, final_atom37_py, final_mask_py, aatype)
println("Wrote PDB: ", pdb_path, " (atoms=", pdb_atoms, ")")

out = Dict{String,Any}(
    "single_max_abs_per_iter" => single_max,
    "pair_max_abs_per_iter" => pair_max,
    "atom14_max_abs_per_iter" => atom14_max,
    "affine_max_abs_per_iter" => affine_max,
    "out_atom37" => final_atom37_py,
    "atom37_mask" => final_mask_py,
    "ca_consecutive_distances" => ca[:distances],
    "ca_distance_mean" => ca[:mean],
    "ca_distance_std" => ca[:std],
    "ca_distance_min" => ca[:min],
    "ca_distance_max" => ca[:max],
    "ca_distance_outlier_fraction" => ca[:outlier_fraction],
)

mkpath(dirname(out_path))
NPZ.npzwrite(out_path, out)
println("Saved Julia pre-evo parity run to ", out_path)
