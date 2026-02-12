# Inference loop and feature preprocessing.
# Decomposed into composable pipeline stages for research use.

using Printf
using Statistics

# ── Named constants for feature dimensions ───────────────────────────────────

const MSA_FEATURE_DIM = 49          # MSA feature channels (23 one-hot + 1 has_del + 1 del_val + 23 profile + 1 del_mean)
const EXTRA_MSA_FEATURE_DIM = 25    # Extra MSA channels (23 one-hot + 1 has_del + 1 del_val)
const DGRAM_NUM_BINS = 15           # Distance gram bins for recycling
const NUM_ATOM_TYPES = 37           # Number of atom types in atom37 representation
const AATYPE_ONE_HOT_DIM = 22       # One-hot encoding dimension for amino acids (21 + unknown)
const NUM_STD_RESIDUES = 21         # Standard amino acid types (20 + unknown gap)
const RESIDUE_ONE_HOT_DIM = 23      # Residue one-hot classes (22 + gap)

# ── Feature building helpers ─────────────────────────────────────────────────

function _one_hot_aatype(aatype::AbstractVector{Int}, num_classes::Int)
    L = length(aatype)
    out = zeros(Float32, num_classes, L, 1)
    for i in 1:L
        idx = clamp(aatype[i], 0, num_classes - 1)
        out[idx + 1, i, 1] = 1f0
    end
    return out
end

function _build_target_and_msa_feat(aatype::AbstractVector{Int}; target_dim::Int=AATYPE_ONE_HOT_DIM)
    L = length(aatype)
    target_feat = if target_dim == AATYPE_ONE_HOT_DIM
        cat(zeros(Float32, 1, L, 1), _one_hot_aatype(aatype, NUM_STD_RESIDUES); dims=1)
    elseif target_dim == NUM_STD_RESIDUES
        _one_hot_aatype(aatype, NUM_STD_RESIDUES)
    else
        error("Unsupported target feature dim $(target_dim); expected $(NUM_STD_RESIDUES) or $(AATYPE_ONE_HOT_DIM)")
    end

    msa_1hot = zeros(Float32, RESIDUE_ONE_HOT_DIM, 1, L, 1)
    for i in 1:L
        idx = clamp(aatype[i], 0, RESIDUE_ONE_HOT_DIM - 1)
        msa_1hot[idx + 1, 1, i, 1] = 1f0
    end
    msa_feat = cat(msa_1hot, zeros(Float32, 1, 1, L, 1), zeros(Float32, 1, 1, L, 1), copy(msa_1hot), zeros(Float32, 1, 1, L, 1); dims=1)
    residue_index = collect(0:(L - 1))
    return target_feat, msa_feat, residue_index
end

function _deletion_value_transform(x::AbstractArray)
    return atan.(Float32.(x) ./ 3f0) .* (2f0 / Float32(π))
end

function _build_target_and_msa_feat(
    aatype::AbstractVector{Int},
    msa_int::AbstractArray,
    deletion_matrix::Union{Nothing,AbstractArray}=nothing,
    ;
    target_dim::Int=AATYPE_ONE_HOT_DIM,
)
    L = length(aatype)
    S = size(msa_int, 1)
    size(msa_int, 2) == L || error("msa length mismatch: expected L=$(L), got $(size(msa_int, 2))")

    target_feat = if target_dim == AATYPE_ONE_HOT_DIM
        cat(zeros(Float32, 1, L, 1), _one_hot_aatype(aatype, NUM_STD_RESIDUES); dims=1)
    elseif target_dim == NUM_STD_RESIDUES
        _one_hot_aatype(aatype, NUM_STD_RESIDUES)
    else
        error("Unsupported target feature dim $(target_dim); expected $(NUM_STD_RESIDUES) or $(AATYPE_ONE_HOT_DIM)")
    end

    msa_1hot = zeros(Float32, RESIDUE_ONE_HOT_DIM, S, L, 1)
    for s in 1:S, i in 1:L
        idx = clamp(Int(msa_int[s, i]), 0, RESIDUE_ONE_HOT_DIM - 1)
        msa_1hot[idx + 1, s, i, 1] = 1f0
    end

    del = deletion_matrix === nothing ? zeros(Float32, S, L) : Float32.(deletion_matrix)
    has_del = reshape(Float32.(del .> 0f0), 1, S, L, 1)
    del_val = reshape(_deletion_value_transform(del), 1, S, L, 1)
    del_mean = reshape(_deletion_value_transform(del), 1, S, L, 1)

    msa_feat = cat(msa_1hot, has_del, del_val, copy(msa_1hot), del_mean; dims=1)
    residue_index = collect(0:(L - 1))
    return target_feat, msa_feat, residue_index
end

function _build_extra_msa_feat(aatype::AbstractVector{Int})
    L = length(aatype)
    msa_1hot = zeros(Float32, RESIDUE_ONE_HOT_DIM, 1, L, 1)
    for i in 1:L
        idx = clamp(aatype[i], 0, RESIDUE_ONE_HOT_DIM - 1)
        msa_1hot[idx + 1, 1, i, 1] = 1f0
    end
    return cat(msa_1hot, zeros(Float32, 1, 1, L, 1), zeros(Float32, 1, 1, L, 1); dims=1)
end

function _build_extra_msa_feat(msa_int::AbstractArray, deletion_matrix::Union{Nothing,AbstractArray}=nothing)
    S = size(msa_int, 1)
    L = size(msa_int, 2)
    msa_1hot = zeros(Float32, RESIDUE_ONE_HOT_DIM, S, L, 1)
    for s in 1:S, i in 1:L
        idx = clamp(Int(msa_int[s, i]), 0, RESIDUE_ONE_HOT_DIM - 1)
        msa_1hot[idx + 1, s, i, 1] = 1f0
    end
    del = deletion_matrix === nothing ? zeros(Float32, S, L) : Float32.(deletion_matrix)
    has_del = reshape(Float32.(del .> 0f0), 1, S, L, 1)
    del_val = reshape(_deletion_value_transform(del), 1, S, L, 1)
    return cat(msa_1hot, has_del, del_val; dims=1)
end

function _sample_msa_rows_deterministic(
    msa_int::AbstractMatrix{<:Integer},
    deletion_matrix::Union{Nothing,AbstractMatrix},
    msa_mask::Union{Nothing,AbstractMatrix},
    num_msa_keep::Int,
    num_extra_keep::Int,
)
    S, L = size(msa_int)
    S == 0 && error("msa must have at least one row")
    num_msa_keep = clamp(num_msa_keep, 1, S)
    mask = msa_mask === nothing ? ones(Float32, S, L) : Float32.(msa_mask)

    nonempty = [sum(mask[i, :]) > 0f0 for i in 1:S]
    sel = sizehint!(Int[1], num_msa_keep)
    for i in 2:S
        length(sel) >= num_msa_keep && break
        nonempty[i] && push!(sel, i)
    end
    for i in 2:S
        length(sel) >= num_msa_keep && break
        !nonempty[i] && push!(sel, i)
    end
    length(sel) == num_msa_keep || error("Failed to select requested msa rows")

    sel_set = Set(sel)
    extra = [i for i in 1:S if !(i in sel_set)]
    if isempty(extra)
        extra = copy(sel)
    end
    if length(extra) > num_extra_keep
        extra = extra[1:num_extra_keep]
    end

    del = deletion_matrix === nothing ? zeros(Float32, S, L) : Float32.(deletion_matrix)
    return (
        Int.(msa_int[sel, :]),
        Float32.(del[sel, :]),
        Float32.(mask[sel, :]),
        Int.(msa_int[extra, :]),
        Float32.(del[extra, :]),
        Float32.(mask[extra, :]),
    )
end

function _relpos_one_hot(residue_index::AbstractVector{Int}, max_relative_feature::Int)
    L = length(residue_index)
    out = zeros(Float32, 2 * max_relative_feature + 1, L, L, 1)
    for i in 1:L, j in 1:L
        idx = clamp((residue_index[i] - residue_index[j]) + max_relative_feature, 0, 2 * max_relative_feature)
        out[idx + 1, i, j, 1] = 1f0
    end
    return out
end

function _multimer_relpos_features(
    residue_index::AbstractVector{Int},
    asym_id::AbstractVector{Int},
    entity_id::AbstractVector{Int},
    sym_id::AbstractVector{Int},
    max_relative_idx::Int,
    max_relative_chain::Int,
)
    L = length(residue_index)
    rel_pos_dim = 2 * max_relative_idx + 2
    rel_chain_dim = 2 * max_relative_chain + 2
    out_dim = rel_pos_dim + 1 + rel_chain_dim
    out = zeros(Float32, out_dim, L, L, 1)
    for i in 1:L, j in 1:L
        asym_same = asym_id[i] == asym_id[j]
        entity_same = entity_id[i] == entity_id[j]

        off = residue_index[i] - residue_index[j]
        clipped = clamp(off + max_relative_idx, 0, 2 * max_relative_idx)
        relpos_idx = asym_same ? clipped : (2 * max_relative_idx + 1)
        out[relpos_idx + 1, i, j, 1] = 1f0

        out[rel_pos_dim + 1, i, j, 1] = entity_same ? 1f0 : 0f0

        rel_sym = sym_id[i] - sym_id[j]
        clipped_chain = clamp(rel_sym + max_relative_chain, 0, 2 * max_relative_chain)
        relchain_idx = entity_same ? clipped_chain : (2 * max_relative_chain + 1)
        out[rel_pos_dim + 1 + relchain_idx + 1, i, j, 1] = 1f0
    end
    return out
end

function _pseudo_beta_from_atom37(aatype::AbstractVector{Int}, atom37_bllc::AbstractArray)
    ca_idx = atom_order["CA"] + 1
    cb_idx = atom_order["CB"] + 1
    gly_idx = restype_order["G"]
    L = length(aatype)
    out = zeros(Float32, 3, L, 1)
    for i in 1:L
        atom_idx = aatype[i] == gly_idx ? ca_idx : cb_idx
        out[:, i, 1] .= atom37_bllc[1, i, atom_idx, :]
    end
    return out
end

function _dgram_from_positions(positions::AbstractArray; num_bins::Int=DGRAM_NUM_BINS, min_bin::Real=3.25f0, max_bin::Real=20.75f0)
    L = size(positions, 2)
    out = zeros(Float32, num_bins, L, L, 1)
    lower_breaks = range(Float32(min_bin), Float32(max_bin); length=num_bins)
    lower2 = lower_breaks .^ 2
    upper2 = vcat(lower2[2:end], Float32[1f8])
    p = permutedims(view(positions, :, :, 1), (2, 1))
    diff = reshape(p, L, 1, 3) .- reshape(p, 1, L, 3)
    d2 = dropdims(sum(diff .^ 2; dims=3); dims=3)
    for k in 1:num_bins
        out[k, :, :, 1] .= (d2 .> lower2[k]) .& (d2 .< upper2[k])
    end
    return out
end

# ── Geometry and output helpers ──────────────────────────────────────────────

"""Cα distance statistics between consecutive residues."""
struct CAMetrics
    count::Int
    mean::Float32
    std::Float32
    min::Float32
    max::Float32
    outlier_fraction::Float32
end

function _ca_distance_metrics(
    atom37::AbstractArray,
    atom37_mask::AbstractArray;
    asym_id::Union{Nothing,AbstractVector}=nothing,
    intra_chain_only::Bool=false,
)
    ca_idx = atom_order["CA"] + 1
    L = size(atom37, 1)
    valid = atom37_mask[:, ca_idx] .> 0.5f0
    d = sizehint!(Float32[], L - 1)
    for i in 1:(L - 1)
        if intra_chain_only && asym_id !== nothing && Int(asym_id[i]) != Int(asym_id[i + 1])
            continue
        end
        if valid[i] && valid[i + 1]
            v = atom37[i + 1, ca_idx, :] .- atom37[i, ca_idx, :]
            push!(d, sqrt(sum(v .^ 2)))
        end
    end
    if isempty(d)
        return CAMetrics(0, NaN32, NaN32, NaN32, NaN32, NaN32)
    end
    return CAMetrics(
        length(d),
        Float32(mean(d)),
        Float32(std(d; corrected=false)),
        Float32(minimum(d)),
        Float32(maximum(d)),
        Float32(sum((d .< 3.2f0) .| (d .> 4.4f0)) / length(d)),
    )
end

function _infer_pdb_path_from_npz(npz_path::AbstractString)
    if endswith(lowercase(npz_path), ".npz")
        return string(first(npz_path, lastindex(npz_path) - 4), ".pdb")
    end
    return string(npz_path, ".pdb")
end

function _write_pdb(
    path::AbstractString,
    atom37::AbstractArray,
    atom37_mask::AbstractArray,
    aatype::AbstractArray;
    bfactor_by_res::Union{Nothing,AbstractVector}=nothing,
    asym_id::Union{Nothing,AbstractVector}=nothing,
    residue_index::Union{Nothing,AbstractVector}=nothing,
)
    chain_alphabet = collect("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
    chain_char_for(i::Int) = chain_alphabet[clamp(i, 1, length(chain_alphabet))]

    function _chain_char(res_i::Int)
        if asym_id === nothing
            return 'A'
        end
        ai = Int(asym_id[res_i])
        return chain_char_for(ai <= 0 ? 1 : ai)
    end

    function _resseq(res_i::Int)
        if residue_index === nothing
            return res_i
        end
        return Int(residue_index[res_i]) + 1
    end

    atom_serial = 1
    open(path, "w") do io
        for i in 1:size(atom37, 1)
            aa_idx0 = Int(aatype[i])
            resname = if 0 <= aa_idx0 < length(restypes)
                restype_1to3[restypes[aa_idx0 + 1]]
            else
                "UNK"
            end
            chain_id = _chain_char(i)
            resseq = _resseq(i)
            for a_idx in 1:length(atom_types)
                atom37_mask[i, a_idx] < 0.5f0 && continue
                atom_name = atom_types[a_idx]
                x = atom37[i, a_idx, 1]
                y = atom37[i, a_idx, 2]
                z = atom37[i, a_idx, 3]
                bfactor = bfactor_by_res === nothing ? 0.0 : Float64(bfactor_by_res[i])
                element = uppercase(first(atom_name, 1))
                line = @sprintf("ATOM  %5d %4s %3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s",
                    atom_serial, atom_name, resname, chain_id, resseq, x, y, z, 1.00, bfactor, element)
                println(io, line)
                atom_serial += 1
            end
            if i < size(atom37, 1) && _chain_char(i + 1) != chain_id
                println(io, @sprintf("TER   %5d      %3s %c%4d", atom_serial, resname, chain_id, resseq))
                atom_serial += 1
            end
        end
        println(io, "END")
    end
    return atom_serial - 1
end

# ── Result struct ────────────────────────────────────────────────────────────

"""
    AF2InferenceResult

Raw inference output from `_infer()` or `run_inference()`.

# Fields
- `atom37`: `(L, 37, 3)` predicted atom coordinates
- `atom37_mask`: `(L, 37)` atom existence mask
- `plddt`: `(L,)` per-residue pLDDT confidence
- `aatype`: `(L,)` integer amino acid types
- `residue_index`: `(L,)` residue indices
- `asym_id`: `(L,)` chain IDs
- `pae`: predicted aligned error matrix (multimer only)
- `ptm`: predicted TM-score (multimer only)
"""
Base.@kwdef struct AF2InferenceResult
    atom37::Array{Float32,3}
    atom37_mask::Array{Float32,2}
    plddt::Vector{Float32}
    masked_msa_logits::Array{Float32}
    distogram_logits::Array{Float32}
    distogram_bin_edges::Vector{Float32}
    experimentally_resolved_logits::Array{Float32}
    lddt_logits::Array{Float32}
    pae::Union{Nothing,Array{Float32}}=nothing
    pae_max::Union{Nothing,Float32}=nothing
    ptm::Union{Nothing,Float32}=nothing
    aatype::Vector{Int}
    residue_index::Vector{Int}
    asym_id::Vector{Int}
    ca_metrics::CAMetrics
    ca_metrics_intra::CAMetrics
end

# ── Output writers ───────────────────────────────────────────────────────────

function _write_fold_pdb(path::AbstractString, result::AF2InferenceResult; bfactor=result.plddt)
    return _write_pdb(
        path,
        result.atom37,
        result.atom37_mask,
        result.aatype;
        bfactor_by_res=bfactor,
        asym_id=result.asym_id,
        residue_index=result.residue_index,
    )
end

function _write_fold_npz(path::AbstractString, result::AF2InferenceResult)
    out_npz = Dict{String,Any}(
        "out_atom37" => result.atom37,
        "atom37_mask" => result.atom37_mask,
        "out_masked_msa_logits" => result.masked_msa_logits,
        "out_distogram_logits" => result.distogram_logits,
        "out_distogram_bin_edges" => result.distogram_bin_edges,
        "out_experimentally_resolved_logits" => result.experimentally_resolved_logits,
        "out_predicted_lddt_logits" => result.lddt_logits,
        "out_plddt" => result.plddt,
        "mean_plddt" => Float32(mean(result.plddt)),
        "min_plddt" => Float32(minimum(result.plddt)),
        "max_plddt" => Float32(maximum(result.plddt)),
        "ca_distance_mean" => result.ca_metrics.mean,
        "ca_distance_std" => result.ca_metrics.std,
        "ca_distance_min" => result.ca_metrics.min,
        "ca_distance_max" => result.ca_metrics.max,
        "ca_distance_outlier_fraction" => result.ca_metrics.outlier_fraction,
        "ca_distance_intra_chain_mean" => result.ca_metrics_intra.mean,
        "ca_distance_intra_chain_std" => result.ca_metrics_intra.std,
        "ca_distance_intra_chain_min" => result.ca_metrics_intra.min,
        "ca_distance_intra_chain_max" => result.ca_metrics_intra.max,
        "ca_distance_intra_chain_outlier_fraction" => result.ca_metrics_intra.outlier_fraction,
    )
    if result.pae !== nothing
        out_npz["predicted_aligned_error"] = result.pae
        out_npz["max_predicted_aligned_error"] = result.pae_max
        out_npz["predicted_tm_score"] = result.ptm
        out_npz["mean_predicted_aligned_error"] = Float32(mean(result.pae))
    end
    NPZ.npzwrite(path, out_npz)
    return path
end

# ── Composable pipeline types ────────────────────────────────────────────────

"""Preprocessed inputs ready for the evoformer. All device tensors on correct device."""
Base.@kwdef struct AF2PreparedInputs{
    A1<:AbstractArray, A2<:AbstractArray, A3<:AbstractArray,
    A4<:AbstractArray, A5<:AbstractArray, A6<:AbstractArray,
    M1<:Union{Nothing,AbstractMatrix}, M2<:Union{Nothing,AbstractMatrix},
    T1<:AbstractArray, T2<:AbstractArray, T3<:AbstractVector,
    T4<:AbstractMatrix, T5<:AbstractArray, T6<:AbstractArray,
}
    # Core features (on device)
    target_feat::A1
    msa_feat::A2
    extra_msa_feat::A3
    pair_mask::A4
    seq_mask::A5
    multichain_mask::A6

    # Masks (CPU, nothing if using defaults)
    msa_mask_input::M1
    extra_msa_mask_input::M2

    # Template data (on device)
    template_rows_first::T1
    template_row_mask::T2
    template_mask::T3

    # Template raw data (CPU, for template embedding)
    template_aatype::T4
    template_all_atom_positions::T5
    template_all_atom_masks::T6

    # Metadata (CPU integers)
    aatype::Vector{Int}
    residue_index::Vector{Int}
    asym_id::Vector{Int}
    entity_id::Vector{Int}
    sym_id::Vector{Int}

    # Config
    num_iters::Int
    use_gpu::Bool
end

"""State carried between recycle iterations."""
Base.@kwdef struct RecycleState{A1<:AbstractArray, A2<:AbstractArray, A3<:AbstractArray}
    msa_first_row::A1   # (c_m, L, 1) — on device
    pair::A2             # (c_z, L, L, 1) — on device
    atom37::A3           # (1, L, NUM_ATOM_TYPES, 3) — CPU
end

# ── Pipeline stage: prepare_inputs ───────────────────────────────────────────

"""
    prepare_inputs(model, dump; num_recycle) → AF2PreparedInputs

Parse a raw feature Dict into typed, device-placed tensors ready for the evoformer.
This is the non-differentiable preprocessing step.
"""
function prepare_inputs(
    model::AF2Model,
    dump::AbstractDict;
    num_recycle::Union{Nothing,Int}=nothing,
)::AF2PreparedInputs
    config = model.config
    use_gpu = _model_on_gpu(model)
    c_m = config.c_m
    c_z = config.c_z

    # ── Parse input features ─────────────────────────────────────────────
    aatype = vec(Int.(dump["aatype"]))
    seq_mask = Float32.(vec(dump["seq_mask"]))
    asym_id = haskey(dump, "asym_id") ? vec(Int.(dump["asym_id"])) : ones(Int, length(aatype))
    entity_id = haskey(dump, "entity_id") ? vec(Int.(dump["entity_id"])) : ones(Int, length(aatype))
    sym_id = haskey(dump, "sym_id") ? vec(Int.(dump["sym_id"])) : ones(Int, length(aatype))

    msa_mask_input = if haskey(dump, "msa_mask_model")
        x = Float32.(dump["msa_mask_model"])
        ndims(x) == 1 ? reshape(x, 1, :) : x
    elseif haskey(dump, "msa_mask")
        x = Float32.(dump["msa_mask"])
        ndims(x) == 1 ? reshape(x, 1, :) : x
    else
        nothing
    end
    extra_msa_mask_input = if haskey(dump, "extra_msa_mask")
        x = Float32.(dump["extra_msa_mask"])
        ndims(x) == 1 ? reshape(x, 1, :) : x
    else
        msa_mask_input
    end

    msa_int = if haskey(dump, "msa")
        x = Int.(dump["msa"])
        ndims(x) == 1 ? reshape(x, 1, :) : x
    else
        nothing
    end
    deletion_matrix = if haskey(dump, "deletion_matrix")
        x = Float32.(dump["deletion_matrix"])
        ndims(x) == 1 ? reshape(x, 1, :) : x
    else
        nothing
    end

    target_feat_override = haskey(dump, "target_feat") ? Float32.(dump["target_feat"]) : nothing
    msa_feat_override = haskey(dump, "msa_feat") ? Float32.(dump["msa_feat"]) : nothing

    extra_msa_int = if haskey(dump, "extra_msa")
        x = Int.(dump["extra_msa"])
        ndims(x) == 1 ? reshape(x, 1, :) : x
    else
        msa_int
    end
    extra_deletion_matrix = if haskey(dump, "extra_deletion_matrix")
        x = Float32.(dump["extra_deletion_matrix"])
        ndims(x) == 1 ? reshape(x, 1, :) : x
    elseif haskey(dump, "extra_deletion_value")
        x = Float32.(dump["extra_deletion_value"])
        x = ndims(x) == 1 ? reshape(x, 1, :) : x
        tan.(x .* (Float32(π) / 2f0)) .* 3f0
    else
        deletion_matrix
    end

    # Match multimer sampling regime
    if config.is_multimer_checkpoint && msa_int !== nothing && target_feat_override === nothing && msa_feat_override === nothing
        num_msa_keep = parse(Int, get(ENV, "AF2_NUM_MSA", "64"))
        num_extra_keep = parse(Int, get(ENV, "AF2_NUM_EXTRA_MSA", "128"))
        msa_int, deletion_matrix, msa_mask_input, extra_msa_int, extra_deletion_matrix, extra_msa_mask_input =
            _sample_msa_rows_deterministic(msa_int, deletion_matrix, msa_mask_input, num_msa_keep, num_extra_keep)
    end

    # ── Build features ───────────────────────────────────────────────────
    preprocess_1d_in_dim = config.preprocess_1d_in_dim
    target_feat, msa_feat, residue_index = if target_feat_override !== nothing && msa_feat_override !== nothing
        tf = target_feat_override
        if ndims(tf) == 3 && size(tf, 1) == 1
            tf = dropdims(tf; dims=1)
        end
        ndims(tf) == 2 || error("target_feat override must be rank-2 (L,C) or rank-3 with leading singleton.")
        size(tf, 2) == preprocess_1d_in_dim || error("target_feat dim mismatch: expected $(preprocess_1d_in_dim), got $(size(tf, 2))")
        Ltf = size(tf, 1)
        tf_first = reshape(permutedims(tf, (2, 1)), preprocess_1d_in_dim, Ltf, 1)

        mf = msa_feat_override
        if ndims(mf) == 4 && size(mf, 1) == 1
            mf = dropdims(mf; dims=1)
        end
        ndims(mf) == 3 || error("msa_feat override must be rank-3 (S,L,$(MSA_FEATURE_DIM)) or rank-4 with leading singleton.")
        mf_first = if size(mf, 3) == MSA_FEATURE_DIM
            reshape(permutedims(mf, (3, 1, 2)), MSA_FEATURE_DIM, size(mf, 1), size(mf, 2), 1)
        elseif size(mf, 1) == MSA_FEATURE_DIM
            reshape(mf, MSA_FEATURE_DIM, size(mf, 2), size(mf, 3), 1)
        else
            error("msa_feat override must have $(MSA_FEATURE_DIM) channels in last or first dimension.")
        end
        ridx = haskey(dump, "residue_index") ? vec(Int.(dump["residue_index"])) : collect(0:(Ltf - 1))
        tf_first, mf_first, ridx
    elseif msa_int === nothing
        _build_target_and_msa_feat(aatype; target_dim=preprocess_1d_in_dim)
    else
        _build_target_and_msa_feat(aatype, msa_int, deletion_matrix; target_dim=preprocess_1d_in_dim)
    end
    extra_msa_feat = if extra_msa_int === nothing
        _build_extra_msa_feat(aatype)
    else
        _build_extra_msa_feat(extra_msa_int, extra_deletion_matrix)
    end
    if extra_msa_mask_input !== nothing
        size(extra_msa_mask_input, 1) == size(extra_msa_feat, 2) || error("extra_msa_mask row count mismatch")
        size(extra_msa_mask_input, 2) == size(extra_msa_feat, 3) || error("extra_msa_mask length mismatch")
        extra_msa_feat .*= reshape(Float32.(extra_msa_mask_input), 1, size(extra_msa_mask_input, 1), size(extra_msa_mask_input, 2), 1)
    end
    if haskey(dump, "residue_index")
        residue_index = vec(Int.(dump["residue_index"]))
    end

    L = length(aatype)

    # ── Template processing ──────────────────────────────────────────────
    has_template_masks_key = haskey(dump, "template_all_atom_masks") || haskey(dump, "template_all_atom_mask")
    has_template_raw = haskey(dump, "template_aatype") &&
                       haskey(dump, "template_all_atom_positions") &&
                       has_template_masks_key
    if model.template_embedding !== nothing && !has_template_raw
        @printf("Warning: template weights are present but input dump has no template atom inputs; proceeding with zero templates.\n")
    end

    template_aatype = has_template_raw ? Int.(dump["template_aatype"]) : zeros(Int, 0, L)
    template_all_atom_positions = has_template_raw ? Float32.(dump["template_all_atom_positions"]) : zeros(Float32, 0, L, NUM_ATOM_TYPES, 3)
    template_all_atom_masks = if has_template_raw
        Float32.(haskey(dump, "template_all_atom_masks") ? dump["template_all_atom_masks"] : dump["template_all_atom_mask"])
    else
        zeros(Float32, 0, L, NUM_ATOM_TYPES)
    end
    template_placeholder_for_undefined = if haskey(dump, "template_placeholder_for_undefined")
        v = dump["template_placeholder_for_undefined"]
        Int(v isa AbstractArray ? v[] : v) != 0
    else
        false
    end

    template_rows_first, template_row_mask, template_mask = if has_template_raw && model.template_embedding !== nothing
        ts_layer = use_gpu ? Flux.cpu(model.template_single) : model.template_single
        rows, row_mask = ts_layer(
            template_aatype,
            template_all_atom_positions,
            template_all_atom_masks;
            placeholder_for_undefined=template_placeholder_for_undefined,
        )
        template_mask_input = if haskey(dump, "template_mask")
            tm = Float32.(vec(dump["template_mask"]))
            length(tm) == size(template_aatype, 1) || error("template_mask length mismatch")
            tm
        else
            ones(Float32, size(template_aatype, 1))
        end
        rows, row_mask, template_mask_input
    else
        zeros(Float32, c_m, 0, L, 1), zeros(Float32, 0, L), zeros(Float32, 0)
    end

    # ── Determine number of recycle iterations ───────────────────────────
    num_iters = if num_recycle !== nothing
        num_recycle + 1
    elseif haskey(dump, "num_recycle")
        Int(dump["num_recycle"]) + 1
    else
        config.kind == :multimer ? 6 : 4
    end

    pair_mask = reshape(seq_mask, L, 1, 1) .* reshape(seq_mask, 1, L, 1)
    multichain_mask = reshape(Float32.(reshape(asym_id, L, 1) .== reshape(asym_id, 1, L)), L, L, 1)

    # ── Move input tensors to GPU if needed ──────────────────────────────
    if use_gpu
        target_feat = CuArray(target_feat)
        msa_feat = CuArray(msa_feat)
        extra_msa_feat = CuArray(extra_msa_feat)
        pair_mask = CuArray(pair_mask)
        seq_mask = CuArray(seq_mask)
        multichain_mask = CuArray(multichain_mask)
        template_rows_first = CuArray(template_rows_first)
        template_row_mask = CuArray(template_row_mask)
    end

    return AF2PreparedInputs(
        target_feat=target_feat,
        msa_feat=msa_feat,
        extra_msa_feat=extra_msa_feat,
        pair_mask=pair_mask,
        seq_mask=seq_mask,
        multichain_mask=multichain_mask,
        msa_mask_input=msa_mask_input,
        extra_msa_mask_input=extra_msa_mask_input,
        template_rows_first=template_rows_first,
        template_row_mask=template_row_mask,
        template_mask=template_mask,
        template_aatype=template_aatype,
        template_all_atom_positions=template_all_atom_positions,
        template_all_atom_masks=template_all_atom_masks,
        aatype=aatype,
        residue_index=residue_index,
        asym_id=asym_id,
        entity_id=entity_id,
        sym_id=sym_id,
        num_iters=num_iters,
        use_gpu=use_gpu,
    )
end

# ── Pipeline stage: initial_recycle_state ────────────────────────────────────

"""
    initial_recycle_state(model, inputs) → RecycleState

Create zero initial state for the recycle loop.
"""
function initial_recycle_state(model::AF2Model, inputs::AF2PreparedInputs)::RecycleState
    c_m = model.config.c_m
    c_z = model.config.c_z
    L = length(inputs.aatype)
    prev_msa = inputs.use_gpu ? CUDA.zeros(Float32, c_m, L, 1) : zeros(Float32, c_m, L, 1)
    prev_pair = inputs.use_gpu ? CUDA.zeros(Float32, c_z, L, L, 1) : zeros(Float32, c_z, L, L, 1)
    prev_atom37 = zeros(Float32, 1, L, NUM_ATOM_TYPES, 3)
    return RecycleState(msa_first_row=prev_msa, pair=prev_pair, atom37=prev_atom37)
end

# ── Pipeline stage: run_evoformer ────────────────────────────────────────────

"""
    run_evoformer(model, inputs, prev) → (msa_act, pair_act)

Run one evoformer cycle: preprocessing projections, recycling features, template
embedding, extra MSA stack, and evoformer blocks.

Returns `(msa_act, pair_act)` tensors on the model's device.
"""
function run_evoformer(model::AF2Model, inputs::AF2PreparedInputs, prev::RecycleState)
    config = model.config
    use_gpu = inputs.use_gpu
    c_m = config.c_m
    c_z = config.c_z
    L = length(inputs.aatype)

    # ── Preprocessing projections ────────────────────────────────────────
    p1 = model.preprocess_1d(inputs.target_feat)
    pmsa = model.preprocess_msa(inputs.msa_feat)
    msa_base = pmsa .+ reshape(p1, c_m, 1, L, 1)

    left = model.left_single(inputs.target_feat)
    right = model.right_single(inputs.target_feat)
    pair_recycle = reshape(left, c_z, L, 1, 1) .+ reshape(right, c_z, 1, L, 1)

    # ── Recycling features ───────────────────────────────────────────────
    prev_pb = _pseudo_beta_from_atom37(inputs.aatype, prev.atom37)
    dgram_feat = _dgram_from_positions(prev_pb)
    dgram_feat_dev = use_gpu ? CuArray(dgram_feat) : dgram_feat
    pair_recycle = pair_recycle .+ model.prev_pos_linear(dgram_feat_dev)
    msa_base[:, 1:1, :, :] .+= reshape(model.prev_msa_first_row_norm(prev.msa_first_row), c_m, 1, L, 1)
    pair_recycle .+= model.prev_pair_norm(prev.pair)
    relpos_feat = if config.relpos_is_multimer
        _multimer_relpos_features(
            inputs.residue_index,
            inputs.asym_id,
            inputs.entity_id,
            inputs.sym_id,
            config.relpos_max_relative_idx,
            config.relpos_max_relative_chain,
        )
    else
        _relpos_one_hot(inputs.residue_index, config.relpos_max_relative_idx)
    end
    relpos_feat_dev = use_gpu ? CuArray(relpos_feat) : relpos_feat
    pair_recycle .+= model.pair_relpos(relpos_feat_dev)

    pair_rr_jl = dropdims(first_to_af2_2d(pair_recycle); dims=1)
    pair_act = reshape(permutedims(pair_rr_jl, (3, 1, 2)), c_z, L, L, 1)

    # ── Template embedding ───────────────────────────────────────────────
    tpair = if model.template_embedding === nothing || size(inputs.template_aatype, 1) == 0
        use_gpu ? CUDA.zeros(Float32, size(pair_recycle)...) : zeros(Float32, size(pair_recycle)...)
    else
        te_layer = use_gpu ? Flux.cpu(model.template_embedding) : model.template_embedding
        pair_recycle_cpu = use_gpu ? Array(pair_recycle) : pair_recycle
        pair_mask_cpu = use_gpu ? Array(inputs.pair_mask) : inputs.pair_mask
        multichain_mask_cpu = use_gpu ? Array(inputs.multichain_mask) : inputs.multichain_mask
        tpair_cpu = if te_layer isa TemplateEmbeddingMultimer
            te_layer(
                pair_recycle_cpu,
                inputs.template_aatype,
                inputs.template_all_atom_positions,
                inputs.template_all_atom_masks,
                pair_mask_cpu;
                template_mask=inputs.template_mask,
                multichain_mask=multichain_mask_cpu,
            )
        else
            te_layer(
                pair_recycle_cpu,
                inputs.template_aatype,
                inputs.template_all_atom_positions,
                inputs.template_all_atom_masks,
                pair_mask_cpu;
                template_mask=inputs.template_mask,
            )
        end
        use_gpu ? CuArray(tpair_cpu) : tpair_cpu
    end
    tpair_jl = dropdims(first_to_af2_2d(tpair); dims=1)
    pair_template_jl = pair_rr_jl .+ tpair_jl
    pair_act = reshape(permutedims(pair_template_jl, (3, 1, 2)), c_z, L, L, 1)

    # ── Extra MSA stack ──────────────────────────────────────────────────
    msa_extra = model.extra_msa_activations(inputs.extra_msa_feat)
    msa_extra_mask = if inputs.extra_msa_mask_input === nothing
        _ones = use_gpu ? CUDA.ones : ones
        _ones(Float32, size(inputs.extra_msa_feat, 2), L, 1)
    else
        size(inputs.extra_msa_mask_input, 1) == size(inputs.extra_msa_feat, 2) || error("extra_msa_mask row count mismatch")
        m = reshape(Float32.(inputs.extra_msa_mask_input), size(inputs.extra_msa_mask_input, 1), size(inputs.extra_msa_mask_input, 2), 1)
        use_gpu ? CuArray(m) : m
    end
    for b in eachindex(model.extra_blocks)
        msa_extra, pair_act = model.extra_blocks[b](msa_extra, pair_act, msa_extra_mask, inputs.pair_mask)
    end

    # ── Build full evoformer input MSA ───────────────────────────────────
    T = size(inputs.template_rows_first, 2)
    msa_act = cat(msa_base, inputs.template_rows_first; dims=2)
    query_msa_mask = if inputs.msa_mask_input === nothing
        _ones = use_gpu ? CUDA.ones : ones
        _ones(Float32, size(msa_base, 2), L, 1)
    else
        size(inputs.msa_mask_input, 1) == size(msa_base, 2) || error("msa_mask row count mismatch")
        m = reshape(Float32.(inputs.msa_mask_input), size(inputs.msa_mask_input, 1), size(inputs.msa_mask_input, 2), 1)
        use_gpu ? CuArray(m) : m
    end
    msa_mask = cat(query_msa_mask, reshape(inputs.template_row_mask, T, L, 1); dims=1)

    # ── Evoformer blocks ─────────────────────────────────────────────────
    for blk in model.blocks
        msa_act, pair_act = blk(msa_act, pair_act, msa_mask, inputs.pair_mask)
    end

    return msa_act, pair_act
end

# ── Pipeline stage: run_heads ────────────────────────────────────────────────

"""
    run_heads(model, msa_act, pair_act, inputs) → NamedTuple

Run structure module and all output heads. Returns a NamedTuple with all outputs
needed for result assembly and recycle state updates.
"""
function run_heads(model::AF2Model, msa_act, pair_act, inputs::AF2PreparedInputs)
    config = model.config
    use_gpu = inputs.use_gpu
    aatype = inputs.aatype
    L = length(aatype)

    single = model.single_activations(view(msa_act, :, 1, :, :))
    n_query_msa = size(inputs.msa_feat, 2)
    masked_msa_input = config.is_multimer_checkpoint ? view(msa_act, :, 1:n_query_msa, :, :) : msa_act
    masked_msa_logits = model.masked_msa_head(masked_msa_input).logits
    distogram_out = model.distogram_head(pair_act)
    distogram_logits = distogram_out.logits
    distogram_bin_edges = use_gpu ? Array(distogram_out.bin_edges) : distogram_out.bin_edges
    experimentally_resolved_logits = model.experimentally_resolved_head(single).logits
    struct_out = model.structure(single, pair_act, reshape(inputs.seq_mask, :, 1), reshape(aatype, :, 1))
    lddt_logits = model.predicted_lddt_head(struct_out.act).logits
    plddt = vec(compute_plddt(lddt_logits))

    pae_logits_cpu = if model.predicted_aligned_error_head === nothing
        nothing
    else
        raw = model.predicted_aligned_error_head(pair_act).logits
        use_gpu ? Array(raw) : raw
    end
    pae_metrics = if pae_logits_cpu === nothing
        nothing
    else
        compute_predicted_aligned_error(
            pae_logits_cpu;
            max_bin=Int(round(model.predicted_aligned_error_head.max_error_bin)),
            no_bins=model.predicted_aligned_error_head.num_bins,
        )
    end
    ptm_score = if pae_logits_cpu === nothing
        nothing
    else
        compute_tm(
            pae_logits_cpu;
            max_bin=Int(round(model.predicted_aligned_error_head.max_error_bin)),
            no_bins=model.predicted_aligned_error_head.num_bins,
        )
    end

    # Atom coordinates
    atom14 = dropdims(view(struct_out.atom_pos, size(struct_out.atom_pos, 1):size(struct_out.atom_pos, 1), :, :, :, :); dims=1)
    atom14_cpu = use_gpu ? Array(atom14) : atom14
    single_cpu = use_gpu ? Array(single) : single
    masks = make_atom14_masks(reshape(aatype, :, 1); mask_type=eltype(single_cpu))
    atom37 = atom14_to_atom37(atom14_cpu, masks)

    _c = use_gpu ? Array : identity
    masked_msa_logits_py = _c(dropdims(permutedims(masked_msa_logits, (4, 2, 3, 1)); dims=1))
    distogram_logits_py = _c(dropdims(first_to_af2_2d(distogram_logits); dims=1))
    experimentally_resolved_logits_py = _c(dropdims(first_to_af2_3d(experimentally_resolved_logits); dims=1))
    lddt_logits_py = _c(dropdims(first_to_af2_3d(lddt_logits); dims=1))
    plddt = use_gpu ? Array(plddt) : plddt

    return (
        atom37=atom37,
        atom37_flat=dropdims(atom37; dims=1),
        atom37_mask=dropdims(permutedims(masks.atom37_atom_exists, (2, 3, 1)); dims=2),
        masked_msa_logits=masked_msa_logits_py,
        distogram_logits=distogram_logits_py,
        distogram_bin_edges=distogram_bin_edges,
        experimentally_resolved_logits=experimentally_resolved_logits_py,
        plddt=plddt,
        lddt_logits=lddt_logits_py,
        pae=pae_metrics !== nothing ? dropdims(pae_metrics.predicted_aligned_error; dims=3) : nothing,
        pae_max=pae_metrics !== nothing ? Float32(pae_metrics.max_predicted_aligned_error) : nothing,
        ptm=ptm_score !== nothing ? Float32(ptm_score) : nothing,
    )
end

# ── Pipeline stage: run_inference ────────────────────────────────────────────

"""
    run_inference(model, inputs) → AF2InferenceResult

Full recycle loop: runs `run_evoformer` and `run_heads` for each iteration,
then assembles the final result.
"""
function run_inference(model::AF2Model, inputs::AF2PreparedInputs)::AF2InferenceResult
    prev = initial_recycle_state(model, inputs)
    heads = nothing

    for i in 1:inputs.num_iters
        msa_act, pair_act = run_evoformer(model, inputs, prev)
        heads = run_heads(model, msa_act, pair_act, inputs)
        prev = RecycleState(
            msa_first_row=view(msa_act, :, 1, :, :),
            pair=pair_act,
            atom37=heads.atom37,
        )
    end

    ca = _ca_distance_metrics(heads.atom37_flat, heads.atom37_mask; asym_id=inputs.asym_id, intra_chain_only=false)
    ca_intra = _ca_distance_metrics(heads.atom37_flat, heads.atom37_mask; asym_id=inputs.asym_id, intra_chain_only=true)

    return AF2InferenceResult(
        atom37=heads.atom37_flat,
        atom37_mask=heads.atom37_mask,
        plddt=heads.plddt,
        masked_msa_logits=heads.masked_msa_logits,
        distogram_logits=heads.distogram_logits,
        distogram_bin_edges=heads.distogram_bin_edges,
        experimentally_resolved_logits=heads.experimentally_resolved_logits,
        lddt_logits=heads.lddt_logits,
        pae=heads.pae,
        pae_max=heads.pae_max,
        ptm=heads.ptm,
        aatype=inputs.aatype,
        residue_index=inputs.residue_index,
        asym_id=inputs.asym_id,
        ca_metrics=ca,
        ca_metrics_intra=ca_intra,
    )
end

# ── Legacy entry point ───────────────────────────────────────────────────────

"""
    _infer(model, dump; num_recycle) → AF2InferenceResult

Full inference from raw feature Dict. Thin wrapper around the composable pipeline:
`prepare_inputs` → `run_inference`.
"""
function _infer(
    model::AF2Model,
    dump::AbstractDict;
    num_recycle::Union{Nothing,Int}=nothing,
)::AF2InferenceResult
    inputs = prepare_inputs(model, dump; num_recycle=num_recycle)
    return run_inference(model, inputs)
end
