using NNlib
using Statistics

"""
    build_soft_sequence_features(seq_logits)

Create minimal AF2-style `target_feat` and single-row `msa_feat` from differentiable
sequence logits in feature-first layout.

Input:
- `seq_logits`: `(21, L, B)` for 20 residues + unknown.

Output dictionary keys:
- `:seq_probs` `(21, L, B)`
- `:target_feat` `(22, L, B)`
- `:msa_feat` `(49, 1, L, B)`
"""
function build_soft_sequence_features(seq_logits::AbstractArray)
    ndims(seq_logits) == 3 || error("seq_logits must have shape (21, L, B)")
    size(seq_logits, 1) == 21 || error("seq_logits first dimension must be 21, got $(size(seq_logits, 1))")

    logits = Float32.(seq_logits)
    L = size(logits, 2)
    B = size(logits, 3)

    seq_probs = NNlib.softmax(logits; dims=1) # (21, L, B)

    target_feat = cat(zeros(Float32, 1, L, B), seq_probs; dims=1) # (22, L, B)

    # AF2 MSA token space is 23 classes (adds gap and bert mask). For minimal soft features,
    # map sequence probabilities to first 21 classes and keep the last 2 classes at zero.
    msa_23 = cat(seq_probs, zeros(Float32, 2, L, B); dims=1) # (23, L, B)
    msa_1hot = reshape(msa_23, 23, 1, L, B)                 # (23, 1, L, B)

    msa_feat = cat(
        msa_1hot,
        zeros(Float32, 1, 1, L, B),
        zeros(Float32, 1, 1, L, B),
        msa_1hot,
        zeros(Float32, 1, 1, L, B);
        dims=1,
    ) # (49, 1, L, B)

    return (; seq_probs, target_feat, msa_feat)
end

"""
    build_hard_sequence_features(aatype)

Create minimal AF2-style `target_feat` and single-row `msa_feat` from integer
`aatype` in `[0, 20]` with shape `(L, B)`.
"""
function build_hard_sequence_features(aatype::AbstractMatrix{<:Integer})
    L, B = size(aatype)
    logits = fill(-12f0, 21, L, B)
    for b in 1:B, i in 1:L
        k = clamp(Int(aatype[i, b]) + 1, 1, 21)
        logits[k, i, b] = 12f0
    end
    return build_soft_sequence_features(logits)
end

"""
    build_basic_masks(aatype; n_msa_seq=1)

Build default `seq_mask`, `msa_mask`, and `residue_index` for AF2-style trunks.
"""
function build_basic_masks(aatype::AbstractMatrix{<:Integer}; n_msa_seq::Int=1)
    L, B = size(aatype)
    n_msa_seq > 0 || error("n_msa_seq must be positive")
    seq_mask = ones(Float32, L, B)
    msa_mask = ones(Float32, n_msa_seq, L, B)
    residue_index = repeat(reshape(collect(0:(L - 1)), L, 1), 1, B)
    return seq_mask, msa_mask, residue_index
end

"""
    mean_plddt_loss(lddt_logits)

Compute mean predicted LDDT from AF2-style logits `(num_bins, L, B)`.
"""
function mean_plddt_loss(lddt_logits::AbstractArray)
    ndims(lddt_logits) == 3 || error("lddt_logits must have shape (num_bins, L, B)")
    plddt = compute_plddt(dropdims(first_to_af2_3d(lddt_logits); dims=1))
    return mean(plddt)
end
