# OpenFold inference utilities.
# Atom14/atom37 data tables and confidence metrics come from ProtInterop.
# This file keeps only AF2-specific wrappers.

using ProtInterop: OF_RESTYPE_ATOM14_TO_ATOM37, OF_RESTYPE_ATOM37_TO_ATOM14,
    OF_RESTYPE_ATOM14_MASK, OF_RESTYPE_ATOM37_MASK

# ── Aliases for backward compatibility ──────────────────────────────────────

const _restype_atom14_to_atom37 = OF_RESTYPE_ATOM14_TO_ATOM37
const _restype_atom37_to_atom14 = OF_RESTYPE_ATOM37_TO_ATOM14
const _restype_atom14_mask = OF_RESTYPE_ATOM14_MASK
const _restype_atom37_mask = OF_RESTYPE_ATOM37_MASK

# ── make_atom14_masks (AF2-specific: functional API, returns NamedTuple) ────

"""
    make_atom14_masks(aatype; mask_type=Float32) → NamedTuple

Compute atom14/atom37 index maps and existence masks from residue types.
`aatype` should be `(L, B)` with 0-based residue indices.
"""
function make_atom14_masks(aatype::AbstractArray; mask_type::Type{<:AbstractFloat}=Float32)
    protein_aatype = aatype .+ 1

    restype_atom14_to_atom37_dev = to_device(_restype_atom14_to_atom37, protein_aatype, Int)
    restype_atom37_to_atom14_dev = to_device(_restype_atom37_to_atom14, protein_aatype, Int)
    restype_atom14_mask_dev = to_device(_restype_atom14_mask, protein_aatype, mask_type)
    restype_atom37_mask_dev = to_device(_restype_atom37_mask, protein_aatype, mask_type)

    residx_atom14_to_atom37 = permutedims(restype_atom14_to_atom37_dev[protein_aatype, :], (3, 1, 2))
    atom14_atom_exists = permutedims(restype_atom14_mask_dev[protein_aatype, :], (3, 1, 2))

    residx_atom37_to_atom14 = permutedims(restype_atom37_to_atom14_dev[protein_aatype, :], (3, 1, 2))
    atom37_atom_exists = permutedims(restype_atom37_mask_dev[protein_aatype, :], (3, 1, 2))

    return (;
        atom14_atom_exists,
        residx_atom14_to_atom37 = residx_atom14_to_atom37 .- 1,
        residx_atom37_to_atom14 = residx_atom37_to_atom14 .- 1,
        atom37_atom_exists,
    )
end

# ── _calculate_expected_aligned_error (AF2-specific helper) ─────────────────

function _calculate_expected_aligned_error(alignment_confidence_breaks, aligned_distance_error_probs)
    bin_centers = ProtInterop._calculate_bin_centers(alignment_confidence_breaks)
    bview = reshape(bin_centers, ntuple(_ -> 1, ndims(aligned_distance_error_probs) - 1)..., length(bin_centers))
    expected = sum(aligned_distance_error_probs .* bview; dims=ndims(aligned_distance_error_probs))
    expected = dropdims(expected; dims=ndims(expected))
    return expected, bin_centers[end]
end

# Confidence metrics (compute_plddt, compute_predicted_aligned_error, compute_tm)
# are now imported from ProtInterop via `using ProtInterop` in the module definition.
# AF2 re-exports them.
