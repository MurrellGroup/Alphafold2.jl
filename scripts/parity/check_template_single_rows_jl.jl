using NPZ
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

if length(ARGS) < 2
    error("Usage: julia scripts/parity/check_template_single_rows_jl.jl <params_model_1.npz> <dump.npz> [tol]")
end

params_path = ARGS[1]
dump_path = ARGS[2]
tol = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 5e-4

arrs = NPZ.npzread(dump_path)
template_aatype = Int.(arrs["template_aatype"])
template_all_atom_positions = Float32.(arrs["template_all_atom_positions"])
template_all_atom_masks = Float32.(arrs["template_all_atom_masks"])
py_tors = Float32.(arrs["torsion_angles_sin_cos"])
py_alt_tors = Float32.(arrs["alt_torsion_angles_sin_cos"])
py_tors_mask = Float32.(arrs["torsion_angles_mask"])
py_rows = Float32.(arrs["template_single_rows"])
py_row_mask = Float32.(arrs["torsion_row_mask"])
placeholder_for_undefined = haskey(arrs, "placeholder_for_undefined") ? (Int(arrs["placeholder_for_undefined"][1]) != 0) : false

m = TemplateSingleRows(size(py_rows, 3))
load_template_single_rows_npz!(m, params_path)

ret = atom37_to_torsion_angles(
    template_aatype,
    template_all_atom_positions,
    template_all_atom_masks;
    placeholder_for_undefined=placeholder_for_undefined,
)
jl_tors = ret[:torsion_angles_sin_cos]
jl_alt_tors = ret[:alt_torsion_angles_sin_cos]
jl_tors_mask = ret[:torsion_angles_mask]

jl_rows_first, jl_row_mask = m(
    template_aatype,
    template_all_atom_positions,
    template_all_atom_masks;
    placeholder_for_undefined=placeholder_for_undefined,
)
jl_rows = dropdims(permutedims(jl_rows_first, (2, 3, 1, 4)); dims=4)

d_tors = jl_tors .- py_tors
d_alt = jl_alt_tors .- py_alt_tors
d_mask = jl_tors_mask .- py_tors_mask
d_rows = jl_rows .- py_rows
d_row_mask = jl_row_mask .- py_row_mask

max_tors = maximum(abs, d_tors)
max_alt = maximum(abs, d_alt)
max_mask = maximum(abs, d_mask)
max_rows = maximum(abs, d_rows)
max_row_mask = maximum(abs, d_row_mask)

@printf("TemplateSingleRows parity\n")
@printf("  torsion_max_abs: %.8g\n", max_tors)
@printf("  alt_torsion_max_abs: %.8g\n", max_alt)
@printf("  torsion_mask_max_abs: %.8g\n", max_mask)
@printf("  rows_max_abs: %.8g\n", max_rows)
@printf("  row_mask_max_abs: %.8g\n", max_row_mask)
@printf("  tol: %.8g\n", tol)

if max(max(max(max_tors, max_alt), max_mask), max(max_rows, max_row_mask)) > tol
    error("Parity check failed")
end

println("PASS")
