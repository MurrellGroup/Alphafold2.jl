# Thin CLI wrapper around Alphafold2.build_monomer_features().
# Usage: julia build_monomer_input_jl.jl <sequence> <out.npz> [num_recycle] [msa_file] [template_pdb_csv] [template_chain_csv]

using NPZ

if !isdefined(Main, :Alphafold2)
    include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
end
using .Alphafold2

function main()
    if length(ARGS) < 2
        error("Usage: julia build_monomer_input_jl.jl <sequence> <out.npz> [num_recycle] [msa_file] [template_pdb_or_csv] [template_chain_or_csv]")
    end

    query_seq = ARGS[1]
    out_path = ARGS[2]
    num_recycle = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
    msa_file = length(ARGS) >= 4 ? ARGS[4] : ""
    template_pdb_arg = length(ARGS) >= 5 ? ARGS[5] : ""
    template_chain_arg = length(ARGS) >= 6 ? ARGS[6] : "A"

    payload = Alphafold2.build_monomer_features(
        query_seq;
        num_recycle=num_recycle,
        msa_file=msa_file,
        template_pdb_arg=template_pdb_arg,
        template_chain_arg=template_chain_arg,
    )

    mkpath(dirname(abspath(out_path)))
    NPZ.npzwrite(out_path, payload)
    println("Saved monomer input NPZ to ", out_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
