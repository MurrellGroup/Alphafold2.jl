# Thin CLI wrapper around Alphafold2.build_multimer_features().
# Usage: julia build_multimer_input_jl.jl <sequences_csv> <out.npz> [num_recycle] [msa_files_csv] [template_pdbs_csv] [template_chains_csv] [pairing_mode] [pairing_seed]

using NPZ

if !isdefined(Main, :Alphafold2)
    include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
end
using .Alphafold2

function _split_csv(arg::AbstractString)
    return [uppercase(strip(x)) for x in split(arg, ",") if !isempty(strip(x))]
end

function main()
    if length(ARGS) < 2
        error("Usage: julia build_multimer_input_jl.jl <sequences_csv> <out.npz> [num_recycle] [msa_files_csv] [template_pdbs_csv] [template_chains_csv] [pairing_mode] [pairing_seed]")
    end

    seqs = _split_csv(ARGS[1])
    out_path = ARGS[2]
    num_recycle = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
    msa_files = if length(ARGS) >= 4
        [String(strip(x)) for x in split(ARGS[4], ",")]
    else
        String[]
    end
    template_pdb_arg = length(ARGS) >= 5 ? ARGS[5] : ""
    template_chain_arg = length(ARGS) >= 6 ? ARGS[6] : ""
    pairing_mode = length(ARGS) >= 7 ? ARGS[7] : get(ENV, "AF2_MULTIMER_PAIRING_MODE", "block diagonal")
    pairing_seed_raw = length(ARGS) >= 8 ? ARGS[8] : get(ENV, "AF2_MULTIMER_PAIRING_SEED", "0")
    pairing_seed = tryparse(Int, strip(pairing_seed_raw))
    pairing_seed === nothing && error("Invalid pairing seed: $(pairing_seed_raw)")

    payload = Alphafold2.build_multimer_features(
        seqs;
        num_recycle=num_recycle,
        msa_files=msa_files,
        template_pdb_arg=template_pdb_arg,
        template_chain_arg=template_chain_arg,
        pairing_mode_raw=pairing_mode,
        pairing_seed=pairing_seed,
    )

    mkpath(dirname(abspath(out_path)))
    NPZ.npzwrite(out_path, payload)
    println("Saved multimer input NPZ to ", out_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
