using Dates
using Printf
using SHA
using TOML

repo_root = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "regression_cases.jl"))
include(joinpath(@__DIR__, "regression_helpers.jl"))

params = default_regression_params(repo_root)
monomer_params = params.monomer
multimer_params = params.multimer

isfile(monomer_params) || error("Monomer params not found: $(monomer_params). Set AF2_MONOMER_PARAMS.")
isfile(multimer_params) || error("Multimer params not found: $(multimer_params). Set AF2_MULTIMER_PARAMS.")

selected = if isempty(ARGS)
    nothing
else
    Set([strip(x) for x in split(ARGS[1], ",") if !isempty(strip(x))])
end

reference_dir = joinpath(repo_root, "test", "regression", "reference_pdbs")
mkpath(reference_dir)

manifest_cases = Dict{String,Any}()

mktempdir() do tmpdir
    for case in pure_julia_regression_cases(repo_root)
        if selected !== nothing && !(case.name in selected)
            continue
        end

        for p in case.msa_files
            isfile(p) || error("Missing MSA file for $(case.name): $(p)")
        end
        for p in case.template_pdbs
            isfile(p) || error("Missing template PDB for $(case.name): $(p)")
        end

        params_path = case.params_kind == :monomer ? monomer_params : multimer_params
        out = run_pure_julia_regression_case(repo_root, case, params_path, joinpath(tmpdir, case.name))

        ref_pdb = joinpath(reference_dir, string(case.name, ".pdb"))
        cp(out.out_pdb, ref_pdb; force=true)

        chains = [string(c) for c in pdb_chain_ids(ref_pdb)]
        geom = npz_geometry_metrics(out.out_npz)
        digest = bytes2hex(SHA.sha256(read(ref_pdb)))

        @printf("Generated %-28s atoms=%d chains=%s plddt=%.4f\n",
            case.name,
            length(parse_pdb_atoms(ref_pdb)),
            join(chains, ","),
            get(geom, "mean_plddt", NaN),
        )

        manifest_cases[case.name] = Dict(
            "model" => String(case.model),
            "description" => case.description,
            "num_recycle" => Int(case.num_recycle),
            "params_kind" => String(case.params_kind),
            "sequence_arg" => case.sequence_arg,
            "msa_files" => case.msa_files,
            "template_pdbs" => case.template_pdbs,
            "template_chains" => case.template_chains,
            "expected_chain_ids" => [string(c) for c in case.expected_chain_ids],
            "observed_chain_ids" => chains,
            "sha256" => digest,
            "geometry" => geom,
        )
    end
end

manifest = Dict(
    "generated_at" => string(now()),
    "generator" => "scripts/regression/generate_reference_pdbs.jl",
    "monomer_params" => abspath(monomer_params),
    "multimer_params" => abspath(multimer_params),
    "cases" => manifest_cases,
)

manifest_path = joinpath(reference_dir, "manifest.toml")
open(manifest_path, "w") do io
    TOML.print(io, manifest)
end

println("Saved reference manifest to ", manifest_path)
println("Reference PDBs directory: ", reference_dir)
