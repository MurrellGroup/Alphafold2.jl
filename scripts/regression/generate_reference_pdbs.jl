# Generate and validate Julia reference PDBs for all 9 regression cases.
#
# Usage:
#   # CPU (deterministic, canonical reference):
#   julia --project=. scripts/regression/generate_reference_pdbs.jl
#
#   # GPU with KernelAbstractions (ONIONop) backend:
#   JULIA_CUDA_HARD_MEMORY_LIMIT=64GiB julia --project=GPU_test \
#       scripts/regression/generate_reference_pdbs.jl --gpu
#
#   # GPU with cuTile (OnionTile) backend — same command, cuTile dispatches via CuArray:
#   JULIA_CUDA_HARD_MEMORY_LIMIT=64GiB julia --project=GPU_test \
#       scripts/regression/generate_reference_pdbs.jl --gpu
#
#   # Run only specific cases:
#   julia --project=. scripts/regression/generate_reference_pdbs.jl multimer_msa_only,monomer_seq_only
#
#   # Overwrite existing reference PDBs (use with caution!):
#   julia --project=. scripts/regression/generate_reference_pdbs.jl --update-refs
#
# Flags:
#   --gpu           Move models to GPU before inference (requires CUDA)
#   --update-refs   Overwrite existing reference PDBs in test/regression/reference_pdbs/
#   [case,case,...] Comma-separated list of case names to run (default: all 9)
#
# Outputs:
#   test/regression/reference_pdbs/<case>.pdb    — reference PDB files
#   test/regression/reference_pdbs/manifest.toml — metadata (sha256, geometry, params)
#
# Validation:
#   - Without --update-refs, compares output against existing reference PDBs
#   - PASS (byte-identical): exact match, no computation changed
#   - WARN (max_abs > 0):    close but not identical (e.g., GPU TF32 differences)
#   - FAIL:                  structural mismatch (atom count or identity differs)
#   - Reports clashes per case (non-bonded atoms < 2.0 Å apart)
#
# Expected clash counts (healthy output):
#   All multimer cases:            0 clashes
#   monomer_seq_only (9 res):      ≤1 clash
#   monomer_msa_only (9 res):      0 clashes
#   monomer_template_only (29 res): ≤3 clashes
#   monomer_template_msa (29 res):  ≤3 clashes

using Dates
using Printf
using SHA
using TOML
using CUDA
using Flux

repo_root = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(repo_root, "src", "Alphafold2.jl"))
using .Alphafold2
include(joinpath(@__DIR__, "regression_cases.jl"))
include(joinpath(@__DIR__, "regression_helpers.jl"))

use_gpu = "--gpu" in ARGS
update_refs = "--update-refs" in ARGS
positional = filter(a -> !startswith(a, "--"), ARGS)
selected = if isempty(positional)
    nothing
else
    Set([strip(x) for x in split(positional[1], ",") if !isempty(strip(x))])
end

params = default_regression_params(repo_root)
monomer_params_path = Alphafold2.resolve_af2_params_path(
    params.monomer;
    repo_id=get(ENV, "AF2_HF_REPO_ID", Alphafold2.AF2_HF_REPO_ID),
    revision=get(ENV, "AF2_HF_REVISION", Alphafold2.AF2_HF_REVISION),
)
multimer_params_path = Alphafold2.resolve_af2_params_path(
    params.multimer;
    repo_id=get(ENV, "AF2_HF_REPO_ID", Alphafold2.AF2_HF_REPO_ID),
    revision=get(ENV, "AF2_HF_REVISION", Alphafold2.AF2_HF_REVISION),
)
println("Resolved monomer params: ", monomer_params_path)
println("Resolved multimer params: ", multimer_params_path)

# Build models once
println("Building monomer model...")
monomer_arrs = Alphafold2.af2_params_read(monomer_params_path)
monomer_model = Alphafold2._build_af2_model(monomer_arrs)
monomer_arrs = nothing  # free params dict
GC.gc()
println("Building multimer model...")
multimer_arrs = Alphafold2.af2_params_read(multimer_params_path)
multimer_model = Alphafold2._build_af2_model(multimer_arrs)
multimer_arrs = nothing
GC.gc()
if use_gpu
    if !CUDA.functional()
        error("--gpu flag specified but CUDA is not functional")
    end
    CUDA.math_mode!(CUDA.FAST_MATH)
    println("Moving monomer model to GPU...")
    monomer_model = Flux.gpu(monomer_model)
    println("Moving multimer model to GPU...")
    multimer_model = Flux.gpu(multimer_model)
    CUDA.reclaim()
    println("Models on GPU: ", CUDA.name(CUDA.device()))
end

reference_dir = joinpath(repo_root, "test", "regression", "reference_pdbs")
mkpath(reference_dir)

manifest_cases = Dict{String,Any}()

function _split_template_path_group(spec::AbstractString)
    return [strip(x) for x in split(spec, "+") if !isempty(strip(x))]
end

mktempdir() do tmpdir
    for case in pure_julia_regression_cases(repo_root)
        if selected !== nothing && !(case.name in selected)
            continue
        end

        for p in case.msa_files
            isfile(p) || error("Missing MSA file for $(case.name): $(p)")
        end
        for spec in case.template_pdbs
            for p in _split_template_path_group(spec)
                isfile(p) || error("Missing template PDB for $(case.name): $(p)")
            end
        end

        model = case.params_kind == :monomer ? monomer_model : multimer_model
        out = run_inprocess_regression_case(model, case, joinpath(tmpdir, case.name); repo_root=repo_root)

        ref_pdb = joinpath(reference_dir, string(case.name, ".pdb"))

        # Compare against existing reference if present (don't overwrite without --update-refs)
        if isfile(ref_pdb) && !update_refs
            cmp = compare_pdb_coordinates(out.out_pdb, ref_pdb)
            ok = get(cmp, :ok, false)
            if ok && cmp[:max_abs] == 0.0
                @printf("%-28s PASS (byte-identical)\n", case.name)
            elseif ok
                @printf("%-28s WARN max_abs=%.9g (not byte-identical, use --update-refs to overwrite)\n", case.name, cmp[:max_abs])
            else
                reason = String(get(cmp, :reason, "unknown"))
                @printf("%-28s FAIL (%s, use --update-refs to overwrite)\n", case.name, reason)
            end
        else
            cp(out.out_pdb, ref_pdb; force=true)
            @printf("%-28s GENERATED\n", case.name)
        end

        out_pdb_for_manifest = update_refs || !isfile(ref_pdb) ? ref_pdb : out.out_pdb
        chains = [string(c) for c in pdb_chain_ids(isfile(ref_pdb) ? ref_pdb : out.out_pdb)]
        geom = npz_geometry_metrics(out.out_npz)
        digest = bytes2hex(SHA.sha256(read(isfile(ref_pdb) ? ref_pdb : out.out_pdb)))
        clashes = check_clashes(out.out_pdb)

        @printf("  atoms=%d chains=%s plddt=%.4f clashes=%d",
            length(parse_pdb_atoms(out.out_pdb)),
            join(chains, ","),
            get(geom, "mean_plddt", NaN),
            clashes.count,
        )
        if clashes.count > 0
            @printf(" (worst=%.3fÅ)", clashes.worst)
        end
        println()

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
    "monomer_params" => monomer_params_path,
    "multimer_params" => multimer_params_path,
    "gpu" => use_gpu,
    "cases" => manifest_cases,
)

manifest_path = joinpath(reference_dir, "manifest.toml")
open(manifest_path, "w") do io
    TOML.print(io, manifest)
end

println("Saved reference manifest to ", manifest_path)
println("Reference PDBs directory: ", reference_dir)
