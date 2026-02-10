using Test
using TOML

repo_root = normpath(joinpath(@__DIR__, ".."))
include(joinpath(repo_root, "scripts", "regression", "regression_cases.jl"))
include(joinpath(repo_root, "scripts", "regression", "regression_helpers.jl"))

@testset "Pure Julia PDB Regression" begin
    if get(ENV, "AF2_RUN_PURE_JULIA_REGRESSION", "0") != "1"
        @info "Skipping pure Julia PDB regression (set AF2_RUN_PURE_JULIA_REGRESSION=1 to enable)."
        @test true
    else
        params = default_regression_params(repo_root)

        reference_dir = joinpath(repo_root, "test", "regression", "reference_pdbs")
        manifest_path = joinpath(reference_dir, "manifest.toml")
        @test isfile(manifest_path)

        manifest = TOML.parsefile(manifest_path)
        haskey(manifest, "cases") || error("Invalid manifest at $(manifest_path): missing [cases]")

        coord_max_abs_tol = parse(Float64, get(ENV, "AF2_PDB_REGRESSION_MAX_ABS_TOL", "5e-3"))
        coord_rms_tol = parse(Float64, get(ENV, "AF2_PDB_REGRESSION_RMS_TOL", "1e-3"))

        # Build models once
        mono_model = Alphafold2._build_af2_model(Alphafold2.af2_params_read(
            Alphafold2.resolve_af2_params_path(params.monomer;
                repo_id=Alphafold2.AF2_HF_REPO_ID, revision=Alphafold2.AF2_HF_REVISION)))
        multi_model = Alphafold2._build_af2_model(Alphafold2.af2_params_read(
            Alphafold2.resolve_af2_params_path(params.multimer;
                repo_id=Alphafold2.AF2_HF_REPO_ID, revision=Alphafold2.AF2_HF_REVISION)))

        mktempdir() do tmpdir
            for case in pure_julia_regression_cases(repo_root)
                @testset "Case $(case.name)" begin
                    mdl = case.params_kind == :monomer ? mono_model : multi_model
                    out = run_inprocess_regression_case(mdl, case, joinpath(tmpdir, case.name);
                        repo_root=repo_root)

                    ref_pdb = joinpath(reference_dir, string(case.name, ".pdb"))
                    @test isfile(ref_pdb)

                    cmp = compare_pdb_coordinates(out.out_pdb, ref_pdb)
                    @test get(cmp, :ok, false)
                    @test get(cmp, :max_abs, Inf) <= coord_max_abs_tol
                    @test get(cmp, :rms, Inf) <= coord_rms_tol

                    chains = collect(pdb_chain_ids(out.out_pdb))
                    @test chains == case.expected_chain_ids

                    geom = npz_geometry_metrics(out.out_npz)

                    if haskey(manifest["cases"], case.name)
                        info = manifest["cases"][case.name]
                        if haskey(info, "geometry") && haskey(info["geometry"], "mean_plddt") && haskey(geom, "mean_plddt")
                            ref_plddt = Float64(info["geometry"]["mean_plddt"])
                            @test abs(geom["mean_plddt"] - ref_plddt) <= 1e-2
                        end
                    end
                end
            end
        end
    end
end
