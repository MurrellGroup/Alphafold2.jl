using Printf

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

function main()
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    monomer_weights = get(
        ENV,
        "AF2_MONOMER_PARAMS",
        joinpath(
            normpath(joinpath(repo_root, "..")),
            "af2_weights_official",
            "params_safetensors",
            "alphafold2_model_1_ptm_dm_2022-12-06.safetensors",
        ),
    )
    msa_path = joinpath(repo_root, "test", "regression", "msa", "monomer_short.a3m")

    model = load_monomer(; filename=monomer_weights)

    r1 = fold(
        "ACDEFGHIK";
        num_recycle=1,
        out_prefix="/tmp/af2_repl_api_case1",
    )
    r2 = fold(
        "ACDEFGHIK";
        msas=msa_path,
        num_recycle=1,
        out_prefix="/tmp/af2_repl_api_case2",
    )

    @printf("case1 pdb=%s mean_plddt=%.4f\n", r1.out_pdb, r1.mean_plddt)
    @printf("case2 pdb=%s mean_plddt=%.4f\n", r2.out_pdb, r2.mean_plddt)

    run_multimer = lowercase(strip(get(ENV, "AF2_SMOKE_INCLUDE_MULTIMER", "false"))) in ("1", "true", "yes", "on")
    if run_multimer
        multimer_weights = get(
            ENV,
            "AF2_MULTIMER_PARAMS",
            joinpath(
                normpath(joinpath(repo_root, "..")),
                "af2_weights_official",
                "params_safetensors",
                "alphafold2_model_1_multimer_v3_dm_2022-12-06.safetensors",
            ),
        )
        multi_model = load_multimer(; filename=multimer_weights)
        rm = fold(
            multi_model,
            ["MKQLEDKVEELLSKNYHLENEVARLKKLV", "MKQLEDKVEELLSKNYHLENEVARLKKLV"];
            num_recycle=1,
            out_prefix="/tmp/af2_repl_api_multimer_case",
        )
        @printf("multimer pdb=%s mean_plddt=%.4f\n", rm.out_pdb, rm.mean_plddt)
    end
end

main()
