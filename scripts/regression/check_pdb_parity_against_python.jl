using Printf

repo_root = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "regression_cases.jl"))
include(joinpath(@__DIR__, "regression_helpers.jl"))

function main()
    out_root = if !isempty(ARGS)
        abspath(ARGS[1])
    else
        mktempdir(prefix="af2_pdb_parity_")
    end
    mkpath(out_root)

    params = default_regression_params(repo_root)
    monomer_params = params.monomer
    multimer_params = params.multimer
    isempty(monomer_params) && error("Monomer params spec is empty.")
    isempty(multimer_params) && error("Multimer params spec is empty.")

    println("out_root=", out_root)
    println("case,out_pdb,python_ref,strict_ok,max_abs,mean_abs,rms,atom_count")

    strict_failures = 0
    for case in pure_julia_regression_cases(repo_root)
        params_path = case.params_kind == :monomer ? monomer_params : multimer_params
        case_dir = joinpath(out_root, case.name)
        out = run_pure_julia_regression_case(repo_root, case, params_path, case_dir)

        py_ref = joinpath(repo_root, "test", "regression", "reference_pdbs", string(case.name, ".python_official.pdb"))
        isfile(py_ref) || error("Missing Python reference PDB for $(case.name): $(py_ref)")

        cmp = compare_pdb_coordinates(out.out_pdb, py_ref)
        ok = get(cmp, :ok, false)
        if ok
            @printf("%s,%s,%s,true,%.9g,%.9g,%.9g,%d\n",
                case.name, out.out_pdb, py_ref, cmp[:max_abs], cmp[:mean_abs], cmp[:rms], cmp[:atom_count])
        else
            strict_failures += 1
            reason = String(get(cmp, :reason, "unknown"))
            @printf("%s,%s,%s,false,NaN,NaN,NaN,0 (%s)\n", case.name, out.out_pdb, py_ref, reason)
        end
    end

    println("strict_failures=", strict_failures)
end

main()
