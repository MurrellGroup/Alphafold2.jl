using Printf
using CUDA
using Flux

repo_root = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(repo_root, "src", "Alphafold2.jl"))
using .Alphafold2
include(joinpath(@__DIR__, "regression_cases.jl"))
include(joinpath(@__DIR__, "regression_helpers.jl"))

function main()
    use_gpu = "--gpu" in ARGS
    positional = filter(a -> !startswith(a, "--"), ARGS)

    out_root = if !isempty(positional)
        abspath(positional[1])
    else
        mktempdir(prefix="af2_pdb_parity_")
    end
    mkpath(out_root)

    params = default_regression_params(repo_root)
    monomer_params_spec = params.monomer
    multimer_params_spec = params.multimer
    isempty(monomer_params_spec) && error("Monomer params spec is empty.")
    isempty(multimer_params_spec) && error("Multimer params spec is empty.")

    # Build models once
    monomer_params_path = Alphafold2.resolve_af2_params_path(
        monomer_params_spec;
        repo_id=get(ENV, "AF2_HF_REPO_ID", Alphafold2.AF2_HF_REPO_ID),
        revision=get(ENV, "AF2_HF_REVISION", Alphafold2.AF2_HF_REVISION),
    )
    multimer_params_path = Alphafold2.resolve_af2_params_path(
        multimer_params_spec;
        repo_id=get(ENV, "AF2_HF_REPO_ID", Alphafold2.AF2_HF_REPO_ID),
        revision=get(ENV, "AF2_HF_REVISION", Alphafold2.AF2_HF_REVISION),
    )

    println("Building monomer model...")
    monomer_model = Alphafold2._build_af2_model(Alphafold2.af2_params_read(monomer_params_path))
    GC.gc()
    println("Building multimer model...")
    multimer_model = Alphafold2._build_af2_model(Alphafold2.af2_params_read(multimer_params_path))
    GC.gc()
    if use_gpu
        if !CUDA.functional()
            error("--gpu flag specified but CUDA is not functional")
        end
        CUDA.math_mode!(CUDA.FAST_MATH)
        println("Moving models to GPU...")
        monomer_model = Flux.gpu(monomer_model)
        multimer_model = Flux.gpu(multimer_model)
        CUDA.reclaim()
        println("Models on GPU: ", CUDA.name(CUDA.device()))
    end

    println("out_root=", out_root)
    println("case,out_pdb,python_ref,strict_ok,max_abs,mean_abs,rms,atom_count")

    strict_failures = 0
    for case in pure_julia_regression_cases(repo_root)
        model = case.params_kind == :monomer ? monomer_model : multimer_model
        case_dir = joinpath(out_root, case.name)
        out = run_inprocess_regression_case(model, case, case_dir; repo_root=repo_root)

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
