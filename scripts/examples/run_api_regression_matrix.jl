using Dates
using Printf
using TOML

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

repo_root = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(repo_root, "scripts", "regression", "regression_cases.jl"))

function _sequence_list(sequence_arg::AbstractString)
    return [String(strip(s)) for s in split(sequence_arg, ",") if !isempty(strip(s))]
end

function _string_or_nothing(v::Vector{String})
    return isempty(v) ? nothing : v[1]
end

function _vector_or_nothing(v::Vector{String})
    return isempty(v) ? nothing : v
end

function _case_call_repr(case::NamedTuple)
    if case.model == :monomer
        msas_repr = isempty(case.msa_files) ? "nothing" : repr(case.msa_files[1])
        templates_repr = isempty(case.template_pdbs) ? "nothing" : repr(case.template_pdbs)
        chains_repr = isempty(case.template_chains) ? "nothing" : repr(case.template_chains)
        return string(
            "fold(monomer_model, ",
            repr(case.sequence_arg),
            "; msas=", msas_repr,
            ", templates=", templates_repr,
            ", template_chains=", chains_repr,
            ", num_recycle=", case.num_recycle,
            ", out_prefix=", repr(case.name), ")",
        )
    end
    seqs = _sequence_list(case.sequence_arg)
    msas_repr = isempty(case.msa_files) ? "nothing" : repr(case.msa_files)
    templates_repr = isempty(case.template_pdbs) ? "nothing" : repr(case.template_pdbs)
    chains_repr = isempty(case.template_chains) ? "nothing" : repr(case.template_chains)
    return string(
        "fold(multimer_model, ",
        repr(seqs),
        "; msas=", msas_repr,
        ", templates=", templates_repr,
        ", template_chains=", chains_repr,
        ", pairing_mode=\"block diagonal\"",
        ", num_recycle=", case.num_recycle,
        ", out_prefix=", repr(case.name), ")",
    )
end

function main()
    out_root = if length(ARGS) >= 1
        abspath(ARGS[1])
    else
        stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        joinpath(repo_root, "test", "regression", "api_matrix_outputs", stamp)
    end
    mkpath(out_root)

    params = default_regression_params(repo_root)
    monomer_spec = params.monomer
    multimer_spec = params.multimer

    println("Loading models...")
    load_time = @elapsed begin
        monomer_model = load_monomer(; filename=monomer_spec)
        multimer_model = load_multimer(; filename=multimer_spec)
    end
    @printf("  Models loaded in %.2fs\n", load_time)
    println("  monomer params: ", monomer_model.params_path)
    println("  multimer params: ", multimer_model.params_path)

    calls_path = joinpath(out_root, "api_calls.txt")
    summary_path = joinpath(out_root, "summary.toml")

    summary_cases = Dict{String,Any}()

    open(calls_path, "w") do calls_io
        for case in pure_julia_regression_cases(repo_root)
            out_prefix = joinpath(out_root, case.name)
            call_repr = _case_call_repr(case)
            println(calls_io, call_repr)
            println("Running ", case.name)
            println("  call: ", call_repr)

            if case.model == :monomer
                fold_time = @elapsed begin
                    result = fold(
                        monomer_model,
                        case.sequence_arg;
                        msas=_string_or_nothing(case.msa_files),
                        templates=_vector_or_nothing(case.template_pdbs),
                        template_chains=_vector_or_nothing(case.template_chains),
                        num_recycle=case.num_recycle,
                        out_prefix=out_prefix,
                    )
                end
                @printf("  done: mean_pLDDT=%.4f  time=%.2fs  pdb=%s\n", result.mean_plddt, fold_time, result.out_pdb)
                summary_cases[case.name] = Dict(
                    "model" => "monomer",
                    "call" => call_repr,
                    "out_npz" => result.out_npz,
                    "out_pdb" => result.out_pdb,
                    "mean_plddt" => result.mean_plddt,
                    "min_plddt" => result.min_plddt,
                    "max_plddt" => result.max_plddt,
                    "mean_pae" => result.mean_pae,
                    "ptm" => result.ptm,
                    "fold_time_s" => fold_time,
                )
            else
                fold_time = @elapsed begin
                    result = fold(
                        multimer_model,
                        _sequence_list(case.sequence_arg);
                        msas=_vector_or_nothing(case.msa_files),
                        templates=_vector_or_nothing(case.template_pdbs),
                        template_chains=_vector_or_nothing(case.template_chains),
                        pairing_mode="block diagonal",
                        num_recycle=case.num_recycle,
                        out_prefix=out_prefix,
                    )
                end
                @printf("  done: mean_pLDDT=%.4f  time=%.2fs  pdb=%s\n", result.mean_plddt, fold_time, result.out_pdb)
                summary_cases[case.name] = Dict(
                    "model" => "multimer",
                    "call" => call_repr,
                    "out_npz" => result.out_npz,
                    "out_pdb" => result.out_pdb,
                    "mean_plddt" => result.mean_plddt,
                    "min_plddt" => result.min_plddt,
                    "max_plddt" => result.max_plddt,
                    "mean_pae" => result.mean_pae,
                    "ptm" => result.ptm,
                    "fold_time_s" => fold_time,
                )
            end
        end
    end

    summary = Dict(
        "generated_at" => string(now()),
        "generator" => "scripts/examples/run_api_regression_matrix.jl",
        "output_root" => out_root,
        "calls_file" => calls_path,
        "monomer_params_path" => monomer_model.params_path,
        "multimer_params_path" => multimer_model.params_path,
        "cases" => summary_cases,
    )

    open(summary_path, "w") do io
        TOML.print(io, summary)
    end

    println("Saved calls file: ", calls_path)
    println("Saved summary file: ", summary_path)
    println("Output root: ", out_root)
end

main()
