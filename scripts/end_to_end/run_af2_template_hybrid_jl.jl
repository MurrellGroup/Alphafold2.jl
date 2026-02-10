using NPZ
using Printf
using Statistics
using NNlib
using CUDA
using cuDNN
using Flux

if !isdefined(Main, :Alphafold2)
    include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
end
using .Alphafold2

function run_af2_template_hybrid(
    params_source,
    dump_path::AbstractString,
    out_path::AbstractString;
    use_gpu::Bool=false,
)
    arrs = Alphafold2.af2_params_read(params_source)
    model = Alphafold2._build_af2_model(arrs)
    if use_gpu
        CUDA.math_mode!(CUDA.FAST_MATH)
        @printf("Moving model layers to GPU...\n")
        model = Flux.gpu(model)
        CUDA.reclaim()
        @printf("Model layers on GPU.\n")
    end

    dump = NPZ.npzread(dump_path)
    num_recycle = haskey(dump, "num_recycle") ? Int(dump["num_recycle"]) : nothing
    result = Alphafold2._infer(model, dump; num_recycle=num_recycle)

    pdb_path = Alphafold2._infer_pdb_path_from_npz(out_path)
    atoms_written = Alphafold2._write_fold_pdb(pdb_path, result)
    Alphafold2._write_fold_npz(out_path, result)

    # Print geometry and confidence (matches legacy output)
    ca = result.ca_metrics
    ca_intra = result.ca_metrics_intra
    is_multimer_output = length(unique(result.asym_id)) > 1
    @printf("Final geometry (native)\n")
    @printf("  count: %d\n", ca[:count])
    @printf("  mean: %.6f A\n", ca[:mean])
    @printf("  std:  %.6f A\n", ca[:std])
    @printf("  min:  %.6f A\n", ca[:min])
    @printf("  max:  %.6f A\n", ca[:max])
    @printf("  outlier_fraction: %.3f\n", ca[:outlier_fraction])
    if is_multimer_output
        @printf("  intra_chain_count: %d\n", ca_intra[:count])
        @printf("  intra_chain_mean: %.6f A\n", ca_intra[:mean])
        @printf("  intra_chain_std:  %.6f A\n", ca_intra[:std])
        @printf("  intra_chain_min:  %.6f A\n", ca_intra[:min])
        @printf("  intra_chain_max:  %.6f A\n", ca_intra[:max])
        @printf("  intra_chain_outlier_fraction: %.3f\n", ca_intra[:outlier_fraction])
    end

    @printf("Final confidence\n")
    @printf("  mean_pLDDT: %.4f\n", mean(result.plddt))
    @printf("  min_pLDDT: %.4f\n", minimum(result.plddt))
    @printf("  max_pLDDT: %.4f\n", maximum(result.plddt))
    if result.pae !== nothing
        @printf("  mean_PAE: %.4f\n", mean(result.pae))
        @printf("  max_PAE_cap: %.4f\n", result.pae_max)
        @printf("  pTM: %.6f\n", result.ptm)
    else
        println("  PAE/pTM unavailable (checkpoint has no predicted_aligned_error_head weights).")
    end

    println("Saved Julia run to ", out_path)
    println("Saved Julia PDB to ", pdb_path, " (atoms=", atoms_written, ")")
end

function main()
    # Filter out --gpu flag from ARGS
    use_gpu = "--gpu" in ARGS
    positional = filter(a -> a != "--gpu", ARGS)
    if length(positional) < 3
        error("Usage: julia run_af2_template_hybrid_jl.jl [--gpu] <params_path_or_hf_filename> <input_dump.npz> <out.npz>")
    end
    params_spec, dump_path, out_path = positional[1], positional[2], positional[3]
    if use_gpu
        if !CUDA.functional()
            error("--gpu flag specified but CUDA is not functional")
        end
        println("GPU mode enabled: ", CUDA.name(CUDA.device()))
    end
    params_path = Alphafold2.resolve_af2_params_path(
        params_spec;
        repo_id=get(ENV, "AF2_HF_REPO_ID", Alphafold2.AF2_HF_REPO_ID),
        revision=get(ENV, "AF2_HF_REVISION", Alphafold2.AF2_HF_REVISION),
    )
    println("Using params file: ", params_path)
    return run_af2_template_hybrid(params_path, dump_path, out_path; use_gpu=use_gpu)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
