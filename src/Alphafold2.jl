module Alphafold2

using LinearAlgebra
using Statistics

using CUDA
using cuDNN
using Flux
using HuggingFaceApi
using NNlib
import Onion
using Onion: @concrete, @layer
using NPZ

# Force cuDNN softmax to use ACCURATE algorithm even with CUDA.FAST_MATH.
# FAST_MATH causes NNlib.softmax → CUDNN_SOFTMAX_FAST which produces NaN
# for large logits (e.g., structure module sidechain atoms with max ~200).
function __init__()
    ext = Base.get_extension(NNlib, :NNlibCUDACUDNNExt)
    if ext !== nothing
        ext.eval(:(softmaxalgo() = cuDNN.CUDNN_SOFTMAX_ACCURATE))
    end
end

include("device_utils.jl")
include("layers.jl")
include("safetensors.jl")
include("weights.jl")
include("tensor_utils.jl")
include("rigid.jl")
include("openfold_utils.jl")
include("residue_constants.jl")
include("openfold_feats.jl")
include("openfold_infer_utils.jl")
include("training_utils.jl")
include("modules/attention.jl")
include("modules/transition.jl")
include("modules/triangle.jl")
include("modules/triangle_attention.jl")
include("modules/outer_product_mean.jl")
include("modules/msa_attention.jl")
include("modules/template_pair_stack.jl")
include("modules/template_embedding.jl")
include("modules/template_single_rows.jl")
include("modules/ipa.jl")
include("modules/evoformer_iteration.jl")
include("modules/fold_iteration_core.jl")
include("modules/generate_affines_core.jl")
include("modules/sidechain.jl")
include("modules/structure_module_core.jl")
include("modules/confidence_heads.jl")
include("modules/output_heads.jl")
include("model_builder.jl")
include("inference.jl")
include("feature_pipeline.jl")
include("high_level_api.jl")

# ── Public API ──
export load_monomer, load_multimer, fold
export AF2Config, AF2Model, AF2InferenceResult
export FoldResult, DEFAULT_MONOMER_WEIGHTS, DEFAULT_MULTIMER_WEIGHTS
export AF2PreparedInputs, RecycleState

# ── Pipeline stages (for research) ──
export prepare_inputs, initial_recycle_state, run_evoformer, run_heads, run_inference

# ── Feature building ──
export build_monomer_features, build_multimer_features

# ── Training / sequence optimization ──
export build_soft_sequence_features, build_hard_sequence_features
export build_basic_masks, mean_plddt_loss

# ── Confidence metrics ──
export compute_predicted_aligned_error, compute_tm, compute_plddt

# ── Device helpers ──
export gpu_available, to_gpu, to_cpu

# ── Layers (for custom models) ──
export LayerNormFirst, LinearFirst

# ── Weight loading (advanced) ──
export af2_params_read, resolve_af2_params_path
export AF2_HF_REPO_ID, AF2_HF_REVISION

end
