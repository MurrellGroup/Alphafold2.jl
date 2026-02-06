#!/usr/bin/env bash
set -euo pipefail

AF2_REPO="${1:-/Users/benmurrell/JuliaM3/AF2JuliaPort/alphafold}"
PARAMS="${2:-/Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz}"
JULIA_BIN="${3:-/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia}"
PTM_PARAMS="${4:-/Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1_ptm.npz}"

ROOT="/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl"
PARITY_DIR="${ROOT}/scripts/parity"
E2E_DIR="${ROOT}/scripts/end_to_end"

if [[ ! -f "${PARAMS}" ]]; then
  echo "Missing params file: ${PARAMS}" >&2
  exit 1
fi

JULIA_ENV_PREFIX=(
  env
  JULIA_PROJECT=/Users/benmurrell/JuliaM3/juliaESM
  JULIA_DEPOT_PATH=/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot
  JULIA_PKG_OFFLINE=true
  JULIA_PKG_PRECOMPILE_AUTO=0
)

echo "== IPA real-weight parity =="
JAX_PLATFORMS=cpu python3.11 "${PARITY_DIR}/dump_ipa_from_params_py.py" \
  --alphafold-repo "${AF2_REPO}" \
  --params "${PARAMS}" \
  --out "${PARITY_DIR}/ipa_real_dump.npz" \
  --n 9
"${JULIA_ENV_PREFIX[@]}" "${JULIA_BIN}" --startup-file=no --history-file=no \
  "${PARITY_DIR}/check_ipa_jl.jl" "${PARITY_DIR}/ipa_real_dump.npz" 3e-4

echo "== Evoformer real-weight parity (blocks 0, 23, 47) =="
for b in 0 23 47; do
  out_dir="${PARITY_DIR}/evo_real_dump_block${b}"
  JAX_PLATFORMS=cpu python3.11 "${PARITY_DIR}/dump_evoformer_iteration_from_params_py.py" \
    --alphafold-repo "${AF2_REPO}" \
    --params "${PARAMS}" \
    --out-dir "${out_dir}" \
    --block "${b}" \
    --n-seq 3 \
    --n-res 7

  "${JULIA_ENV_PREFIX[@]}" "${JULIA_BIN}" --startup-file=no --history-file=no \
    "${PARITY_DIR}/check_evoformer_iteration_jl.jl" "${out_dir}" 1e-3

done

echo "== FoldIterationCore real-weight parity =="
JAX_PLATFORMS=cpu python3.11 "${PARITY_DIR}/dump_fold_iteration_core_from_params_py.py" \
  --alphafold-repo "${AF2_REPO}" \
  --params "${PARAMS}" \
  --out "${PARITY_DIR}/fold_core_real_dump.npz" \
  --n 9
"${JULIA_ENV_PREFIX[@]}" "${JULIA_BIN}" --startup-file=no --history-file=no \
  "${PARITY_DIR}/check_fold_iteration_core_jl.jl" "${PARITY_DIR}/fold_core_real_dump.npz" 2e-3

echo "== GenerateAffinesCore real-weight parity (8 layers) =="
JAX_PLATFORMS=cpu python3.11 "${PARITY_DIR}/dump_generate_affines_core_from_params_py.py" \
  --alphafold-repo "${AF2_REPO}" \
  --params "${PARAMS}" \
  --out "${PARITY_DIR}/generate_affines_core_real_dump.npz" \
  --n 9
"${JULIA_ENV_PREFIX[@]}" "${JULIA_BIN}" --startup-file=no --history-file=no \
  "${PARITY_DIR}/check_generate_affines_core_jl.jl" "${PARITY_DIR}/generate_affines_core_real_dump.npz" 2e-2

echo "== MultiRigidSidechain real-weight parity =="
JAX_PLATFORMS=cpu python3.11 "${PARITY_DIR}/dump_sidechain_from_params_py.py" \
  --alphafold-repo "${AF2_REPO}" \
  --params "${PARAMS}" \
  --out "${PARITY_DIR}/sidechain_real_dump.npz" \
  --n 9
"${JULIA_ENV_PREFIX[@]}" "${JULIA_BIN}" --startup-file=no --history-file=no \
  "${PARITY_DIR}/check_sidechain_jl.jl" "${PARITY_DIR}/sidechain_real_dump.npz" 4e-3

echo "== StructureModuleCore real-weight parity =="
JAX_PLATFORMS=cpu python3.11 "${PARITY_DIR}/dump_structure_module_core_from_params_py.py" \
  --alphafold-repo "${AF2_REPO}" \
  --params "${PARAMS}" \
  --out "${PARITY_DIR}/structure_module_core_real_dump.npz" \
  --n 9
"${JULIA_ENV_PREFIX[@]}" "${JULIA_BIN}" --startup-file=no --history-file=no \
  "${PARITY_DIR}/check_structure_module_core_jl.jl" "${PARITY_DIR}/structure_module_core_real_dump.npz" 2e-2

echo "== TemplateSingleRows + TemplateEmbedding real-weight parity =="
TEMPLATE_INPUT_NPZ="${E2E_DIR}/af2_template_glucagon_pre_evo_r1_for_parity.npz"
JAX_PLATFORMS=cpu python3.11 "${E2E_DIR}/run_af2_template_case_py.py" \
  --alphafold-repo "${AF2_REPO}" \
  --params "${PARAMS}" \
  --template-pdb "${AF2_REPO}/alphafold/common/testdata/glucagon.pdb" \
  --template-chain A \
  --num-recycle 1 \
  --out "${E2E_DIR}/af2_template_glucagon_py_r1_for_parity.npz" \
  --dump-pre-evo "${TEMPLATE_INPUT_NPZ}"

JAX_PLATFORMS=cpu python3.11 "${PARITY_DIR}/dump_template_single_rows_from_params_py.py" \
  --alphafold-repo "${AF2_REPO}" \
  --params "${PARAMS}" \
  --template-npz "${TEMPLATE_INPUT_NPZ}" \
  --out "${PARITY_DIR}/template_single_rows_real_dump.npz"
"${JULIA_ENV_PREFIX[@]}" "${JULIA_BIN}" --startup-file=no --history-file=no \
  "${PARITY_DIR}/check_template_single_rows_jl.jl" "${PARAMS}" "${PARITY_DIR}/template_single_rows_real_dump.npz" 5e-4

JAX_PLATFORMS=cpu python3.11 "${PARITY_DIR}/dump_template_embedding_from_params_py.py" \
  --alphafold-repo "${AF2_REPO}" \
  --params "${PARAMS}" \
  --input-npz "${TEMPLATE_INPUT_NPZ}" \
  --out "${PARITY_DIR}/template_embedding_real_dump.npz"
"${JULIA_ENV_PREFIX[@]}" "${JULIA_BIN}" --startup-file=no --history-file=no \
  "${PARITY_DIR}/check_template_embedding_jl.jl" "${PARAMS}" "${PARITY_DIR}/template_embedding_real_dump.npz" 5e-4

echo "== PredictedLDDT real-weight parity =="
JAX_PLATFORMS=cpu python3.11 "${PARITY_DIR}/dump_predicted_lddt_from_params_py.py" \
  --alphafold-repo "${AF2_REPO}" \
  --params "${PARAMS}" \
  --out "${PARITY_DIR}/predicted_lddt_real_dump.npz" \
  --n-res 17
"${JULIA_ENV_PREFIX[@]}" "${JULIA_BIN}" --startup-file=no --history-file=no \
  "${PARITY_DIR}/check_predicted_lddt_jl.jl" "${PARAMS}" "${PARITY_DIR}/predicted_lddt_real_dump.npz" 5e-4 5e-4

echo "== Distogram / MaskedMSA / ExperimentallyResolved real-weight parity =="
JAX_PLATFORMS=cpu python3.11 "${PARITY_DIR}/dump_distogram_from_params_py.py" \
  --alphafold-repo "${AF2_REPO}" \
  --params "${PARAMS}" \
  --out "${PARITY_DIR}/distogram_real_dump.npz" \
  --n-res 17
"${JULIA_ENV_PREFIX[@]}" "${JULIA_BIN}" --startup-file=no --history-file=no \
  "${PARITY_DIR}/check_distogram_jl.jl" "${PARAMS}" "${PARITY_DIR}/distogram_real_dump.npz" 5e-4 5e-6

JAX_PLATFORMS=cpu python3.11 "${PARITY_DIR}/dump_masked_msa_from_params_py.py" \
  --alphafold-repo "${AF2_REPO}" \
  --params "${PARAMS}" \
  --out "${PARITY_DIR}/masked_msa_real_dump.npz" \
  --n-seq 5 \
  --n-res 17
"${JULIA_ENV_PREFIX[@]}" "${JULIA_BIN}" --startup-file=no --history-file=no \
  "${PARITY_DIR}/check_masked_msa_jl.jl" "${PARAMS}" "${PARITY_DIR}/masked_msa_real_dump.npz" 5e-4

JAX_PLATFORMS=cpu python3.11 "${PARITY_DIR}/dump_experimentally_resolved_from_params_py.py" \
  --alphafold-repo "${AF2_REPO}" \
  --params "${PARAMS}" \
  --out "${PARITY_DIR}/experimentally_resolved_real_dump.npz" \
  --n-res 17
"${JULIA_ENV_PREFIX[@]}" "${JULIA_BIN}" --startup-file=no --history-file=no \
  "${PARITY_DIR}/check_experimentally_resolved_jl.jl" "${PARAMS}" "${PARITY_DIR}/experimentally_resolved_real_dump.npz" 5e-4

if [[ -f "${PTM_PARAMS}" ]]; then
  echo "== PredictedAlignedError real-weight parity (PTM) =="
  JAX_PLATFORMS=cpu python3.11 "${PARITY_DIR}/dump_predicted_aligned_error_py.py" \
    --alphafold-repo "${AF2_REPO}" \
    --params "${PTM_PARAMS}" \
    --out "${PARITY_DIR}/predicted_aligned_error_real_dump.npz" \
    --n-res 17
  "${JULIA_ENV_PREFIX[@]}" "${JULIA_BIN}" --startup-file=no --history-file=no \
    "${PARITY_DIR}/check_predicted_aligned_error_jl.jl" "${PARITY_DIR}/predicted_aligned_error_real_dump.npz" 5e-4 5e-4
else
  echo "== PredictedAlignedError synthetic parity (PTM params not found) =="
  JAX_PLATFORMS=cpu python3.11 "${PARITY_DIR}/dump_predicted_aligned_error_py.py" \
    --alphafold-repo "${AF2_REPO}" \
    --out "${PARITY_DIR}/predicted_aligned_error_synth_dump.npz" \
    --n-res 17 \
    --c-z 128
  "${JULIA_ENV_PREFIX[@]}" "${JULIA_BIN}" --startup-file=no --history-file=no \
    "${PARITY_DIR}/check_predicted_aligned_error_jl.jl" "${PARITY_DIR}/predicted_aligned_error_synth_dump.npz" 5e-4 5e-4
fi

echo "All configured parity checks passed."
