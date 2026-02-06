#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JULIA_BIN="${JULIA_BIN:-/Users/benmurrell/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia}"
JULIA_PROJECT_ENV="${JULIA_PROJECT:-/Users/benmurrell/JuliaM3/juliaESM}"
JULIA_DEPOT_ENV="${JULIA_DEPOT_PATH:-${ROOT}/.julia_depot:/Users/benmurrell/JuliaM3/juliaESM/.julia_depot}"
PARAMS_PATH="${1:-/Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights/params_model_1.npz}"

run_julia() {
  env \
    JULIA_PROJECT="${JULIA_PROJECT_ENV}" \
    JULIA_DEPOT_PATH="${JULIA_DEPOT_ENV}" \
    JULIA_PKG_OFFLINE="${JULIA_PKG_OFFLINE:-true}" \
    JULIA_PKG_PRECOMPILE_AUTO="0" \
    "${JULIA_BIN}" --startup-file=no --history-file=no "$@"
}

echo "== Running unit + AD smoke tests =="
run_julia "${ROOT}/test/runtests.jl"

if [[ -f "${PARAMS_PATH}" ]]; then
  echo "== Running full-model Zygote gradient check =="
  run_julia "${ROOT}/scripts/gradients/check_full_model_zygote.jl" "${PARAMS_PATH}" ACDEFGHIK
else
  echo "Skipping full-model Zygote gradient check (missing params: ${PARAMS_PATH})"
fi

echo "Smoke checks passed."
