#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"
JULIA_PROJECT_ENV="${JULIA_PROJECT:-${ROOT}}"

run_julia() {
  env \
    JULIA_PROJECT="${JULIA_PROJECT_ENV}" \
    JULIA_PKG_PRECOMPILE_AUTO="0" \
    "${JULIA_BIN}" --startup-file=no --history-file=no "$@"
}

echo "== Running unit tests =="
run_julia "${ROOT}/test/runtests.jl"

PARAMS_PATH="${1:-}"
if [[ -n "${PARAMS_PATH}" && -f "${PARAMS_PATH}" ]]; then
  echo "== Running full-model Zygote gradient check =="
  run_julia "${ROOT}/scripts/gradients/check_full_model_zygote.jl" "${PARAMS_PATH}" ACDEFGHIK
else
  echo "Skipping full-model Zygote gradient check (no params path provided or file missing)"
fi

echo "Smoke checks passed."
