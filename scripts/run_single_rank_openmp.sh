#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BUILD_DIR="${BUILD_DIR:-$project_root/build/openmp}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-threads}"

"$project_root/scripts/run_openmp.sh" "${1:-16}" "${2:-${1:-16}}" "${@:3}"
