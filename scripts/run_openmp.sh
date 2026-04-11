#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-$project_root/build}"
exe="$build_dir/laplace_fem_2d"

nx="${1:-16}"
ny="${2:-$nx}"
threads="${OMP_NUM_THREADS:-4}"

export OMP_NUM_THREADS="$threads"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-threads}"

"$exe" "$nx" "$ny" "${@:3}"
