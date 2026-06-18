#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mpi_exec="${MPI_EXEC:-/home/xuanming/lib/mpich-4.3.2/bin/mpiexec}"
mpi_ranks="${MPI_RANKS:-1}"

export BUILD_DIR="${BUILD_DIR:-$project_root/build-cuda}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-threads}"
nx="${1:-16}"
ny="${2:-${1:-16}}"
partition_dir="${PARTITION_DIR:-$project_root/Mesh/nx=${nx}_ny=${ny}_np=${mpi_ranks}}"
solver_tol="${SOLVER_TOL:-1e-9}"
export LAPLACE_FEM_OUTPUT_DIR="${LAPLACE_FEM_OUTPUT_DIR:-$BUILD_DIR}"

if [[ "$mpi_ranks" -gt 1 ]]; then
  "$mpi_exec" -np "$mpi_ranks" "$BUILD_DIR/laplace_fem_2d" --partition-dir "$partition_dir" --solver-tol "$solver_tol" "${@:3}"
else
  "$BUILD_DIR/laplace_fem_2d" --partition-dir "$partition_dir" --solver-tol "$solver_tol" "${@:3}"
fi
