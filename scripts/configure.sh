#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-$project_root/build}"
trilinos_root="${TRILINOS_ROOT:-/home/xuanming/lib/Trilinos}"
backend="${LAPLACE_FEM_BACKEND:-OPENMP}"
mpi_compiler="${MPI_CXX_COMPILER:-/home/xuanming/lib/mpich-4.3.2/bin/mpicxx}"
compiler="${CXX_COMPILER:-$trilinos_root/bin/nvcc_wrapper}"

export NVCC_WRAPPER_DEFAULT_COMPILER="${NVCC_WRAPPER_DEFAULT_COMPILER:-$mpi_compiler}"

cmake -S "$project_root" -B "$build_dir" \
  -DTrilinos_ROOT="$trilinos_root" \
  -DCMAKE_CXX_COMPILER="$compiler" \
  -DKokkos_LAUNCH_COMPILER=OFF \
  -DLAPLACE_FEM_BACKEND="$backend"
