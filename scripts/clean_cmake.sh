#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-$project_root/build}"

rm -rf \
  "$build_dir/CMakeCache.txt" \
  "$build_dir/CMakeFiles" \
  "$build_dir/Makefile" \
  "$build_dir/cmake_install.cmake" \
  "$build_dir/liblaplace_fem_core.a" \
  "$build_dir/laplace_fem_2d"
