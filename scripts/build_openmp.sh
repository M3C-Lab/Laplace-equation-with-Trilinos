#!/usr/bin/env bash
set -euo pipefail

jobs="${JOBS:-4}"

cmake --preset openmp
cmake --build --preset openmp -j"$jobs"
