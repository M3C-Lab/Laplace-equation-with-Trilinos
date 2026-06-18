#!/usr/bin/env bash
set -euo pipefail

jobs="${JOBS:-4}"

cmake --preset cuda
cmake --build --preset cuda -j"$jobs"
