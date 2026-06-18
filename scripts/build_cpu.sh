#!/usr/bin/env bash
set -euo pipefail

jobs="${JOBS:-4}"

cmake --preset cpu
cmake --build --preset cpu -j"$jobs"
