#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-$project_root/build-cpu}"
exe="$build_dir/laplace_preprocess"
mesh_root="$project_root/Mesh"

nx="${1:-16}"
ny="${2:-$nx}"
parts="${3:-${MPI_RANKS:-1}}"
output_dir="${PARTITION_DIR:-$mesh_root/nx=${nx}_ny=${ny}_np=${parts}}"
mesh_file="$output_dir/mesh.h5"
last_partition_file="$output_dir/partition.rank$((parts - 1)).h5"

mkdir -p "$mesh_root"

if [[ -f "$mesh_file" && -f "$last_partition_file" ]]; then
  echo "Reusing existing partitioned mesh at $output_dir"
  exit 0
fi

"$exe" --nx "$nx" --ny "$ny" --parts "$parts" --output-dir "$output_dir"
