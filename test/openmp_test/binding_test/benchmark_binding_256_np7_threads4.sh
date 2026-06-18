#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output_root="$project_root/test/openmp_test"
runscript="$project_root/scripts/run_openmp.sh"
summary_file="$output_root/nx=256_ny=256_np=7_threads=4_binding_summary.txt"
sleep_seconds="${SLEEP_SECONDS:-5}"
solver_tol="${SOLVER_TOL:-1e-9}"
mpi_ranks=7
omp_threads=4
nx=256
ny=256

mkdir -p "$output_root"

if [[ ! -x "$runscript" ]]; then
  echo "Run script not found: $runscript" >&2
  exit 1
fi

strategies=(
  "spread_threads:spread:threads"
  "spread_cores:spread:cores"
  "close_threads:close:threads"
  "close_cores:close:cores"
  "false_threads:false:threads"
  "false_cores:false:cores"
)

{
  echo "OpenMP binding strategy summary"
  echo "nx = $nx"
  echo "ny = $ny"
  echo "np = $mpi_ranks"
  echo "threads = $omp_threads"
  echo "solver_tolerance = $solver_tol"
  echo "sleep_seconds = $sleep_seconds"
  echo
} > "$summary_file"

for strategy in "${strategies[@]}"; do
  IFS=":" read -r tag proc_bind places <<< "$strategy"
  report_file="$output_root/nx=256_ny=256_np=7_threads=4_binding=${tag}.txt"

  echo "[binding=$tag] starting"
  output="$(MPI_RANKS="$mpi_ranks" OMP_NUM_THREADS="$omp_threads" \
    OMP_PROC_BIND="$proc_bind" OMP_PLACES="$places" SOLVER_TOL="$solver_tol" \
    "$runscript" "$nx" "$ny" --benchmark 2>&1)"

  mesh_load="$(awk '/mesh_load = / {print $3}' <<< "$output" | tail -n 1)"
  assembly="$(awk '/assembly = / {print $3}' <<< "$output" | tail -n 1)"
  linear_solve="$(awk '/linear_solve = / {print $3}' <<< "$output" | tail -n 1)"
  iterations="$(awk '/Belos iterations:/ {print $3}' <<< "$output" | tail -n 1)"

  if [[ -z "$mesh_load" || -z "$assembly" || -z "$linear_solve" || -z "$iterations" ]]; then
    echo "Failed to parse benchmark output for binding=$tag" >&2
    echo "$output" >&2
    exit 1
  fi

  {
    echo "OpenMP binding benchmark"
    echo "nx = $nx"
    echo "ny = $ny"
    echo "np = $mpi_ranks"
    echo "threads = $omp_threads"
    echo "OMP_PROC_BIND = $proc_bind"
    echo "OMP_PLACES = $places"
    echo "solver_tolerance = $solver_tol"
    echo
    echo "$output"
  } > "$report_file"

  {
    echo "binding = $tag"
    echo "  OMP_PROC_BIND = $proc_bind"
    echo "  OMP_PLACES = $places"
    echo "  iterations = $iterations"
    echo "  mesh_load = $mesh_load"
    echo "  assembly = $assembly"
    echo "  linear_solve = $linear_solve"
    echo
  } >> "$summary_file"

  echo "[binding=$tag] mesh_load=$mesh_load assembly=$assembly linear_solve=$linear_solve"
  if [[ "$tag" != "false_cores" ]]; then
    echo "[binding=$tag] sleeping ${sleep_seconds}s"
    sleep "$sleep_seconds"
  fi
done

echo "Benchmark reports written to $output_root"
