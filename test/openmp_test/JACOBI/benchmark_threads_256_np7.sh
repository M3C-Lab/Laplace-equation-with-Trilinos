#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
output_root="$project_root/test/openmp_test/JACOBI"
runscript="$project_root/scripts/run_openmp.sh"
summary_file="$output_root/nx=256_ny=256_np=7_threads_1_20_summary.txt"
sleep_seconds="${SLEEP_SECONDS:-5}"
solver_tol="${SOLVER_TOL:-1e-9}"
runs="${RUNS:-10}"
mpi_ranks=7
nx=256
ny=256
prec_type="${LAPLACE_FEM_PREC_TYPE:-JACOBI}"

mkdir -p "$output_root"

if [[ ! -x "$runscript" ]]; then
  echo "Run script not found: $runscript" >&2
  exit 1
fi

{
  echo "OpenMP thread scan summary"
  echo "preconditioner = $prec_type"
  echo "nx = $nx"
  echo "ny = $ny"
  echo "np = $mpi_ranks"
  echo "solver_tolerance = $solver_tol"
  echo "sleep_seconds = $sleep_seconds"
  echo "runs = $runs"
  echo
} > "$summary_file"

for thread_count in $(seq 1 20); do
  report_file="$output_root/nx=256_ny=256_np=7_threads=${thread_count}.txt"
  {
    echo "OpenMP thread benchmark"
    echo "preconditioner = $prec_type"
    echo "nx = $nx"
    echo "ny = $ny"
    echo "np = $mpi_ranks"
    echo "threads = $thread_count"
    echo "solver_tolerance = $solver_tol"
    echo "runs = $runs"
  } > "$report_file"

  mesh_sum="0"
  assembly_sum="0"
  solve_sum="0"
  last_iterations=""

  echo "[threads=$thread_count] starting"
  for run_id in $(seq 1 "$runs"); do
    output="$(MPI_RANKS="$mpi_ranks" OMP_NUM_THREADS="$thread_count" SOLVER_TOL="$solver_tol" \
      LAPLACE_FEM_PREC_TYPE="$prec_type" "$runscript" "$nx" "$ny" --benchmark 2>&1)"

    mesh_load="$(awk '/mesh_load = / {print $3}' <<< "$output" | tail -n 1)"
    assembly="$(awk '/assembly = / {print $3}' <<< "$output" | tail -n 1)"
    linear_solve="$(awk '/linear_solve = / {print $3}' <<< "$output" | tail -n 1)"
    iterations="$(awk '/Belos iterations:/ {print $3}' <<< "$output" | tail -n 1)"

    if [[ -z "$mesh_load" || -z "$assembly" || -z "$linear_solve" || -z "$iterations" ]]; then
      echo "Failed to parse benchmark output for threads=$thread_count run=$run_id" >&2
      echo "$output" >&2
      exit 1
    fi

    {
      echo
      echo "run $(printf '%02d' "$run_id")"
      echo "  iterations = $iterations"
      echo "  mesh_load = $mesh_load"
      echo "  assembly = $assembly"
      echo "  linear_solve = $linear_solve"
    } >> "$report_file"

    mesh_sum="$(awk -v a="$mesh_sum" -v b="$mesh_load" 'BEGIN {printf "%.15g", a + b}')"
    assembly_sum="$(awk -v a="$assembly_sum" -v b="$assembly" 'BEGIN {printf "%.15g", a + b}')"
    solve_sum="$(awk -v a="$solve_sum" -v b="$linear_solve" 'BEGIN {printf "%.15g", a + b}')"
    last_iterations="$iterations"

    echo "[threads=$thread_count run=$run_id/$runs] mesh_load=$mesh_load assembly=$assembly linear_solve=$linear_solve"
    if [[ "$run_id" -lt "$runs" ]]; then
      echo "[threads=$thread_count run=$run_id/$runs] sleeping ${sleep_seconds}s"
      sleep "$sleep_seconds"
    fi
  done

  mesh_avg="$(awk -v sum="$mesh_sum" -v count="$runs" 'BEGIN {printf "%.15g", sum / count}')"
  assembly_avg="$(awk -v sum="$assembly_sum" -v count="$runs" 'BEGIN {printf "%.15g", sum / count}')"
  solve_avg="$(awk -v sum="$solve_sum" -v count="$runs" 'BEGIN {printf "%.15g", sum / count}')"

  {
    echo
    echo "Averages over $runs runs"
    echo "  iterations = $last_iterations"
    echo "  mesh_load_avg = $mesh_avg"
    echo "  assembly_avg = $assembly_avg"
    echo "  linear_solve_avg = $solve_avg"
  } >> "$report_file"

  {
    echo "threads = $thread_count"
    echo "  iterations = $last_iterations"
    echo "  mesh_load_avg = $mesh_avg"
    echo "  assembly_avg = $assembly_avg"
    echo "  linear_solve_avg = $solve_avg"
    echo
  } >> "$summary_file"

  echo "[threads=$thread_count] mesh_load_avg=$mesh_avg assembly_avg=$assembly_avg linear_solve_avg=$solve_avg"
  if [[ "$thread_count" -lt 20 ]]; then
    echo "[threads=$thread_count] sleeping ${sleep_seconds}s"
    sleep "$sleep_seconds"
  fi
done

echo "Benchmark reports written to $output_root"
