#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mesh_root="$project_root/Mesh"
output_root="$project_root/test/cuda_test/np1_scale_test"
build_dir="${BUILD_DIR:-$project_root/build-cuda}"
exe="$build_dir/laplace_fem_2d"
runscript="$project_root/scripts/run_cuda.sh"
runs="${RUNS:-10}"
solver_tol="${SOLVER_TOL:-1e-9}"
summary_file="$output_root/cuda_np1_scale_summary.txt"
lock_dir="$output_root/.benchmark_lock"
sleep_seconds="${SLEEP_SECONDS:-5}"
mpi_exec="${MPI_EXEC:-/home/xuanming/lib/mpich-4.3.2/bin/mpiexec}"

mkdir -p "$output_root"

if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "Another CUDA benchmark run appears to be active: $lock_dir" >&2
  exit 1
fi

cleanup() {
  rmdir "$lock_dir" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

if [[ ! -x "$exe" ]]; then
  echo "Executable not found: $exe" >&2
  exit 1
fi

if [[ ! -x "$runscript" ]]; then
  echo "Run script not found: $runscript" >&2
  exit 1
fi

mesh_dirs=(
  "$mesh_root/nx=256_ny=256_np=1"
  "$mesh_root/nx=512_ny=512_np=1"
  "$mesh_root/nx=1024_ny=1024_np=1"
)

for mesh_dir in "${mesh_dirs[@]}"; do
  if [[ ! -d "$mesh_dir" ]]; then
    echo "Required mesh directory not found: $mesh_dir" >&2
    exit 1
  fi
done

{
  echo "CUDA np=1 scale benchmark summary"
  echo "runs = $runs"
  echo "solver_tolerance = $solver_tol"
  echo "mpi_exec = $mpi_exec"
  echo "cases = nx=256_ny=256_np=1 nx=512_ny=512_np=1 nx=1024_ny=1024_np=1"
  echo
} > "$summary_file"

for mesh_dir in "${mesh_dirs[@]}"; do
  mesh_name="$(basename "$mesh_dir")"
  nx="$(sed -E 's/^nx=([0-9]+)_ny=([0-9]+)_np=([0-9]+)$/\1/' <<< "$mesh_name")"
  ny="$(sed -E 's/^nx=([0-9]+)_ny=([0-9]+)_np=([0-9]+)$/\2/' <<< "$mesh_name")"
  np=1
  report_file="$output_root/${mesh_name}.txt"

  mesh_sum="0"
  assembly_sum="0"
  solve_sum="0"

  {
    echo "CUDA np=1 scale benchmark report"
    echo "mesh = $mesh_name"
    echo "mesh_dir = $mesh_dir"
    echo "runs = $runs"
    echo "solver_tolerance = $solver_tol"
    echo "mpi_exec = $mpi_exec"
    echo
    echo "Per-run timings (seconds)"
  } > "$report_file"

  echo "[case] $mesh_name"

  for run_id in $(seq 1 "$runs"); do
    echo "  [run $run_id/$runs] starting"
    output="$(
      MPI_EXEC="$mpi_exec" \
      MPI_RANKS="$np" \
      OMP_NUM_THREADS=1 \
      SOLVER_TOL="$solver_tol" \
      "$runscript" "$nx" "$ny" --benchmark 2>&1
    )"

    mesh_load="$(awk '/mesh_load = / {print $3}' <<< "$output" | tail -n 1)"
    assembly="$(awk '/assembly = / {print $3}' <<< "$output" | tail -n 1)"
    linear_solve="$(awk '/linear_solve = / {print $3}' <<< "$output" | tail -n 1)"

    if [[ -z "$mesh_load" || -z "$assembly" || -z "$linear_solve" ]]; then
      echo "Failed to parse timing output for $mesh_name run $run_id" >&2
      echo "$output" >&2
      exit 1
    fi

    printf 'run %02d: mesh_load=%s assembly=%s linear_solve=%s\n' \
      "$run_id" "$mesh_load" "$assembly" "$linear_solve" >> "$report_file"
    echo "  [run $run_id/$runs] mesh_load=$mesh_load assembly=$assembly linear_solve=$linear_solve"

    mesh_sum="$(awk -v a="$mesh_sum" -v b="$mesh_load" 'BEGIN {printf "%.15g", a + b}')"
    assembly_sum="$(awk -v a="$assembly_sum" -v b="$assembly" 'BEGIN {printf "%.15g", a + b}')"
    solve_sum="$(awk -v a="$solve_sum" -v b="$linear_solve" 'BEGIN {printf "%.15g", a + b}')"

    if [[ "$run_id" -lt "$runs" ]]; then
      echo "  [run $run_id/$runs] sleeping ${sleep_seconds}s"
      sleep "$sleep_seconds"
    fi
  done

  mesh_avg="$(awk -v sum="$mesh_sum" -v count="$runs" 'BEGIN {printf "%.15g", sum / count}')"
  assembly_avg="$(awk -v sum="$assembly_sum" -v count="$runs" 'BEGIN {printf "%.15g", sum / count}')"
  solve_avg="$(awk -v sum="$solve_sum" -v count="$runs" 'BEGIN {printf "%.15g", sum / count}')"

  {
    echo
    echo "Averages over $runs runs (seconds)"
    echo "mesh_load_avg = $mesh_avg"
    echo "assembly_avg = $assembly_avg"
    echo "linear_solve_avg = $solve_avg"
  } >> "$report_file"

  {
    echo "$mesh_name"
    echo "  mesh_load_avg = $mesh_avg"
    echo "  assembly_avg = $assembly_avg"
    echo "  linear_solve_avg = $solve_avg"
    echo
  } >> "$summary_file"

  echo "[case] $mesh_name completed"
  echo "[case] $mesh_name sleeping ${sleep_seconds}s"
  sleep "$sleep_seconds"
done

echo "CUDA np=1 scale benchmark reports written to $output_root"
