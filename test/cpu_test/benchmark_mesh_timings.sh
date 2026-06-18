#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mesh_root="$project_root/Mesh"
output_root="$project_root/test/cpu_test"
build_dir="${BUILD_DIR:-$project_root/build-cpu}"
exe="$build_dir/laplace_fem_2d"
runscript="$project_root/scripts/run_cpu.sh"
runs="${RUNS:-10}"
solver_tol="${SOLVER_TOL:-1e-9}"
summary_file="$output_root/cpu_benchmark_summary.txt"
lock_dir="$output_root/.benchmark_lock"
sleep_seconds="${SLEEP_SECONDS:-5}"
case_nx_list="${CASE_NX_LIST:-128 256 512}"

mkdir -p "$output_root"

if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "Another benchmark run appears to be active: $lock_dir" >&2
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

mesh_dirs=()
while IFS= read -r mesh_dir; do
  mesh_name="$(basename "$mesh_dir")"
  nx="$(sed -E 's/^nx=([0-9]+)_ny=([0-9]+)_np=([0-9]+)$/\1/' <<< "$mesh_name")"
  ny="$(sed -E 's/^nx=([0-9]+)_ny=([0-9]+)_np=([0-9]+)$/\2/' <<< "$mesh_name")"
  if [[ " $case_nx_list " == *" $nx "* && "$nx" == "$ny" ]]; then
    mesh_dirs+=("$mesh_dir")
  fi
done < <(find "$mesh_root" -mindepth 1 -maxdepth 1 -type d | sort -V)

if [[ "${#mesh_dirs[@]}" -eq 0 ]]; then
  echo "No mesh directories found under $mesh_root" >&2
  exit 1
fi

{
  echo "CPU benchmark summary"
  echo "runs = $runs"
  echo "solver_tolerance = $solver_tol"
  echo "case_nx_list = $case_nx_list"
  echo
} > "$summary_file"

for mesh_dir in "${mesh_dirs[@]}"; do
  mesh_name="$(basename "$mesh_dir")"
  nx="$(sed -E 's/^nx=([0-9]+)_ny=([0-9]+)_np=([0-9]+)$/\1/' <<< "$mesh_name")"
  ny="$(sed -E 's/^nx=([0-9]+)_ny=([0-9]+)_np=([0-9]+)$/\2/' <<< "$mesh_name")"
  np="$(sed -E 's/^nx=([0-9]+)_ny=([0-9]+)_np=([0-9]+)$/\3/' <<< "$mesh_name")"
  report_file="$output_root/${mesh_name}.txt"

  mesh_sum="0"
  assembly_sum="0"
  solve_sum="0"

  {
    echo "Mesh benchmark report"
    echo "mesh = $mesh_name"
    echo "mesh_dir = $mesh_dir"
    echo "runs = $runs"
    echo "solver_tolerance = $solver_tol"
    echo
    echo "Per-run timings (seconds)"
  } > "$report_file"

  echo "[case] $mesh_name"

  for run_id in $(seq 1 "$runs"); do
    echo "  [run $run_id/$runs] starting"
    output="$(MPI_RANKS="$np" SOLVER_TOL="$solver_tol" "$runscript" "$nx" "$ny" --benchmark 2>&1)"

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
    if [[ "$run_id" -lt "$runs" ]]; then
      echo "  [run $run_id/$runs] sleeping ${sleep_seconds}s"
      sleep "$sleep_seconds"
    fi

    mesh_sum="$(awk -v a="$mesh_sum" -v b="$mesh_load" 'BEGIN {printf "%.15g", a + b}')"
    assembly_sum="$(awk -v a="$assembly_sum" -v b="$assembly" 'BEGIN {printf "%.15g", a + b}')"
    solve_sum="$(awk -v a="$solve_sum" -v b="$linear_solve" 'BEGIN {printf "%.15g", a + b}')"
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

echo "Benchmark reports written to $output_root"
