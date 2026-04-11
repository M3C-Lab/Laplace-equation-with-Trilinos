#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runs="${RUNS:-10}"
nx="${NX:-256}"
ny="${NY:-256}"

extract_total_seconds() {
  awk '/^TIMER total_seconds / {print $3}' <<<"$1" | tail -n 1
}

run_case() {
  local label="$1"
  local configure_preset="$2"
  local build_preset="$3"
  local runner="$4"
  local omp_threads="$5"
  local sum="0.0"

  echo "== $label =="
  cmake --preset "$configure_preset" >/dev/null
  cmake --build --preset "$build_preset" -j"${JOBS:-4}" >/dev/null

  for ((i = 1; i <= runs; ++i)); do
    local output
    output="$(OMP_NUM_THREADS="$omp_threads" "$runner" "$nx" "$ny" --benchmark 2>&1)"
    local seconds
    seconds="$(extract_total_seconds "$output")"
    if [[ -z "$seconds" ]]; then
      echo "$output"
      echo "Failed to extract TIMER total_seconds for $label run $i" >&2
      exit 1
    fi
    printf 'run %02d: %s s\n' "$i" "$seconds"
    sum="$(awk -v a="$sum" -v b="$seconds" 'BEGIN { printf "%.12f", a + b }')"
  done

  local average
  average="$(awk -v s="$sum" -v n="$runs" 'BEGIN { printf "%.6f", s / n }')"
  printf 'average: %s s\n\n' "$average"
  printf '%s,%s\n' "$label" "$average" >> "$project_root/build/benchmark_results.csv"
}

mkdir -p "$project_root/build"
cat > "$project_root/build/benchmark_results.csv" <<'EOF'
mode,average_seconds
EOF

run_case "single-rank-serial" "single-rank-serial" "single-rank-serial" \
  "$project_root/scripts/run_single_rank_serial.sh" "1"
run_case "single-rank-openmp" "single-rank-openmp" "single-rank-openmp" \
  "$project_root/scripts/run_single_rank_openmp.sh" "${OPENMP_THREADS:-4}"
run_case "single-rank-cuda" "single-rank-cuda" "single-rank-cuda" \
  "$project_root/scripts/run_single_rank_cuda.sh" "1"

echo "Saved averages to $project_root/build/benchmark_results.csv"
echo "Comparison against the first mode:"
awk -F, '
  NR == 2 { baseline = $2 }
  NR >= 2 {
    speedup = (baseline > 0 && $2 > 0) ? baseline / $2 : 0
    printf "%s: average = %.6f s, speedup = %.3fx\n", $1, $2, speedup
  }
' "$project_root/build/benchmark_results.csv"
