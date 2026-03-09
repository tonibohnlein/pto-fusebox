#!/bin/bash
# Run all benchmarks and report results.
# Usage: ./run_benchmarks.sh [build_dir] [benchmark_dir]

BUILD_DIR="${1:-build}"
BENCH_DIR="${2:-benchmarks}"
SOLVER="${BUILD_DIR}/mlsys"

if [ ! -f "$SOLVER" ]; then
    echo "Solver not found at $SOLVER. Build first:"
    echo "  mkdir -p build && cd build && cmake .. && cmake --build . -j\$(nproc)"
    exit 1
fi

if [ ! -d "$BENCH_DIR" ]; then
    echo "Benchmark directory $BENCH_DIR not found."
    echo "Copy the benchmarks folder into the project root."
    exit 1
fi

echo "Solver: $SOLVER"
echo "Benchmarks: $BENCH_DIR"
echo ""

total_score=0
count=0

for f in "$BENCH_DIR"/*.json; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .json)
    out="${BUILD_DIR}/${name}_sol.json"
    
    start=$(date +%s%N)
    result=$(timeout 120 "$SOLVER" "$f" "$out" 2>/dev/null | grep "Total:")
    end=$(date +%s%N)
    ms=$(( (end - start) / 1000000 ))
    
    cost=$(echo "$result" | sed 's/Total: //')
    printf "%-20s %12s  (%5dms)\n" "$name" "$cost" "$ms"
    count=$((count + 1))
done

echo ""
echo "$count benchmarks completed."