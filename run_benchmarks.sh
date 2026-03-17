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

CSV_FILE="benchmark_$(date +%Y-%m-%d_%H-%M-%S).csv"
echo "Instance,Partition,Build,Sol-Evo,Final" > "$CSV_FILE"

for f in "$BENCH_DIR"/*.json; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .json)
    out="${BUILD_DIR}/${name}_sol.json"
    
    start=$(date +%s%N)
    # Capture stderr as well to get the Summary section
    output=$(timeout 120 "$SOLVER" "$f" "$out" 2>&1)
    end=$(date +%s%N)
    ms=$(( (end - start) / 1000000 ))
    
    # Parse costs from the Summary section at the end of output
    partition_cost=$(echo "$output" | grep -E "^\s+Partition:" | tail -n 1 | awk '{print $2}')
    build_cost=$(echo "$output" | grep -E "^\s+Build:" | tail -n 1 | awk '{print $2}')
    sol_evo_cost=$(echo "$output" | grep -E "^\s+Sol-Evo:" | tail -n 1 | awk '{print $2}')
    final_cost=$(echo "$output" | grep -E "^\s+Final:" | tail -n 1 | awk '{print $2}')
    
    # If parsing failed, fallback to "Total:" if available
    if [ -z "$final_cost" ]; then
        final_cost=$(echo "$output" | grep "Total:" | tail -n 1 | sed 's/Total: //')
    fi
    
    # Clean up any potential carriage returns or extra whitespace
    partition_cost=$(echo "$partition_cost" | tr -d '\r' | xargs)
    build_cost=$(echo "$build_cost" | tr -d '\r' | xargs)
    sol_evo_cost=$(echo "$sol_evo_cost" | tr -d '\r' | xargs)
    final_cost=$(echo "$final_cost" | tr -d '\r' | xargs)
    
    printf "%-20s %12s  (%5dms)\n" "$name" "$final_cost" "$ms"
    
    # Record to CSV
    echo "$name,$partition_cost,$build_cost,$sol_evo_cost,$final_cost" >> "$CSV_FILE"
    
    if [ -f "$out" ]; then
        python3 "$(dirname "$0")/visualize.py" solution "$f" "$out" "${BUILD_DIR}/${name}_sol.dot" > /dev/null 2>&1
    fi
    
    count=$((count + 1))
done

echo ""
echo "$count benchmarks completed. Results written to $CSV_FILE"