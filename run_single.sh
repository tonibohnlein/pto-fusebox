#!/bin/bash
# Run a single benchmark instance and generate its visualization.
# Usage: ./run_single.sh <path_to_benchmark.json> [build_dir]

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_benchmark.json> [build_dir]"
    exit 1
fi

INSTANCE="$1"
BUILD_DIR="${2:-build}"
SOLVER="${BUILD_DIR}/mlsys"

if [ ! -f "$SOLVER" ]; then
    echo "Solver not found at $SOLVER. Build it first."
    exit 1
fi

if [ ! -f "$INSTANCE" ]; then
    echo "Instance not found: $INSTANCE"
    exit 1
fi

name=$(basename "$INSTANCE" .json)
out="${BUILD_DIR}/${name}_sol.json"
dot_out="${BUILD_DIR}/${name}_sol.dot"

echo "Running solver on $name..."
start=$(date +%s%N)
"$SOLVER" "$INSTANCE" "$out"
end=$(date +%s%N)
ms=$(( (end - start) / 1000000 ))

echo "Solver finished in ${ms}ms."

if [ -f "$out" ]; then
    echo "Generating visualization at $dot_out..."
    python3 "$(dirname "$0")/visualize.py" solution "$INSTANCE" "$out" "$dot_out"
    echo "Done. You can render the dot file using:"
    echo "  dot -Tpng $dot_out -o ${BUILD_DIR}/${name}_sol.png"
else
    echo "Error: Output solution file $out was not generated."
    exit 1
fi
