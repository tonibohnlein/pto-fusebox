#!/bin/bash
# solve_and_verify.sh — Run solver on one instance, verify with reference evaluator.
#
# Usage:
#   ./scripts/solve_and_verify.sh benchmarks/custom-enc-dec-2x3.json
#   ./scripts/solve_and_verify.sh benchmarks/mlsys-2026-17.json

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <instance.json>"
    exit 1
fi

INSTANCE="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
BUILD="$ROOT/build"
SOLVER="$BUILD/mlsys"
EVALUATOR="$BUILD/reference_evaluator"

NAME=$(basename "$INSTANCE" .json)
SOL="/tmp/${NAME}_sol.json"

for exe in "$SOLVER" "$EVALUATOR"; do
    if [[ ! -x "$exe" ]]; then
        echo "ERROR: $exe not found. Run: cd build && cmake .. && cmake --build . -j\$(nproc)"
        exit 1
    fi
done

echo "=== Solving $NAME ==="
"$SOLVER" "$INSTANCE" "$SOL"
echo ""

if [[ ! -f "$SOL" ]]; then
    echo "ERROR: solver did not produce $SOL"
    exit 1
fi

echo "=== Verifying with reference evaluator ==="
"$EVALUATOR" "$INSTANCE" "$SOL"
