#!/bin/bash
# render_fusion_dag.sh — solve an instance with the 910B model and render the
# fusion-DAG solution to DOT + PNG via scripts/visualize.py.
#
# Usage:
#   ./scripts/render_fusion_dag.sh benchmarks/910b-2mm-chained.json
#   ./scripts/render_fusion_dag.sh benchmarks/910b-2mm-accumulate.json
#
# Output: fusion_dag/<name>.dot, fusion_dag/<name>.png (+ <name>_sol.json).
# The 910B parallel model activates when the instance JSON carries the optional
# num_cube_cores / num_vector_cores / cube_capacity fields (else single-context).

set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 <instance.json>"; exit 1; }

INSTANCE="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOLVER="$ROOT/build/mlsys"
OUT="$ROOT/fusion_dag"
NAME="$(basename "$INSTANCE" .json)"
mkdir -p "$OUT"

[[ -x "$SOLVER" ]] || { echo "ERROR: $SOLVER not built (cd build && cmake --build . -j)"; exit 1; }

"$SOLVER" "$INSTANCE" "$OUT/${NAME}_sol.json" | tail -3
python3 "$ROOT/scripts/visualize.py" solution "$INSTANCE" "$OUT/${NAME}_sol.json" "$OUT/${NAME}.dot"
dot -Tpng "$OUT/${NAME}.dot" -o "$OUT/${NAME}.png"
echo "Rendered $OUT/${NAME}.png"
