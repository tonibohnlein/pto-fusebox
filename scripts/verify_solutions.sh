#!/bin/bash
# verify_solutions.sh — Run solver on all benchmarks and verify with reference evaluator.
#
# Usage:
#   ./scripts/verify_solutions.sh                  # all benchmarks
#   ./scripts/verify_solutions.sh custom-enc*.json  # specific pattern
#
# Requires: build/mlsys and build/reference_evaluator (cmake --build build)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
BUILD="$ROOT/build"
BENCH="$ROOT/benchmarks"
SOLVER="$BUILD/mlsys"
EVALUATOR="$BUILD/reference_evaluator"
OUT_DIR="$BUILD/verify_tmp"

mkdir -p "$OUT_DIR"

# Check executables exist
for exe in "$SOLVER" "$EVALUATOR"; do
    if [[ ! -x "$exe" ]]; then
        echo "ERROR: $exe not found. Run: cd build && cmake .. && cmake --build . -j\$(nproc)"
        exit 1
    fi
done

# Determine which benchmarks to run
if [[ $# -gt 0 ]]; then
    PATTERNS=("$@")
    FILES=()
    for pat in "${PATTERNS[@]}"; do
        for f in "$BENCH"/$pat; do
            [[ -f "$f" ]] && FILES+=("$f")
        done
    done
else
    FILES=("$BENCH"/*.json)
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No benchmark files found."
    exit 1
fi

PASS=0
FAIL=0
ERRORS=()

echo "=== Verifying ${#FILES[@]} benchmarks ==="
echo ""

for instance in "${FILES[@]}"; do
    name=$(basename "$instance" .json)
    sol="$OUT_DIR/${name}_sol.json"

    printf "%-45s " "$name"

    # Run solver (capture stderr for warnings, suppress stdout)
    solver_stderr=$("$SOLVER" "$instance" "$sol" 2>&1) || {
        printf "SOLVER_ERROR\n"
        FAIL=$((FAIL + 1))
        ERRORS+=("$name: solver failed")
        continue
    }

    if [[ ! -f "$sol" ]]; then
        printf "NO_OUTPUT\n"
        FAIL=$((FAIL + 1))
        ERRORS+=("$name: no solution file generated")
        continue
    fi

    # Check for solver warnings
    warnings=$(echo "$solver_stderr" | grep -c "WARNING" || true)

    # Run reference evaluator
    eval_output=$("$EVALUATOR" "$instance" "$sol" 2>&1) || {
        printf "EVAL_ERROR\n"
        FAIL=$((FAIL + 1))
        ERRORS+=("$name: evaluator failed")
        continue
    }

    # Parse evaluator output
    if echo "$eval_output" | grep -q "Cost calculation seems correct"; then
        cost=$(echo "$eval_output" | grep "Computed cost:" | awk '{print $NF}')
        if [[ $warnings -gt 0 ]]; then
            printf "OK  cost=%-12s (warnings=%d)\n" "$cost" "$warnings"
        else
            printf "OK  cost=%s\n" "$cost"
        fi
        PASS=$((PASS + 1))
    elif echo "$eval_output" | grep -q "ERROR"; then
        error_msg=$(echo "$eval_output" | grep "ERROR" | head -1)
        printf "COST_MISMATCH\n"
        FAIL=$((FAIL + 1))
        ERRORS+=("$name: $error_msg")
    else
        printf "UNKNOWN\n"
        FAIL=$((FAIL + 1))
        ERRORS+=("$name: unexpected evaluator output: $eval_output")
    fi
done

echo ""
echo "=== Results: $PASS passed, $FAIL failed out of $((PASS + FAIL)) ==="

if [[ ${#ERRORS[@]} -gt 0 ]]; then
    echo ""
    echo "Failures:"
    for err in "${ERRORS[@]}"; do
        echo "  - $err"
    done
    exit 1
fi
