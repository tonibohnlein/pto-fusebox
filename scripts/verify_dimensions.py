#!/usr/bin/env python3
"""Verify tensor dimension consistency for operation graphs.

Checks:
  - Pointwise ops: all inputs and outputs must have identical (width, height).
  - MatMul ops: inner dimensions must match (LHS.width == RHS.height),
    and output dimensions must be (LHS.height, RHS.width).
"""

import json
import sys
from pathlib import Path


def verify(path: str) -> list[str]:
    with open(path) as f:
        prob = json.load(f)

    widths = prob["widths"]
    heights = prob["heights"]
    op_types = prob["op_types"]
    inputs = prob["inputs"]
    outputs = prob["outputs"]

    errors: list[str] = []

    for op_idx, op_type in enumerate(op_types):
        inp_ids = inputs[op_idx]
        out_ids = outputs[op_idx]

        if op_type == "Pointwise":
            # All inputs and outputs must share the same dimensions.
            all_ids = inp_ids + out_ids
            ref_w, ref_h = widths[all_ids[0]], heights[all_ids[0]]
            for tid in all_ids[1:]:
                if widths[tid] != ref_w or heights[tid] != ref_h:
                    errors.append(
                        f"op{op_idx} (Pointwise): dimension mismatch — "
                        f"tensor {all_ids[0]} is {widths[all_ids[0]]}x{heights[all_ids[0]]}, "
                        f"but tensor {tid} is {widths[tid]}x{heights[tid]}"
                    )

        elif op_type == "MatMul":
            if len(inp_ids) != 2 or len(out_ids) != 1:
                errors.append(
                    f"op{op_idx} (MatMul): expected 2 inputs and 1 output, "
                    f"got {len(inp_ids)} inputs and {len(out_ids)} outputs"
                )
                continue

            lhs, rhs = inp_ids
            out = out_ids[0]
            lhs_w, lhs_h = widths[lhs], heights[lhs]
            rhs_w, rhs_h = widths[rhs], heights[rhs]
            out_w, out_h = widths[out], heights[out]

            # Inner dimensions: LHS.width == RHS.height
            if lhs_w != rhs_h:
                errors.append(
                    f"op{op_idx} (MatMul): inner dimension mismatch — "
                    f"LHS (tensor {lhs}) width={lhs_w}, "
                    f"RHS (tensor {rhs}) height={rhs_h}"
                )

            # Output height == LHS height
            if out_h != lhs_h:
                errors.append(
                    f"op{op_idx} (MatMul): output height mismatch — "
                    f"output (tensor {out}) height={out_h}, "
                    f"LHS (tensor {lhs}) height={lhs_h}"
                )

            # Output width == RHS width
            if out_w != rhs_w:
                errors.append(
                    f"op{op_idx} (MatMul): output width mismatch — "
                    f"output (tensor {out}) width={out_w}, "
                    f"RHS (tensor {rhs}) width={rhs_w}"
                )

        else:
            errors.append(f"op{op_idx}: unknown op type '{op_type}'")

    return errors


def main():
    if len(sys.argv) < 2:
        paths = sorted(Path("benchmarks").glob("*.json"))
        if not paths:
            print("Usage: verify_dimensions.py <file.json> [...]")
            sys.exit(1)
    else:
        paths = [Path(p) for p in sys.argv[1:]]

    any_error = False
    for path in paths:
        errors = verify(str(path))
        if errors:
            any_error = True
            print(f"\n{path}:")
            for e in errors:
                print(f"  ERROR: {e}")
        else:
            print(f"{path}: OK")

    sys.exit(1 if any_error else 0)


if __name__ == "__main__":
    main()
