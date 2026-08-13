"""Shared command-line runner for the Torch frontend examples."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch
from pto_fusebox import export_and_normalize, extract_solver_regions, solve_graph
from torch import nn

Example = tuple[nn.Module, tuple[torch.Tensor, ...]]


def run_examples(examples: Mapping[str, Example]) -> None:
    """Capture examples and optionally pass their supported regions to Fusebox."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(examples), help="run only one example")
    parser.add_argument("--json", action="store_true", help="print normalized graph JSON")
    parser.add_argument("--solver", type=Path, help="optional path to a built mlsys_mixed solver")
    args = parser.parse_args()

    selected = examples if args.case is None else {args.case: examples[args.case]}
    for name, (module, example_args) in selected.items():
        graph = export_and_normalize(module.eval(), example_args)
        regions = extract_solver_regions(graph)
        op_kinds = " -> ".join(op.kind for op in graph.ops)
        opaque = [f"{op.kind}: {op.opaque_reason}" for op in graph.ops if not op.supported]

        print(f"{name}: {len(graph.ops)} ops, {len(regions)} schedulable region(s)")
        print(f"  DAG: {op_kinds}")
        if graph.patterns:
            print(f"  annotations: {', '.join(pattern.kind for pattern in graph.patterns)}")
        if opaque:
            print(f"  opaque boundaries: {'; '.join(opaque)}")
        if args.json:
            print(graph.to_json(), end="")
        if args.solver is not None:
            result = solve_graph(graph, solver_binary=args.solver)
            for region in result.regions:
                print(f"  {region.region.id}: {region.status}")
                for diagnostic in region.diagnostics:
                    print(f"    {diagnostic}")
