"""Shared command-line runner for the Torch frontend examples."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch
from pto_fusebox import (
    can_emit_region,
    emit_pypto_callable,
    emit_pypto_region,
    export_and_normalize,
    extract_solver_regions,
    solve_graph,
)
from torch import nn

Example = tuple[nn.Module, tuple[torch.Tensor, ...]]


def run_examples(examples: Mapping[str, Example]) -> None:
    """Capture examples and optionally pass their supported regions to Fusebox."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(examples), help="run only one example")
    parser.add_argument(
        "--json", action="store_true", help="print normalized graph JSON"
    )
    parser.add_argument(
        "--solver", type=Path, help="optional path to a built mlsys_mixed solver"
    )
    parser.add_argument(
        "--solver-workers",
        type=int,
        default=2,
        help="solver search workers (default: 2)",
    )
    parser.add_argument(
        "--emit-source",
        action="store_true",
        help="print generated PyPTO DSL for every source-ready solved region",
    )
    parser.add_argument(
        "--emit-callable",
        action="store_true",
        help="print importable @pl.inline PyPTO for each source-ready region",
    )
    args = parser.parse_args()
    if args.emit_source and args.emit_callable:
        parser.error("choose only one of --emit-source and --emit-callable")
    if (args.emit_source or args.emit_callable) and args.solver is None:
        parser.error("source emission requires --solver")

    selected = examples if args.case is None else {args.case: examples[args.case]}
    for name, (module, example_args) in selected.items():
        graph = export_and_normalize(module.eval(), example_args)
        regions = extract_solver_regions(graph)
        op_kinds = " -> ".join(op.kind for op in graph.ops)
        opaque = [
            f"{op.kind}: {op.opaque_reason}" for op in graph.ops if not op.supported
        ]

        print(
            f"{name}: {len(graph.ops)} ops, {len(regions)} supported region candidate(s)"
        )
        print(f"  DAG: {op_kinds}")
        if graph.patterns:
            print(
                f"  annotations: {', '.join(pattern.kind for pattern in graph.patterns)}"
            )
        if opaque:
            print(f"  opaque boundaries: {'; '.join(opaque)}")
        if args.json:
            print(graph.to_json(), end="")
        if args.solver is not None:
            result = solve_graph(
                graph,
                solver_binary=args.solver,
                solver_workers=args.solver_workers,
                require_source_codegen=args.emit_source or args.emit_callable,
            )
            op_by_id = graph.op_map()
            for region in result.regions:
                print(f"  {region.region.id}: {region.status}")
                for diagnostic in region.diagnostics:
                    print(f"    {diagnostic}")
                if region.status != "solved" or region.solution is None:
                    continue
                for index, step in enumerate(region.solution["steps"]):
                    solver_ops = step["ops"]
                    tile = step["launch"]["tile"]
                    latency = step["latency_cycles"]
                    kinds = " -> ".join(
                        op_by_id[region.solver_op_to_graph[solver_op]].kind
                        for solver_op in solver_ops
                    )
                    print(
                        f"    step {index}: {kinds}; tile={tile}; latency={latency:.1f}"
                    )
                if args.emit_source or args.emit_callable:
                    if not can_emit_region(graph, region):
                        print("    PyPTO source: unavailable for the selected schedule")
                    else:
                        emitted = (
                            emit_pypto_callable(
                                graph,
                                region,
                                function_name=f"{name}_{region.region.id}",
                            )
                            if args.emit_callable
                            else emit_pypto_region(
                                graph,
                                region,
                                program_name=f"{name}_{region.region.id}",
                            )
                        )
                        print("    PyPTO source:")
                        print(emitted.source, end="")
