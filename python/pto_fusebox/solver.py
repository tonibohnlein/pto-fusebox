"""Subprocess bridge from normalized regions to the existing C++ solver."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ir import SOLUTION_SCHEMA, NormalizedGraph
from .regions import LoweredProblem, SolverRegion, extract_solver_regions
from .target import TargetProfile, resolve_target


@dataclass(frozen=True)
class RegionSolveResult:
    region: SolverRegion
    status: str
    problem: Mapping[str, Any] | None
    solution: Mapping[str, Any] | None
    solver_op_to_graph: tuple[str, ...]
    solver_tensor_to_value: tuple[str, ...]
    diagnostics: tuple[str, ...]
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None


@dataclass(frozen=True)
class SolveResult:
    graph: NormalizedGraph
    target: str
    regions: tuple[RegionSolveResult, ...]
    graph_diagnostics: tuple[str, ...]
    solver_binary: str
    whole_graph_supported: bool

    @property
    def regions_solved(self) -> bool:
        return bool(self.regions) and all(
            region.status == "solved" for region in self.regions
        )

    @property
    def successful(self) -> bool:
        """Backward-compatible alias for analytic region-solving success."""

        return self.regions_solved

    @property
    def whole_graph_codegen_ready(self) -> bool:
        """Whether the whole graph and every selected schedule are source-ready."""

        # Local import keeps the subprocess bridge independent of source
        # rendering during module initialization.
        from .source import can_emit_region

        return (
            self.whole_graph_supported
            and self.regions_solved
            and all(can_emit_region(self.graph, region) for region in self.regions)
        )


def solve_graph(
    graph: NormalizedGraph,
    *,
    target: str | TargetProfile = "ascend910b",
    solver_binary: str | os.PathLike[str] | None = None,
    solver_workers: int | None = None,
) -> SolveResult:
    """Partition, lower, and solve every supported region in ``graph``.

    This function never builds PTO-Fusebox. A solver executable must already
    exist or be supplied explicitly, keeping compilation and graph capture as
    separate, reproducible steps.
    """

    if solver_workers is not None and solver_workers <= 0:
        raise ValueError("solver_workers must be a positive integer")
    profile = resolve_target(target)
    values = graph.value_map()
    whole_graph_supported = all(
        profile.admission_reason(op, values) is None for op in graph.ops
    )
    regions = extract_solver_regions(graph, profile)
    lowered_by_region: dict[str, LoweredProblem] = {}
    declined_by_region: dict[str, RegionSolveResult] = {}
    for region in regions:
        try:
            lowered_by_region[region.id] = region.lower(graph, profile)
        except ValueError as error:
            declined_by_region[region.id] = RegionSolveResult(
                region=region,
                status="declined",
                problem=None,
                solution=None,
                solver_op_to_graph=(),
                solver_tensor_to_value=(),
                diagnostics=(*region.diagnostics, str(error)),
            )
    executable = _resolve_solver_binary(solver_binary) if lowered_by_region else None
    region_results = [
        declined_by_region[region.id]
        if region.id in declined_by_region
        else _solve_region(
            executable,
            region,
            lowered_by_region[region.id],
            solver_workers=solver_workers,
        )
        for region in regions
    ]
    return SolveResult(
        graph=graph,
        target=profile.name,
        regions=tuple(region_results),
        graph_diagnostics=graph.diagnostics,
        solver_binary="" if executable is None else str(executable),
        whole_graph_supported=whole_graph_supported,
    )


def _solve_region(
    executable: Path | None,
    region: SolverRegion,
    lowered: LoweredProblem,
    *,
    solver_workers: int | None,
) -> RegionSolveResult:
    if executable is None:
        raise AssertionError("a lowerable region requires a solver executable")
    with tempfile.TemporaryDirectory(prefix="pto-fusebox-") as directory:
        root = Path(directory)
        problem_path = root / "problem.json"
        solution_path = root / "solution.json"
        problem_path.write_text(
            json.dumps(lowered.problem, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        command = [str(executable)]
        if solver_workers is not None:
            command.extend(("--threads", str(solver_workers)))
        command.extend((str(problem_path), str(solution_path)))
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            return RegionSolveResult(
                region=region,
                status="error",
                problem=lowered.problem,
                solution=None,
                solver_op_to_graph=lowered.solver_op_to_graph,
                solver_tensor_to_value=lowered.solver_tensor_to_value,
                diagnostics=(
                    *region.diagnostics,
                    "C++ solver returned a non-zero status",
                ),
                stdout=process.stdout,
                stderr=process.stderr,
                returncode=process.returncode,
            )
        if not solution_path.is_file():
            return RegionSolveResult(
                region=region,
                status="error",
                problem=lowered.problem,
                solution=None,
                solver_op_to_graph=lowered.solver_op_to_graph,
                solver_tensor_to_value=lowered.solver_tensor_to_value,
                diagnostics=(
                    *region.diagnostics,
                    "C++ solver did not create a solution file",
                ),
                stdout=process.stdout,
                stderr=process.stderr,
                returncode=process.returncode,
            )
        solution = json.loads(solution_path.read_text(encoding="utf-8"))
        if not isinstance(solution, dict):
            raise ValueError("solver solution JSON must contain an object")
        schema = solution.get("schema_version")
        if schema != SOLUTION_SCHEMA:
            raise ValueError(
                f"unsupported solver solution schema {schema!r}; expected {SOLUTION_SCHEMA!r}"
            )
        infeasible_reason = _infeasible_solution_reason(
            solution, len(lowered.solver_op_to_graph)
        )
        if infeasible_reason is not None:
            return RegionSolveResult(
                region=region,
                status="infeasible",
                problem=lowered.problem,
                solution=solution,
                solver_op_to_graph=lowered.solver_op_to_graph,
                solver_tensor_to_value=lowered.solver_tensor_to_value,
                diagnostics=(*region.diagnostics, infeasible_reason),
                stdout=process.stdout,
                stderr=process.stderr,
                returncode=process.returncode,
            )
        return RegionSolveResult(
            region=region,
            status="solved",
            problem=lowered.problem,
            solution=solution,
            solver_op_to_graph=lowered.solver_op_to_graph,
            solver_tensor_to_value=lowered.solver_tensor_to_value,
            diagnostics=region.diagnostics,
            stdout=process.stdout,
            stderr=process.stderr,
            returncode=process.returncode,
        )


def _resolve_solver_binary(value: str | os.PathLike[str] | None) -> Path:
    if value is not None:
        candidates = [Path(value)]
    elif os.environ.get("PTO_FUSEBOX_SOLVER"):
        candidates = [Path(os.environ["PTO_FUSEBOX_SOLVER"])]
    else:
        root = Path(__file__).resolve().parents[2]
        candidates = [root / "build" / "mlsys_mixed", root / "build" / "mlsys"]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    rendered = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        f"no built PTO-Fusebox solver found ({rendered}); build it explicitly or pass solver_binary"
    )


def _infeasible_solution_reason(
    solution: Mapping[str, Any], num_ops: int
) -> str | None:
    steps = solution.get("steps")
    if not isinstance(steps, list) or not steps:
        return "solver response contains no complete schedule"
    covered: set[int] = set()
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            return f"solver step {index} is not an object"
        subgraph = step.get("ops")
        launch = step.get("launch")
        latency = step.get("latency_cycles")
        if (
            not isinstance(subgraph, list)
            or not subgraph
            or any(
                not isinstance(item, int) or item < 0 or item >= num_ops
                for item in subgraph
            )
            or len(set(subgraph)) != len(subgraph)
        ):
            return f"solver step {index} references an invalid subgraph"
        covered.update(subgraph)
        if not isinstance(launch, Mapping):
            return f"solver step {index} has no launch descriptor"
        granularity = launch.get("tile")
        if (
            not isinstance(granularity, list)
            or len(granularity) != 3
            or any(not isinstance(item, int) or item <= 0 for item in granularity)
        ):
            return f"solver step {index} has no feasible tile granularity"
        if not isinstance(latency, (int, float)) or not math.isfinite(float(latency)):
            return f"solver step {index} has no finite latency"
    if covered != set(range(num_ops)):
        return "solver response does not cover every normalized operation"
    return None
