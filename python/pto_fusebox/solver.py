"""Subprocess bridge from normalized regions to the existing C++ solver."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .ir import SOLUTION_SCHEMA, NormalizedGraph
from .regions import LoweredProblem, SolverRegion, extract_solver_regions
from .target import TargetProfile, resolve_target

SOURCE_CANDIDATE_SUMMARY_SCHEMA = "pto_fusebox.source_candidate_summaries.v1"


@dataclass(frozen=True)
class SourceCandidateSummary:
    """One solver-discovered partition with source-readiness evidence.

    ``schedule`` preserves serialized step order. ``memory`` preserves each
    step's complete typed plan, including its physical frames, allocations,
    high-water marks, and traffic fields.
    """

    id: str
    rank: int
    selected: bool
    modeled_cost_cycles: float
    partition: tuple[tuple[int, ...], ...]
    schedule: tuple[Mapping[str, Any], ...]
    memory: tuple[Mapping[str, Any], ...]
    solution: Mapping[str, Any]
    source_ready: bool
    rejection_reason: str | None


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
    candidate_summaries: tuple[SourceCandidateSummary, ...] = ()


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
        """Backward-compatible alias for region-solving success."""

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
    require_source_codegen: bool = False,
    collect_candidate_summaries: bool = False,
) -> SolveResult:
    """Partition, lower, and solve every supported region in ``graph``.

    This function never builds PTO-Fusebox. A solver executable must already
    exist or be supplied explicitly, keeping compilation and graph capture as
    separate, reproducible steps. Set ``require_source_codegen`` when the
    selected schedule will be rendered as standalone PyPTO DSL. Source-oriented
    solving applies the stricter source-realization constraint from the first
    candidate search. Analytic solving is unchanged when the flag is false.
    Set ``collect_candidate_summaries`` to retain the selected and alternative
    solver partitions with modeled costs, complete schedules and memory plans,
    plus an explicit source-readiness result and rejection reason for each.
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
            lowered = region.lower(graph, profile)
            lowered_by_region[region.id] = lowered
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
    region_results: list[RegionSolveResult] = []
    for region in regions:
        if region.id in declined_by_region:
            region_results.append(declined_by_region[region.id])
            continue
        lowered = lowered_by_region[region.id]
        if require_source_codegen:
            problem = dict(lowered.problem)
            problem["require_source_codegen"] = True
            lowered = replace(lowered, problem=problem)
        solved = _solve_region(
            executable,
            graph,
            region,
            lowered,
            solver_workers=solver_workers,
            collect_candidate_summaries=collect_candidate_summaries,
        )
        if require_source_codegen and solved.status == "solved":
            from .source import can_emit_region

            if not can_emit_region(graph, solved):
                solved = replace(
                    solved,
                    status="infeasible",
                    diagnostics=(
                        *solved.diagnostics,
                        "source-constrained solver result is not PyPTO-emittable",
                    ),
                )
        region_results.append(solved)
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
    graph: NormalizedGraph,
    region: SolverRegion,
    lowered: LoweredProblem,
    *,
    solver_workers: int | None,
    collect_candidate_summaries: bool,
) -> RegionSolveResult:
    if executable is None:
        raise AssertionError("a lowerable region requires a solver executable")
    with tempfile.TemporaryDirectory(prefix="pto-fusebox-") as directory:
        root = Path(directory)
        problem_path = root / "problem.json"
        solution_path = root / "solution.json"
        candidates_path = root / "source_candidates.json"
        problem_path.write_text(
            json.dumps(lowered.problem, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        command = [str(executable)]
        if solver_workers is not None:
            command.extend(("--threads", str(solver_workers)))
        if collect_candidate_summaries:
            command.extend(("--candidate-output", str(candidates_path)))
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
        result = RegionSolveResult(
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
        if collect_candidate_summaries:
            result = replace(
                result,
                candidate_summaries=_read_candidate_summaries(
                    candidates_path, graph, result
                ),
            )
        return result


def _read_candidate_summaries(
    path: Path,
    graph: NormalizedGraph,
    selected_result: RegionSolveResult,
) -> tuple[SourceCandidateSummary, ...]:
    if not path.is_file():
        raise ValueError("solver did not create the requested candidate summary file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source candidate summary JSON must contain an object")
    if payload.get("schema_version") != SOURCE_CANDIDATE_SUMMARY_SCHEMA:
        raise ValueError("unsupported source candidate summary schema")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("source candidate summary contains no candidates")
    selected_candidate_id = payload.get("selected_candidate_id")
    if not isinstance(selected_candidate_id, str):
        raise ValueError("source candidate summary has no selected candidate ID")

    from .source import SourceEmissionError, emit_pypto_region

    summaries: list[SourceCandidateSummary] = []
    candidate_ids: set[str] = set()
    for expected_rank, raw in enumerate(raw_candidates):
        if not isinstance(raw, dict):
            raise ValueError(f"source candidate {expected_rank} must be an object")
        candidate_id = raw.get("id")
        rank = raw.get("rank")
        selected = raw.get("selected")
        modeled_cost = raw.get("modeled_cost_cycles")
        solution = raw.get("solution")
        if (
            not isinstance(candidate_id, str)
            or candidate_id in candidate_ids
            or not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank != expected_rank
            or not isinstance(selected, bool)
            or not isinstance(modeled_cost, (int, float))
            or isinstance(modeled_cost, bool)
            or not math.isfinite(float(modeled_cost))
            or not isinstance(solution, dict)
            or solution.get("schema_version") != SOLUTION_SCHEMA
        ):
            raise ValueError(f"source candidate {expected_rank} is malformed")
        candidate_ids.add(candidate_id)
        raw_steps = solution.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError(f"source candidate {expected_rank} has no schedule steps")
        schedule: list[Mapping[str, Any]] = []
        partition: list[tuple[int, ...]] = []
        memory: list[Mapping[str, Any]] = []
        serialized_cost = 0.0
        for step_index, step in enumerate(raw_steps):
            if not isinstance(step, dict):
                raise ValueError(
                    f"source candidate {expected_rank} step {step_index} is malformed"
                )
            ops = step.get("ops")
            plan = step.get("plan")
            latency = step.get("latency_cycles")
            if (
                not isinstance(ops, list)
                or any(not isinstance(op, int) or isinstance(op, bool) for op in ops)
                or not isinstance(plan, dict)
                or not isinstance(latency, (int, float))
                or isinstance(latency, bool)
                or not math.isfinite(float(latency))
            ):
                raise ValueError(
                    f"source candidate {expected_rank} step {step_index} is malformed"
                )
            schedule.append(step)
            partition.append(tuple(ops))
            memory.append(plan)
            serialized_cost += float(latency)
        if not math.isclose(
            float(modeled_cost), serialized_cost, rel_tol=1e-12, abs_tol=1e-9
        ):
            raise ValueError(
                f"source candidate {expected_rank} modeled cost differs from its steps"
            )
        candidate_result = replace(
            selected_result,
            solution=solution,
            candidate_summaries=(),
        )
        rejection_reason: str | None = None
        try:
            emit_pypto_region(
                graph,
                candidate_result,
                program_name=f"FuseboxCandidate{expected_rank}",
            )
        except SourceEmissionError as error:
            rejection_reason = str(error)
        summaries.append(
            SourceCandidateSummary(
                id=candidate_id,
                rank=rank,
                selected=selected,
                modeled_cost_cycles=float(modeled_cost),
                partition=tuple(partition),
                schedule=tuple(schedule),
                memory=tuple(memory),
                solution=solution,
                source_ready=rejection_reason is None,
                rejection_reason=rejection_reason,
            )
        )
    if (
        sum(candidate.selected for candidate in summaries) != 1
        or not summaries[0].selected
        or summaries[0].id != selected_candidate_id
        or summaries[0].solution != selected_result.solution
    ):
        raise ValueError("source candidate selected marker is inconsistent")
    return tuple(summaries)


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
