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
class SourceCandidateCutSummary:
    """One solver-created GM materialization between kernel steps."""

    value: str
    bytes: int
    producer_step: int
    consumer_steps: tuple[int, ...]


@dataclass(frozen=True)
class SourceCandidateStepSummary:
    """Execution-shape and traffic evidence for one scheduled kernel step.

    A drain site is one explicit static drain operation. Executions and bytes
    include its loop multiplicity; the traffic map distinguishes on-chip
    intermediate drains from externally visible GM traffic.
    """

    index: int
    kind: str
    submissions: int
    device_programs: int
    drain_sites: int
    drain_executions: int
    drain_bytes: int
    traffic_bytes: Mapping[str, float]


@dataclass(frozen=True)
class SourceCandidateExecutionSummary:
    """Candidate-level launches, cuts, drains, and modeled port traffic."""

    submissions: int
    device_programs: int
    cuts: tuple[SourceCandidateCutSummary, ...]
    cut_bytes: int
    drain_sites: int
    drain_executions: int
    drain_bytes: int
    steps: tuple[SourceCandidateStepSummary, ...]


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
    execution: SourceCandidateExecutionSummary
    solution: Mapping[str, Any]
    source_ready: bool
    rejection_reason: str | None


def region_for_source_candidate(
    region: RegionSolveResult,
    candidate: SourceCandidateSummary,
) -> RegionSolveResult:
    """Bind one frozen source candidate without invoking the solver again."""

    matching = tuple(
        item
        for item in region.candidate_summaries
        if item.id == candidate.id and item.solution == candidate.solution
    )
    if len(matching) != 1:
        raise ValueError(
            f"source candidate {candidate.id!r} does not belong to region "
            f"{region.region.id!r}"
        )
    if not candidate.source_ready:
        raise ValueError(
            f"source candidate {candidate.id!r} is not source-ready: "
            f"{candidate.rejection_reason}"
        )
    return replace(
        region,
        status="solved",
        solution=candidate.solution,
        candidate_summaries=(),
        diagnostics=region.region.diagnostics,
        stdout="",
        stderr="",
        returncode=0,
    )


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
                execution=_candidate_execution_summary(
                    graph, selected_result, tuple(schedule)
                ),
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


def _candidate_execution_summary(
    graph: NormalizedGraph,
    region: RegionSolveResult,
    schedule: tuple[Mapping[str, Any], ...],
) -> SourceCandidateExecutionSummary:
    """Derive source execution evidence without invoking PyPTO or the solver."""

    step_for_solver_op: dict[int, int] = {}
    step_summaries: list[SourceCandidateStepSummary] = []
    for step_index, step in enumerate(schedule):
        for solver_op in step["ops"]:
            step_for_solver_op[int(solver_op)] = step_index

    solver_for_graph_op = {
        graph_op: solver_op
        for solver_op, graph_op in enumerate(region.solver_op_to_graph)
    }
    consumers: dict[str, set[int]] = {}
    for op in graph.ops:
        solver_op = solver_for_graph_op.get(op.id)
        if solver_op is None:
            continue
        consumer_step = step_for_solver_op[solver_op]
        for value in op.inputs:
            consumers.setdefault(value, set()).add(consumer_step)

    for step_index, step in enumerate(schedule):
        step_summaries.append(
            _candidate_step_summary(
                step_index,
                step,
                graph,
                region,
                consumers,
            )
        )

    cuts: list[SourceCandidateCutSummary] = []
    for value in graph.values:
        if value.producer is None:
            continue
        producer_solver_op = solver_for_graph_op.get(value.producer)
        if producer_solver_op is None:
            continue
        producer_step = step_for_solver_op[producer_solver_op]
        consumer_steps = tuple(
            sorted(
                step for step in consumers.get(value.id, set()) if step != producer_step
            )
        )
        if not consumer_steps:
            continue
        cuts.append(
            SourceCandidateCutSummary(
                value=value.id,
                bytes=_static_value_bytes(value.shape, value.dtype),
                producer_step=producer_step,
                consumer_steps=consumer_steps,
            )
        )

    return SourceCandidateExecutionSummary(
        submissions=sum(step.submissions for step in step_summaries),
        device_programs=sum(step.device_programs for step in step_summaries),
        cuts=tuple(cuts),
        cut_bytes=sum(cut.bytes for cut in cuts),
        drain_sites=sum(step.drain_sites for step in step_summaries),
        drain_executions=sum(step.drain_executions for step in step_summaries),
        drain_bytes=sum(step.drain_bytes for step in step_summaries),
        steps=tuple(step_summaries),
    )


def _candidate_step_summary(
    index: int,
    step: Mapping[str, Any],
    graph: NormalizedGraph,
    region: RegionSolveResult,
    consumers: Mapping[str, set[int]],
) -> SourceCandidateStepSummary:
    kind = str(step["kind"])
    plan = step["plan"]
    if not isinstance(plan, Mapping):
        raise ValueError(f"candidate step {index} has no plan")
    launch = step["launch"]
    if not isinstance(launch, Mapping):
        raise ValueError(f"candidate step {index} has no launch")

    submissions = 1
    device_programs = 2 if kind == "mixed" else 1
    drain_sites = 0
    drain_executions = 0
    drain_bytes = 0
    traffic: Mapping[str, float] = {}
    if kind == "cube":
        split = int(plan.get("split_k", 1))
        for matmul in plan.get("matmuls", []):
            if not isinstance(matmul, Mapping):
                continue
            drain = matmul.get("final_drain")
            if not isinstance(drain, Mapping) or not drain.get("required"):
                continue
            tile_count = int(drain.get("tile_count", 0))
            spatial_tiles = int(plan.get("spatial_tiles", 0))
            drain_sites += 1
            drain_executions += tile_count * spatial_tiles * split
            drain_bytes += int(drain.get("bytes", 0)) * spatial_tiles * split
        first = plan.get("first_partial_then_atomic")
        zero = plan.get("aiv_zero_seed_then_atomic")
        if split > 1 and (
            isinstance(first, Mapping)
            and first.get("present")
            or isinstance(zero, Mapping)
            and zero.get("present")
        ):
            submissions = 2
            device_programs = 2
        if region.problem is not None:
            traffic = _cube_candidate_traffic(plan, region.problem)
    else:
        # Vector and mixed plans each have one final source drain site.  Its
        # logical byte volume is the region output, independent of the number
        # of physical trips that execute that site.
        step_solver_ops = {int(op) for op in step["ops"]}
        step_outputs: set[str] = set()
        graph_ops = graph.op_map()
        region_outputs = set(region.region.output_values)
        for solver_op in step_solver_ops:
            graph_op = graph_ops[region.solver_op_to_graph[solver_op]]
            for output in graph_op.outputs:
                if output in region_outputs or any(
                    consumer != index for consumer in consumers.get(output, set())
                ):
                    step_outputs.add(output)
        drain_sites = len(step_outputs)
        logical_work = int(plan.get("spatial_tiles", plan.get("work_units", 1)))
        drain_executions = drain_sites * logical_work
        if kind == "vector" and plan.get("kind") == "multi_pass":
            replay_passes = plan.get("replay_passes")
            if not isinstance(replay_passes, list):
                raise ValueError("multi-pass candidate omits replay_passes")
            drain_executions = 0
            for replay in replay_passes:
                if not isinstance(replay, Mapping):
                    raise ValueError("multi-pass candidate has a malformed replay pass")
                outputs = replay.get("output_tensors")
                if not isinstance(outputs, list):
                    raise ValueError("multi-pass candidate replay omits output_tensors")
                executions = 1
                if replay.get("kind") == "apply":
                    loop = replay.get("loop")
                    tail = replay.get("tail")
                    if not isinstance(loop, Mapping) or not isinstance(tail, Mapping):
                        raise ValueError(
                            "multi-pass APPLY replay omits loop or tail geometry"
                        )
                    executions = int(loop.get("trip_count", 0)) + int(
                        tail.get("present") is True
                    )
                drain_executions += len(outputs) * logical_work * executions
        values = graph.value_map()
        drain_bytes = sum(
            _static_value_bytes(values[value].shape, values[value].dtype)
            for value in step_outputs
        )
        breakdown = plan.get("cost_breakdown")
        if isinstance(breakdown, Mapping):
            raw_traffic = breakdown.get("traffic_bytes")
            if isinstance(raw_traffic, Mapping):
                traffic = {
                    str(port): float(value)
                    for port, value in raw_traffic.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
        if kind == "vector" and region.problem is not None:
            traffic = _vector_candidate_traffic(
                plan,
                region.problem,
                step_solver_ops,
            )

    return SourceCandidateStepSummary(
        index=index,
        kind=kind,
        submissions=submissions,
        device_programs=device_programs,
        drain_sites=drain_sites,
        drain_executions=drain_executions,
        drain_bytes=drain_bytes,
        traffic_bytes=traffic,
    )


def _cube_candidate_traffic(
    plan: Mapping[str, Any], problem: Mapping[str, Any]
) -> Mapping[str, float]:
    """Return loop-weighted GM traffic for a homogeneous cube step."""

    dtypes = problem.get("dtypes")
    if not isinstance(dtypes, list):
        return {}
    widths = _problem_dtype_widths(dtypes)
    work_units = int(plan.get("work_units", 0))
    if work_units <= 0:
        return {}
    gm_l1_per_work = sum(
        int(resident.get("bytes", 0))
        for resident in plan.get("resident_boundaries", [])
        if isinstance(resident, Mapping)
    )
    l0c_gm = 0
    for matmul in plan.get("matmuls", []):
        if not isinstance(matmul, Mapping):
            return {}
        grid = matmul.get("output_grid")
        retained = matmul.get("retained_panels")
        if (
            not isinstance(grid, list)
            or len(grid) != 2
            or not isinstance(retained, Mapping)
        ):
            return {}
        for operand, repeats in (("lhs", int(grid[1])), ("rhs", int(grid[0]))):
            if int(matmul.get(f"{operand}_producer", -1)) >= 0:
                continue
            if int(matmul.get(f"{operand}_resident_boundary", -1)) >= 0:
                continue
            region = matmul.get(operand)
            if not isinstance(region, Mapping):
                return {}
            tensor = int(region.get("tensor", -1))
            if tensor not in widths:
                return {}
            gm_l1_per_work += (
                int(region.get("height", 0))
                * int(region.get("width", 0))
                * widths[tensor]
                * (1 if retained.get(operand) is True else repeats)
            )
        drain = matmul.get("final_drain")
        if (
            isinstance(drain, Mapping)
            and drain.get("required") is True
            and drain.get("target_l1") is False
        ):
            l0c_gm += int(drain.get("bytes", 0)) * work_units
    zero = plan.get("aiv_zero_seed_then_atomic")
    ub_gm = (
        int(zero.get("seed_bytes", 0))
        if isinstance(zero, Mapping) and zero.get("present") is True
        else 0
    )
    return {
        "gm_l1": float(gm_l1_per_work * work_units),
        "gm_ub": 0.0,
        "l0c_gm": float(l0c_gm),
        "ub_gm": float(ub_gm),
    }


def _vector_candidate_traffic(
    plan: Mapping[str, Any],
    problem: Mapping[str, Any],
    step_ops: set[int],
) -> Mapping[str, float]:
    """Return model-equivalent GM traffic for pointwise and multi-pass vectors."""

    dtypes = problem.get("dtypes")
    heights = problem.get("heights")
    widths_raw = problem.get("widths")
    inputs = problem.get("inputs")
    outputs = problem.get("outputs")
    if (
        not isinstance(dtypes, list)
        or not isinstance(heights, list)
        or not isinstance(widths_raw, list)
        or not isinstance(inputs, list)
        or not isinstance(outputs, list)
    ):
        return {}
    element_widths = _problem_dtype_widths(dtypes)
    if len(element_widths) != len(dtypes):
        return {}

    produced: set[int] = set()
    consumed_outside: set[int] = set()
    for op_index, op_outputs in enumerate(outputs):
        if op_index in step_ops and isinstance(op_outputs, list):
            produced.update(int(tensor) for tensor in op_outputs)
    for op_index, op_inputs in enumerate(inputs):
        if op_index not in step_ops and isinstance(op_inputs, list):
            consumed_outside.update(int(tensor) for tensor in op_inputs)
    required = {
        int(tensor)
        for tensor in problem.get("required_outputs", [])
        if isinstance(tensor, int) and not isinstance(tensor, bool)
    }
    boundary_outputs = produced & (consumed_outside | required)
    reduced_axis = int(plan.get("axis", 0))
    free_tile = int(plan.get("free_tile", 0))
    m_partition = plan.get("m_partition")
    n_partition = plan.get("n_partition")
    if not isinstance(m_partition, Mapping) or not isinstance(n_partition, Mapping):
        return {}
    free_regions = int(
        m_partition.get("parts", 0)
        if reduced_axis == 1
        else n_partition.get("parts", 0)
    )

    def streamed_bytes(tensor: int, covered_extent: int, chunks: int) -> int:
        tensor_free = int(heights[tensor] if reduced_axis == 1 else widths_raw[tensor])
        tensor_reduced = int(
            widths_raw[tensor] if reduced_axis == 1 else heights[tensor]
        )
        free_total = (
            free_regions
            if tensor_free == 1
            else free_regions * min(tensor_free, free_tile)
        )
        reduced_total = chunks if tensor_reduced == 1 else covered_extent
        return free_total * reduced_total * element_widths[tensor]

    gm_ub = 0
    ub_gm = 0
    if plan.get("kind") == "multi_pass":
        passes = plan.get("replay_passes")
        if not isinstance(passes, list):
            return {}
        chunk = int(plan.get("chunk", 0))
        for replay in passes:
            if not isinstance(replay, Mapping):
                return {}
            lifetimes = replay.get("input_lifetimes", [])
            pass_outputs = replay.get("output_tensors", [])
            if not isinstance(lifetimes, list) or not isinstance(pass_outputs, list):
                return {}
            tensors = [
                int(lifetime["tensor"])
                for lifetime in lifetimes
                if isinstance(lifetime, Mapping)
                and isinstance(lifetime.get("tensor"), int)
            ]
            segments: list[tuple[int, int]] = []
            init = replay.get("init")
            loop = replay.get("loop")
            tail = replay.get("tail")
            if isinstance(init, Mapping) and init.get("present") is True:
                segments.append((int(init.get("extent", 0)), 1))
            if isinstance(loop, Mapping) and int(loop.get("trip_count", 0)) > 0:
                segments.append(
                    (chunk * int(loop["trip_count"]), int(loop["trip_count"]))
                )
            if isinstance(tail, Mapping) and tail.get("present") is True:
                segments.append((int(tail.get("extent", 0)), 1))
            for extent, chunks in segments:
                gm_ub += sum(
                    streamed_bytes(tensor, extent, chunks) for tensor in tensors
                )
                if replay.get("kind") == "apply":
                    ub_gm += sum(
                        streamed_bytes(int(tensor), extent, chunks)
                        for tensor in pass_outputs
                        if int(tensor) in boundary_outputs
                    )
            if replay.get("kind") == "reduction":
                ub_gm += sum(
                    streamed_bytes(int(tensor), 0, 1)
                    for tensor in pass_outputs
                    if int(tensor) in boundary_outputs
                )
    elif plan.get("kind") in ("pointwise", "materialized"):
        phases = plan.get("phases")
        strip = plan.get("strip")
        strip_grid = plan.get("strip_grid")
        if (
            not isinstance(phases, list)
            or not isinstance(strip, list)
            or not isinstance(strip_grid, list)
        ):
            return {}
        body = next(
            (
                phase
                for phase in phases
                if isinstance(phase, Mapping) and phase.get("name") == "body"
            ),
            None,
        )
        if not isinstance(body, Mapping):
            return {}
        multiplicity = (
            int(plan.get("work_units", 0)) * int(strip_grid[0]) * int(strip_grid[1])
        )
        for lifetime in body.get("input_lifetimes", []):
            if not isinstance(lifetime, Mapping):
                continue
            tensor = int(lifetime.get("tensor", -1))
            rows = 1 if int(heights[tensor]) == 1 else int(strip[0])
            cols = 1 if int(widths_raw[tensor]) == 1 else int(strip[1])
            gm_ub += multiplicity * rows * cols * element_widths[tensor]
        for tensor in boundary_outputs:
            rows = 1 if int(heights[tensor]) == 1 else int(strip[0])
            cols = 1 if int(widths_raw[tensor]) == 1 else int(strip[1])
            ub_gm += multiplicity * rows * cols * element_widths[tensor]
    else:
        phases = plan.get("phases")
        if not isinstance(phases, list) or reduced_axis not in (1, 2):
            return {}
        chunk = int(plan.get("chunk", 0))
        stream_passes = int(plan.get("stream_passes", 0))
        for phase in phases:
            if not isinstance(phase, Mapping):
                return {}
            name = phase.get("name")
            if name not in ("stats", "apply", "finalize"):
                continue
            lifetimes = phase.get("input_lifetimes", [])
            if not isinstance(lifetimes, list):
                return {}
            tensors = [
                int(lifetime["tensor"])
                for lifetime in lifetimes
                if isinstance(lifetime, Mapping)
                and isinstance(lifetime.get("tensor"), int)
            ]
            segments: list[tuple[int, int]] = []
            init = phase.get("init")
            loop = phase.get("loop")
            tail = phase.get("tail")
            if isinstance(init, Mapping) and init.get("present") is True:
                segments.append((int(init.get("extent", 0)), 1))
            if isinstance(loop, Mapping) and int(loop.get("trip_count", 0)) > 0:
                trips = int(loop["trip_count"])
                segments.append((chunk * trips, trips))
            if isinstance(tail, Mapping) and tail.get("present") is True:
                segments.append((int(tail.get("extent", 0)), 1))
            for extent, chunks in segments:
                gm_ub += sum(
                    streamed_bytes(tensor, extent, chunks) for tensor in tensors
                )
                if name == "apply":
                    ub_gm += sum(
                        streamed_bytes(tensor, extent, chunks)
                        for tensor in boundary_outputs
                    )
        if stream_passes != 2:
            ub_gm += sum(streamed_bytes(tensor, 0, 1) for tensor in boundary_outputs)
    return {"gm_l1": 0.0, "gm_ub": float(gm_ub), "l0c_gm": 0.0, "ub_gm": float(ub_gm)}


def _problem_dtype_widths(dtypes: list[Any]) -> dict[int, int]:
    widths = {
        "bool": 1,
        "int8": 1,
        "int16": 2,
        "fp16": 2,
        "float16": 2,
        "bf16": 2,
        "bfloat16": 2,
        "int32": 4,
        "fp32": 4,
        "float32": 4,
        "int64": 8,
        "fp64": 8,
        "float64": 8,
    }
    return {
        index: widths[dtype.lower()]
        for index, dtype in enumerate(dtypes)
        if isinstance(dtype, str) and dtype.lower() in widths
    }


def _static_value_bytes(shape: tuple[Any, ...], dtype: str) -> int:
    elements = 1
    for dimension in shape:
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            return 0
        elements *= dimension
    widths = {
        "bool": 1,
        "int8": 1,
        "int16": 2,
        "float16": 2,
        "bfloat16": 2,
        "int32": 4,
        "float32": 4,
        "int64": 8,
        "float64": 8,
    }
    try:
        return elements * widths[dtype.lower()]
    except KeyError as error:
        raise ValueError(
            f"candidate summary has unsupported dtype {dtype!r}"
        ) from error


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
