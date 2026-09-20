"""Developer-facing enumeration of homogeneous cube model candidates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .ir import SOLUTION_SCHEMA
from .solver import RegionSolveResult


CUBE_PLAN_SWEEP_SCHEMA = "pto_fusebox.cube_plan_sweep.v3"


@dataclass(frozen=True)
class CubeCandidateGrid:
    """The fixed outer grid enumerated by the cube model."""

    parts_m: int
    parts_n: int
    split_k: int
    sequential_k_limit: int
    realized_sequential_k: int
    inner_k: int

    @property
    def work_units(self) -> int:
        """Return the number of spatial and split-K tasks."""

        return self.parts_m * self.parts_n * self.split_k


@dataclass(frozen=True)
class CubePlanCandidate:
    """One feasible cube candidate with an ordinary forced solution payload."""

    id: str
    selected: bool
    problem_sha256: str
    grid: CubeCandidateGrid
    latency_cycles: float
    cores_used: int
    compute_bound: bool
    ddr_traffic_cycles: float
    l1_l0_extract_cycles: float
    uses_model_ahead_split_k: bool
    geometry: CubeCandidateGeometry
    execution: CubeCandidateExecution
    solution: Mapping[str, Any]


@dataclass(frozen=True)
class CubeCandidateExecution:
    """Observable launch, drain, and modeled GM traffic for one cube plan.

    A drain site is one explicit ``final_drain`` in the static schedule,
    including an intermediate L0C-to-L1 drain. Executions and bytes include
    the site's tile and work-unit multiplicity; ``traffic_bytes`` separately
    records whether the drain reaches GM.
    """

    submissions: int
    device_programs: int
    cuts: int
    drain_sites: int
    drain_executions: int
    drain_bytes: int
    traffic_bytes: Mapping[str, int]


@dataclass(frozen=True)
class CubeCandidateGeometry:
    """Physical dimensions that distinguish executable cube candidates."""

    matmuls: tuple[CubeMatmulGeometry, ...]
    sequential_k_window: int
    inner_k: int
    spatial_n_tile: int
    active_tasks: int
    trips_per_task: int
    physical_l0_tile: tuple[int, int, int]
    contiguous_request_bytes: Mapping[str, int]
    physical_constraints: Mapping[str, bool]


@dataclass(frozen=True)
class CubeMatmulGeometry:
    """Per-request physical geometry retained inside one cube candidate."""

    solver_op: int
    is_sink: bool
    sequential_k_window: int
    inner_k: int
    physical_l0_tile: tuple[int, int, int]
    contiguous_request_bytes: Mapping[str, int]
    physical_constraints: Mapping[str, bool]


@dataclass(frozen=True)
class CubePlanRejection:
    """One enumerated cube point rejected before source emission."""

    parts_m: int
    parts_n: int
    split_k: int
    sequential_k_limit: int
    inner_k: int
    reason: str


@dataclass(frozen=True)
class CubePlanSweep:
    """All feasible candidates and the model-selected candidate."""

    selected_candidate_id: str
    candidates: tuple[CubePlanCandidate, ...]
    rejections: tuple[CubePlanRejection, ...] = ()
    stdout: str = ""
    stderr: str = ""

    @property
    def selected(self) -> CubePlanCandidate:
        """Return the unique model-selected candidate."""

        for candidate in self.candidates:
            if candidate.id == self.selected_candidate_id:
                return candidate
        raise ValueError(
            f"selected cube candidate {self.selected_candidate_id!r} is missing"
        )


def enumerate_cube_plans(
    region: RegionSolveResult,
    *,
    sweep_binary: str | os.PathLike[str] | None = None,
    source_oriented: bool = False,
) -> CubePlanSweep:
    """Enumerate every feasible cube plan for one lowered homogeneous cube region.

    The C++ sweep executable reuses the production model and serializes each
    fixed candidate as ``solution.v9``. This function validates that envelope;
    it does not re-price or re-plan the candidate in Python.

    Args:
        region: A region result that carries the exact lowered problem.
        sweep_binary: An explicitly built ``cube_plan_sweep`` executable.
        source_oriented: Enumerate the larger source-first surface, including
            grid-stride replay and explicit sequential-K limits.  The returned
            candidates remain bound to the caller's original lowered problem;
            this flag changes search only, not graph semantics.

    Returns:
        The validated candidate sweep.

    Raises:
        ValueError: If the region or sweep payload violates the contract.
        FileNotFoundError: If no sweep executable is available.
        RuntimeError: If the C++ sweep executable fails.
    """

    if region.problem is None:
        raise ValueError(f"region {region.region.id} has no lowered problem")
    executable = _resolve_sweep_binary(sweep_binary)
    canonical_problem = json.dumps(
        region.problem, sort_keys=True, separators=(",", ":")
    )
    problem_sha256 = hashlib.sha256(canonical_problem.encode()).hexdigest()
    sweep_problem = dict(region.problem)
    if source_oriented:
        sweep_problem["require_source_codegen"] = True
    canonical_sweep_problem = json.dumps(
        sweep_problem, sort_keys=True, separators=(",", ":")
    )
    with tempfile.TemporaryDirectory(prefix="pto-fusebox-cube-sweep-") as directory:
        root = Path(directory)
        problem_path = root / "problem.json"
        output_path = root / "sweep.json"
        problem_path.write_text(canonical_sweep_problem + "\n", encoding="utf-8")
        process = subprocess.run(
            [str(executable), str(problem_path), str(output_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            detail = process.stderr.strip() or process.stdout.strip()
            raise RuntimeError(
                f"cube plan sweep failed with status {process.returncode}: {detail}"
            )
        if not output_path.is_file():
            raise RuntimeError("cube plan sweep did not create its output file")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    return _parse_sweep(
        payload,
        problem=sweep_problem,
        problem_sha256=problem_sha256,
        stdout=process.stdout,
        stderr=process.stderr,
    )


def region_for_cube_candidate(
    region: RegionSolveResult, candidate: CubePlanCandidate
) -> RegionSolveResult:
    """Bind a swept candidate to its original region for typed replay."""

    if region.problem is None:
        raise ValueError(f"region {region.region.id} has no lowered problem")
    canonical_problem = json.dumps(
        region.problem, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(canonical_problem.encode()).hexdigest()
    if digest != candidate.problem_sha256:
        raise ValueError(
            f"cube candidate {candidate.id} belongs to a different lowered problem"
        )
    return replace(
        region,
        status="solved",
        solution=candidate.solution,
        diagnostics=region.region.diagnostics,
        stdout="",
        stderr="",
        returncode=0,
    )


def _resolve_sweep_binary(value: str | os.PathLike[str] | None) -> Path:
    if value is not None:
        candidates = [Path(value)]
    elif os.environ.get("PTO_FUSEBOX_CUBE_SWEEP"):
        candidates = [Path(os.environ["PTO_FUSEBOX_CUBE_SWEEP"])]
    else:
        root = Path(__file__).resolve().parents[2]
        candidates = [root / "build" / "cube_plan_sweep"]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    rendered = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        f"no built cube plan sweep found ({rendered}); build cube_plan_sweep explicitly"
    )


def _parse_sweep(
    payload: Any,
    *,
    problem: Mapping[str, Any],
    problem_sha256: str,
    stdout: str,
    stderr: str,
) -> CubePlanSweep:
    if not isinstance(payload, Mapping):
        raise ValueError("cube plan sweep JSON must contain an object")
    if payload.get("schema_version") != CUBE_PLAN_SWEEP_SCHEMA:
        raise ValueError(
            "unsupported cube plan sweep schema "
            f"{payload.get('schema_version')!r}; expected {CUBE_PLAN_SWEEP_SCHEMA!r}"
        )
    selected_id = payload.get("selected_candidate_id")
    raw_candidates = payload.get("candidates")
    raw_rejections = payload.get("rejections", [])
    if not isinstance(selected_id, str) or not selected_id:
        raise ValueError("cube plan sweep has no selected candidate id")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("cube plan sweep contains no candidates")
    if not isinstance(raw_rejections, list):
        raise ValueError("cube plan sweep rejections must be a list")

    candidates = tuple(
        _parse_candidate(
            item,
            problem=problem,
            problem_sha256=problem_sha256,
            index=index,
        )
        for index, item in enumerate(raw_candidates)
    )
    ids = [candidate.id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("cube plan sweep candidate ids are not unique")
    selected = [candidate for candidate in candidates if candidate.selected]
    if len(selected) != 1 or selected[0].id != selected_id:
        raise ValueError("cube plan sweep selected-candidate markers disagree")
    rejections = tuple(
        _parse_rejection(item, index=index) for index, item in enumerate(raw_rejections)
    )
    return CubePlanSweep(
        selected_candidate_id=selected_id,
        candidates=candidates,
        rejections=rejections,
        stdout=stdout,
        stderr=stderr,
    )


def _parse_rejection(payload: Any, *, index: int) -> CubePlanRejection:
    field = f"rejections[{index}]"
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field} is not an object")
    grid = payload.get("grid")
    reason = payload.get("reason")
    if not isinstance(grid, Mapping):
        raise ValueError(f"{field}.grid is not an object")
    if not isinstance(reason, str) or not reason:
        raise ValueError(f"{field}.reason must be a non-empty string")
    return CubePlanRejection(
        parts_m=_positive_int(grid.get("parts_m"), f"{field}.parts_m"),
        parts_n=_positive_int(grid.get("parts_n"), f"{field}.parts_n"),
        split_k=_positive_int(grid.get("split_k"), f"{field}.split_k"),
        sequential_k_limit=_positive_int(
            grid.get("sequential_k_limit"), f"{field}.sequential_k_limit"
        ),
        inner_k=_nonnegative_int(grid.get("inner_k"), f"{field}.inner_k"),
        reason=reason,
    )


def _parse_candidate(
    payload: Any,
    *,
    problem: Mapping[str, Any],
    problem_sha256: str,
    index: int,
) -> CubePlanCandidate:
    field = f"candidates[{index}]"
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field} is not an object")
    candidate_id = payload.get("id")
    selected = payload.get("selected")
    grid = payload.get("enumerated_grid")
    model = payload.get("model")
    solution = payload.get("solution")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError(f"{field}.id must be a non-empty string")
    if not isinstance(selected, bool):
        raise ValueError(f"{field}.selected must be boolean")
    if not isinstance(grid, Mapping) or not isinstance(model, Mapping):
        raise ValueError(f"{field} has no grid or model descriptor")
    parsed_grid = CubeCandidateGrid(
        parts_m=_positive_int(grid.get("parts_m"), f"{field}.parts_m"),
        parts_n=_positive_int(grid.get("parts_n"), f"{field}.parts_n"),
        split_k=_positive_int(grid.get("split_k"), f"{field}.split_k"),
        sequential_k_limit=_positive_int(
            grid.get("sequential_k_limit"), f"{field}.sequential_k_limit"
        ),
        realized_sequential_k=_positive_int(
            grid.get("realized_sequential_k"), f"{field}.realized_sequential_k"
        ),
        inner_k=_nonnegative_int(grid.get("inner_k"), f"{field}.inner_k"),
    )
    if (
        not isinstance(solution, Mapping)
        or solution.get("schema_version") != SOLUTION_SCHEMA
    ):
        raise ValueError(f"{field}.solution is not {SOLUTION_SCHEMA}")
    steps = solution.get("steps")
    if not isinstance(steps, list) or len(steps) != 1:
        raise ValueError(f"{field}.solution must contain exactly one step")
    return CubePlanCandidate(
        id=candidate_id,
        selected=selected,
        problem_sha256=problem_sha256,
        grid=parsed_grid,
        latency_cycles=_finite_float(
            model.get("latency_cycles"), f"{field}.latency_cycles"
        ),
        cores_used=_positive_int(model.get("cores_used"), f"{field}.cores_used"),
        compute_bound=_bool(model.get("compute_bound"), f"{field}.compute_bound"),
        ddr_traffic_cycles=_finite_float(
            model.get("ddr_traffic_cycles"), f"{field}.ddr_traffic_cycles"
        ),
        l1_l0_extract_cycles=_finite_float(
            model.get("l1_l0_extract_cycles"),
            f"{field}.l1_l0_extract_cycles",
        ),
        uses_model_ahead_split_k=_bool(
            model.get("uses_model_ahead_split_k"),
            f"{field}.uses_model_ahead_split_k",
        ),
        geometry=_cube_candidate_geometry(solution, problem, field=field),
        execution=_cube_execution_summary(solution, problem, field=field),
        solution=solution,
    )


def _cube_candidate_geometry(
    solution: Mapping[str, Any],
    problem: Mapping[str, Any],
    *,
    field: str,
) -> CubeCandidateGeometry:
    """Recompute layout, replay, and physical-admission facts from the plan."""

    step = solution["steps"][0]
    plan = step.get("plan") if isinstance(step, Mapping) else None
    if not isinstance(plan, Mapping):
        raise ValueError(f"{field}.plan is malformed")
    matmuls = plan.get("matmuls")
    if not isinstance(matmuls, list) or not matmuls:
        raise ValueError(f"{field}.matmuls must contain cube requests")
    n_partition = plan.get("n_partition")
    if not isinstance(n_partition, Mapping):
        raise ValueError(f"{field}.n_partition is malformed")
    spatial_n = _positive_int(n_partition.get("big"), f"{field}.spatial_n_tile")
    replay = plan.get("spatial_replay")
    work_units = _positive_int(plan.get("work_units"), f"{field}.work_units")
    if not isinstance(replay, Mapping):
        raise ValueError(f"{field}.spatial_replay is missing")
    if replay.get("present") is True:
        active_tasks = _positive_int(
            replay.get("active_tasks"), f"{field}.active_tasks"
        )
        trips = _positive_int(replay.get("trips_per_task"), f"{field}.trips_per_task")
    else:
        active_tasks, trips = work_units, 1

    dtypes = problem.get("dtypes")
    inputs = problem.get("inputs")
    lhs_flags = problem.get("matmul_lhs_k_contiguous")
    rhs_flags = problem.get("matmul_rhs_k_contiguous")
    if (
        not isinstance(dtypes, list)
        or not isinstance(inputs, list)
        or not isinstance(lhs_flags, list)
        or not isinstance(rhs_flags, list)
    ):
        raise ValueError(f"{field}.problem layout metadata is malformed")
    element_bytes = {
        "int8": 1,
        "fp16": 2,
        "float16": 2,
        "bf16": 2,
        "bfloat16": 2,
        "int32": 4,
        "fp32": 4,
        "float32": 4,
    }
    geometries: list[CubeMatmulGeometry] = []
    for request_index, matmul in enumerate(matmuls):
        request_field = f"{field}.matmuls[{request_index}]"
        if not isinstance(matmul, Mapping):
            raise ValueError(f"{request_field} is not an object")
        op_index = matmul.get("op")
        variants = matmul.get("output_variants")
        k_loop = matmul.get("k_loop")
        if (
            not isinstance(op_index, int)
            or isinstance(op_index, bool)
            or op_index < 0
            or op_index >= len(inputs)
            or not isinstance(inputs[op_index], list)
            or len(inputs[op_index]) != 2
            or op_index >= len(lhs_flags)
            or op_index >= len(rhs_flags)
            or not isinstance(variants, list)
            or not variants
            or not isinstance(k_loop, Mapping)
        ):
            raise ValueError(f"{request_field} geometry is malformed")
        first = variants[0]
        init = first.get("l0_init") if isinstance(first, Mapping) else None
        tile = init.get("tile") if isinstance(init, Mapping) else None
        if (
            not isinstance(tile, list)
            or len(tile) != 3
            or any(not isinstance(item, int) or isinstance(item, bool) for item in tile)
        ):
            raise ValueError(f"{request_field}.l0_init.tile is malformed")
        physical_m, physical_n, inner_k = (
            _positive_int(item, f"{request_field}.l0_tile") for item in tile
        )
        sequential_k = _positive_int(
            k_loop.get("l1_window_k"), f"{request_field}.sequential_k_window"
        )
        lhs_tensor, rhs_tensor = inputs[op_index]
        try:
            lhs_bytes = element_bytes[str(dtypes[lhs_tensor]).lower()]
            rhs_bytes = element_bytes[str(dtypes[rhs_tensor]).lower()]
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError(f"{request_field} operand dtype is unsupported") from error
        lhs_k_contiguous = _bool(
            lhs_flags[op_index], f"{request_field}.lhs_k_contiguous"
        )
        rhs_k_contiguous = _bool(
            rhs_flags[op_index], f"{request_field}.rhs_k_contiguous"
        )
        lhs_dtype = str(dtypes[lhs_tensor]).lower()
        rhs_dtype = str(dtypes[rhs_tensor]).lower()
        int8 = lhs_dtype == rhs_dtype == "int8"
        geometries.append(
            CubeMatmulGeometry(
                solver_op=op_index,
                is_sink=_bool(matmul.get("is_sink"), f"{request_field}.is_sink"),
                sequential_k_window=sequential_k,
                inner_k=inner_k,
                physical_l0_tile=(physical_m, physical_n, inner_k),
                contiguous_request_bytes={
                    "lhs": (inner_k if lhs_k_contiguous else physical_m) * lhs_bytes,
                    "rhs": (inner_k if rhs_k_contiguous else physical_n) * rhs_bytes,
                },
                physical_constraints={
                    "bf16_m_box_16": lhs_dtype != "bf16" or physical_m % 16 == 0,
                    "int8_k_box_32": not int8 or inner_k % 32 == 0,
                    "int8_n_box_32": not int8 or physical_n % 32 == 0,
                },
            )
        )
    sinks = [geometry for geometry in geometries if geometry.is_sink]
    if len(sinks) != 1:
        raise ValueError(f"{field}.matmuls must contain one sink request")
    sink = sinks[0]
    return CubeCandidateGeometry(
        matmuls=tuple(geometries),
        sequential_k_window=sink.sequential_k_window,
        inner_k=sink.inner_k,
        spatial_n_tile=spatial_n,
        active_tasks=active_tasks,
        trips_per_task=trips,
        physical_l0_tile=sink.physical_l0_tile,
        contiguous_request_bytes=sink.contiguous_request_bytes,
        physical_constraints={
            key: all(geometry.physical_constraints[key] for geometry in geometries)
            for key in sink.physical_constraints
        },
    )


def _cube_execution_summary(
    solution: Mapping[str, Any],
    problem: Mapping[str, Any],
    *,
    field: str,
) -> CubeCandidateExecution:
    """Derive launch, drain, and four-port traffic from a cube plan."""

    step = solution["steps"][0]
    if not isinstance(step, Mapping) or step.get("kind") != "cube":
        raise ValueError(f"{field}.solution step is not a cube plan")
    plan = step.get("plan")
    if not isinstance(plan, Mapping):
        raise ValueError(f"{field}.solution cube plan is missing")
    work_units = _positive_int(plan.get("work_units"), f"{field}.work_units")
    dtypes = problem.get("dtypes")
    if not isinstance(dtypes, list):
        raise ValueError(f"{field}.problem dtypes are missing")
    dtype_bytes = {
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

    def region_bytes(region: Any, name: str) -> int:
        if not isinstance(region, Mapping):
            raise ValueError(f"{field}.{name} region is missing")
        tensor = region.get("tensor")
        if (
            not isinstance(tensor, int)
            or isinstance(tensor, bool)
            or tensor < 0
            or tensor >= len(dtypes)
            or not isinstance(dtypes[tensor], str)
        ):
            raise ValueError(f"{field}.{name}.tensor is outside problem dtypes")
        try:
            element_bytes = dtype_bytes[dtypes[tensor].lower()]
        except KeyError as error:
            raise ValueError(
                f"{field}.{name} has unsupported dtype {dtypes[tensor]!r}"
            ) from error
        height = _positive_int(region.get("height"), f"{field}.{name}.height")
        width = _positive_int(region.get("width"), f"{field}.{name}.width")
        return height * width * element_bytes

    gm_l1_per_work = 0
    residents = plan.get("resident_boundaries", [])
    if not isinstance(residents, list):
        raise ValueError(f"{field}.resident_boundaries must be a list")
    for resident_index, resident in enumerate(residents):
        if not isinstance(resident, Mapping):
            raise ValueError(
                f"{field}.resident_boundaries[{resident_index}] is not an object"
            )
        gm_l1_per_work += _positive_int(
            resident.get("bytes"),
            f"{field}.resident_boundaries[{resident_index}].bytes",
        )

    matmuls = plan.get("matmuls")
    if not isinstance(matmuls, list) or not matmuls:
        raise ValueError(f"{field}.matmuls must be a non-empty list")
    drain_sites = 0
    drain_executions = 0
    drain_bytes = 0
    l0c_gm = 0
    for matmul_index, matmul in enumerate(matmuls):
        name = f"matmuls[{matmul_index}]"
        if not isinstance(matmul, Mapping):
            raise ValueError(f"{field}.{name} is not an object")
        output_grid = matmul.get("output_grid")
        if not isinstance(output_grid, list) or len(output_grid) != 2:
            raise ValueError(f"{field}.{name}.output_grid must have two entries")
        tiles_m = _positive_int(output_grid[0], f"{field}.{name}.output_grid[0]")
        tiles_n = _positive_int(output_grid[1], f"{field}.{name}.output_grid[1]")
        retained = matmul.get("retained_panels")
        if not isinstance(retained, Mapping):
            raise ValueError(f"{field}.{name}.retained_panels is missing")
        for operand, repeats in (("lhs", tiles_n), ("rhs", tiles_m)):
            producer = matmul.get(f"{operand}_producer")
            resident = matmul.get(f"{operand}_resident_boundary")
            if not isinstance(producer, int) or not isinstance(resident, int):
                raise ValueError(f"{field}.{name}.{operand} ownership is malformed")
            if producer >= 0 or resident >= 0:
                continue
            held = retained.get(operand)
            if not isinstance(held, bool):
                raise ValueError(f"{field}.{name}.retained_panels.{operand} is invalid")
            gm_l1_per_work += region_bytes(matmul.get(operand), f"{name}.{operand}") * (
                1 if held else repeats
            )

        drain = matmul.get("final_drain")
        if not isinstance(drain, Mapping):
            raise ValueError(f"{field}.{name}.final_drain is missing")
        if drain.get("required") is True:
            per_work_bytes = _positive_int(
                drain.get("bytes"), f"{field}.{name}.final_drain.bytes"
            )
            per_work_tiles = _positive_int(
                drain.get("tile_count"), f"{field}.{name}.final_drain.tile_count"
            )
            drain_sites += 1
            drain_executions += per_work_tiles * work_units
            drain_bytes += per_work_bytes * work_units
            if drain.get("target_l1") is False:
                l0c_gm += per_work_bytes * work_units

    policy = plan.get("split_merge_policy")
    split = policy not in (None, "none")
    zero = plan.get("aiv_zero_seed_then_atomic")
    ub_gm = 0
    if isinstance(zero, Mapping) and zero.get("present") is True:
        seed_bytes = zero.get("seed_bytes")
        if (
            not isinstance(seed_bytes, int)
            or isinstance(seed_bytes, bool)
            or seed_bytes < 0
        ):
            raise ValueError(f"{field}.aiv_zero_seed_then_atomic.seed_bytes is invalid")
        ub_gm = seed_bytes
    submissions = 2 if split else 1
    return CubeCandidateExecution(
        submissions=submissions,
        device_programs=submissions,
        cuts=0,
        drain_sites=drain_sites,
        drain_executions=drain_executions,
        drain_bytes=drain_bytes,
        traffic_bytes={
            "gm_l1": gm_l1_per_work * work_units,
            "gm_ub": 0,
            "l0c_gm": l0c_gm,
            "ub_gm": ub_gm,
        },
    )


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _finite_float(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value
