"""Developer-facing mixed active-group cost and schedule sweep."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .solver import RegionSolveResult
from .schedule import scheduled_region
from .schedule.schema import MixedKernelPlan


MIXED_GROUP_SWEEP_SCHEMA = "pto_fusebox.mixed_group_sweep.v2"
MIXED_GROUP_SWEEP_AVAILABILITY_SCHEMA = "pto_fusebox.mixed_group_sweep_availability.v1"


@dataclass(frozen=True)
class MixedCostBreakdown:
    """Production mixed roofline components for one active-group choice."""

    cube_phase_cycles: float
    vector_phase_cycles: float
    gm_l1_bytes: float
    gm_ub_bytes: float
    l0c_gm_bytes: float
    ub_gm_bytes: float
    gm_l1_effective_parallelism: float
    gm_ub_effective_parallelism: float
    l0c_gm_effective_parallelism: float
    ub_gm_effective_parallelism: float
    gm_l1_cycles: float
    gm_ub_cycles: float
    l0c_gm_cycles: float
    ub_gm_cycles: float
    ddr_wall_cycles: float
    pipeline_wall_cycles: float
    kernel_fill_cycles: float
    group_overhead_cycles: float
    total_cycles: float


@dataclass(frozen=True)
class MixedGroupTile:
    """Exact model-selected tile/grid provenance shared by sweep candidates."""

    height: int
    width: int
    contraction: int
    parts_m: int
    parts_n: int


@dataclass(frozen=True)
class MixedGroupCandidate:
    """One uniformly assigned active-group candidate."""

    id: str
    selected: bool
    problem_sha256: str
    solution_sha256: str
    tile: MixedGroupTile
    groups: int
    trips_per_group: int
    pipeline_stages: int
    overlap_implementable: bool
    cube_stage_peak_l1_bytes: int
    cube_stage_peak_l0a_bytes: int
    cube_stage_peak_l0b_bytes: int
    source_l1_allocation_bytes: int
    vector_stage_peak_ub_bytes: int
    breakdown: MixedCostBreakdown
    fifos: tuple[Mapping[str, Any], ...]
    stages: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class MixedGroupSweep:
    """All group-count candidates for the model-selected mixed tile."""

    selected_candidate_id: str
    selection_resolution_cycles: float
    tile: MixedGroupTile
    candidates: tuple[MixedGroupCandidate, ...]
    stdout: str = ""
    stderr: str = ""

    @property
    def selected(self) -> MixedGroupCandidate:
        for candidate in self.candidates:
            if candidate.id == self.selected_candidate_id:
                return candidate
        raise ValueError(
            f"selected mixed candidate {self.selected_candidate_id!r} is missing"
        )


@dataclass(frozen=True)
class MixedGroupSweepAvailability:
    """Whether the selected solution can be swept, with a stable reason code."""

    available: bool
    code: str | None
    reason: str | None
    selected_step_kinds: tuple[str, ...]
    selected_partition: tuple[tuple[int, ...], ...]
    protocol: str | None = None
    topology_stages: int | None = None
    transfers: int | None = None
    vector_to_cube_transfers: int | None = None
    cube_to_vector_transfers: int | None = None
    active_groups: int | None = None
    trips_per_group: int | None = None
    pipeline_stages: int | None = None
    rejection_counts: Mapping[str, int] | None = None
    closest_tile: MixedGroupTile | None = None
    required_vec_bytes: int | None = None
    available_vec_bytes: int | None = None
    required_l1_bytes: int | None = None
    available_l1_bytes: int | None = None
    required_l0a_bytes: int | None = None
    available_l0a_bytes: int | None = None
    required_l0b_bytes: int | None = None
    available_l0b_bytes: int | None = None


class MixedGroupSweepUnavailable(RuntimeError):
    """A mixed sweep cannot be formed for the selected source solution."""

    def __init__(self, availability: MixedGroupSweepAvailability):
        if (
            availability.available
            or availability.code is None
            or availability.reason is None
        ):
            raise ValueError("an unavailable mixed sweep requires a code and reason")
        super().__init__(f"{availability.code}: {availability.reason}")
        self.availability = availability


def _cost_breakdown_payload(candidate: MixedGroupCandidate) -> dict[str, Any]:
    """Serialize one swept candidate in the ordinary solution shape."""

    breakdown = candidate.breakdown
    return {
        "active_groups": candidate.groups,
        "trips_per_group": candidate.trips_per_group,
        "pipeline_stages": candidate.pipeline_stages,
        "overlap_implementable": candidate.overlap_implementable,
        "cube_phase_cycles": breakdown.cube_phase_cycles,
        "vector_phase_cycles": breakdown.vector_phase_cycles,
        "traffic_bytes": {
            "gm_l1": breakdown.gm_l1_bytes,
            "gm_ub": breakdown.gm_ub_bytes,
            "l0c_gm": breakdown.l0c_gm_bytes,
            "ub_gm": breakdown.ub_gm_bytes,
        },
        "effective_parallelism": {
            "gm_l1": breakdown.gm_l1_effective_parallelism,
            "gm_ub": breakdown.gm_ub_effective_parallelism,
            "l0c_gm": breakdown.l0c_gm_effective_parallelism,
            "ub_gm": breakdown.ub_gm_effective_parallelism,
        },
        "traffic_cycles": {
            "gm_l1": breakdown.gm_l1_cycles,
            "gm_ub": breakdown.gm_ub_cycles,
            "l0c_gm": breakdown.l0c_gm_cycles,
            "ub_gm": breakdown.ub_gm_cycles,
        },
        "ddr_wall_cycles": breakdown.ddr_wall_cycles,
        "pipeline_wall_cycles": breakdown.pipeline_wall_cycles,
        "kernel_fill_cycles": breakdown.kernel_fill_cycles,
        "group_overhead_cycles": breakdown.group_overhead_cycles,
        "total_cycles": breakdown.total_cycles,
    }


def mixed_group_sweep_availability(
    region: RegionSolveResult,
    *,
    sweep_binary: str | os.PathLike[str] | None = None,
) -> MixedGroupSweepAvailability:
    """Inspect sweep prerequisites, optionally probing the complete op set.

    Without ``sweep_binary`` this is a cheap inspection of the selected
    solution.  Supplying the diagnostic binary also explains why a solution
    containing GM cuts could not be represented as one whole mixed region.
    """

    solution = region.solution
    if region.status != "solved" or not isinstance(solution, Mapping):
        return MixedGroupSweepAvailability(
            available=False,
            code="region_not_solved",
            reason=f"region {region.region.id} has status {region.status!r}",
            selected_step_kinds=(),
            selected_partition=(),
        )
    raw_steps = solution.get("steps")
    if not isinstance(raw_steps, list):
        return MixedGroupSweepAvailability(
            available=False,
            code="solution_steps_missing",
            reason="selected solution has no step list",
            selected_step_kinds=(),
            selected_partition=(),
        )
    kinds = tuple(
        str(step.get("kind", "<missing>")) if isinstance(step, Mapping) else "<invalid>"
        for step in raw_steps
    )
    partition = tuple(
        tuple(int(op) for op in step.get("ops", ()))
        for step in raw_steps
        if isinstance(step, Mapping)
        and isinstance(step.get("ops"), list)
        and all(isinstance(op, int) and not isinstance(op, bool) for op in step["ops"])
    )
    if len(raw_steps) != 1 or kinds != ("mixed",):
        selected = MixedGroupSweepAvailability(
            available=False,
            code="selected_solution_is_not_one_mixed_region",
            reason=(
                "mixed group sweeps preserve one selected mixed tile, but the "
                f"solver selected {len(raw_steps)} steps with kinds {kinds!r}"
            ),
            selected_step_kinds=kinds,
            selected_partition=partition,
        )
        if sweep_binary is None or region.problem is None:
            return selected
        return _probe_whole_region_availability(
            region,
            selected=selected,
            sweep_binary=sweep_binary,
        )
    try:
        step = scheduled_region(region).steps[0]
    except (TypeError, ValueError) as error:
        return MixedGroupSweepAvailability(
            available=False,
            code="selected_schedule_invalid",
            reason=str(error),
            selected_step_kinds=kinds,
            selected_partition=partition,
        )
    if not isinstance(step.plan, MixedKernelPlan):
        return MixedGroupSweepAvailability(
            available=False,
            code="selected_step_has_no_mixed_plan",
            reason="selected mixed step does not carry a MixedKernelPlan",
            selected_step_kinds=kinds,
            selected_partition=partition,
        )
    if not step.plan.source_codegen_ready:
        return MixedGroupSweepAvailability(
            available=False,
            code="selected_mixed_plan_is_not_source_ready",
            reason="selected mixed plan is not source-codegen ready",
            selected_step_kinds=kinds,
            selected_partition=partition,
        )
    return MixedGroupSweepAvailability(
        available=True,
        code=None,
        reason=None,
        selected_step_kinds=kinds,
        selected_partition=partition,
    )


def enumerate_mixed_group_plans(
    region: RegionSolveResult,
    *,
    sweep_binary: str | os.PathLike[str] | None = None,
) -> MixedGroupSweep:
    """Enumerate group counts through the production C++ mixed cost model."""

    availability = mixed_group_sweep_availability(region)
    executable: Path | None = None
    if (
        not availability.available
        and availability.code == "selected_solution_is_not_one_mixed_region"
        and region.problem is not None
    ):
        executable = _resolve_sweep_binary(sweep_binary)
        availability = mixed_group_sweep_availability(region, sweep_binary=executable)
    if not availability.available:
        raise MixedGroupSweepUnavailable(availability)
    if region.problem is None:
        raise ValueError(f"region {region.region.id} has no lowered problem")
    if executable is None:
        executable = _resolve_sweep_binary(sweep_binary)
    canonical_problem = json.dumps(
        region.problem, sort_keys=True, separators=(",", ":")
    )
    problem_sha256 = hashlib.sha256(canonical_problem.encode()).hexdigest()
    if region.solution is None:
        raise ValueError(f"region {region.region.id} has no solver solution")
    solution_sha256 = _canonical_sha256(region.solution)
    with tempfile.TemporaryDirectory(prefix="pto-fusebox-mixed-sweep-") as directory:
        root = Path(directory)
        problem_path = root / "problem.json"
        output_path = root / "sweep.json"
        problem_path.write_text(canonical_problem + "\n", encoding="utf-8")
        process = subprocess.run(
            [str(executable), str(problem_path), str(output_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            detail = process.stderr.strip() or process.stdout.strip()
            code = (
                "whole_region_has_no_feasible_mixed_candidate"
                if "found no selected candidate" in detail
                else "mixed_group_sweep_process_failed"
            )
            raise MixedGroupSweepUnavailable(
                MixedGroupSweepAvailability(
                    available=False,
                    code=code,
                    reason=(
                        f"sweep process exited with status {process.returncode}: {detail}"
                    ),
                    selected_step_kinds=availability.selected_step_kinds,
                    selected_partition=availability.selected_partition,
                )
            )
        if not output_path.is_file():
            raise RuntimeError("mixed group sweep did not create its output file")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    unavailable = _availability_from_payload(
        payload,
        selected_step_kinds=availability.selected_step_kinds,
        selected_partition=availability.selected_partition,
    )
    if unavailable is not None:
        raise MixedGroupSweepUnavailable(unavailable)
    sweep = _parse_sweep(
        payload,
        problem_sha256=problem_sha256,
        solution_sha256=solution_sha256,
        stdout=process.stdout,
        stderr=process.stderr,
    )
    _validate_sweep_against_region(region, sweep)
    return sweep


def _probe_whole_region_availability(
    region: RegionSolveResult,
    *,
    selected: MixedGroupSweepAvailability,
    sweep_binary: str | os.PathLike[str],
) -> MixedGroupSweepAvailability:
    """Run the diagnostic sweep on the whole lowered problem."""

    executable = _resolve_sweep_binary(sweep_binary)
    canonical_problem = json.dumps(
        region.problem, sort_keys=True, separators=(",", ":")
    )
    with tempfile.TemporaryDirectory(prefix="pto-fusebox-mixed-probe-") as directory:
        root = Path(directory)
        problem_path = root / "problem.json"
        output_path = root / "sweep.json"
        problem_path.write_text(canonical_problem + "\n", encoding="utf-8")
        process = subprocess.run(
            [str(executable), str(problem_path), str(output_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0 or not output_path.is_file():
            detail = process.stderr.strip() or process.stdout.strip()
            return replace(
                selected,
                code="mixed_group_sweep_process_failed",
                reason=(
                    f"sweep process exited with status {process.returncode}: {detail}"
                ),
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    unavailable = _availability_from_payload(
        payload,
        selected_step_kinds=selected.selected_step_kinds,
        selected_partition=selected.selected_partition,
    )
    if unavailable is not None:
        return unavailable
    return replace(
        selected,
        code="selected_solution_prefers_gm_cut",
        reason=(
            "a whole-region mixed sweep is feasible, but the selected solution "
            "has a lower-cost partition containing GM cuts"
        ),
    )


def _availability_from_payload(
    payload: Any,
    *,
    selected_step_kinds: tuple[str, ...],
    selected_partition: tuple[tuple[int, ...], ...],
) -> MixedGroupSweepAvailability | None:
    if not isinstance(payload, Mapping):
        return None
    if payload.get("schema_version") != MIXED_GROUP_SWEEP_AVAILABILITY_SCHEMA:
        return None
    if payload.get("available") is not False:
        raise ValueError("mixed sweep availability payload must be unavailable")
    code = payload.get("code")
    reason = payload.get("reason")
    if not isinstance(code, str) or not isinstance(reason, str):
        raise ValueError("unavailable mixed sweep omits its code or reason")
    raw_tile = payload.get("closest_tile")
    closest_tile = (
        _parse_tile(raw_tile, field="closest_tile")
        if isinstance(raw_tile, Mapping)
        else None
    )

    def optional_int(field: str) -> int | None:
        value = payload.get(field)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"unavailable mixed sweep has invalid {field}")
        return value

    raw_rejection_counts = payload.get("rejection_counts")
    rejection_counts = (
        {
            str(key): int(value)
            for key, value in raw_rejection_counts.items()
            if isinstance(key, str)
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        }
        if isinstance(raw_rejection_counts, Mapping)
        else None
    )

    return MixedGroupSweepAvailability(
        available=False,
        code=code,
        reason=reason,
        selected_step_kinds=selected_step_kinds,
        selected_partition=selected_partition,
        protocol=payload.get("protocol")
        if isinstance(payload.get("protocol"), str)
        else None,
        topology_stages=optional_int("topology_stages"),
        transfers=optional_int("transfers"),
        vector_to_cube_transfers=optional_int("vector_to_cube_transfers"),
        cube_to_vector_transfers=optional_int("cube_to_vector_transfers"),
        active_groups=optional_int("active_groups"),
        trips_per_group=optional_int("trips_per_group"),
        pipeline_stages=optional_int("pipeline_stages"),
        rejection_counts=rejection_counts,
        closest_tile=closest_tile,
        required_vec_bytes=optional_int("required_vec_bytes"),
        available_vec_bytes=optional_int("available_vec_bytes"),
        required_l1_bytes=optional_int("required_l1_bytes"),
        available_l1_bytes=optional_int("available_l1_bytes"),
        required_l0a_bytes=optional_int("required_l0a_bytes"),
        available_l0a_bytes=optional_int("available_l0a_bytes"),
        required_l0b_bytes=optional_int("required_l0b_bytes"),
        available_l0b_bytes=optional_int("available_l0b_bytes"),
    )


def region_for_mixed_group_candidate(
    region: RegionSolveResult,
    candidate: MixedGroupCandidate,
) -> RegionSolveResult:
    """Bind a swept group count to the selected tile without replanning."""

    if region.problem is None or region.solution is None:
        raise ValueError(f"region {region.region.id} has no problem or solution")
    canonical_problem = json.dumps(
        region.problem, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(canonical_problem.encode()).hexdigest()
    if digest != candidate.problem_sha256:
        raise ValueError(
            f"mixed candidate {candidate.id} belongs to a different lowered problem"
        )
    if _canonical_sha256(region.solution) != candidate.solution_sha256:
        raise ValueError(
            f"mixed candidate {candidate.id} belongs to a different solver plan"
        )
    solution = copy.deepcopy(region.solution)
    steps = solution.get("steps")
    if not isinstance(steps, list) or len(steps) != 1:
        raise ValueError("mixed group candidate requires exactly one solution step")
    step = steps[0]
    plan = step.get("plan")
    launch = step.get("launch")
    if not isinstance(plan, dict) or not isinstance(launch, dict):
        raise ValueError("mixed solution omits its plan or launch")
    if _tile_from_launch(launch) != candidate.tile:
        raise ValueError("mixed candidate tile differs from the solution launch")
    if plan.get("spatial_tiles") != candidate.groups * candidate.trips_per_group:
        raise ValueError("mixed candidate does not cover the solution work items")
    protocol = plan.get("protocol")
    if protocol not in {"one_way", "single_round_trip_bundle"}:
        raise ValueError("mixed candidate cannot bind this protocol")
    plan["active_groups"] = candidate.groups
    plan["min_trips_per_group"] = candidate.trips_per_group
    plan["max_trips_per_group"] = candidate.trips_per_group
    plan["pipeline_stages"] = candidate.pipeline_stages
    plan["requested_skew_depth"] = (
        candidate.pipeline_stages - 1 if candidate.pipeline_stages <= 2 else 2
    )
    plan["model_overlap_granted"] = candidate.overlap_implementable
    plan["overlap_implementable"] = candidate.overlap_implementable
    plan["pipeline_fill_absorbed"] = (
        protocol == "single_round_trip_bundle"
        and candidate.overlap_implementable
        and plan.get("algorithm") == "generic"
    )
    plan["cube_stage_peak_l1_bytes"] = candidate.cube_stage_peak_l1_bytes
    plan["cube_stage_peak_l0a_bytes"] = candidate.cube_stage_peak_l0a_bytes
    plan["cube_stage_peak_l0b_bytes"] = candidate.cube_stage_peak_l0b_bytes
    plan["source_l1_allocation_bytes"] = candidate.source_l1_allocation_bytes
    plan["vector_stage_peak_ub_bytes"] = candidate.vector_stage_peak_ub_bytes
    plan["fifos"] = [dict(fifo) for fifo in candidate.fifos]
    plan["cost_breakdown"] = _cost_breakdown_payload(candidate)
    vector_lanes = plan.get("vector_lanes")
    if not isinstance(vector_lanes, int) or isinstance(vector_lanes, bool):
        raise ValueError("mixed candidate solution has no vector lane count")
    launch["cores"] = candidate.groups * (1 + vector_lanes)
    step["latency_cycles"] = candidate.breakdown.total_cycles
    return replace(
        region,
        status="solved",
        solution=solution,
        diagnostics=region.region.diagnostics,
        stdout="",
        stderr="",
        returncode=0,
    )


def _resolve_sweep_binary(value: str | os.PathLike[str] | None) -> Path:
    if value is not None:
        candidates = [Path(value)]
    elif os.environ.get("PTO_FUSEBOX_MIXED_SWEEP"):
        candidates = [Path(os.environ["PTO_FUSEBOX_MIXED_SWEEP"])]
    else:
        root = Path(__file__).resolve().parents[2]
        candidates = [root / "build" / "mixed_group_sweep"]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    rendered = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        f"no built mixed group sweep found ({rendered}); "
        "build mixed_group_sweep explicitly"
    )


def _parse_sweep(
    payload: Any,
    *,
    problem_sha256: str,
    solution_sha256: str,
    stdout: str,
    stderr: str,
) -> MixedGroupSweep:
    if not isinstance(payload, Mapping):
        raise ValueError("mixed group sweep JSON must contain an object")
    if payload.get("schema_version") != MIXED_GROUP_SWEEP_SCHEMA:
        raise ValueError(
            "unsupported mixed group sweep schema "
            f"{payload.get('schema_version')!r}; expected {MIXED_GROUP_SWEEP_SCHEMA!r}"
        )
    selected_id = payload.get("selected_candidate_id")
    selection_resolution_cycles = _finite_float(
        payload.get("selection_resolution_cycles"),
        "selection_resolution_cycles",
    )
    if selection_resolution_cycles <= 0:
        raise ValueError("selection_resolution_cycles must be positive")
    tile = _parse_tile(payload.get("tile"), field="tile")
    raw_candidates = payload.get("candidates")
    if not isinstance(selected_id, str) or not selected_id:
        raise ValueError("mixed group sweep has no selected candidate id")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("mixed group sweep contains no candidates")
    candidates = tuple(
        _parse_candidate(
            item,
            problem_sha256=problem_sha256,
            solution_sha256=solution_sha256,
            tile=tile,
            index=index,
        )
        for index, item in enumerate(raw_candidates)
    )
    ids = [candidate.id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("mixed group sweep candidate ids are not unique")
    selected = [candidate for candidate in candidates if candidate.selected]
    if len(selected) != 1 or selected[0].id != selected_id:
        raise ValueError("mixed group sweep selected-candidate markers disagree")
    return MixedGroupSweep(
        selected_candidate_id=selected_id,
        selection_resolution_cycles=selection_resolution_cycles,
        tile=tile,
        candidates=candidates,
        stdout=stdout,
        stderr=stderr,
    )


def _parse_candidate(
    payload: Any,
    *,
    problem_sha256: str,
    solution_sha256: str,
    tile: MixedGroupTile,
    index: int,
) -> MixedGroupCandidate:
    field = f"candidates[{index}]"
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field} is not an object")
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError(f"{field}.model is not an object")
    candidate_id = payload.get("id")
    selected = payload.get("selected")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError(f"{field}.id must be a non-empty string")
    if not isinstance(selected, bool):
        raise ValueError(f"{field}.selected must be boolean")
    breakdown = MixedCostBreakdown(
        **{
            name: _finite_float(model.get(name), f"{field}.model.{name}")
            for name in MixedCostBreakdown.__dataclass_fields__
        }
    )
    fifos = payload.get("fifos")
    stages = payload.get("stages")
    if not isinstance(fifos, list) or not all(
        isinstance(item, Mapping) for item in fifos
    ):
        raise ValueError(f"{field}.fifos must be a list of objects")
    if not isinstance(stages, list) or not all(
        isinstance(item, Mapping) for item in stages
    ):
        raise ValueError(f"{field}.stages must be a list of objects")
    return MixedGroupCandidate(
        id=candidate_id,
        selected=selected,
        problem_sha256=problem_sha256,
        solution_sha256=solution_sha256,
        tile=tile,
        groups=_positive_int(payload.get("groups"), f"{field}.groups"),
        trips_per_group=_positive_int(
            payload.get("trips_per_group"), f"{field}.trips_per_group"
        ),
        pipeline_stages=_positive_int(
            payload.get("pipeline_stages"), f"{field}.pipeline_stages"
        ),
        overlap_implementable=_bool(
            payload.get("overlap_implementable"),
            f"{field}.overlap_implementable",
        ),
        cube_stage_peak_l1_bytes=_nonnegative_int(
            payload.get("cube_stage_peak_l1_bytes"),
            f"{field}.cube_stage_peak_l1_bytes",
        ),
        cube_stage_peak_l0a_bytes=_nonnegative_int(
            payload.get("cube_stage_peak_l0a_bytes"),
            f"{field}.cube_stage_peak_l0a_bytes",
        ),
        cube_stage_peak_l0b_bytes=_nonnegative_int(
            payload.get("cube_stage_peak_l0b_bytes"),
            f"{field}.cube_stage_peak_l0b_bytes",
        ),
        source_l1_allocation_bytes=_nonnegative_int(
            payload.get("source_l1_allocation_bytes"),
            f"{field}.source_l1_allocation_bytes",
        ),
        vector_stage_peak_ub_bytes=_nonnegative_int(
            payload.get("vector_stage_peak_ub_bytes"),
            f"{field}.vector_stage_peak_ub_bytes",
        ),
        breakdown=breakdown,
        fifos=tuple(
            _parse_fifo(item, field=f"{field}.fifos[{i}]")
            for i, item in enumerate(fifos)
        ),
        stages=tuple(
            _parse_stage(item, field=f"{field}.stages[{i}]")
            for i, item in enumerate(stages)
        ),
    )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _parse_tile(value: Any, *, field: str) -> MixedGroupTile:
    if not isinstance(value, Mapping) or set(value) != {
        "h",
        "w",
        "k",
        "parts_m",
        "parts_n",
    }:
        raise ValueError(f"{field} must contain exactly h/w/k/parts_m/parts_n")
    return MixedGroupTile(
        height=_positive_int(value.get("h"), f"{field}.h"),
        width=_positive_int(value.get("w"), f"{field}.w"),
        contraction=_positive_int(value.get("k"), f"{field}.k"),
        parts_m=_positive_int(value.get("parts_m"), f"{field}.parts_m"),
        parts_n=_positive_int(value.get("parts_n"), f"{field}.parts_n"),
    )


def _tile_from_launch(launch: Mapping[str, Any]) -> MixedGroupTile:
    tile = launch.get("tile")
    parts = launch.get("parts")
    if (
        not isinstance(tile, Sequence)
        or isinstance(tile, (str, bytes))
        or len(tile) != 3
        or not isinstance(parts, Sequence)
        or isinstance(parts, (str, bytes))
        or len(parts) != 2
    ):
        raise ValueError("mixed solution launch has malformed tile or parts")
    return MixedGroupTile(
        height=_positive_int(tile[1], "solution.launch.tile[1]"),
        width=_positive_int(tile[0], "solution.launch.tile[0]"),
        contraction=_positive_int(tile[2], "solution.launch.tile[2]"),
        parts_m=_positive_int(parts[0], "solution.launch.parts[0]"),
        parts_n=_positive_int(parts[1], "solution.launch.parts[1]"),
    )


def _parse_fifo(value: Any, *, field: str) -> Mapping[str, Any]:
    required = {
        "tensor",
        "pipe_id",
        "direction",
        "wire_dtype",
        "bundle",
        "spatial_m",
        "spatial_n",
        "valid_rows",
        "valid_cols",
        "slot_bytes",
        "slot_count",
        "reserved_bytes",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{field} has an incomplete FIFO identity")
    direction = value.get("direction")
    if direction not in {"cube_to_vector", "vector_to_cube"}:
        raise ValueError(f"{field}.direction is unsupported")
    wire_dtype = value.get("wire_dtype")
    if not isinstance(wire_dtype, str) or not wire_dtype:
        raise ValueError(f"{field}.wire_dtype must be a non-empty string")
    return {
        "tensor": _nonnegative_int(value.get("tensor"), f"{field}.tensor"),
        "pipe_id": _nonnegative_int(value.get("pipe_id"), f"{field}.pipe_id"),
        "direction": direction,
        "wire_dtype": wire_dtype,
        "bundle": _integer(value.get("bundle"), f"{field}.bundle"),
        "spatial_m": _bool(value.get("spatial_m"), f"{field}.spatial_m"),
        "spatial_n": _bool(value.get("spatial_n"), f"{field}.spatial_n"),
        "valid_rows": _positive_int(value.get("valid_rows"), f"{field}.valid_rows"),
        "valid_cols": _positive_int(value.get("valid_cols"), f"{field}.valid_cols"),
        "slot_bytes": _positive_int(value.get("slot_bytes"), f"{field}.slot_bytes"),
        "slot_count": _positive_int(value.get("slot_count"), f"{field}.slot_count"),
        "reserved_bytes": _positive_int(
            value.get("reserved_bytes"), f"{field}.reserved_bytes"
        ),
    }


def _parse_stage(value: Any, *, field: str) -> Mapping[str, Any]:
    required = {
        "engine",
        "topology_stage",
        "ops",
        "valid_rows",
        "valid_cols",
        "cube_window_k",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{field} has an incomplete stage identity")
    engine = value.get("engine")
    if engine not in {"cube", "vector"}:
        raise ValueError(f"{field}.engine is unsupported")
    ops = value.get("ops")
    windows = value.get("cube_window_k")
    if not isinstance(ops, list) or not ops or not isinstance(windows, list):
        raise ValueError(f"{field} has malformed operations or cube windows")
    return {
        "engine": engine,
        "topology_stage": _nonnegative_int(
            value.get("topology_stage"), f"{field}.topology_stage"
        ),
        "ops": [_nonnegative_int(op, f"{field}.ops") for op in ops],
        "valid_rows": _positive_int(value.get("valid_rows"), f"{field}.valid_rows"),
        "valid_cols": _positive_int(value.get("valid_cols"), f"{field}.valid_cols"),
        "cube_window_k": [
            _positive_int(window, f"{field}.cube_window_k") for window in windows
        ],
    }


def _validate_sweep_against_region(
    region: RegionSolveResult, sweep: MixedGroupSweep
) -> None:
    scheduled = scheduled_region(region)
    if len(scheduled.steps) != 1:
        raise ValueError("mixed group sweep requires exactly one scheduled step")
    step = scheduled.steps[0]
    if not isinstance(step.plan, MixedKernelPlan):
        raise ValueError("mixed group sweep requires a mixed scheduled plan")
    plan = step.plan
    expected_tile = MixedGroupTile(
        height=step.launch.tile_h,
        width=step.launch.tile_w,
        contraction=step.launch.tile_k,
        parts_m=step.launch.parts_m,
        parts_n=step.launch.parts_n,
    )
    if sweep.tile != expected_tile:
        raise ValueError("mixed group sweep tile differs from the selected solver plan")
    if sweep.selected.groups != plan.active_groups:
        raise ValueError(
            "mixed group sweep selected group count differs from the solver plan"
        )
    expected_fifos = tuple(
        {
            "tensor": fifo.tensor,
            "pipe_id": fifo.pipe_id,
            "direction": fifo.direction.value,
            "wire_dtype": fifo.wire_dtype,
            "bundle": fifo.bundle,
            "spatial_m": fifo.spatial_m,
            "spatial_n": fifo.spatial_n,
            "valid_rows": fifo.valid_rows,
            "valid_cols": fifo.valid_cols,
            "slot_bytes": fifo.slot_bytes,
            "slot_count": fifo.slot_count,
            "reserved_bytes": fifo.reserved_bytes,
        }
        for fifo in plan.fifos
    )
    expected_stages = tuple(
        {
            "engine": stage.engine.value,
            "topology_stage": stage.topology_stage,
            "ops": list(stage.ops),
            "valid_rows": stage.valid_rows,
            "valid_cols": stage.valid_cols,
            "cube_window_k": list(stage.cube_window_k),
        }
        for stage in plan.stages
    )
    if sweep.selected.fifos != expected_fifos:
        raise ValueError("mixed sweep selected candidate has stale FIFO provenance")
    expected_fifo_frames = tuple(
        {
            key: value
            for key, value in fifo.items()
            if key not in {"slot_count", "reserved_bytes"}
        }
        for fifo in expected_fifos
    )
    for candidate in sweep.candidates:
        if candidate.tile != expected_tile:
            raise ValueError(
                f"mixed candidate {candidate.id} has stale tile provenance"
            )
        if (
            candidate.cube_stage_peak_l1_bytes != plan.cube_stage_peak_l1_bytes
            or candidate.vector_stage_peak_ub_bytes != plan.vector_stage_peak_ub_bytes
        ):
            raise ValueError(
                f"mixed candidate {candidate.id} has stale memory provenance"
            )
        candidate_fifo_frames = tuple(
            {
                key: value
                for key, value in fifo.items()
                if key not in {"slot_count", "reserved_bytes"}
            }
            for fifo in candidate.fifos
        )
        if candidate_fifo_frames != expected_fifo_frames or any(
            fifo["reserved_bytes"] != fifo["slot_bytes"] * fifo["slot_count"]
            for fifo in candidate.fifos
        ):
            raise ValueError(
                f"mixed candidate {candidate.id} has stale FIFO provenance"
            )
        if candidate.stages != expected_stages:
            raise ValueError(
                f"mixed candidate {candidate.id} has stale stage provenance"
            )


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
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
