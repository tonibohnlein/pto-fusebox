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


def enumerate_mixed_group_plans(
    region: RegionSolveResult,
    *,
    sweep_binary: str | os.PathLike[str] | None = None,
) -> MixedGroupSweep:
    """Enumerate group counts through the production C++ mixed cost model."""

    if region.problem is None:
        raise ValueError(f"region {region.region.id} has no lowered problem")
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
            raise RuntimeError(
                f"mixed group sweep failed with status {process.returncode}: {detail}"
            )
        if not output_path.is_file():
            raise RuntimeError("mixed group sweep did not create its output file")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    sweep = _parse_sweep(
        payload,
        problem_sha256=problem_sha256,
        solution_sha256=solution_sha256,
        stdout=process.stdout,
        stderr=process.stderr,
    )
    _validate_sweep_against_region(region, sweep)
    return sweep


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
    launch["cores"] = candidate.groups * 3
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
        if candidate.fifos != expected_fifos:
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
