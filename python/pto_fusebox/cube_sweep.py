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


CUBE_PLAN_SWEEP_SCHEMA = "pto_fusebox.cube_plan_sweep.v1"


@dataclass(frozen=True)
class CubeCandidateGrid:
    """The fixed outer grid enumerated by the cube model."""

    parts_m: int
    parts_n: int
    split_k: int

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
    solution: Mapping[str, Any]


@dataclass(frozen=True)
class CubePlanSweep:
    """All feasible candidates and the model-selected candidate."""

    selected_candidate_id: str
    candidates: tuple[CubePlanCandidate, ...]
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
) -> CubePlanSweep:
    """Enumerate every feasible cube plan for one lowered homogeneous cube region.

    The C++ sweep executable reuses the production model and serializes each
    fixed candidate as ``solution.v5``. This function validates that envelope;
    it does not re-price or re-plan the candidate in Python.

    Args:
        region: A region result that carries the exact lowered problem.
        sweep_binary: An explicitly built ``cube_plan_sweep`` executable.

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
    with tempfile.TemporaryDirectory(prefix="pto-fusebox-cube-sweep-") as directory:
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
                f"cube plan sweep failed with status {process.returncode}: {detail}"
            )
        if not output_path.is_file():
            raise RuntimeError("cube plan sweep did not create its output file")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    return _parse_sweep(
        payload,
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
    if not isinstance(selected_id, str) or not selected_id:
        raise ValueError("cube plan sweep has no selected candidate id")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("cube plan sweep contains no candidates")

    candidates = tuple(
        _parse_candidate(item, problem_sha256=problem_sha256, index=index)
        for index, item in enumerate(raw_candidates)
    )
    ids = [candidate.id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("cube plan sweep candidate ids are not unique")
    selected = [candidate for candidate in candidates if candidate.selected]
    if len(selected) != 1 or selected[0].id != selected_id:
        raise ValueError("cube plan sweep selected-candidate markers disagree")
    return CubePlanSweep(
        selected_candidate_id=selected_id,
        candidates=candidates,
        stdout=stdout,
        stderr=stderr,
    )


def _parse_candidate(
    payload: Any, *, problem_sha256: str, index: int
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
        solution=solution,
    )


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
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
