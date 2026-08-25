"""Public Torch/FX capture and PTO-Fusebox scheduling API."""

from .bindings import InputBindingError, bind_emitted_inputs

from .cube_sweep import (
    CUBE_PLAN_SWEEP_SCHEMA,
    CubeCandidateGrid,
    CubePlanCandidate,
    CubePlanSweep,
    enumerate_cube_plans,
    region_for_cube_candidate,
)
from .ir import (
    NORMALIZED_GRAPH_SCHEMA,
    PROBLEM_SCHEMA,
    SOLUTION_SCHEMA,
    GraphPattern,
    GraphPatternBinding,
    NormalizedGraph,
    NormalizedOp,
    NormalizedValue,
    ShapeDimension,
)
from .normalize import export_and_normalize, normalize_exported
from .regions import SolverRegion, extract_solver_regions
from .schedule import (
    AxisPartition,
    KernelKind,
    KernelStep,
    ScheduleContractError,
    ScheduledRegion,
    scheduled_region,
)
from .solver import RegionSolveResult, SolveResult, solve_graph
from .source import (
    EmittedPyPTOCallable,
    EmittedPyPTOSource,
    PyPTOABIArgument,
    PyPTORuntimeValidShapeArgument,
    RuntimeValidShapeSpec,
    SourceEmissionError,
    can_emit_region,
    emit_pypto_callable,
    emit_pypto_region,
)
from .target import Ascend910BTarget, TargetProfile

__all__ = [
    "NORMALIZED_GRAPH_SCHEMA",
    "PROBLEM_SCHEMA",
    "SOLUTION_SCHEMA",
    "Ascend910BTarget",
    "AxisPartition",
    "CUBE_PLAN_SWEEP_SCHEMA",
    "CubeCandidateGrid",
    "CubePlanCandidate",
    "CubePlanSweep",
    "EmittedPyPTOCallable",
    "EmittedPyPTOSource",
    "GraphPattern",
    "GraphPatternBinding",
    "InputBindingError",
    "KernelKind",
    "KernelStep",
    "NormalizedGraph",
    "NormalizedOp",
    "NormalizedValue",
    "PyPTOABIArgument",
    "PyPTORuntimeValidShapeArgument",
    "RegionSolveResult",
    "RuntimeValidShapeSpec",
    "ScheduleContractError",
    "ScheduledRegion",
    "ShapeDimension",
    "SolveResult",
    "SolverRegion",
    "SourceEmissionError",
    "TargetProfile",
    "can_emit_region",
    "bind_emitted_inputs",
    "export_and_normalize",
    "emit_pypto_callable",
    "emit_pypto_region",
    "enumerate_cube_plans",
    "extract_solver_regions",
    "normalize_exported",
    "region_for_cube_candidate",
    "scheduled_region",
    "solve_graph",
]
