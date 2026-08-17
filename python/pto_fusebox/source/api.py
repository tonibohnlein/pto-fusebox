"""Public entry points for deterministic PyPTO source emission."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ..ir import NormalizedGraph
from ..lowered import LoweredContractError, lowered_region
from ..schedule import KernelKind, ScheduleContractError, scheduled_region
from ..solver import RegionSolveResult
from .common import EmissionContext, SourceEmissionError, class_name, interface


@dataclass(frozen=True)
class EmittedPyPTOSource:
    """One deterministic PyPTO program and the schedule step it implements."""

    program_name: str
    region_id: str
    kind: KernelKind
    source: str


def can_emit_region(graph: NormalizedGraph, result: RegionSolveResult) -> bool:
    """Return whether the installed backend can emit this exact region.

    Readiness deliberately runs the same graph-aware validation and rendering
    path as :func:`emit_pypto_region`; it is not a weaker schedule-family hint.
    """

    try:
        context = _emission_context(graph, result)
        _render(context, "FuseboxReadinessProbe")
    except (LoweredContractError, ScheduleContractError, SourceEmissionError):
        return False
    return True


def emit_pypto_region(
    graph: NormalizedGraph,
    result: RegionSolveResult,
    *,
    program_name: str | None = None,
) -> EmittedPyPTOSource:
    """Emit one solver-owned homogeneous schedule as ordinary PyPTO DSL."""

    try:
        context = _emission_context(graph, result)
    except (LoweredContractError, ScheduleContractError) as error:
        raise SourceEmissionError(str(error)) from error
    chosen_name = class_name(program_name or f"fused_{result.region.id}")
    source = _render(context, chosen_name)
    return EmittedPyPTOSource(
        program_name=chosen_name,
        region_id=context.region_id,
        kind=context.step.kind,
        source=source,
    )


def _emission_context(
    graph: NormalizedGraph, result: RegionSolveResult
) -> EmissionContext:
    schedule = scheduled_region(result)
    lowered = lowered_region(result)
    if len(schedule.steps) != 1:
        raise SourceEmissionError(
            "source v1 requires exactly one selected kernel step; "
            f"the solver selected {len(schedule.steps)}"
        )
    if schedule.region_id != lowered.region_id:
        raise SourceEmissionError("problem and solution region identities disagree")
    return EmissionContext(
        graph=graph,
        lowered=lowered,
        schedule=schedule,
        step=schedule.steps[0],
        interface=interface(graph, lowered),
    )


def _render(context: EmissionContext, program_name: str) -> str:
    from .cube import emit_cube
    from .vector import emit_vector

    if context.step.kind is KernelKind.VECTOR:
        source = emit_vector(context, program_name)
    elif context.step.kind is KernelKind.CUBE:
        source = emit_cube(context, program_name)
    else:
        raise SourceEmissionError("mixed PyPTO source emission is not implemented yet")
    ast.parse(source)
    if "auto_fuse" in source or "auto_tile" in source:
        raise AssertionError("generated source must encode the plan directly")
    return source
