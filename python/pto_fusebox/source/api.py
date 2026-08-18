"""Public entry points for deterministic PyPTO source emission."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ..ir import NormalizedGraph, normalized_graph_sha256
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
    if normalized_graph_sha256(graph) != lowered.normalized_graph_sha256:
        raise SourceEmissionError(
            "supplied normalized graph does not match the graph used to solve the region"
        )
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
    tree = ast.parse(source)
    if _has_automatic_scheduling_tag(tree):
        raise SourceEmissionError("generated source must encode the plan directly")
    return source


def _has_automatic_scheduling_tag(tree: ast.AST) -> bool:
    """Return whether a generated function requests compiler-side scheduling."""

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if not (
                isinstance(decorator.func, ast.Attribute)
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == "pl"
                and decorator.func.attr == "function"
            ):
                continue
            for keyword in decorator.keywords:
                if keyword.arg != "attrs" or not isinstance(keyword.value, ast.Dict):
                    continue
                for key in keyword.value.keys:
                    if isinstance(key, ast.Constant) and key.value in {
                        "auto_fuse",
                        "auto_tile",
                    }:
                        return True
    return False
