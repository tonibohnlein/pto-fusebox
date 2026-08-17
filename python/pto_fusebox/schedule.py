"""Typed, fail-closed view of a PTO-Fusebox solver solution."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .solver import RegionSolveResult


class ScheduleContractError(ValueError):
    """Raised when solver JSON is incomplete or internally inconsistent."""


class KernelKind(Enum):
    VECTOR = "vector"
    CUBE = "cube"
    MIXED = "mixed"


@dataclass(frozen=True)
class AxisPartition:
    big: int
    small: int
    num_big: int
    parts: int

    @classmethod
    def from_json(cls, value: Any, *, field: str) -> AxisPartition:
        item = _mapping(value, field)
        result = cls(
            big=_positive_int(item.get("big"), f"{field}.big"),
            small=_positive_int(item.get("small"), f"{field}.small"),
            num_big=_nonnegative_int(item.get("num_big"), f"{field}.num_big"),
            parts=_positive_int(item.get("parts"), f"{field}.parts"),
        )
        if result.small > result.big or result.num_big > result.parts:
            raise ScheduleContractError(f"{field} is not a valid balanced partition")
        return result


@dataclass(frozen=True)
class KernelStep:
    index: int
    kind: KernelKind
    solver_ops: tuple[int, ...]
    graph_ops: tuple[str, ...]
    op_order: tuple[int, ...]
    tile_w: int
    tile_h: int
    tile_k: int
    parts_m: int
    parts_n: int
    split: int
    cores: int
    latency: float
    plan: Mapping[str, Any]


@dataclass(frozen=True)
class ScheduledRegion:
    result: RegionSolveResult
    tensor_values: tuple[str, ...]
    steps: tuple[KernelStep, ...]


def scheduled_region(result: RegionSolveResult) -> ScheduledRegion:
    """Validate and type one solved region without rediscovering its schedule."""

    if result.status != "solved" or result.solution is None or result.problem is None:
        raise ScheduleContractError(f"region {result.region.id} is not solved")
    solution = result.solution
    subgraphs = _sequence(solution.get("subgraphs"), "subgraphs")
    count = len(subgraphs)
    if count == 0:
        raise ScheduleContractError("solution contains no kernel steps")
    fields = {
        name: _sequence(solution.get(name), name)
        for name in (
            "granularities",
            "parts",
            "splits",
            "cores",
            "op_order",
            "subgraph_latencies",
            "vector_stream",
            "cube_schedule",
            "mixed_schedule",
        )
    }
    for name, values in fields.items():
        if len(values) != count:
            raise ScheduleContractError(
                f"solution has {count} subgraphs but {len(values)} {name} entries"
            )

    graph_mapping = result.solver_op_to_graph
    tensor_values = result.solver_tensor_to_value
    steps: list[KernelStep] = []
    covered: set[int] = set()
    for index in range(count):
        solver_ops = tuple(
            _bounded_int(item, f"subgraphs[{index}]", len(graph_mapping))
            for item in _sequence(subgraphs[index], f"subgraphs[{index}]")
        )
        if not solver_ops or len(set(solver_ops)) != len(solver_ops):
            raise ScheduleContractError(
                f"subgraphs[{index}] must contain distinct operations"
            )
        if covered.intersection(solver_ops):
            raise ScheduleContractError(
                f"subgraphs[{index}] overlaps an earlier kernel step"
            )
        covered.update(solver_ops)

        order = tuple(
            _bounded_int(item, f"op_order[{index}]", len(graph_mapping))
            for item in _sequence(fields["op_order"][index], f"op_order[{index}]")
        )
        if set(order) != set(solver_ops) or len(order) != len(solver_ops):
            raise ScheduleContractError(
                f"op_order[{index}] is not a permutation of its subgraph"
            )

        granularity = _int_tuple(
            fields["granularities"][index], 3, f"granularities[{index}]"
        )
        parts = _int_tuple(fields["parts"][index], 2, f"parts[{index}]")
        vector = fields["vector_stream"][index]
        cube = fields["cube_schedule"][index]
        mixed = fields["mixed_schedule"][index]
        present = [vector is not None, cube is not None, mixed is not None]
        if sum(present) != 1:
            raise ScheduleContractError(
                f"kernel step {index} must carry exactly one vector, cube, or mixed plan"
            )
        if mixed is not None:
            kind, plan = KernelKind.MIXED, _mapping(mixed, f"mixed_schedule[{index}]")
        elif cube is not None:
            kind, plan = KernelKind.CUBE, _mapping(cube, f"cube_schedule[{index}]")
        else:
            kind, plan = KernelKind.VECTOR, _mapping(vector, f"vector_stream[{index}]")

        latency = fields["subgraph_latencies"][index]
        if not isinstance(latency, (int, float)) or not math.isfinite(float(latency)):
            raise ScheduleContractError(f"subgraph_latencies[{index}] is not finite")
        steps.append(
            KernelStep(
                index=index,
                kind=kind,
                solver_ops=solver_ops,
                graph_ops=tuple(graph_mapping[op] for op in solver_ops),
                op_order=order,
                tile_w=granularity[0],
                tile_h=granularity[1],
                tile_k=granularity[2],
                parts_m=parts[0],
                parts_n=parts[1],
                split=_positive_int(fields["splits"][index], f"splits[{index}]"),
                cores=_positive_int(fields["cores"][index], f"cores[{index}]"),
                latency=float(latency),
                plan=plan,
            )
        )

    if covered != set(range(len(graph_mapping))):
        raise ScheduleContractError(
            "kernel steps do not cover every lowered solver operation"
        )
    return ScheduledRegion(
        result=result, tensor_values=tensor_values, steps=tuple(steps)
    )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScheduleContractError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ScheduleContractError(f"{field} must be an array")
    return value


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ScheduleContractError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ScheduleContractError(f"{field} must be a non-negative integer")
    return value


def _bounded_int(value: Any, field: str, bound: int) -> int:
    result = _nonnegative_int(value, field)
    if result >= bound:
        raise ScheduleContractError(f"{field} contains out-of-range operation {result}")
    return result


def _int_tuple(value: Any, size: int, field: str) -> tuple[int, ...]:
    items = _sequence(value, field)
    if len(items) != size:
        raise ScheduleContractError(f"{field} must contain exactly {size} integers")
    return tuple(_positive_int(item, field) for item in items)
