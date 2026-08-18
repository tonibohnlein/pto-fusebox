"""Typed view of the solver problem consumed by source emission."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .ir import PROBLEM_SCHEMA
from .solver import RegionSolveResult


class LoweredContractError(ValueError):
    """Raised when a solved region's lowered problem is internally inconsistent."""


@dataclass(frozen=True)
class LoweredTensor:
    """One concrete tensor known to the C++ scheduling problem."""

    index: int
    value_id: str
    width: int
    height: int
    dtype: str
    alias_of: str | None
    synthetic: bool

    @property
    def byte_width(self) -> int:
        widths = {
            "bool": 1,
            "BOOL": 1,
            "int8": 1,
            "INT8": 1,
            "fp16": 2,
            "FP16": 2,
            "bf16": 2,
            "BF16": 2,
            "int16": 2,
            "INT16": 2,
            "fp32": 4,
            "FP32": 4,
            "int32": 4,
            "INT32": 4,
        }
        try:
            return widths[self.dtype]
        except KeyError as error:
            raise LoweredContractError(
                f"tensor {self.index} has unsupported dtype {self.dtype!r}"
            ) from error


@dataclass(frozen=True)
class LoweredOperation:
    """One concrete solver operation, including compiler-generated cast hops."""

    index: int
    graph_op_id: str
    op_type: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]


@dataclass(frozen=True)
class LoweredRegion:
    """Typed problem-side half of the model-to-emitter contract."""

    region_id: str
    normalized_graph_sha256: str
    region_inputs: tuple[str, ...]
    region_outputs: tuple[str, ...]
    output_allocation_owners: tuple[str, ...]
    tensors: tuple[LoweredTensor, ...]
    operations: tuple[LoweredOperation, ...]
    required_outputs: tuple[int, ...]

    def tensor(self, index: int) -> LoweredTensor:
        return self.tensors[index]

    def operation(self, index: int) -> LoweredOperation:
        return self.operations[index]


def lowered_region(result: RegionSolveResult) -> LoweredRegion:
    """Decode the concrete problem paired with a solver schedule."""

    if result.status != "solved" or result.problem is None:
        raise LoweredContractError(f"region {result.region.id} is not solved")
    problem = result.problem
    if problem.get("schema_version") != PROBLEM_SCHEMA:
        raise LoweredContractError(
            f"problem schema must be {PROBLEM_SCHEMA!r}, got "
            f"{problem.get('schema_version')!r}"
        )

    widths = _sequence(problem.get("widths"), "problem.widths")
    heights = _sequence(problem.get("heights"), "problem.heights")
    dtypes = _sequence(problem.get("dtypes"), "problem.dtypes")
    tensor_count = len(result.solver_tensor_to_value)
    for name, values in (("widths", widths), ("heights", heights), ("dtypes", dtypes)):
        if len(values) != tensor_count:
            raise LoweredContractError(
                f"problem.{name} has {len(values)} entries for {tensor_count} tensors"
            )

    frontend = _mapping(problem.get("frontend_mapping"), "problem.frontend_mapping")
    region_id = frontend.get("region_id")
    if region_id != result.region.id:
        raise LoweredContractError(
            f"problem region id {region_id!r} does not match {result.region.id!r}"
        )
    graph_sha256 = frontend.get("normalized_graph_sha256")
    if (
        not isinstance(graph_sha256, str)
        or len(graph_sha256) != 64
        or any(character not in "0123456789abcdef" for character in graph_sha256)
    ):
        raise LoweredContractError(
            "problem.frontend_mapping.normalized_graph_sha256 must be a "
            "lowercase SHA-256 digest"
        )
    mapped_ops = _string_sequence(
        frontend.get("solver_op_to_graph"),
        "problem.frontend_mapping.solver_op_to_graph",
    )
    mapped_tensors = _string_sequence(
        frontend.get("solver_tensor_to_value"),
        "problem.frontend_mapping.solver_tensor_to_value",
    )
    region_inputs = _string_sequence(
        frontend.get("region_inputs"),
        "problem.frontend_mapping.region_inputs",
    )
    region_outputs = _string_sequence(
        frontend.get("region_outputs"),
        "problem.frontend_mapping.region_outputs",
    )
    output_allocation_owners = _string_sequence(
        frontend.get("region_output_allocation_owners"),
        "problem.frontend_mapping.region_output_allocation_owners",
    )
    if region_inputs != result.region.input_values:
        raise LoweredContractError(
            "problem region inputs differ from the solved region"
        )
    if region_outputs != result.region.output_values:
        raise LoweredContractError(
            "problem region outputs differ from the solved region"
        )
    if len(output_allocation_owners) != len(region_outputs):
        raise LoweredContractError(
            "problem output-allocation owners do not cover the region outputs"
        )
    aliases = _sequence(
        frontend.get("solver_tensor_alias_of"),
        "problem.frontend_mapping.solver_tensor_alias_of",
    )
    synthetic = _sequence(
        frontend.get("solver_tensor_synthetic"),
        "problem.frontend_mapping.solver_tensor_synthetic",
    )
    if mapped_ops != result.solver_op_to_graph:
        raise LoweredContractError(
            "problem solver-op mapping differs from solve result"
        )
    if mapped_tensors != result.solver_tensor_to_value:
        raise LoweredContractError(
            "problem solver-tensor mapping differs from solve result"
        )
    if len(aliases) != tensor_count or len(synthetic) != tensor_count:
        raise LoweredContractError("problem tensor metadata has the wrong length")

    tensors: list[LoweredTensor] = []
    for index, value_id in enumerate(mapped_tensors):
        alias = aliases[index]
        if alias is not None and not isinstance(alias, str):
            raise LoweredContractError(
                f"problem tensor alias {index} must be a string or null"
            )
        if not isinstance(synthetic[index], bool):
            raise LoweredContractError(
                f"problem tensor synthetic flag {index} must be a boolean"
            )
        dtype = dtypes[index]
        if not isinstance(dtype, str) or not dtype:
            raise LoweredContractError(f"problem dtype {index} must be a string")
        tensors.append(
            LoweredTensor(
                index=index,
                value_id=value_id,
                width=_positive_int(widths[index], f"problem.widths[{index}]"),
                height=_positive_int(heights[index], f"problem.heights[{index}]"),
                dtype=dtype,
                alias_of=alias,
                synthetic=synthetic[index],
            )
        )

    raw_inputs = _sequence(problem.get("inputs"), "problem.inputs")
    raw_outputs = _sequence(problem.get("outputs"), "problem.outputs")
    op_types = _sequence(problem.get("op_types"), "problem.op_types")
    op_count = len(result.solver_op_to_graph)
    for name, values in (
        ("inputs", raw_inputs),
        ("outputs", raw_outputs),
        ("op_types", op_types),
    ):
        if len(values) != op_count:
            raise LoweredContractError(
                f"problem.{name} has {len(values)} entries for {op_count} operations"
            )
    operations: list[LoweredOperation] = []
    for index, graph_op_id in enumerate(mapped_ops):
        op_type = op_types[index]
        if not isinstance(op_type, str) or not op_type:
            raise LoweredContractError(f"problem op type {index} must be a string")
        operations.append(
            LoweredOperation(
                index=index,
                graph_op_id=graph_op_id,
                op_type=op_type,
                inputs=_indices(
                    raw_inputs[index], tensor_count, f"problem.inputs[{index}]"
                ),
                outputs=_indices(
                    raw_outputs[index], tensor_count, f"problem.outputs[{index}]"
                ),
            )
        )

    required_outputs = _indices(
        problem.get("required_outputs"), tensor_count, "problem.required_outputs"
    )
    owner_indices: list[int] = []
    for owner in output_allocation_owners:
        matches = [
            tensor.index
            for tensor in tensors
            if tensor.value_id == owner and not tensor.synthetic
        ]
        if len(matches) != 1:
            raise LoweredContractError(
                f"output-allocation owner {owner!r} does not map to one concrete tensor"
            )
        if matches[0] not in owner_indices:
            owner_indices.append(matches[0])
    if tuple(owner_indices) != required_outputs:
        raise LoweredContractError(
            "problem required outputs differ from output-allocation owners"
        )
    return LoweredRegion(
        region_id=result.region.id,
        normalized_graph_sha256=graph_sha256,
        region_inputs=region_inputs,
        region_outputs=region_outputs,
        output_allocation_owners=output_allocation_owners,
        tensors=tuple(tensors),
        operations=tuple(operations),
        required_outputs=required_outputs,
    )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LoweredContractError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise LoweredContractError(f"{field} must be an array")
    return value


def _string_sequence(value: Any, field: str) -> tuple[str, ...]:
    items = _sequence(value, field)
    if not all(isinstance(item, str) for item in items):
        raise LoweredContractError(f"{field} must contain only strings")
    return tuple(items)


def _indices(value: Any, bound: int, field: str) -> tuple[int, ...]:
    result: list[int] = []
    for item in _sequence(value, field):
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < bound:
            raise LoweredContractError(
                f"{field} contains invalid tensor index {item!r}"
            )
        result.append(item)
    return tuple(result)


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LoweredContractError(f"{field} must be a positive integer")
    return value
