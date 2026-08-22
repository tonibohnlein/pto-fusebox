"""Versioned, deterministic graph IR used at the Torch/Fusebox boundary."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

NORMALIZED_GRAPH_SCHEMA = "pto_fusebox.normalized_graph.v1"
PROBLEM_SCHEMA = "pto_fusebox.problem.v1"
SOLUTION_SCHEMA = "pto_fusebox.solution.v5"

_UNARY_NORMALIZED_OPS = {
    "abs",
    "cast",
    "exp",
    "log",
    "max",
    "neg",
    "rsqrt",
    "sqrt",
    "sum",
    "transpose_view",
    "view",
}
_BINARY_NORMALIZED_OPS = {"maximum", "minimum", "matmul"}
_SCALAR_BINARY_NORMALIZED_OPS = {"add", "div", "mul", "sub"}


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class ShapeDimension:
    """A symbolic dimension retained for diagnostics but not statically lowered."""

    expression: str
    minimum: int | None = None
    maximum: int | None = None

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "expression": self.expression,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> ShapeDimension:
        return cls(
            expression=str(value["expression"]),
            minimum=_optional_int(value.get("minimum")),
            maximum=_optional_int(value.get("maximum")),
        )


Dimension = int | ShapeDimension


@dataclass(frozen=True)
class NormalizedValue:
    id: str
    name: str
    shape: tuple[Dimension, ...]
    dtype: str
    role: str
    strides: tuple[Dimension, ...] | None = None
    storage_offset: Dimension | None = None
    producer: str | None = None
    target: str | None = None
    alias_of: str | None = None

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "name": self.name,
            "shape": [_dimension_to_json(dim) for dim in self.shape],
            "dtype": self.dtype,
            "role": self.role,
            "strides": (
                None
                if self.strides is None
                else [_dimension_to_json(stride) for stride in self.strides]
            ),
            "storage_offset": (
                None
                if self.storage_offset is None
                else _dimension_to_json(self.storage_offset)
            ),
            "producer": self.producer,
            "target": self.target,
            "alias_of": self.alias_of,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> NormalizedValue:
        return cls(
            id=str(value["id"]),
            name=str(value["name"]),
            shape=tuple(_dimension_from_json(dim) for dim in value["shape"]),
            dtype=str(value["dtype"]),
            role=str(value["role"]),
            strides=(
                None
                if value.get("strides") is None
                else tuple(_dimension_from_json(stride) for stride in value["strides"])
            ),
            storage_offset=(
                None
                if value.get("storage_offset") is None
                else _dimension_from_json(value["storage_offset"])
            ),
            producer=_optional_str(value.get("producer")),
            target=_optional_str(value.get("target")),
            alias_of=_optional_str(value.get("alias_of")),
        )


@dataclass(frozen=True)
class NormalizedOp:
    id: str
    kind: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: Mapping[str, JsonValue] = field(default_factory=dict)
    supported: bool = True
    opaque_reason: str | None = None
    metadata_only: bool = False

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "kind": self.kind,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attributes": _canonical_json_value(dict(self.attributes)),
            "supported": self.supported,
            "opaque_reason": self.opaque_reason,
            "metadata_only": self.metadata_only,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> NormalizedOp:
        attributes = value.get("attributes", {})
        if not isinstance(attributes, Mapping):
            raise ValueError("normalized op attributes must be an object")
        return cls(
            id=str(value["id"]),
            kind=str(value["kind"]),
            inputs=tuple(str(item) for item in value["inputs"]),
            outputs=tuple(str(item) for item in value["outputs"]),
            attributes=dict(attributes),
            supported=bool(value.get("supported", True)),
            opaque_reason=_optional_str(value.get("opaque_reason")),
            metadata_only=bool(value.get("metadata_only", False)),
        )


@dataclass(frozen=True)
class GraphPatternBinding:
    """One semantically named value supplied by a generated algorithm."""

    op: str
    value: str

    def to_json(self) -> dict[str, JsonValue]:
        return {"op": self.op, "value": self.value}

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> GraphPatternBinding:
        return cls(op=str(value["op"]), value=str(value["value"]))


@dataclass(frozen=True)
class GraphPattern:
    kind: str
    ops: tuple[str, ...]
    apply_substitutions: tuple[str, ...] = ()
    apply_bindings: tuple[GraphPatternBinding, ...] = ()

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind,
            "ops": list(self.ops),
            "apply_substitutions": list(self.apply_substitutions),
            "apply_bindings": [binding.to_json() for binding in self.apply_bindings],
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> GraphPattern:
        return cls(
            kind=str(value["kind"]),
            ops=tuple(str(item) for item in value["ops"]),
            apply_substitutions=tuple(
                str(item) for item in value.get("apply_substitutions", [])
            ),
            apply_bindings=tuple(
                GraphPatternBinding.from_json(item)
                for item in value.get("apply_bindings", [])
            ),
        )


@dataclass(frozen=True)
class NormalizedGraph:
    values: tuple[NormalizedValue, ...]
    ops: tuple[NormalizedOp, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    patterns: tuple[GraphPattern, ...] = ()
    constraints: Mapping[str, Mapping[str, int | None]] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()
    input_tree_spec: JsonValue = None
    output_tree_spec: JsonValue = None
    output_specs: tuple[Mapping[str, JsonValue], ...] = ()
    schema_version: str = NORMALIZED_GRAPH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != NORMALIZED_GRAPH_SCHEMA:
            raise ValueError(
                f"unsupported normalized graph schema {self.schema_version!r}; "
                f"expected {NORMALIZED_GRAPH_SCHEMA!r}"
            )
        _validate_graph(self)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "values": [value.to_json() for value in self.values],
            "ops": [op.to_json() for op in self.ops],
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "patterns": [pattern.to_json() for pattern in self.patterns],
            "constraints": _canonical_json_value(dict(self.constraints)),
            "diagnostics": list(self.diagnostics),
            "input_tree_spec": _canonical_json_value(self.input_tree_spec),
            "output_tree_spec": _canonical_json_value(self.output_tree_spec),
            "output_specs": [
                _canonical_json_value(dict(spec)) for spec in self.output_specs
            ],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        separators = (",", ":") if indent is None else None
        return json.dumps(
            self.to_dict(), sort_keys=True, indent=indent, separators=separators
        ) + ("\n" if indent is not None else "")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NormalizedGraph:
        schema = value.get("schema_version")
        if schema != NORMALIZED_GRAPH_SCHEMA:
            raise ValueError(
                f"unsupported normalized graph schema {schema!r}; expected {NORMALIZED_GRAPH_SCHEMA!r}"
            )
        constraints = value.get("constraints", {})
        if not isinstance(constraints, Mapping):
            raise ValueError("normalized graph constraints must be an object")
        return cls(
            values=tuple(NormalizedValue.from_json(item) for item in value["values"]),
            ops=tuple(NormalizedOp.from_json(item) for item in value["ops"]),
            inputs=tuple(str(item) for item in value["inputs"]),
            outputs=tuple(str(item) for item in value["outputs"]),
            patterns=tuple(
                GraphPattern.from_json(item) for item in value.get("patterns", [])
            ),
            constraints={str(key): dict(item) for key, item in constraints.items()},
            diagnostics=tuple(str(item) for item in value.get("diagnostics", [])),
            input_tree_spec=_canonical_json_value(value.get("input_tree_spec")),
            output_tree_spec=_canonical_json_value(value.get("output_tree_spec")),
            output_specs=tuple(dict(spec) for spec in value.get("output_specs", [])),
            schema_version=str(schema),
        )

    @classmethod
    def from_json(cls, text: str) -> NormalizedGraph:
        value = json.loads(text)
        if not isinstance(value, Mapping):
            raise ValueError("normalized graph JSON must contain an object")
        return cls.from_dict(value)

    def value_map(self) -> dict[str, NormalizedValue]:
        return {value.id: value for value in self.values}

    def op_map(self) -> dict[str, NormalizedOp]:
        return {op.id: op for op in self.ops}


def normalized_graph_sha256(graph: NormalizedGraph) -> str:
    """Return the deterministic identity of one complete normalized graph."""

    payload = graph.to_json(indent=None).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_graph(graph: NormalizedGraph) -> None:
    value_ids = [value.id for value in graph.values]
    op_ids = [op.id for op in graph.ops]
    if len(value_ids) != len(set(value_ids)):
        raise ValueError("normalized graph contains duplicate value ids")
    if len(op_ids) != len(set(op_ids)):
        raise ValueError("normalized graph contains duplicate op ids")
    known_values, known_ops = set(value_ids), set(op_ids)
    _validate_values(graph, known_values, known_ops)
    _validate_ops(graph, known_values)
    _validate_interface(graph, known_values)
    _validate_patterns(graph, known_ops)
    _validate_output_specs(graph)


def _validate_values(
    graph: NormalizedGraph, known_values: set[str], known_ops: set[str]
) -> None:
    op_outputs = {op.id: set(op.outputs) for op in graph.ops}
    for value in graph.values:
        if any(isinstance(dim, int) and dim <= 0 for dim in value.shape):
            raise ValueError(f"value {value.id} has a non-positive shape dimension")
        if value.strides is not None and len(value.strides) != len(value.shape):
            raise ValueError(
                f"value {value.id} has rank {len(value.shape)} but {len(value.strides)} strides"
            )
        if value.producer is not None and value.producer not in known_ops:
            raise ValueError(
                f"value {value.id} names unknown producer {value.producer}"
            )
        if value.producer is not None and value.id not in op_outputs[value.producer]:
            raise ValueError(
                f"value {value.id} is absent from producer {value.producer} outputs"
            )
        if value.alias_of is not None and value.alias_of not in known_values:
            raise ValueError(f"value {value.id} aliases unknown value {value.alias_of}")
    _validate_aliases(graph)


def _validate_ops(graph: NormalizedGraph, known_values: set[str]) -> None:
    value_map = graph.value_map()
    op_index = {op.id: index for index, op in enumerate(graph.ops)}
    produced_values: set[str] = set()
    for op in graph.ops:
        _canonical_json_value(dict(op.attributes))
        missing = (set(op.inputs) | set(op.outputs)) - known_values
        if missing:
            raise ValueError(f"op {op.id} references unknown values {sorted(missing)}")
        if op.supported and op.opaque_reason is not None:
            raise ValueError(f"supported op {op.id} cannot carry an opaque reason")
        if not op.supported and not op.opaque_reason:
            raise ValueError(f"unsupported op {op.id} must carry an opaque reason")
        _validate_known_op_arity(op)
        for value_id in op.outputs:
            if value_id in produced_values:
                raise ValueError(f"value {value_id} is produced by more than one op")
            produced_values.add(value_id)
            if value_map[value_id].producer != op.id:
                raise ValueError(
                    f"value {value_id} does not name its producing op {op.id}"
                )
        for value_id in op.inputs:
            producer = value_map[value_id].producer
            if producer is not None and op_index[producer] >= op_index[op.id]:
                raise ValueError(
                    f"op {op.id} is not topologically ordered after {producer}"
                )


def _validate_interface(graph: NormalizedGraph, known_values: set[str]) -> None:
    for value_id in (*graph.inputs, *graph.outputs):
        if value_id not in known_values:
            raise ValueError(f"graph interface references unknown value {value_id}")
    for value_id in graph.inputs:
        if graph.value_map()[value_id].producer is not None:
            raise ValueError(f"graph input {value_id} cannot have a producer")
    graph_inputs = set(graph.inputs)
    for value in graph.values:
        if value.producer is None and value.id not in graph_inputs:
            raise ValueError(
                f"producerless value {value.id} must be declared as a graph input"
            )


def _validate_known_op_arity(op: NormalizedOp) -> None:
    if not op.supported:
        return
    if op.kind in _UNARY_NORMALIZED_OPS:
        expected_inputs = 1
    elif op.kind in _BINARY_NORMALIZED_OPS:
        expected_inputs = 2
    elif op.kind in _SCALAR_BINARY_NORMALIZED_OPS:
        scalars = op.attributes.get("scalars", [])
        if not isinstance(scalars, list):
            raise ValueError(f"op {op.id} scalar operands must be a list")
        positions: set[int] = set()
        for scalar in scalars:
            if not isinstance(scalar, Mapping):
                raise ValueError(f"op {op.id} has a malformed scalar operand")
            position = scalar.get("position")
            if type(position) is not int or position not in {0, 1}:
                raise ValueError(f"op {op.id} has an invalid scalar operand position")
            if position in positions:
                raise ValueError(f"op {op.id} has duplicate scalar operand positions")
            positions.add(position)
        expected_inputs = 2 - len(scalars)
        if expected_inputs < 1:
            raise ValueError(f"op {op.id} must have at least one tensor input")
    else:
        return
    if len(op.inputs) != expected_inputs or len(op.outputs) != 1:
        raise ValueError(
            f"op {op.id} ({op.kind}) expects {expected_inputs} tensor input(s) "
            f"and one output, got {len(op.inputs)} input(s) and "
            f"{len(op.outputs)} output(s)"
        )


def _validate_patterns(graph: NormalizedGraph, known_ops: set[str]) -> None:
    for pattern in graph.patterns:
        binding_ops = {binding.op for binding in pattern.apply_bindings}
        missing = (
            set(pattern.ops) | set(pattern.apply_substitutions) | binding_ops
        ) - known_ops
        if missing:
            raise ValueError(
                f"pattern {pattern.kind} references unknown ops {sorted(missing)}"
            )
        if not set(pattern.apply_substitutions).issubset(pattern.ops):
            raise ValueError(
                f"pattern {pattern.kind} substitutions must belong to its op set"
            )
        if not binding_ops.issubset(pattern.ops):
            raise ValueError(
                f"pattern {pattern.kind} bindings must belong to its op set"
            )
        if len(binding_ops) != len(pattern.apply_bindings):
            raise ValueError(f"pattern {pattern.kind} has duplicate binding operations")
        if binding_ops and binding_ops != set(pattern.apply_substitutions):
            raise ValueError(
                f"pattern {pattern.kind} named bindings must cover its substitutions"
            )


def _validate_output_specs(graph: NormalizedGraph) -> None:
    if len(graph.output_specs) != len(graph.outputs):
        raise ValueError("normalized graph output specs must match ordered outputs")
    for index, spec in enumerate(graph.output_specs):
        if spec.get("value") != graph.outputs[index]:
            raise ValueError(
                f"output spec {index} does not match normalized output ordering"
            )


def _validate_aliases(graph: NormalizedGraph) -> None:
    values = graph.value_map()
    for value in graph.values:
        visited: set[str] = set()
        current = value.id
        while values[current].alias_of is not None:
            if current in visited:
                raise ValueError(
                    f"normalized graph contains an alias cycle through {value.id}"
                )
            visited.add(current)
            alias = values[current].alias_of
            if alias is None:
                break
            current = alias


def _dimension_to_json(value: Dimension) -> JsonValue:
    return value if isinstance(value, int) else value.to_json()


def _dimension_from_json(value: Any) -> Dimension:
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid shape dimension")
    if isinstance(value, int):
        return value
    if isinstance(value, Mapping):
        return ShapeDimension.from_json(value)
    raise ValueError(f"invalid shape dimension {value!r}")


def _canonical_json_value(value: Any) -> JsonValue:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("normalized graph JSON cannot contain a non-finite float")
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _canonical_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_json_value(item) for item in value]
    raise TypeError(f"value {value!r} is not JSON serializable")


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
