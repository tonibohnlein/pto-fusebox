"""Conservative convex-region extraction and solver problem lowering."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .ir import PROBLEM_SCHEMA, NormalizedGraph, NormalizedOp, NormalizedValue, ShapeDimension
from .target import TargetProfile, resolve_target


@dataclass(frozen=True)
class LoweredProblem:
    region_id: str
    problem: Mapping[str, Any]
    solver_op_to_graph: tuple[str, ...]
    solver_tensor_to_value: tuple[str, ...]


@dataclass(frozen=True)
class SolverRegion:
    id: str
    op_ids: tuple[str, ...]
    input_values: tuple[str, ...]
    output_values: tuple[str, ...]
    target: str
    diagnostics: tuple[str, ...] = ()

    def lower(self, graph: NormalizedGraph, target: str | TargetProfile | None = None) -> LoweredProblem:
        return lower_solver_region(graph, self, resolve_target(target or self.target))


def extract_solver_regions(  # noqa: PLR0912 -- each branch is one explicit region contract.
    graph: NormalizedGraph, target: str | TargetProfile = "ascend910b"
) -> list[SolverRegion]:
    """Return deterministic maximal convex supported regions.

    Unsupported and target-inadmissible nodes remain in ``graph`` and act as
    mandatory boundaries. A supported edge cannot reconnect nodes across an
    alternate path containing such a boundary: nodes carry the complete set of
    opaque ancestors, and only equal signatures may join one region.
    """

    profile = resolve_target(target)
    values = graph.value_map()
    op_by_id = graph.op_map()
    op_index = {op.id: index for index, op in enumerate(graph.ops)}
    admitted: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    for op in graph.ops:
        reason = profile.admission_reason(op, values)
        admitted[op.id] = reason is None
        if reason is not None:
            reasons[op.id] = reason

    signatures: dict[str, frozenset[str]] = {}
    for op in graph.ops:
        ancestors: set[str] = set()
        for value_id in op.inputs:
            producer = values[value_id].producer
            if producer is None:
                continue
            ancestors.update(signatures.get(producer, frozenset()))
            if not admitted.get(producer, False):
                ancestors.add(producer)
        signatures[op.id] = frozenset(ancestors)

    parent = {op.id: op.id for op in graph.ops if admitted[op.id]}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(first: str, second: str) -> None:
        left, right = find(first), find(second)
        if left == right:
            return
        if op_index[left] > op_index[right]:
            left, right = right, left
        parent[right] = left

    for op in graph.ops:
        if not admitted[op.id]:
            continue
        for value_id in op.inputs:
            producer = values[value_id].producer
            if (
                producer is not None
                and admitted.get(producer, False)
                and signatures[producer] == signatures[op.id]
            ):
                union(producer, op.id)

    components: dict[str, list[str]] = {}
    for op in graph.ops:
        if admitted[op.id]:
            components.setdefault(find(op.id), []).append(op.id)

    consumers = _consumers(graph)
    graph_outputs = set(graph.outputs)
    regions: list[SolverRegion] = []
    sorted_components = sorted(components.values(), key=lambda ids: min(op_index[item] for item in ids))
    for ids in sorted_components:
        region_ops = set(ids)
        if not any(not op_by_id[op_id].metadata_only for op_id in ids):
            continue
        inputs: list[str] = []
        outputs: list[str] = []
        for op_id in ids:
            op = op_by_id[op_id]
            for value_id in op.inputs:
                producer = values[value_id].producer
                if producer not in region_ops and value_id not in inputs:
                    inputs.append(value_id)
            for value_id in op.outputs:
                outside = any(consumer not in region_ops for consumer in consumers[value_id])
                if (value_id in graph_outputs or outside) and value_id not in outputs:
                    outputs.append(value_id)
        diagnostics = tuple(
            f"boundary {barrier}: {reasons[barrier]}"
            for barrier in sorted(signatures[ids[0]], key=lambda item: op_index[item])
            if barrier in reasons
        )
        regions.append(
            SolverRegion(
                id=f"region{len(regions):04d}",
                op_ids=tuple(ids),
                input_values=tuple(inputs),
                output_values=tuple(outputs),
                target=profile.name,
                diagnostics=diagnostics,
            )
        )
    return regions


def lower_solver_region(
    graph: NormalizedGraph, region: SolverRegion, target: TargetProfile
) -> LoweredProblem:
    graph_values = graph.value_map()
    ops = graph.op_map()
    graph_compute_ops = [ops[op_id] for op_id in region.op_ids if not ops[op_id].metadata_only]
    if not graph_compute_ops:
        raise ValueError(f"{region.id} contains no solver operations")
    canonical_graph_ops = [_canonicalize_metadata_inputs(op, graph_values, ops) for op in graph_compute_ops]
    compute_ops, values, solver_op_to_graph = _expand_native_casts(canonical_graph_ops, graph_values, target)
    if not compute_ops:
        raise ValueError(f"{region.id} contains no operations after target lowering")

    ordered_value_ids: list[str] = []
    for op in compute_ops:
        for value_id in (*op.inputs, *op.outputs):
            if value_id not in ordered_value_ids:
                ordered_value_ids.append(value_id)
    ordered_values = [values[value_id] for value_id in ordered_value_ids]
    static_shapes: dict[str, tuple[int, int]] = {}
    for value in ordered_values:
        static_shapes[value.id] = _solver_shape(value)

    tensor_index = {value.id: index for index, value in enumerate(ordered_values)}
    required_value_ids = [_allocation_owner(value_id, values, ops) for value_id in region.output_values]
    graph_op_indices: dict[str, list[int]] = {}
    for index, graph_op in enumerate(solver_op_to_graph):
        graph_op_indices.setdefault(graph_op, []).append(index)
    problem: dict[str, Any] = dict(target.problem_fields())
    problem.update(
        {
            "schema_version": PROBLEM_SCHEMA,
            "widths": [static_shapes[value.id][0] for value in ordered_values],
            "heights": [static_shapes[value.id][1] for value in ordered_values],
            "dtypes": [_solver_dtype(value.dtype) for value in ordered_values],
            "inputs": [[tensor_index[item] for item in op.inputs] for op in compute_ops],
            "outputs": [[tensor_index[op.outputs[0]]] for op in compute_ops],
            "op_types": [_solver_op_type(op) for op in compute_ops],
            "vec_slopes": [_vector_cost(op)[0] for op in compute_ops],
            "vec_fixed_costs": [_vector_cost(op)[1] for op in compute_ops],
            "vector_primitive_families": [_vector_primitive(op, values, static_shapes) for op in compute_ops],
            "vector_op_geometries": [_vector_geometry(op, values, static_shapes) for op in compute_ops],
            "vector_op_capabilities": [_vector_capability(op) for op in compute_ops],
            "mixed_vector_semantics": [_mixed_semantic(op) for op in compute_ops],
            "mixed_emit_compatible": [True for _ in compute_ops],
            "required_outputs": [
                tensor_index[value_id]
                for value_id in dict.fromkeys(required_value_ids)
                if value_id in tensor_index
            ],
            "p4_patterns": [],
            "frontend_mapping": {
                "region_id": region.id,
                "solver_op_to_graph": list(solver_op_to_graph),
                "solver_tensor_to_value": [value.id for value in ordered_values],
                "solver_tensor_alias_of": [value.alias_of for value in ordered_values],
                "solver_tensor_synthetic": [value.id not in graph_values for value in ordered_values],
                "region_inputs": list(region.input_values),
                "region_outputs": list(region.output_values),
                "region_output_allocation_owners": required_value_ids,
            },
        }
    )
    region_ops = set(region.op_ids)
    for pattern in graph.patterns:
        if set(pattern.ops).issubset(region_ops) and all(item in graph_op_indices for item in pattern.ops):
            problem["p4_patterns"].append(
                {
                    "kind": pattern.kind,
                    "ops": [index for item in pattern.ops for index in graph_op_indices[item]],
                    "apply_substitutions": [
                        index for item in pattern.apply_substitutions for index in graph_op_indices[item]
                    ],
                }
            )
    return LoweredProblem(
        region_id=region.id,
        problem=problem,
        solver_op_to_graph=solver_op_to_graph,
        solver_tensor_to_value=tuple(value.id for value in ordered_values),
    )


def _allocation_owner(
    value_id: str,
    values: Mapping[str, NormalizedValue],
    ops: Mapping[str, NormalizedOp],
) -> str:
    """Resolve a metadata alias to the value whose allocation carries it."""

    visited: set[str] = set()
    current = value_id
    while current not in visited:
        visited.add(current)
        value = values[current]
        if value.alias_of is not None:
            current = value.alias_of
            continue
        if value.producer is None:
            return current
        producer = ops[value.producer]
        if not producer.metadata_only or len(producer.inputs) != 1:
            return current
        current = producer.inputs[0]
    raise ValueError(f"metadata alias cycle reaches {value_id}")


def _canonicalize_metadata_inputs(
    op: NormalizedOp,
    values: Mapping[str, NormalizedValue],
    ops: Mapping[str, NormalizedOp],
) -> NormalizedOp:
    inputs: list[str] = []
    for value_id in op.inputs:
        value = values[value_id]
        if (
            value.producer is not None
            and value.producer in ops
            and ops[value.producer].kind == "transpose_view"
        ):
            inputs.append(value_id)
        else:
            inputs.append(_allocation_owner(value_id, values, ops))
    if tuple(inputs) == op.inputs:
        return op
    return NormalizedOp(
        id=op.id,
        kind=op.kind,
        inputs=tuple(inputs),
        outputs=op.outputs,
        attributes=op.attributes,
        supported=op.supported,
        opaque_reason=op.opaque_reason,
        metadata_only=op.metadata_only,
    )


def _expand_native_casts(
    graph_ops: list[NormalizedOp],
    graph_values: Mapping[str, NormalizedValue],
    target: TargetProfile,
) -> tuple[list[NormalizedOp], dict[str, NormalizedValue], tuple[str, ...]]:
    values = dict(graph_values)
    expanded: list[NormalizedOp] = []
    graph_mapping: list[str] = []
    for op in graph_ops:
        if op.kind != "cast":
            expanded.append(op)
            graph_mapping.append(op.id)
            continue
        source = values[op.inputs[0]]
        output = values[op.outputs[0]]
        path = target.native_cast_path(source.dtype, output.dtype)
        if path is None or not path:
            if source.dtype == output.dtype:
                continue
            raise ValueError(f"{op.id} has no native cast path from {source.dtype} to {output.dtype}")
        previous = source.id
        for hop, dtype in enumerate(path):
            final = hop + 1 == len(path)
            output_id = output.id if final else f"{op.id}.native_value{hop:02d}"
            native_op_id = f"{op.id}.native{hop:02d}"
            if not final:
                values[output_id] = NormalizedValue(
                    id=output_id,
                    name=f"{output.name}_native_{dtype}",
                    shape=output.shape,
                    dtype=dtype,
                    role="intermediate",
                    strides=output.strides,
                    storage_offset=0,
                    producer=native_op_id,
                )
            expanded.append(
                NormalizedOp(
                    id=native_op_id,
                    kind="cast",
                    inputs=(previous,),
                    outputs=(output_id,),
                    attributes={
                        "dtype": dtype,
                        "native_cast_hop": hop,
                        "normalized_op": op.id,
                    },
                )
            )
            graph_mapping.append(op.id)
            previous = output_id
    return expanded, values, tuple(graph_mapping)


def _consumers(graph: NormalizedGraph) -> dict[str, list[str]]:
    result = {value.id: [] for value in graph.values}
    for op in graph.ops:
        for value_id in op.inputs:
            result[value_id].append(op.id)
    return result


def _solver_shape(value: NormalizedValue) -> tuple[int, int]:
    expressions = [dim.expression for dim in value.shape if isinstance(dim, ShapeDimension)]
    if expressions:
        raise ValueError(
            f"value {value.id} has schedule-defining symbolic dimensions {expressions}; "
            "static solver lowering is intentionally deferred"
        )
    dims: list[int] = []
    for dim in value.shape:
        if not isinstance(dim, int):
            raise AssertionError("symbolic dimensions must decline before static lowering")
        dims.append(dim)
    if not dims:
        return (1, 1)
    if len(dims) == 1:
        return (dims[0], 1)
    return (dims[-1], math.prod(dims[:-1]))


def _solver_dtype(dtype: str) -> str:
    names = {
        "float32": "FP32",
        "float16": "FP16",
        "bfloat16": "BF16",
        "int32": "INT32",
        "int16": "INT16",
        "int8": "INT8",
        "bool": "BOOL",
    }
    if dtype not in names:
        raise ValueError(f"unsupported solver dtype {dtype!r}")
    return names[dtype]


def _solver_op_type(op: NormalizedOp) -> str:
    if op.kind == "matmul":
        return "MatMul"
    if op.kind in {"sum", "max"}:
        return "Reduction"
    return "Pointwise"


def _vector_cost(op: NormalizedOp) -> tuple[float, float]:
    return {
        "add": (2.0, 24.0),
        "sub": (2.0, 24.0),
        "maximum": (2.0, 24.0),
        "minimum": (2.0, 24.0),
        "mul": (2.0, 25.0),
        "div": (4.0, 30.0),
        "exp": (2.0, 31.0),
        "log": (2.0, 32.0),
        "abs": (1.0, 24.0),
        "sqrt": (1.0, 24.0),
        "rsqrt": (1.0, 24.0),
        "neg": (1.0, 24.0),
        "cast": (1.0, 24.0),
    }.get(op.kind, (0.0, 0.0))


def _vector_primitive(
    op: NormalizedOp,
    values: Mapping[str, NormalizedValue],
    shapes: Mapping[str, tuple[int, int]],
) -> str:
    if op.kind == "matmul":
        return "generic"
    if op.kind == "sum":
        return "row_sum"
    if op.kind == "max":
        return "row_extrema"
    if op.kind in {"add", "sub", "maximum", "minimum"}:
        scalar = bool(op.attributes.get("scalars"))
        if op.kind in {"add", "sub"} and scalar:
            return "scalar_add"
        if op.kind == "maximum" and scalar:
            return "scalar_max"
        if op.kind == "minimum" and scalar:
            return "scalar_min"
        return "add"
    if op.kind == "mul":
        return "scalar_mul" if op.attributes.get("scalars") else "mul"
    if op.kind == "neg":
        return "scalar_mul"
    return {
        "div": "div",
        "exp": "exp",
        "log": "log",
        "abs": "abs",
        "sqrt": "sqrt",
        "rsqrt": "rsqrt",
    }.get(op.kind, "generic")


def _vector_geometry(
    op: NormalizedOp,
    values: Mapping[str, NormalizedValue],
    shapes: Mapping[str, tuple[int, int]],
) -> str:
    if op.kind == "matmul" or not op.outputs:
        return "generic"
    output_width, output_height = shapes[op.outputs[0]]
    for value_id in op.inputs:
        width, height = shapes[value_id]
        if height == output_height and width == 1 and output_width > 1:
            return "row_expand"
        if width == output_width and height == 1 and output_height > 1:
            return "col_expand"
    return "flat"


def _vector_capability(op: NormalizedOp) -> str:
    if op.kind == "matmul":
        return "generic"
    if op.kind == "sum":
        return "reduction_sum"
    if op.kind == "max":
        return "reduction_max"
    return "elementwise"


def _mixed_semantic(op: NormalizedOp) -> str:
    if op.kind in {"neg", "exp", "mul", "cast"}:
        return op.kind
    if op.kind == "add" and _has_exact_scalar(op, value=1, positions={0, 1}):
        return "scalar_add"
    if op.kind == "div" and _has_exact_scalar(op, value=1, positions={0}):
        return "recip"
    return "none"


def _has_exact_scalar(op: NormalizedOp, *, value: int, positions: set[int]) -> bool:
    scalars = op.attributes.get("scalars")
    if not isinstance(scalars, list) or len(scalars) != 1:
        return False
    scalar = scalars[0]
    if not isinstance(scalar, Mapping):
        return False
    scalar_value = scalar.get("value")
    return (
        isinstance(scalar_value, (int, float))
        and not isinstance(scalar_value, bool)
        and scalar_value == value
        and scalar.get("position") in positions
    )
