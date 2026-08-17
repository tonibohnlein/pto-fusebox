"""Normalize ``torch.export`` graphs into the public PTO-Fusebox graph IR."""

from __future__ import annotations

import json
import math
import operator
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from .ir import (
    GraphPattern,
    GraphPatternBinding,
    NormalizedGraph,
    NormalizedOp,
    NormalizedValue,
    ShapeDimension,
)

_POINTWISE = {
    "aten.add.Tensor": "add",
    "aten.add.Scalar": "add",
    "aten.sub.Tensor": "sub",
    "aten.sub.Scalar": "sub",
    "aten.mul.Tensor": "mul",
    "aten.mul.Scalar": "mul",
    "aten.div.Tensor": "div",
    "aten.div.Scalar": "div",
    "aten.maximum.default": "maximum",
    "aten.minimum.default": "minimum",
    "aten.exp.default": "exp",
    "aten.log.default": "log",
    "aten.abs.default": "abs",
    "aten.sqrt.default": "sqrt",
    "aten.rsqrt.default": "rsqrt",
    "aten.neg.default": "neg",
}
_REDUCTIONS = {
    "aten.sum.dim_IntList": "sum",
    "aten.amax.default": "max",
    "aten.max.dim": "max",
}
_VIEWS = {
    "aten.view.default",
    "aten.squeeze.default",
    "aten.squeeze.dim",
    "aten.unsqueeze.default",
}
_COPY_CAPABLE_VIEWS = {"aten.reshape.default", "aten.contiguous.default"}
_TRANSPOSES = {
    "aten.t.default",
    "aten.transpose.int",
    "aten.permute.default",
}
_DROPPED_METADATA = {
    "aten.sym_size.int",
    "aten._assert_scalar.default",
    "aten._assert_tensor_metadata.default",
}


def export_and_normalize(
    module: Any,
    args: tuple[Any, ...],
    *,
    kwargs: Mapping[str, Any] | None = None,
    dynamic_shapes: Any = None,
    strict: bool = True,
) -> NormalizedGraph:
    """Export ``module`` with PyTorch and normalize the resulting program.

    ``torch.export.ExportedProgram`` is the authoritative input to the
    normalizer.  This helper is intentionally thin so capture policy stays in
    PyTorch while PTO-Fusebox owns graph semantics after export.
    """

    import torch  # noqa: PLC0415 -- Torch is an optional frontend dependency.

    program = torch.export.export(
        module,
        args,
        kwargs=dict(kwargs or {}),
        dynamic_shapes=dynamic_shapes,
        strict=strict,
    )
    return normalize_exported(program)


def normalize_exported(program: Any) -> NormalizedGraph:
    """Normalize a ``torch.export.ExportedProgram`` without embedding weights."""

    import torch  # noqa: PLC0415 -- Torch is an optional frontend dependency.

    if not isinstance(program, torch.export.ExportedProgram):
        raise TypeError("normalize_exported expects torch.export.ExportedProgram")
    return _ExportNormalizer(program).run()


class _ExportNormalizer:
    def __init__(self, program: Any) -> None:
        self.program = program
        self.values: list[NormalizedValue] = []
        self.ops: list[NormalizedOp] = []
        self.patterns: list[GraphPattern] = []
        self.diagnostics: list[str] = []
        self.node_values: dict[Any, tuple[str, ...]] = {}
        self.input_ids: list[str] = []
        self.output_ids: list[str] = []
        self._value_counter = 0
        self._op_counter = 0
        self._constraints, self._symbol_names = _range_constraints(program)
        self._input_specs = _input_spec_map(program)

    def run(self) -> NormalizedGraph:
        for node in self.program.graph_module.graph.nodes:
            if node.op == "placeholder":
                self._placeholder(node)
            elif node.op == "call_function":
                self._call_function(node)
            elif node.op == "get_attr":
                self._get_attr(node)
            elif node.op == "output":
                self.output_ids.extend(self._flatten_output_values(node.args))
            else:
                self._opaque(node, f"unsupported FX node kind {node.op!r}")

        self._mark_mutated_outputs()
        return NormalizedGraph(
            values=tuple(self.values),
            ops=tuple(self.ops),
            inputs=tuple(self.input_ids),
            outputs=tuple(self.output_ids),
            patterns=tuple(self.patterns),
            constraints=self._constraints,
            diagnostics=tuple(self.diagnostics),
            input_tree_spec=_serialize_tree_spec(self.program.call_spec.in_spec),
            output_tree_spec=_serialize_tree_spec(self.program.call_spec.out_spec),
            output_specs=_output_specs(self.program, self.output_ids),
        )

    def _placeholder(self, node: Any) -> None:
        role, target = self._input_specs.get(node.name, ("user_input", None))
        value_id = self._add_value(
            node.name, node.meta.get("val"), role=role, target=target
        )
        self.node_values[node] = (value_id,)
        self.input_ids.append(value_id)

    def _get_attr(self, node: Any) -> None:
        value_id = self._add_value(
            node.name, node.meta.get("val"), role="constant", target=str(node.target)
        )
        self.node_values[node] = (value_id,)
        self.input_ids.append(value_id)

    def _call_function(self, node: Any) -> None:
        target = _target_name(node.target)
        if target in _DROPPED_METADATA:
            self.node_values[node] = ()
            return
        if node.target is operator.getitem or target == "operator.getitem":
            self._getitem(node)
            return
        if target in _POINTWISE:
            self._pointwise(node, _POINTWISE[target], target)
            return
        if target == "aten.reciprocal.default":
            self._reciprocal(node)
            return
        if target in _REDUCTIONS:
            self._reduction(node, _REDUCTIONS[target], target)
            return
        if target == "aten.mean.dim":
            self._mean(node)
            return
        if target in {"aten.mm.default", "aten.matmul.default"}:
            self._matmul(node, target)
            return
        if target == "aten.linear.default":
            self._linear(node)
            return
        if target in {"aten.softmax.int", "aten._softmax.default"}:
            self._softmax(node)
            return
        if target in {"aten.to.dtype", "aten._to_copy.default"}:
            self._cast(node, target)
            return
        if target in _TRANSPOSES:
            self._view(node, target, transpose=True)
            return
        if target in _COPY_CAPABLE_VIEWS:
            self._opaque(
                node, f"{target} may copy storage and is not a metadata-only alias"
            )
            return
        if target in _VIEWS:
            self._view(node, target, transpose=False)
            return
        self._opaque(node, f"unsupported operator {target}")

    def _getitem(self, node: Any) -> None:
        if len(node.args) < 2 or node.args[0] not in self.node_values:
            self._opaque(node, "getitem source is not a normalized tuple value")
            return
        try:
            index = int(node.args[1])
            self.node_values[node] = (self.node_values[node.args[0]][index],)
        except (IndexError, TypeError, ValueError):
            self._opaque(node, "getitem index is not a static tuple index")

    def _pointwise(self, node: Any, kind: str, target: str) -> None:
        tensor_inputs = self._tensor_inputs(node.args)
        if not tensor_inputs:
            self._opaque(node, f"{target} has no tensor operand")
            return
        if node.kwargs:
            names = ", ".join(sorted(str(name) for name in node.kwargs))
            self._opaque(node, f"{target} has unsupported keyword arguments: {names}")
            return
        attrs: dict[str, Any] = {"source_operator": target}
        scalar_args: list[dict[str, Any]] = []
        for index, arg in enumerate(node.args):
            if not self._is_tensor_arg(arg) and _is_json_scalar(arg):
                scalar_args.append({"position": index, "value": arg})
        if scalar_args:
            attrs["scalars"] = scalar_args
        self._single_output_op(node, kind, tensor_inputs, attrs)

    def _reduction(self, node: Any, kind: str, target: str) -> None:
        tensor_inputs = self._tensor_inputs(node.args[:1])
        if len(tensor_inputs) != 1:
            self._opaque(node, f"{target} requires one tensor input")
            return
        if node.kwargs:
            names = ", ".join(sorted(str(name) for name in node.kwargs))
            self._opaque(node, f"{target} has unsupported keyword arguments: {names}")
            return
        input_shape = self._value(tensor_inputs[0]).shape
        dims = _normalize_dims(
            node.args[1] if len(node.args) > 1 else None, len(input_shape)
        )
        keepdim = (
            bool(node.args[2])
            if len(node.args) > 2
            else bool(node.kwargs.get("keepdim", False))
        )
        if dims != (len(input_shape) - 1,) or not keepdim:
            self._opaque(
                node, "only last-axis reductions with keepdim=True are supported"
            )
            return
        if (
            target == "aten.max.dim"
            and _metadata_output_count(node.meta.get("val")) != 2
        ):
            self._opaque(node, "unexpected aten.max.dim result signature")
            return
        self._single_output_op(
            node,
            kind,
            tensor_inputs,
            {"axis": -1, "keepdim": True, "source_operator": target},
            output_index=0,
        )
        if target == "aten.max.dim":
            # The index result is deliberately opaque. A consumer of it creates
            # a hard region boundary rather than silently dropping semantics.
            value_ids = list(self.node_values[node])
            index_meta = _metadata_outputs(node.meta.get("val"))[1]
            index_id = self._add_value(
                f"{node.name}_indices", index_meta, role="intermediate"
            )
            opaque_id = self._add_op(
                "opaque",
                tensor_inputs,
                (index_id,),
                {"source_operator": target, "result": "indices"},
                supported=False,
                opaque_reason="argmax indices are data-dependent",
            )
            self._set_producer(index_id, opaque_id)
            value_ids.append(index_id)
            self.node_values[node] = tuple(value_ids)

    def _reciprocal(self, node: Any) -> None:
        tensor_inputs = self._tensor_inputs(node.args[:1])
        if len(tensor_inputs) != 1:
            self._opaque(node, "aten.reciprocal.default requires one tensor input")
            return
        self._single_output_op(
            node,
            "div",
            tensor_inputs,
            {
                "source_operator": "aten.reciprocal.default",
                "scalars": [{"position": 0, "value": 1}],
            },
        )

    def _mean(self, node: Any) -> None:
        tensor_inputs = self._tensor_inputs(node.args[:1])
        if len(tensor_inputs) != 1:
            self._opaque(node, "aten.mean.dim requires one tensor input")
            return
        if node.kwargs:
            names = ", ".join(sorted(str(name) for name in node.kwargs))
            self._opaque(
                node, f"aten.mean.dim has unsupported keyword arguments: {names}"
            )
            return
        source = self._value(tensor_inputs[0])
        dims = _normalize_dims(
            node.args[1] if len(node.args) > 1 else None, len(source.shape)
        )
        keepdim = (
            bool(node.args[2])
            if len(node.args) > 2
            else bool(node.kwargs.get("keepdim", False))
        )
        if dims != (len(source.shape) - 1,) or not keepdim:
            self._opaque(node, "only last-axis mean with keepdim=True is supported")
            return
        extent = source.shape[-1]
        if not isinstance(extent, int) or extent <= 0:
            self._opaque(node, "mean reduction extent must be statically known")
            return
        result_meta = _metadata_outputs(node.meta.get("val"))[0]
        reduced_id = self._add_value(
            f"{node.name}_sum", result_meta, role="intermediate"
        )
        sum_op = self._add_op(
            "sum",
            tensor_inputs,
            (reduced_id,),
            {"axis": -1, "keepdim": True, "source_operator": "aten.mean.dim"},
        )
        self._set_producer(reduced_id, sum_op)
        output_id = self._add_value(node.name, result_meta, role="intermediate")
        mul_op = self._add_op(
            "mul",
            (reduced_id,),
            (output_id,),
            {
                "scalars": [{"position": 1, "value": 1.0 / extent}],
                "lowered_from": "mean",
            },
        )
        self._set_producer(output_id, mul_op)
        self.node_values[node] = (output_id,)

    def _matmul(self, node: Any, target: str) -> None:
        tensor_inputs = self._tensor_inputs(node.args[:2])
        if len(tensor_inputs) != 2:
            self._opaque(node, f"{target} requires two tensor inputs")
            return
        lhs = self._value(tensor_inputs[0])
        rhs = self._value(tensor_inputs[1])
        if len(lhs.shape) != 2 or len(rhs.shape) != 2:
            self._opaque(node, "only rank-2 matmul/mm is supported")
            return
        attributes = {
            "source_operator": target,
            "lhs_transposed": self._is_transpose_value(lhs.id),
            "rhs_transposed": self._is_transpose_value(rhs.id),
        }
        self._single_output_op(node, "matmul", tensor_inputs, attributes)

    def _linear(self, node: Any) -> None:
        tensor_inputs = self._tensor_inputs(node.args[:3])
        if len(tensor_inputs) < 2:
            self._opaque(node, "aten.linear requires activation and weight tensors")
            return
        activation = self._value(tensor_inputs[0])
        weight = self._value(tensor_inputs[1])
        if len(activation.shape) < 2 or len(weight.shape) != 2:
            self._opaque(node, "linear supports rank>=2 activation and rank-2 weight")
            return
        result_meta = _metadata_outputs(node.meta.get("val"))[0]
        result_shape = _shape_from_meta(
            result_meta, self._constraints, self._symbol_names
        )
        if result_shape is None:
            self._opaque(node, "linear output has no tensor metadata")
            return
        # Torch stores linear weights as [N,K]. The normalized matmul consumes a
        # zero-copy [K,N] view and retains the storage orientation as an exact
        # semantic attribute for the eventual PyPTO source backend.
        transposed_weight = self._add_value_from_shape(
            f"{node.name}_weight_t",
            tuple(reversed(weight.shape)),
            weight.dtype,
            role="alias",
            alias_of=weight.id,
            strides=None if weight.strides is None else tuple(reversed(weight.strides)),
            storage_offset=weight.storage_offset,
        )
        view_op = self._add_op(
            "transpose_view",
            (weight.id,),
            (transposed_weight,),
            {"permutation": [1, 0], "source_operator": "aten.linear.default"},
            metadata_only=True,
        )
        self._set_producer(transposed_weight, view_op)
        matmul_id = self._add_value_from_shape(
            f"{node.name}_matmul",
            result_shape,
            _dtype_from_meta(result_meta),
            role="intermediate",
        )
        matmul_op = self._add_op(
            "matmul",
            (activation.id, transposed_weight),
            (matmul_id,),
            {
                "rhs_transposed": True,
                "flatten_leading_dimensions": len(activation.shape) > 2,
                "source_operator": "aten.linear.default",
            },
        )
        self._set_producer(matmul_id, matmul_op)
        if len(tensor_inputs) == 2:
            self.node_values[node] = (matmul_id,)
            return
        output_id = self._add_value_from_shape(
            node.name, result_shape, _dtype_from_meta(result_meta), role="intermediate"
        )
        add_op = self._add_op(
            "add",
            (matmul_id, tensor_inputs[2]),
            (output_id,),
            {"source_operator": "aten.linear.default", "linear_bias": True},
        )
        self._set_producer(output_id, add_op)
        self.node_values[node] = (output_id,)

    def _softmax(self, node: Any) -> None:
        tensor_inputs = self._tensor_inputs(node.args[:1])
        if len(tensor_inputs) != 1:
            self._opaque(node, "softmax requires one tensor input")
            return
        source = self._value(tensor_inputs[0])
        dim = (
            int(node.args[1]) if len(node.args) > 1 else int(node.kwargs.get("dim", -1))
        )
        if dim < 0:
            dim += len(source.shape)
        if dim != len(source.shape) - 1:
            self._opaque(node, "only last-axis softmax is supported")
            return
        if not source.shape:
            self._opaque(node, "softmax scalar input is unsupported")
            return
        output_meta = _metadata_outputs(node.meta.get("val"))[0]
        output_dtype = _dtype_from_meta(output_meta)
        half_to_float = len(node.args) > 2 and bool(node.args[2])
        if half_to_float or output_dtype != source.dtype:
            self._opaque(
                node,
                "dtype-changing softmax must be represented as explicit casts",
            )
            return
        reduced_shape = (*source.shape[:-1], 1)
        max_id = self._add_value_from_shape(
            f"{node.name}_max", reduced_shape, source.dtype, role="intermediate"
        )
        sub_id = self._add_value_from_shape(
            f"{node.name}_shift", source.shape, source.dtype, role="intermediate"
        )
        exp_id = self._add_value_from_shape(
            f"{node.name}_exp", source.shape, source.dtype, role="intermediate"
        )
        sum_id = self._add_value_from_shape(
            f"{node.name}_sum", reduced_shape, source.dtype, role="intermediate"
        )
        output_id = self._add_value(node.name, output_meta, role="intermediate")
        op_ids = (
            self._add_op("max", (source.id,), (max_id,), {"axis": -1, "keepdim": True}),
            self._add_op("sub", (source.id, max_id), (sub_id,), {}),
            self._add_op("exp", (sub_id,), (exp_id,), {}),
            self._add_op("sum", (exp_id,), (sum_id,), {"axis": -1, "keepdim": True}),
            self._add_op("div", (exp_id, sum_id), (output_id,), {}),
        )
        for value_id, op_id in zip((max_id, sub_id, exp_id, sum_id, output_id), op_ids):
            self._set_producer(value_id, op_id)
        self.patterns.append(
            GraphPattern(
                kind="softmax_flash",
                ops=op_ids,
                apply_substitutions=(op_ids[0], op_ids[3]),
                apply_bindings=(
                    GraphPatternBinding(op=op_ids[0], value="running_max"),
                    GraphPatternBinding(op=op_ids[3], value="running_sum"),
                ),
            )
        )
        self.node_values[node] = (output_id,)

    def _cast(self, node: Any, target: str) -> None:
        tensor_inputs = self._tensor_inputs(node.args[:1])
        if len(tensor_inputs) != 1:
            self._opaque(node, f"{target} requires one tensor input")
            return
        source = self._value(tensor_inputs[0])
        output_meta = _metadata_outputs(node.meta.get("val"))[0]
        destination_dtype = _dtype_from_meta(output_meta)
        copy_requested = target == "aten._to_copy.default" or (
            len(node.args) > 3 and bool(node.args[3])
        )
        if source.dtype == destination_dtype:
            if copy_requested:
                self._opaque(
                    node, "same-dtype to(copy=True) requires an explicit copy lowering"
                )
                return
            output_id = self._add_value(
                node.name,
                output_meta,
                role="alias",
                alias_of=source.id,
            )
            op_id = self._add_op(
                "view",
                tensor_inputs,
                (output_id,),
                {"source_operator": target, "same_dtype_to": True},
                metadata_only=True,
            )
            self._set_producer(output_id, op_id)
            self.node_values[node] = (output_id,)
            return
        self._single_output_op(
            node,
            "cast",
            tensor_inputs,
            {
                "dtype": destination_dtype,
                "source_operator": target,
            },
        )

    def _view(self, node: Any, target: str, *, transpose: bool) -> None:
        tensor_inputs = self._tensor_inputs(node.args[:1])
        if len(tensor_inputs) != 1:
            self._opaque(node, f"{target} requires one tensor input")
            return
        source = self._value(tensor_inputs[0])
        result_meta = _metadata_outputs(node.meta.get("val"))[0]
        result_shape = _shape_from_meta(
            result_meta, self._constraints, self._symbol_names
        )
        if result_shape is None or _static_numel(source.shape) != _static_numel(
            result_shape
        ):
            self._opaque(node, "view is not a metadata-only element-preserving alias")
            return
        permutation = _permutation(node, len(source.shape)) if transpose else None
        if transpose:
            users = list(node.users)
            allowed_user = len(users) == 1 and _target_name(users[0].target) in {
                "aten.mm.default",
                "aten.matmul.default",
            }
            if len(source.shape) != 2 or permutation != [1, 0] or not allowed_user:
                self._opaque(
                    node,
                    "only an immediately-consumed rank-2 matmul transpose is supported",
                )
                return
        elif _flattened_solver_shape(source.shape) != _flattened_solver_shape(
            result_shape
        ):
            self._opaque(
                node,
                "view changes the two-dimensional solver geometry",
            )
            return
        output_id = self._add_value(
            node.name, result_meta, role="alias", alias_of=source.id
        )
        attrs: dict[str, Any] = {"source_operator": target}
        if transpose:
            attrs["permutation"] = permutation
        op_id = self._add_op(
            "transpose_view" if transpose else "view",
            tensor_inputs,
            (output_id,),
            attrs,
            metadata_only=True,
        )
        self._set_producer(output_id, op_id)
        self.node_values[node] = (output_id,)

    def _single_output_op(
        self,
        node: Any,
        kind: str,
        inputs: Sequence[str],
        attributes: Mapping[str, Any],
        *,
        output_index: int = 0,
    ) -> None:
        outputs = _metadata_outputs(node.meta.get("val"))
        if output_index >= len(outputs):
            self._opaque(node, f"{kind} result has no tensor metadata")
            return
        output_id = self._add_value(
            node.name, outputs[output_index], role="intermediate"
        )
        op_id = self._add_op(kind, inputs, (output_id,), attributes)
        self._set_producer(output_id, op_id)
        self.node_values[node] = (output_id,)

    def _opaque(self, node: Any, reason: str) -> None:
        outputs = _metadata_outputs(node.meta.get("val"))
        if not outputs:
            self.node_values[node] = ()
            self.diagnostics.append(f"{node.name}: {reason}")
            return
        output_ids = tuple(
            self._add_value(
                node.name if len(outputs) == 1 else f"{node.name}_{index}",
                meta,
                role="intermediate",
            )
            for index, meta in enumerate(outputs)
        )
        op_id = self._add_op(
            "opaque",
            self._tensor_inputs(node.args),
            output_ids,
            {"source_operator": _target_name(node.target)},
            supported=False,
            opaque_reason=reason,
        )
        for output_id in output_ids:
            self._set_producer(output_id, op_id)
        self.node_values[node] = output_ids
        self.diagnostics.append(f"{node.name}: {reason}")

    def _add_value(
        self,
        name: str,
        meta: Any,
        *,
        role: str,
        target: str | None = None,
        alias_of: str | None = None,
    ) -> str:
        shape = _shape_from_meta(meta, self._constraints, self._symbol_names)
        if shape is None:
            shape = ()
        return self._add_value_from_shape(
            name,
            shape,
            _dtype_from_meta(meta),
            role=role,
            target=target,
            alias_of=alias_of,
            strides=_strides_from_meta(meta, self._constraints, self._symbol_names),
            storage_offset=_storage_offset_from_meta(
                meta, self._constraints, self._symbol_names
            ),
        )

    def _add_value_from_shape(
        self,
        name: str,
        shape: Sequence[Any],
        dtype: str,
        *,
        role: str,
        target: str | None = None,
        alias_of: str | None = None,
        strides: Sequence[Any] | None = None,
        storage_offset: int | ShapeDimension | None = 0,
    ) -> str:
        value_id = f"v{self._value_counter:04d}"
        self._value_counter += 1
        self.values.append(
            NormalizedValue(
                id=value_id,
                name=str(name),
                shape=tuple(shape),
                dtype=dtype,
                role=role,
                strides=_contiguous_strides(shape)
                if strides is None
                else tuple(strides),
                storage_offset=storage_offset,
                target=target,
                alias_of=alias_of,
            )
        )
        return value_id

    def _add_op(
        self,
        kind: str,
        inputs: Sequence[str],
        outputs: Sequence[str],
        attributes: Mapping[str, Any],
        *,
        supported: bool = True,
        opaque_reason: str | None = None,
        metadata_only: bool = False,
    ) -> str:
        op_id = f"op{self._op_counter:04d}"
        self._op_counter += 1
        self.ops.append(
            NormalizedOp(
                id=op_id,
                kind=kind,
                inputs=tuple(inputs),
                outputs=tuple(outputs),
                attributes=dict(attributes),
                supported=supported,
                opaque_reason=opaque_reason,
                metadata_only=metadata_only,
            )
        )
        return op_id

    def _set_producer(self, value_id: str, op_id: str) -> None:
        for index, value in enumerate(self.values):
            if value.id == value_id:
                self.values[index] = replace(value, producer=op_id)
                return
        raise KeyError(value_id)

    def _value(self, value_id: str) -> NormalizedValue:
        for value in self.values:
            if value.id == value_id:
                return value
        raise KeyError(value_id)

    def _tensor_inputs(self, args: Any) -> tuple[str, ...]:
        result: list[str] = []
        for item in _walk(args):
            if item in self.node_values:
                result.extend(self.node_values[item])
        return tuple(result)

    def _is_tensor_arg(self, arg: Any) -> bool:
        return arg in self.node_values if _hashable(arg) else False

    def _flatten_output_values(self, args: Any) -> tuple[str, ...]:
        output: list[str] = []
        for item in _walk(args):
            if item in self.node_values:
                output.extend(self.node_values[item])
        return tuple(output)

    def _is_transpose_value(self, value_id: str) -> bool:
        value = self._value(value_id)
        if value.producer is None:
            return False
        return (
            next(op for op in self.ops if op.id == value.producer).kind
            == "transpose_view"
        )

    def _mark_mutated_outputs(self) -> None:
        specs = getattr(self.program.graph_signature, "output_specs", ())
        for spec in specs:
            if "MUTATION" not in str(spec.kind):
                continue
            name = getattr(spec.arg, "name", None)
            fx_node = next(
                (
                    node
                    for node in self.program.graph_module.graph.nodes
                    if node.name == name
                ),
                None,
            )
            if fx_node is None or fx_node not in self.node_values:
                continue
            for value_id in self.node_values[fx_node]:
                value = self._value(value_id)
                if value.producer is None:
                    continue
                for index, op in enumerate(self.ops):
                    if op.id == value.producer:
                        reason = "mutated state is an automatic scheduling boundary"
                        self.ops[index] = replace(
                            op, supported=False, opaque_reason=reason
                        )
                        self.diagnostics.append(f"{op.id}: {reason}")


def _target_name(target: Any) -> str:
    if target is operator.getitem:
        return "operator.getitem"
    return str(target)


def _input_spec_map(program: Any) -> dict[str, tuple[str, str | None]]:
    roles: dict[str, tuple[str, str | None]] = {}
    for spec in getattr(program.graph_signature, "input_specs", ()):
        name = getattr(spec.arg, "name", None)
        if name is None:
            continue
        kind = str(spec.kind).split(".")[-1].lower()
        role = {
            "user_input": "user_input",
            "parameter": "parameter",
            "buffer": "buffer",
            "constant_tensor": "constant",
        }.get(kind, "external")
        roles[name] = (role, None if spec.target is None else str(spec.target))
    return roles


def _range_constraints(
    program: Any,
) -> tuple[dict[str, dict[str, int | None]], dict[str, str]]:
    constraints: dict[str, dict[str, int | None]] = {}
    symbol_names: dict[str, str] = {}
    for index, (symbol, value_range) in enumerate(
        getattr(program, "range_constraints", {}).items()
    ):
        canonical = f"s{index}"
        symbol_names[str(symbol)] = canonical
        constraints[canonical] = {
            "minimum": _bound_to_int(getattr(value_range, "lower", None)),
            "maximum": _bound_to_int(getattr(value_range, "upper", None)),
        }
    return constraints, symbol_names


def _shape_from_meta(
    meta: Any,
    constraints: Mapping[str, Mapping[str, int | None]],
    symbol_names: Mapping[str, str],
) -> tuple[int | ShapeDimension, ...] | None:
    shape = getattr(meta, "shape", None)
    if shape is None:
        return None
    result: list[int | ShapeDimension] = []
    for dim in shape:
        if isinstance(dim, int):
            result.append(int(dim))
            continue
        expression = _canonical_symbol_expression(str(dim), symbol_names)
        bounds = constraints.get(expression, {})
        result.append(
            ShapeDimension(
                expression=expression,
                minimum=bounds.get("minimum"),
                maximum=bounds.get("maximum"),
            )
        )
    return tuple(result)


def _strides_from_meta(
    meta: Any,
    constraints: Mapping[str, Mapping[str, int | None]],
    symbol_names: Mapping[str, str],
) -> tuple[int | ShapeDimension, ...] | None:
    stride_method = getattr(meta, "stride", None)
    if not callable(stride_method):
        return None
    try:
        strides = stride_method()
    except (RuntimeError, TypeError):
        return None
    if not isinstance(strides, Sequence):
        return None
    return _dimensions_from_sequence(strides, constraints, symbol_names)


def _storage_offset_from_meta(
    meta: Any,
    constraints: Mapping[str, Mapping[str, int | None]],
    symbol_names: Mapping[str, str],
) -> int | ShapeDimension | None:
    storage_offset_method = getattr(meta, "storage_offset", None)
    if not callable(storage_offset_method):
        return None
    try:
        storage_offset = storage_offset_method()
    except (RuntimeError, TypeError):
        return None
    return _dimensions_from_sequence((storage_offset,), constraints, symbol_names)[0]


def _dimensions_from_sequence(
    values: Sequence[Any],
    constraints: Mapping[str, Mapping[str, int | None]],
    symbol_names: Mapping[str, str],
) -> tuple[int | ShapeDimension, ...]:
    result: list[int | ShapeDimension] = []
    for value in values:
        if isinstance(value, int):
            result.append(int(value))
            continue
        expression = _canonical_symbol_expression(str(value), symbol_names)
        bounds = constraints.get(expression, {})
        result.append(
            ShapeDimension(
                expression=expression,
                minimum=bounds.get("minimum"),
                maximum=bounds.get("maximum"),
            )
        )
    return tuple(result)


def _contiguous_strides(shape: Sequence[Any]) -> tuple[int, ...] | None:
    if any(not isinstance(dim, int) for dim in shape):
        return None
    stride = 1
    result: list[int] = []
    for dim in reversed(shape):
        result.append(stride)
        stride *= int(dim)
    return tuple(reversed(result))


def _serialize_tree_spec(spec: Any) -> Any:
    import torch.utils._pytree as pytree  # noqa: PLC0415 -- Optional Torch frontend state.

    return json.loads(pytree.treespec_dumps(spec))


def _output_specs(
    program: Any, output_ids: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    specs = getattr(program.graph_signature, "output_specs", ())
    if len(specs) != len(output_ids):
        raise ValueError(
            f"exported graph has {len(specs)} output specs for {len(output_ids)} normalized outputs"
        )
    return tuple(
        {
            "kind": str(spec.kind).split(".")[-1].lower(),
            "target": None if spec.target is None else str(spec.target),
            "value": output_ids[index],
        }
        for index, spec in enumerate(specs)
    )


def _dtype_from_meta(meta: Any) -> str:
    dtype = getattr(meta, "dtype", None)
    return "unknown" if dtype is None else str(dtype).removeprefix("torch.")


def _metadata_outputs(meta: Any) -> tuple[Any, ...]:
    if isinstance(meta, (tuple, list)):
        return tuple(item for item in meta if getattr(item, "shape", None) is not None)
    return (meta,) if getattr(meta, "shape", None) is not None else ()


def _metadata_output_count(meta: Any) -> int:
    return len(_metadata_outputs(meta))


def _normalize_dims(value: Any, rank: int) -> tuple[int, ...]:
    if isinstance(value, int):
        values = (value,)
    elif isinstance(value, (tuple, list)):
        values = tuple(int(item) for item in value)
    else:
        return ()
    return tuple(item + rank if item < 0 else item for item in values)


def _permutation(node: Any, rank: int) -> list[int]:
    target = _target_name(node.target)
    if target == "aten.t.default":
        return [1, 0]
    if target == "aten.transpose.int":
        first, second = int(node.args[1]), int(node.args[2])
        if first < 0:
            first += rank
        if second < 0:
            second += rank
        result = list(range(rank))
        result[first], result[second] = result[second], result[first]
        return result
    return [int(item) for item in node.args[1]]


def _static_numel(shape: Sequence[Any]) -> int | None:
    if not all(isinstance(dim, int) for dim in shape):
        return None
    return math.prod(int(dim) for dim in shape)


def _flattened_solver_shape(shape: Sequence[Any]) -> tuple[Any, Any] | None:
    if not shape:
        return (1, 1)
    if any(not isinstance(dim, int) for dim in shape):
        return None
    if len(shape) == 1:
        return (1, shape[0])
    return (math.prod(shape[:-1]), shape[-1])


def _canonical_symbol_expression(expression: str, names: Mapping[str, str]) -> str:
    result = expression
    for source in sorted(names, key=len, reverse=True):
        result = re.sub(rf"\b{re.escape(source)}\b", names[source], result)
    return result


def _walk(value: Any) -> Iterable[Any]:
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk(item)
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _walk(value[key])
    else:
        yield value


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _hashable(value: Any) -> bool:
    try:
        hash(value)
        return True
    except TypeError:
        return False


def _bound_to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    # SymPy infinity may convert through implementation-specific sentinels.
    return numeric if abs(numeric) < (1 << 62) else None
