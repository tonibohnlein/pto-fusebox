"""Bindings from normalized Torch inputs to an emitted PyPTO region ABI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import nn

from .ir import NormalizedGraph

if TYPE_CHECKING:
    from .source import EmittedPyPTOCallable, EmittedPyPTOSource


class InputBindingError(ValueError):
    """Raised when a normalized input cannot be bound to emitted source."""


def bind_emitted_inputs(
    module: nn.Module,
    graph: NormalizedGraph,
    emitted: EmittedPyPTOSource | EmittedPyPTOCallable,
    user_inputs: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Return tensors in the exact order of the emitted region signature.

    Torch positional arguments, lifted parameters, and lifted buffers are first
    associated with stable normalized value IDs.  The result is then ordered by
    the emitted object's ``input_value_ids``.  This avoids assuming that solver
    lowering preserves the original Torch argument order.

    Constant tensors that are not module parameters or buffers remain outside
    this initial binding contract and fail closed.
    """

    values = graph.value_map()
    user_value_ids = tuple(
        value_id for value_id in graph.inputs if values[value_id].role == "user_input"
    )
    if len(user_value_ids) != len(user_inputs):
        raise InputBindingError(
            "Torch user-input count differs from the normalized graph: "
            f"expected {len(user_value_ids)}, got {len(user_inputs)}"
        )

    by_value_id = dict(zip(user_value_ids, user_inputs, strict=True))
    for value_id in graph.inputs:
        value = values[value_id]
        if value.role == "parameter":
            if value.target is None:
                raise InputBindingError(f"parameter {value_id} has no module target")
            by_value_id[value_id] = module.get_parameter(value.target)
        elif value.role == "buffer":
            if value.target is None:
                raise InputBindingError(f"buffer {value_id} has no module target")
            by_value_id[value_id] = module.get_buffer(value.target)
        elif value.role not in {"user_input", "parameter", "buffer"}:
            raise InputBindingError(
                f"normalized input {value_id} has unsupported binding role {value.role!r}"
            )

    missing = [
        value_id for value_id in emitted.input_value_ids if value_id not in by_value_id
    ]
    if missing:
        raise InputBindingError(
            f"emitted ABI references unbound normalized values {missing}"
        )
    bound = tuple(by_value_id[value_id] for value_id in emitted.input_value_ids)
    for value_id, tensor in zip(emitted.input_value_ids, bound, strict=True):
        _validate_tensor_binding(graph, value_id, tensor, direction="input")
    return bound


def bind_emitted_call(
    module: nn.Module,
    graph: NormalizedGraph,
    emitted: EmittedPyPTOSource | EmittedPyPTOCallable,
    user_inputs: Sequence[torch.Tensor],
    outputs: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Bind and validate a complete emitted call in ABI direction order.

    Device harnesses should use this helper rather than concatenate a guessed
    input tuple with output buffers. It validates exact input/output counts,
    normalized value directions, dtypes, physical shapes, and accidental
    input/output tensor aliasing before the runtime sees the call.
    """

    input_ids = emitted.input_value_ids
    output_ids = emitted.output_value_ids
    overlap = set(input_ids) & set(output_ids)
    if overlap:
        raise InputBindingError(
            f"emitted ABI uses normalized values as both input and output: {sorted(overlap)}"
        )
    if len(outputs) != len(output_ids):
        raise InputBindingError(
            "emitted output count differs from supplied buffers: "
            f"expected {len(output_ids)}, got {len(outputs)}"
        )
    inputs = bind_emitted_inputs(module, graph, emitted, user_inputs)
    for value_id, tensor in zip(output_ids, outputs, strict=True):
        _validate_tensor_binding(graph, value_id, tensor, direction="output")
    if any(
        _tensors_alias(input_tensor, output_tensor)
        for input_tensor in inputs
        for output_tensor in outputs
    ):
        raise InputBindingError(
            "emitted pure-output ABI cannot bind aliased input and output tensors"
        )
    return (*inputs, *outputs)


def _tensors_alias(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if lhs is rhs:
        return True
    try:
        lhs_ptr = lhs.untyped_storage().data_ptr()
        rhs_ptr = rhs.untyped_storage().data_ptr()
    except RuntimeError:
        return False
    return lhs_ptr != 0 and lhs_ptr == rhs_ptr


def _validate_tensor_binding(
    graph: NormalizedGraph,
    value_id: str,
    tensor: torch.Tensor,
    *,
    direction: str,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise InputBindingError(
            f"emitted {direction} {value_id} must bind a torch.Tensor"
        )
    value = graph.value_map().get(value_id)
    if value is None:
        raise InputBindingError(
            f"emitted {direction} references unknown normalized value {value_id}"
        )
    expected_dtypes = {
        "bool": torch.bool,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    expected_dtype = expected_dtypes.get(value.dtype)
    if expected_dtype is None:
        raise InputBindingError(
            f"emitted {direction} {value_id} has unsupported dtype {value.dtype!r}"
        )
    if tensor.dtype != expected_dtype:
        raise InputBindingError(
            f"emitted {direction} {value_id} expects dtype {expected_dtype}, got {tensor.dtype}"
        )
    if not all(isinstance(dim, int) for dim in value.shape):
        raise InputBindingError(
            f"emitted {direction} {value_id} does not have a concrete physical shape"
        )
    expected_shape = tuple(value.shape)
    if tuple(tensor.shape) != expected_shape:
        raise InputBindingError(
            f"emitted {direction} {value_id} expects shape {expected_shape}, "
            f"got {tuple(tensor.shape)}"
        )
