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
    return tuple(by_value_id[value_id] for value_id in emitted.input_value_ids)
