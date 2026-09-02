"""Public entry points for deterministic PyPTO source emission."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..ir import NormalizedGraph, NormalizedOp, normalized_graph_sha256
from ..lowered import LoweredContractError, LoweredRegion, lowered_region
from ..schedule import (
    KernelKind,
    KernelStep,
    ScheduleContractError,
    ScheduledRegion,
    VectorKernelPlan,
    scheduled_region,
)
from ..solver import RegionSolveResult, SolveResult
from .common import (
    EmissionContext,
    Interface,
    SourceEmissionError,
    SourceWriter,
    class_name,
    emit_return,
    identifier,
    interface,
    program_preamble,
    pypto_dtype,
    solver_tensor_for_value,
    static_shape,
)


@dataclass(frozen=True)
class PyPTOABIArgument:
    """One stable normalized-value-to-PyPTO-name ABI binding."""

    value_id: str
    name: str


@dataclass(frozen=True)
class RuntimeValidShapeSpec:
    """Request one runtime logical extent over a static physical frame.

    The first source contract supports only the outer/free axis of one
    homogeneous vector region.  The solver still plans the full concrete
    shape; native PyPTO orchestration supplies the active prefix at runtime.
    """

    axis: int = 0
    argument_name: str = "valid_rows"


@dataclass(frozen=True)
class PyPTORuntimeValidShapeArgument:
    """One scalar ABI argument bounding a statically planned tensor axis."""

    name: str
    axis: int
    physical_extent: int


@dataclass(frozen=True)
class EmittedPyPTOSource:
    """One deterministic PyPTO program and its normalized region ABI.

    ``input_value_ids`` is in the exact order used by the generated ``main``
    signature.  Runtime and device harnesses must bind tensors through these
    stable normalized value IDs instead of assuming that Torch positional
    input order survives solver lowering.
    """

    program_name: str
    region_id: str
    kinds: tuple[KernelKind, ...]
    input_value_ids: tuple[str, ...]
    output_value_ids: tuple[str, ...]
    source: str

    @property
    def kind(self) -> KernelKind | None:
        """Return the uniform kind, or ``None`` for a cross-kind program."""

        unique = set(self.kinds)
        return next(iter(unique)) if len(unique) == 1 else None


@dataclass(frozen=True)
class EmittedPyPTOCallable:
    """One callable static schedule for use by native PyPTO orchestration.

    The source defines a module-level ``@pl.inline`` function.  Calling it from
    a native orchestration function expands the exact solver-owned SPMD task
    graph at that call site; it does not invoke a second scheduler.
    """

    function_name: str
    region_id: str
    kinds: tuple[KernelKind, ...]
    input_arguments: tuple[PyPTOABIArgument, ...]
    runtime_valid_shapes: tuple[PyPTORuntimeValidShapeArgument, ...]
    output_arguments: tuple[PyPTOABIArgument, ...]
    source: str

    @property
    def input_value_ids(self) -> tuple[str, ...]:
        """Return normalized input IDs in callable signature order."""

        return tuple(argument.value_id for argument in self.input_arguments)

    @property
    def output_value_ids(self) -> tuple[str, ...]:
        """Return normalized output IDs in callable signature order."""

        return tuple(argument.value_id for argument in self.output_arguments)

    @property
    def kind(self) -> KernelKind | None:
        """Return the uniform kind, or ``None`` for a cross-kind callable."""

        unique = set(self.kinds)
        return next(iter(unique)) if len(unique) == 1 else None


@dataclass(frozen=True)
class EmittedPyPTOStaticBundle:
    """Graph-linked manifest of static callables and native operations.

    ``graph`` remains the authoritative description of operations and values.
    ``native_op_ids`` is the exact ordered complement of the operations owned
    by ``callables``. Native PyPTO orchestration retains the corresponding
    source implementation and uses normalized value IDs to wire it to the
    generated static callables. The manifest does not serialize or regenerate
    dynamic control flow, paged gathers, TopK, routing, or metadata operations.
    """

    graph: NormalizedGraph
    callables: tuple[EmittedPyPTOCallable, ...]
    native_op_ids: tuple[str, ...]

    @property
    def native_operations(self) -> tuple[NormalizedOp, ...]:
        """Return native operations in normalized graph order."""

        operations = self.graph.op_map()
        return tuple(operations[op_id] for op_id in self.native_op_ids)


def can_emit_region(graph: NormalizedGraph, result: RegionSolveResult) -> bool:
    """Return whether the installed backend can emit this exact region.

    Readiness deliberately runs the same graph-aware validation and rendering
    path as :func:`emit_pypto_region`; it is not a weaker schedule-family hint.
    """

    try:
        lowered, schedule, region_interface = _emission_contract(graph, result)
        _render_region(
            graph,
            result.problem,
            lowered,
            schedule,
            region_interface,
            "FuseboxReadinessProbe",
        )
    except (LoweredContractError, ScheduleContractError, SourceEmissionError):
        return False
    return True


def emit_pypto_region(
    graph: NormalizedGraph,
    result: RegionSolveResult,
    *,
    program_name: str | None = None,
) -> EmittedPyPTOSource:
    """Emit one solver-owned schedule as ordinary PyPTO DSL."""

    try:
        lowered, schedule, region_interface = _emission_contract(graph, result)
    except (LoweredContractError, ScheduleContractError) as error:
        raise SourceEmissionError(str(error)) from error
    chosen_name = class_name(program_name or f"fused_{result.region.id}")
    source = _render_region(
        graph,
        result.problem,
        lowered,
        schedule,
        region_interface,
        chosen_name,
    )
    return EmittedPyPTOSource(
        program_name=chosen_name,
        region_id=lowered.region_id,
        kinds=tuple(step.kind for step in schedule.steps),
        input_value_ids=tuple(region_interface.input_arguments),
        output_value_ids=tuple(region_interface.output_arguments),
        source=source,
    )


def emit_pypto_callable(
    graph: NormalizedGraph,
    result: RegionSolveResult,
    *,
    function_name: str | None = None,
    runtime_valid_shape: RuntimeValidShapeSpec | None = None,
) -> EmittedPyPTOCallable:
    """Emit one solver-owned schedule as a callable PyPTO inline fragment.

    Native PyPTO orchestration remains responsible for runtime control flow and
    output allocation.  The callable expands only the selected static task
    graph, including every SPMD launch and dependency carried by the schedule.
    """

    try:
        lowered, schedule, region_interface = _emission_contract(graph, result)
    except (LoweredContractError, ScheduleContractError) as error:
        raise SourceEmissionError(str(error)) from error
    chosen_name = identifier(function_name or f"fused_{result.region.id}")
    program_source = _render_region(
        graph,
        result.problem,
        lowered,
        schedule,
        region_interface,
        class_name(chosen_name),
    )
    source = _program_as_inline_callable(program_source, chosen_name)
    runtime_arguments: tuple[PyPTORuntimeValidShapeArgument, ...] = ()
    if runtime_valid_shape is not None:
        source, runtime_argument = _add_runtime_valid_shape(
            source,
            graph,
            lowered,
            schedule,
            region_interface,
            runtime_valid_shape,
        )
        runtime_arguments = (runtime_argument,)
    return EmittedPyPTOCallable(
        function_name=chosen_name,
        region_id=lowered.region_id,
        kinds=tuple(step.kind for step in schedule.steps),
        input_arguments=tuple(
            PyPTOABIArgument(value_id, argument)
            for value_id, argument in region_interface.input_arguments.items()
        ),
        runtime_valid_shapes=runtime_arguments,
        output_arguments=tuple(
            PyPTOABIArgument(value_id, argument)
            for value_id, argument in region_interface.output_arguments.items()
        ),
        source=source,
    )


def emit_pypto_static_bundle(
    graph: NormalizedGraph,
    result: SolveResult,
    *,
    function_prefix: str = "fused",
) -> EmittedPyPTOStaticBundle:
    """Emit every solved static region while preserving native boundaries.

    This is a graph-linked manifest for native PyPTO orchestration. Every
    region must be solved and source-emittable. Every operation outside those
    regions remains in ``bundle.graph`` and is named by ``native_op_ids`` in
    graph order. Callers retain the original native implementation of those
    operations; this function does not attempt to reconstruct it.
    """

    if normalized_graph_sha256(graph) != normalized_graph_sha256(result.graph):
        raise SourceEmissionError(
            "solve result belongs to a different normalized graph"
        )
    if not result.regions:
        raise SourceEmissionError("static bundle requires at least one solver region")
    prefix = identifier(function_prefix)
    callables: list[EmittedPyPTOCallable] = []
    for region in result.regions:
        if region.status != "solved":
            raise SourceEmissionError(
                f"region {region.region.id} is not solved: {region.status}"
            )
        callables.append(
            emit_pypto_callable(
                graph,
                region,
                function_name=f"{prefix}_{region.region.id}",
            )
        )

    graph_op_ids = {op.id for op in graph.ops}
    static_op_ids: set[str] = set()
    for region in result.regions:
        overlap = static_op_ids.intersection(region.region.op_ids)
        if overlap:
            repeated = ", ".join(sorted(overlap))
            raise SourceEmissionError(
                f"solver regions overlap on normalized operations: {repeated}"
            )
        unknown = set(region.region.op_ids).difference(graph_op_ids)
        if unknown:
            missing = ", ".join(sorted(unknown))
            raise SourceEmissionError(
                f"solver region references unknown normalized operations: {missing}"
            )
        static_op_ids.update(region.region.op_ids)

    native_op_ids = tuple(op.id for op in graph.ops if op.id not in static_op_ids)
    represented_op_ids = static_op_ids.union(native_op_ids)
    if represented_op_ids != graph_op_ids:
        raise SourceEmissionError(
            "static bundle does not cover every normalized operation"
        )
    for value in graph.values:
        if value.producer is not None and value.producer not in represented_op_ids:
            raise SourceEmissionError(
                f"normalized value {value.id!r} has an unrepresented producer "
                f"{value.producer!r}"
            )
    return EmittedPyPTOStaticBundle(graph, tuple(callables), native_op_ids)


def _emission_contract(
    graph: NormalizedGraph, result: RegionSolveResult
) -> tuple[LoweredRegion, ScheduledRegion, Interface]:
    if result.problem is None:
        raise SourceEmissionError("source emission requires a lowered problem")
    schedule = scheduled_region(result)
    lowered = lowered_region(result)
    if schedule.region_id != lowered.region_id:
        raise SourceEmissionError("problem and solution region identities disagree")
    if normalized_graph_sha256(graph) != lowered.normalized_graph_sha256:
        raise SourceEmissionError(
            "supplied normalized graph does not match the graph used to solve the region"
        )
    return lowered, schedule, interface(graph, lowered)


def _render_region(  # noqa: PLR0913
    graph: NormalizedGraph,
    problem: Mapping[str, Any] | None,
    lowered: LoweredRegion,
    schedule: ScheduledRegion,
    region_interface: Interface,
    program_name: str,
) -> str:
    if problem is None:
        raise SourceEmissionError("source emission requires an object problem")
    if len(schedule.steps) == 1:
        context = EmissionContext(
            graph=graph,
            problem=problem,
            lowered=lowered,
            schedule=schedule,
            step=schedule.steps[0],
            interface=region_interface,
        )
        return _render(context, program_name)
    return _render_steps(
        graph,
        problem,
        lowered,
        schedule,
        region_interface,
        program_name,
    )


def _render_steps(  # noqa: PLR0913
    graph: NormalizedGraph,
    problem: Mapping[str, Any],
    lowered: LoweredRegion,
    schedule: ScheduledRegion,
    region_interface: Interface,
    program_name: str,
) -> str:
    """Compose cut homogeneous steps as dependency-linked SPMD launches."""

    producer = _tensor_producers(lowered)
    consumers = _tensor_consumers(lowered)
    step_by_op = {
        operation: step.index
        for step in schedule.steps
        for operation in step.solver_ops
    }
    if len(step_by_op) != len(lowered.operations):
        raise SourceEmissionError("selected steps do not cover every lowered operation")

    boundary_tensors: set[int] = set()
    for tensor, producer_op in enumerate(producer):
        if producer_op is None:
            continue
        producer_step = step_by_op[producer_op]
        consumer_steps = {
            step_by_op[consumer]
            for consumer in consumers[tensor]
            if step_by_op[consumer] != producer_step
        }
        if any(consumer_step <= producer_step for consumer_step in consumer_steps):
            raise SourceEmissionError(
                f"inter-step tensor {tensor} is consumed before its producer launch"
            )
        if consumer_steps:
            boundary_tensors.add(tensor)
    boundary_tensors.update(lowered.required_outputs)

    values = graph.value_map()
    tensor_variables: dict[int, str] = {}
    for tensor_index in sorted(boundary_tensors):
        tensor = lowered.tensor(tensor_index)
        if tensor.synthetic or tensor.value_id not in values:
            raise SourceEmissionError(
                f"inter-step tensor {tensor_index} has no source-level graph value"
            )
        if tensor_index in lowered.required_outputs:
            output_value = _region_output_for_tensor(
                lowered, region_interface, tensor_index
            )
            tensor_variables[tensor_index] = region_interface.output_arguments[
                output_value
            ]
        else:
            tensor_variables[tensor_index] = f"intermediate_tensor_{tensor_index}"

    writer = program_preamble(program_name, region_interface, graph)
    for tensor_index in sorted(boundary_tensors):
        if tensor_index in lowered.required_outputs:
            continue
        tensor = lowered.tensor(tensor_index)
        shape = static_shape(values[tensor.value_id], field="inter-step tensor")
        writer.line(
            2,
            f"{tensor_variables[tensor_index]} = pl.create_tensor("
            f"[{shape[0]}, {shape[1]}], dtype={pypto_dtype(tensor.dtype)})",
        )

    for step in schedule.steps:
        step_interface = _step_interface(
            step,
            lowered,
            producer,
            consumers,
            step_by_op,
            boundary_tensors,
            tensor_variables,
            region_interface,
        )
        context = EmissionContext(
            graph=graph,
            problem=problem,
            lowered=lowered,
            schedule=schedule,
            step=step,
            interface=step_interface,
        )
        step_source = _render(context, f"{program_name}Step{step.index}")
        _append_spmd_statement(
            writer,
            step_source,
            step.index,
            protected=set(step_interface.input_arguments.values())
            | set(step_interface.output_arguments.values()),
        )
    emit_return(writer, region_interface)
    return writer.render()


def _step_interface(  # noqa: PLR0913
    step: KernelStep,
    lowered: LoweredRegion,
    producer: tuple[int | None, ...],
    consumers: tuple[tuple[int, ...], ...],
    step_by_op: dict[int, int],
    boundary_tensors: set[int],
    tensor_variables: dict[int, str],
    region_interface: Interface,
) -> Interface:
    step_ops = set(step.solver_ops)
    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}
    owners: dict[str, str] = {}
    for operation_index in step.op_order:
        operation = lowered.operation(operation_index)
        for tensor_index in operation.inputs:
            producer_op = producer[tensor_index]
            if producer_op in step_ops:
                continue
            tensor = lowered.tensor(tensor_index)
            if producer_op is None:
                input_value = (
                    tensor.alias_of if tensor.alias_of is not None else tensor.value_id
                )
                try:
                    inputs[input_value] = region_interface.input_arguments[input_value]
                except KeyError as error:
                    raise SourceEmissionError(
                        f"step {step.index} input {input_value!r} is absent from the region ABI"
                    ) from error
            else:
                inputs[tensor.value_id] = tensor_variables[tensor_index]
        for tensor_index in operation.outputs:
            if tensor_index not in boundary_tensors:
                continue
            if (
                not any(
                    step_by_op[consumer] != step.index
                    for consumer in consumers[tensor_index]
                )
                and tensor_index not in lowered.required_outputs
            ):
                continue
            tensor = lowered.tensor(tensor_index)
            if tensor_index in lowered.required_outputs:
                output_value = _region_output_for_tensor(
                    lowered, region_interface, tensor_index
                )
                outputs[output_value] = tensor_variables[tensor_index]
                owners[output_value] = region_interface.output_allocation_owners[
                    output_value
                ]
            else:
                outputs[tensor.value_id] = tensor_variables[tensor_index]
                owners[tensor.value_id] = tensor.value_id
    if not outputs:
        raise SourceEmissionError(
            f"step {step.index} has no materialized boundary output"
        )
    return Interface(inputs, outputs, owners)


def _append_spmd_statement(
    writer: SourceWriter,
    source: str,
    step_index: int,
    *,
    protected: set[str],
) -> None:
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if len(classes) != 1:
        raise SourceEmissionError(
            "one homogeneous step must emit exactly one program class"
        )
    functions = [node for node in classes[0].body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise SourceEmissionError(
            "one homogeneous step must emit exactly one orchestration function"
        )
    function = functions[0]
    if (
        len(function.body) != 2
        or not isinstance(function.body[0], ast.For)
        or not isinstance(function.body[1], ast.Return)
    ):
        raise SourceEmissionError(
            "one homogeneous step must contain only one SPMD loop and its return"
        )
    loop = function.body[0]
    if not (
        isinstance(loop.iter, ast.Call)
        and isinstance(loop.iter.func, ast.Attribute)
        and isinstance(loop.iter.func.value, ast.Name)
        and loop.iter.func.value.id == "pl"
        and loop.iter.func.attr == "spmd"
    ):
        raise SourceEmissionError("homogeneous step loop must be a pl.spmd grid")
    local_names = {
        node.id
        for node in ast.walk(loop)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id not in protected
    }

    class PrefixLocals(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:  # noqa: N802
            if node.id in local_names:
                node.id = f"step_{step_index}_{node.id}"
            return node

    loop = PrefixLocals().visit(loop)
    ast.fix_missing_locations(loop)
    lines = ast.unparse(loop).splitlines()
    for line in lines:
        leading = len(line) - len(line.lstrip(" "))
        writer.line(2 + leading // 4, line.lstrip())


def _region_output_for_tensor(
    lowered: LoweredRegion,
    region_interface: Interface,
    tensor_index: int,
) -> str:
    matches = [
        output_value
        for output_value, owner in region_interface.output_allocation_owners.items()
        if solver_tensor_for_value(lowered, owner) == tensor_index
    ]
    if len(matches) != 1:
        raise SourceEmissionError(
            f"solver tensor {tensor_index} does not map to one region output"
        )
    return matches[0]


def _tensor_producers(lowered: LoweredRegion) -> tuple[int | None, ...]:
    producers: list[int | None] = [None] * len(lowered.tensors)
    for operation in lowered.operations:
        for tensor in operation.outputs:
            producers[tensor] = operation.index
    return tuple(producers)


def _tensor_consumers(lowered: LoweredRegion) -> tuple[tuple[int, ...], ...]:
    consumers: list[list[int]] = [[] for _ in lowered.tensors]
    for operation in lowered.operations:
        for tensor in operation.inputs:
            consumers[tensor].append(operation.index)
    return tuple(tuple(items) for items in consumers)


def _render(context: EmissionContext, program_name: str) -> str:
    from .cube import emit_cube
    from .mixed import emit_mixed
    from .vector import emit_vector

    if context.step.kind is KernelKind.VECTOR:
        source = emit_vector(context, program_name)
    elif context.step.kind is KernelKind.CUBE:
        source = emit_cube(context, program_name)
    elif context.step.kind is KernelKind.MIXED:
        source = emit_mixed(context, program_name)
    else:
        raise SourceEmissionError(f"unknown kernel kind {context.step.kind.value!r}")
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


def _program_as_inline_callable(source: str, function_name: str) -> str:
    """Convert one generated orchestration program into an inline callable."""

    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if len(classes) != 1:
        raise SourceEmissionError("generated source must contain exactly one program")
    program = classes[0]
    functions = [node for node in program.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "main":
        raise SourceEmissionError(
            "generated program must contain exactly one main orchestration function"
        )
    unexpected = [node for node in program.body if node is not functions[0]]
    if unexpected:
        kinds = ", ".join(type(node).__name__ for node in unexpected)
        raise SourceEmissionError(
            "generated program class contains unexpected members: " + kinds
        )
    function = functions[0]
    if not function.args.args or function.args.args[0].arg != "self":
        raise SourceEmissionError("generated main function must start with self")
    function.name = function_name
    function.args.args = function.args.args[1:]
    _namespace_inline_locals(function, function_name)
    function.decorator_list = [
        ast.Attribute(
            value=ast.Name(id="pl", ctx=ast.Load()),
            attr="inline",
            ctx=ast.Load(),
        )
    ]
    module = ast.Module(
        body=[node for node in tree.body if not isinstance(node, ast.ClassDef)]
        + [function],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    rendered = ast.unparse(module) + "\n"
    if _has_automatic_scheduling_tag(module):
        raise SourceEmissionError("callable source must encode the plan directly")
    return rendered


def _namespace_inline_locals(function: ast.FunctionDef, function_name: str) -> None:
    """Give compiler-generated locals a callable-specific namespace.

    PyPTO expands ``@pl.inline`` functions into their caller.  Two separately
    generated callables therefore cannot safely reuse orchestration-local
    names: a later expansion may rebind an earlier tile at a different type.
    Parameters are the public callable ABI and deliberately remain unchanged.
    """

    parameters = {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }
    if function.args.vararg is not None:
        parameters.add(function.args.vararg.arg)
    if function.args.kwarg is not None:
        parameters.add(function.args.kwarg.arg)

    assigned = {
        node.id
        for statement in function.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    }
    locals_to_rename = sorted(assigned - parameters)
    replacements = {
        name: _inline_local_name(function_name, name) for name in locals_to_rename
    }
    replacement_names = set(replacements.values())
    if len(replacement_names) != len(replacements) or replacement_names & parameters:
        raise SourceEmissionError(
            "callable-local namespace collides with the generated function ABI"
        )

    class _RenameLocals(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:  # noqa: N802 - AST API.
            replacement = replacements.get(node.id)
            if replacement is None:
                return node
            return ast.copy_location(
                ast.Name(id=replacement, ctx=node.ctx),
                node,
            )

    for index, statement in enumerate(function.body):
        function.body[index] = _RenameLocals().visit(statement)


def _inline_local_name(function_name: str, local_name: str) -> str:
    """Return an injective callable/local namespace encoding."""

    return f"fusebox_local_{len(function_name)}_{function_name}_{local_name}"


def _add_runtime_valid_shape(  # noqa: PLR0913 -- explicit ABI contract inputs.
    source: str,
    graph: NormalizedGraph,
    lowered: LoweredRegion,
    schedule: ScheduledRegion,
    region_interface: Interface,
    spec: RuntimeValidShapeSpec,
) -> tuple[str, PyPTORuntimeValidShapeArgument]:
    """Make one vector free-axis valid extent a runtime scalar.

    Physical tensor types, tiles, allocations, grids, and pipelines remain
    byte-for-byte those selected for the concrete solver problem.  Only direct
    GM loads of non-broadcast operands receive a clamped runtime valid extent;
    PyPTO propagates that logical shape through vector operations and stores.
    """

    if spec.axis != 0:
        raise SourceEmissionError(
            "runtime valid_shape currently supports only outer/free axis 0"
        )
    if identifier(spec.argument_name) != spec.argument_name:
        raise SourceEmissionError(
            "runtime valid_shape argument must be a valid Python identifier"
        )
    if len(schedule.steps) != 1 or schedule.steps[0].kind is not KernelKind.VECTOR:
        raise SourceEmissionError(
            "runtime valid_shape currently requires one homogeneous vector step"
        )
    if not isinstance(schedule.steps[0].plan, VectorKernelPlan):
        raise SourceEmissionError("runtime valid_shape requires a vector plan")

    values = graph.value_map()
    output_extents = {
        static_shape(values[value_id], field="runtime-valid output")[spec.axis]
        for value_id in region_interface.output_values
    }
    if len(output_extents) != 1:
        raise SourceEmissionError(
            "runtime valid_shape requires outputs with one common physical extent"
        )
    (physical_extent,) = output_extents
    dynamic_inputs: set[str] = set()
    for value_id, argument in region_interface.input_arguments.items():
        shape = static_shape(values[value_id], field="runtime-valid input")
        extent = shape[spec.axis]
        if extent == physical_extent:
            dynamic_inputs.add(argument)
        elif extent != 1:
            raise SourceEmissionError(
                "runtime valid_shape input extent must match the output frame or be broadcast"
            )
    if not dynamic_inputs:
        raise SourceEmissionError(
            "runtime valid_shape has no non-broadcast input on its selected axis"
        )
    if spec.argument_name in {
        *region_interface.input_arguments.values(),
        *region_interface.output_arguments.values(),
    }:
        raise SourceEmissionError(
            "runtime valid_shape argument collides with tensor ABI"
        )

    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise SourceEmissionError("callable source must contain exactly one function")
    function = functions[0]
    existing_identifiers = {argument.arg for argument in function.args.args} | {
        node.id for node in ast.walk(function) if isinstance(node, ast.Name)
    }
    if spec.argument_name in existing_identifiers:
        raise SourceEmissionError(
            "runtime valid_shape argument collides with a generated function identifier"
        )
    output_names = set(region_interface.output_arguments.values())
    output_positions = [
        index
        for index, argument in enumerate(function.args.args)
        if argument.arg in output_names
    ]
    if len(output_positions) != len(output_names):
        raise SourceEmissionError(
            "callable output ABI differs from its region interface"
        )
    insert_at = min(output_positions)
    function.args.args.insert(
        insert_at,
        ast.arg(
            arg=spec.argument_name,
            annotation=ast.Subscript(
                value=ast.Attribute(
                    value=ast.Name(id="pl", ctx=ast.Load()),
                    attr="Scalar",
                    ctx=ast.Load(),
                ),
                slice=ast.Attribute(
                    value=ast.Name(id="pl", ctx=ast.Load()),
                    attr="INDEX",
                    ctx=ast.Load(),
                ),
                ctx=ast.Load(),
            ),
        ),
    )

    rewritten = 0

    class RuntimeValidLoadRewriter(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
            nonlocal rewritten
            self.generic_visit(node)
            if not _is_pl_call(node, "load") or len(node.args) < 4:
                return node
            source_argument = node.args[0]
            if not (
                isinstance(source_argument, ast.Name)
                and source_argument.id in dynamic_inputs
            ):
                return node
            offsets = node.args[1]
            valid_shape = node.args[3]
            if not (
                isinstance(offsets, (ast.List, ast.Tuple))
                and isinstance(valid_shape, (ast.List, ast.Tuple))
                and len(offsets.elts) == 2
                and len(valid_shape.elts) == 2
            ):
                raise SourceEmissionError(
                    "runtime valid_shape requires rank-two positional pl.load geometry"
                )
            original_valid = valid_shape.elts[spec.axis]
            offset = offsets.elts[spec.axis]
            valid_shape.elts[spec.axis] = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="pl", ctx=ast.Load()),
                    attr="max",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="pl", ctx=ast.Load()),
                            attr="min",
                            ctx=ast.Load(),
                        ),
                        args=[
                            ast.BinOp(
                                left=ast.Name(id=spec.argument_name, ctx=ast.Load()),
                                op=ast.Sub(),
                                right=offset,
                            ),
                            original_valid,
                        ],
                        keywords=[],
                    ),
                    ast.Constant(value=0),
                ],
                keywords=[],
            )
            rewritten += 1
            return node

    RuntimeValidLoadRewriter().visit(function)
    if rewritten == 0:
        raise SourceEmissionError("runtime valid_shape did not reach any vector load")
    ast.fix_missing_locations(tree)
    return (
        ast.unparse(tree) + "\n",
        PyPTORuntimeValidShapeArgument(
            name=spec.argument_name,
            axis=spec.axis,
            physical_extent=physical_extent,
        ),
    )


def _is_pl_call(node: ast.Call, name: str) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pl"
        and node.func.attr == name
    )
