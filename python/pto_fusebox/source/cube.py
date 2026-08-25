"""Mechanical replay of homogeneous cube schedules as PyPTO DSL."""

from __future__ import annotations

from collections.abc import Mapping

from ..schedule import CubeKernelPlan
from ..schedule.schema import (
    CubeAxisBinding,
    CubeMatmulPlan,
    CubeOutputTileVariant,
    CubeSpatialPolicy,
    CubeSplitMergePolicy,
    CubeTensorRegionPlan,
    L0MatmulPlan,
)
from .common import (
    EmissionContext,
    SourceEmissionError,
    SourceWriter,
    emit_partition_indices,
    emit_return,
    pypto_dtype,
    program_header,
    program_preamble,
    solver_tensor_for_value,
    static_shape,
    validate_grid,
    validate_partition_extent,
)


def emit_cube(
    context: EmissionContext,
    program_name: str,
) -> str:
    """Emit one homogeneous cube schedule."""

    graph = context.graph
    lowered = context.lowered
    step = context.step
    io = context.interface
    plan = step.plan
    if not isinstance(plan, CubeKernelPlan):
        raise SourceEmissionError("cube step does not carry a cube plan")
    if not plan.emit_compatible:
        raise SourceEmissionError("cube plan is not marked emit-compatible")
    if plan.spatial_policy is not CubeSpatialPolicy.UNIFORM:
        raise SourceEmissionError(
            "cube source v1 supports only uniform spatial partitions"
        )
    if step.split != plan.split_k:
        raise SourceEmissionError("cube launch split differs from its plan")
    if plan.split_k > 1:
        return _emit_split_cube_dag(context, program_name, plan)
    if plan.split_merge_policy is not CubeSplitMergePolicy.NONE:
        raise SourceEmissionError("non-split cube plan carries a merge policy")
    matmuls = plan.matmuls
    if (
        len(matmuls) > 1
        or plan.resident_boundaries
        or any(
            matmul.output_grid != (1, 1)
            or matmul.retained_panels.lhs
            or matmul.retained_panels.rhs
            or matmul.storage_dtype != "fp32"
            or any(
                variant.l0_init.tile[2] != matmul.k_loop.chunk
                for variant in matmul.output_variants
            )
            for matmul in matmuls
        )
    ):
        return _emit_full_window_cube_dag(context, program_name, plan)
    if len(matmuls) != 1 or len(step.solver_ops) != 1:
        raise SourceEmissionError("cube source v1 supports exactly one matmul")
    matmul = matmuls[0]
    solver_op = step.solver_ops[0]
    if plan.execution_order != (solver_op,):
        raise SourceEmissionError(
            "cube execution order does not match the selected one-matmul step"
        )
    if matmul.instance != 0 or matmul.op != solver_op:
        raise SourceEmissionError("cube request identity does not match its solver op")
    if matmul.lhs_producer != -1 or matmul.rhs_producer != -1:
        raise SourceEmissionError("cube source v1 requires external matmul operands")
    if matmul.output_grid != (1, 1):
        raise SourceEmissionError(
            "cube source v1 requires one L0 output tile per region"
        )
    if matmul.accumulator_dtype != "fp32" or matmul.storage_dtype != "fp32":
        raise SourceEmissionError(
            "cube source v1 currently supports FP32 accumulation and storage"
        )

    m_partition = plan.m_partition
    n_partition = plan.n_partition
    validate_grid(step, plan.work_units, m_partition, n_partition)
    if m_partition.big != m_partition.small or n_partition.big != n_partition.small:
        raise SourceEmissionError(
            "cube source v1 does not yet emit ragged spatial regions"
        )

    graph_op = graph.op_map()[lowered.operation(solver_op).graph_op_id]
    if graph_op.kind != "matmul":
        raise SourceEmissionError("cube source v1 requires a matmul")
    lhs_transposed = graph_op.attributes.get("lhs_transposed") is True
    rhs_transposed = graph_op.attributes.get("rhs_transposed") is True
    lowered_op = lowered.operation(solver_op)
    op_inputs = list(lowered_op.inputs)
    op_outputs = list(lowered_op.outputs)
    if len(op_inputs) != 2 or len(op_outputs) != 1:
        raise SourceEmissionError("cube matmul must have two inputs and one output")
    output_rows, output_cols = static_shape(
        graph.value_map()[io.output_value], field="cube output"
    )
    validate_partition_extent(m_partition, output_rows, "cube.m_partition")
    validate_partition_extent(n_partition, output_cols, "cube.n_partition")
    lhs_arg = _argument_for_cube_tensor(context, op_inputs[0])
    rhs_arg = _argument_for_cube_tensor(context, op_inputs[1])
    if solver_tensor_for_value(lowered, io.output_allocation_owner) != op_outputs[0]:
        raise SourceEmissionError("cube matmul result must be the region output")

    k_loop = matmul.k_loop
    chunk = k_loop.chunk
    full_chunks = k_loop.full_chunks
    tail = k_loop.tail
    stages = k_loop.pipeline_stages
    if (
        step.sequential_tiles != (k_loop.l1_window_k,)
        or step.launch.tile_k != k_loop.l1_window_k
    ):
        raise SourceEmissionError(
            "cube launch and sequential tile differ from its L1 K window"
        )
    contraction = matmul.contraction
    if full_chunks * chunk + tail != contraction or full_chunks == 0:
        raise SourceEmissionError(
            "cube K-window descriptor does not cover the contraction"
        )
    if stages not in {1, 2} or (full_chunks == 1 and stages != 1):
        raise SourceEmissionError("cube K-window pipeline stage count is unsupported")
    if tail >= chunk:
        raise SourceEmissionError("cube K-window tail must be smaller than its chunk")
    output_tile = list(matmul.output_tile)
    if output_tile != [m_partition.big, n_partition.big]:
        raise SourceEmissionError(
            "cube output tile does not match its spatial partition"
        )
    _validate_l0_variant(matmul, output_tile, chunk, tail, full_chunks)
    _validate_lowered_l0_capacity(context, matmul)

    writer = program_header(
        program_name,
        io,
        graph,
        m_partition.parts * n_partition.parts,
        kernel_name_hint=context.region_id + "_cube",
    )
    indent = 3
    coordinates = emit_partition_indices(writer, indent, m_partition, n_partition)
    _emit_cube_window(
        writer,
        indent,
        lhs_arg,
        rhs_arg,
        coordinates.row,
        coordinates.col,
        output_tile,
        chunk,
        "0",
        first=True,
        lhs_transposed=lhs_transposed,
        rhs_transposed=rhs_transposed,
    )
    if full_chunks > 1:
        loop = "pl.pipeline" if stages > 1 else "pl.range"
        stage = f", stage={stages}" if stages > 1 else ""
        writer.line(indent, f"for k_window in {loop}(1, {full_chunks}{stage}):")
        _emit_cube_window(
            writer,
            indent + 1,
            lhs_arg,
            rhs_arg,
            coordinates.row,
            coordinates.col,
            output_tile,
            chunk,
            f"k_window * {chunk}",
            first=False,
            lhs_transposed=lhs_transposed,
            rhs_transposed=rhs_transposed,
        )
    if tail:
        _emit_cube_window(
            writer,
            indent,
            lhs_arg,
            rhs_arg,
            coordinates.row,
            coordinates.col,
            output_tile,
            tail,
            str(full_chunks * chunk),
            first=False,
            suffix="_tail",
            lhs_transposed=lhs_transposed,
            rhs_transposed=rhs_transposed,
        )
    writer.line(
        indent,
        f"{io.output_argument} = pl.store(accumulator, "
        f"[{coordinates.row}, {coordinates.col}], "
        f"{io.output_argument})",
    )
    emit_return(writer, io)
    return writer.render()


def _emit_split_cube_dag(
    context: EmissionContext,
    program_name: str,
    plan: CubeKernelPlan,
) -> str:
    """Emit a cube DAG whose unique sink uses a solver-selected split-K protocol."""

    graph = context.graph
    lowered = context.lowered
    step = context.step
    io = context.interface
    sinks = tuple(matmul for matmul in plan.matmuls if matmul.is_sink)
    if len(sinks) != 1:
        raise SourceEmissionError("split-K source requires exactly one sink matmul")
    matmul = sinks[0]
    if len(io.output_values) != 1:
        raise SourceEmissionError("split-K source requires exactly one output")
    if matmul.accumulator_dtype != "fp32" or matmul.storage_dtype != "fp32":
        raise SourceEmissionError(
            "split-K source currently requires FP32 accumulation and storage"
        )
    if plan.spatial_policy is not CubeSpatialPolicy.UNIFORM:
        raise SourceEmissionError("split-K source requires a uniform spatial grid")
    if (
        plan.m_partition.big != plan.m_partition.small
        or plan.n_partition.big != plan.n_partition.small
    ):
        raise SourceEmissionError("split-K source requires uniform region extents")
    if not plan.matmuls or plan.matmuls[-1] is not matmul:
        raise SourceEmissionError("split-K sink must be the final cube request")
    split_requests = tuple(
        request
        for request in plan.matmuls
        if request.effective_contraction != request.contraction
    )
    if split_requests != (matmul,):
        raise SourceEmissionError(
            "split-K source requires exactly one split accumulator at the sink"
        )
    if matmul.effective_contraction * plan.split_k != matmul.contraction:
        raise SourceEmissionError("split-K sink shares do not cover its contraction")
    if not matmul.final_drain.atomic or any(
        request.final_drain.atomic for request in plan.matmuls if request is not matmul
    ):
        raise SourceEmissionError(
            "split-K source requires one atomic sink drain and serial upstream drains"
        )
    if (
        matmul.output.height_binding is CubeAxisBinding.PARALLEL_K
        or matmul.output.width_binding is CubeAxisBinding.PARALLEL_K
    ):
        raise SourceEmissionError("split-K sink output cannot be partitioned along K")
    if plan.work_units != plan.spatial_tiles * plan.split_k:
        raise SourceEmissionError("split-K work count differs from grid times split")
    validate_grid(step, plan.spatial_tiles, plan.m_partition, plan.n_partition)
    output_value = io.output_values[0]
    if (
        solver_tensor_for_value(lowered, io.output_allocation_owners[output_value])
        != matmul.output.tensor
    ):
        raise SourceEmissionError("split-K sink differs from the region output")
    output_rows, output_cols = static_shape(
        graph.value_map()[output_value], field="split-K output"
    )
    validate_partition_extent(plan.m_partition, output_rows, "cube.m_partition")
    validate_partition_extent(plan.n_partition, output_cols, "cube.n_partition")
    for request in plan.matmuls:
        _validate_lowered_l0_capacity(context, request)

    writer = program_preamble(program_name, io, graph)
    if plan.split_merge_policy is CubeSplitMergePolicy.FIRST_PARTIAL_THEN_ATOMIC:
        descriptor = plan.first_partial_then_atomic
        if (
            not descriptor.present
            or descriptor.first_work_units != plan.spatial_tiles
            or descriptor.atomic_work_units != plan.spatial_tiles * (plan.split_k - 1)
            or plan.aiv_zero_seed_then_atomic.present
        ):
            raise SourceEmissionError(
                "FirstPartialThenAtomic source descriptor is inconsistent"
            )
        writer.line(
            2,
            f"with pl.spmd({descriptor.first_work_units}, "
            f"name_hint={context.region_id + '_cube_first'!r}) as first_partial_task:",
        )
        writer.line(3, "region_index = pl.tile.get_block_idx()")
        _emit_cube_dag_body(
            writer,
            3,
            context,
            plan,
            split_index="0",
            atomic_sink=False,
            split_sink=matmul,
        )
        writer.line(
            2,
            f"with pl.spmd({descriptor.atomic_work_units}, "
            f"name_hint={context.region_id + '_cube_atomic_rest'!r}, "
            "deps=[first_partial_task]) as atomic_rest_task:",
        )
        writer.line(3, "split_work_index = pl.tile.get_block_idx()")
        writer.line(3, f"region_index = split_work_index // {plan.split_k - 1}")
        writer.line(3, f"split_index = split_work_index % {plan.split_k - 1} + 1")
        _emit_cube_dag_body(
            writer,
            3,
            context,
            plan,
            split_index="split_index",
            atomic_sink=True,
            split_sink=matmul,
        )
    elif plan.split_merge_policy is CubeSplitMergePolicy.AIV_ZERO_SEED_THEN_ATOMIC:
        descriptor = plan.aiv_zero_seed_then_atomic
        if (
            not descriptor.present
            or descriptor.seed_work_units != plan.spatial_tiles
            or descriptor.atomic_work_units != plan.work_units
            or descriptor.seed_bytes != plan.spatial_tiles * matmul.final_drain.bytes
            or plan.first_partial_then_atomic.present
        ):
            raise SourceEmissionError(
                "AivZeroSeedThenAtomic source descriptor is inconsistent"
            )
        writer.line(
            2,
            f"with pl.spmd({descriptor.seed_work_units}, "
            f"name_hint={context.region_id + '_cube_zero_seed'!r}) as zero_seed_task:",
        )
        writer.line(3, "region_index = pl.tile.get_block_idx()")
        _emit_split_cube_zero_seed(writer, 3, context, plan, matmul)
        writer.line(
            2,
            f"with pl.spmd({descriptor.atomic_work_units}, "
            f"name_hint={context.region_id + '_cube_atomic_all'!r}, "
            "deps=[zero_seed_task]) as atomic_all_task:",
        )
        writer.line(3, "split_work_index = pl.tile.get_block_idx()")
        writer.line(3, f"region_index = split_work_index // {plan.split_k}")
        writer.line(3, f"split_index = split_work_index % {plan.split_k}")
        _emit_cube_dag_body(
            writer,
            3,
            context,
            plan,
            split_index="split_index",
            atomic_sink=True,
            split_sink=matmul,
        )
    else:
        raise SourceEmissionError("split-K plan carries no supported merge policy")
    emit_return(writer, io)
    return writer.render()


def _emit_split_cube_zero_seed(
    writer: SourceWriter,
    indent: int,
    context: EmissionContext,
    plan: CubeKernelPlan,
    matmul: CubeMatmulPlan,
) -> None:
    coordinates = emit_partition_indices(
        writer, indent, plan.m_partition, plan.n_partition
    )
    output_row, output_col = _cube_region_offsets(
        matmul.output.height_binding,
        matmul.output.width_binding,
        coordinates.row,
        coordinates.col,
    )
    output_argument = context.interface.output_argument
    tiles_m, tiles_n = matmul.output_grid
    for tile_m in range(tiles_m):
        local_row = tile_m * matmul.output_tile[0]
        tile_height = min(matmul.output_tile[0], matmul.output.height - local_row)
        for tile_n in range(tiles_n):
            local_col = tile_n * matmul.output_tile[1]
            tile_width = min(matmul.output_tile[1], matmul.output.width - local_col)
            writer.line(
                indent,
                f"{output_argument} = pl.assemble({output_argument}, "
                f"pl.full([{tile_height}, {tile_width}], dtype=pl.FP32, value=0.0), "
                f"[{_add_offset(output_row, local_row)}, "
                f"{_add_offset(output_col, local_col)}])",
            )


def _emit_split_cube_output_tile(  # noqa: PLR0913
    writer: SourceWriter,
    indent: int,
    matmul: CubeMatmulPlan,
    variant: CubeOutputTileVariant,
    lhs: str,
    rhs: str,
    lhs_row: str,
    lhs_col: str,
    rhs_row: str,
    rhs_col: str,
    output_row: int,
    output_col: int,
    output_height: int,
    output_width: int,
    tile_index: int,
) -> str:
    """Replay the serialized child-K loop inside every outer L1 window."""

    outer = matmul.k_loop
    if (
        outer.chunk <= 0
        or outer.full_chunks <= 0
        or outer.full_chunks * outer.chunk + outer.tail != matmul.effective_contraction
        or outer.tail >= outer.chunk
        or outer.pipeline_stages not in {1, 2}
    ):
        raise SourceEmissionError(
            f"split-K request {matmul.instance} outer K loop is stale"
        )
    prefix = f"matmul_{matmul.instance}_tile_{tile_index}"
    accumulator = f"{prefix}_accumulator"

    def emit_outer_window(
        level: int,
        suffix: str,
        outer_offset: str,
        outer_extent: int,
        child,
        *,
        first_outer: bool,
    ) -> None:
        child_loop = child.k_loop
        if (
            child_loop.chunk <= 0
            or child_loop.full_chunks <= 0
            or child_loop.full_chunks * child_loop.chunk + child_loop.tail
            != outer_extent
            or child_loop.tail >= child_loop.chunk
            or child_loop.pipeline_stages not in {1, 2}
        ):
            raise SourceEmissionError(
                f"split-K request {matmul.instance} child K loop is stale"
            )

        def emit_child(child_index: int, child_extent: int) -> None:
            child_offset = child_index * child_loop.chunk
            k_offset = _add_expression(outer_offset, str(child_offset))
            child_suffix = f"{suffix}_{child_index}"
            writer.line(
                level,
                f"{prefix}_lhs_{child_suffix} = pl.slice({lhs}, "
                f"[{output_height}, {child_extent}], "
                f"[{_add_offset(lhs_row, output_row)}, "
                f"{_add_expression(lhs_col, k_offset)}])",
            )
            writer.line(
                level,
                f"{prefix}_rhs_{child_suffix} = pl.slice({rhs}, "
                f"[{child_extent}, {output_width}], "
                f"[{_add_expression(rhs_row, k_offset)}, "
                f"{_add_offset(rhs_col, output_col)}])",
            )
            if first_outer and child_index == 0:
                writer.line(
                    level,
                    f"{accumulator} = pl.matmul({prefix}_lhs_{child_suffix}, "
                    f"{prefix}_rhs_{child_suffix}, out_dtype=pl.FP32)",
                )
            else:
                writer.line(
                    level,
                    f"{accumulator} = pl.matmul_acc({accumulator}, "
                    f"{prefix}_lhs_{child_suffix}, {prefix}_rhs_{child_suffix})",
                )

        for child_index in range(child_loop.full_chunks):
            emit_child(child_index, child_loop.chunk)
        if child_loop.tail:
            emit_child(child_loop.full_chunks, child_loop.tail)

    emit_outer_window(
        indent,
        "init",
        "0",
        outer.chunk,
        variant.l0_init,
        first_outer=True,
    )
    if outer.full_chunks > 1:
        iterator = "pl.pipeline" if outer.pipeline_stages > 1 else "pl.range"
        stage = f", stage={outer.pipeline_stages}" if outer.pipeline_stages > 1 else ""
        writer.line(
            indent,
            f"for {prefix}_outer_k in {iterator}(1, {outer.full_chunks}{stage}):",
        )
        emit_outer_window(
            indent + 1,
            "rolled",
            f"{prefix}_outer_k * {outer.chunk}",
            outer.chunk,
            variant.l0_rolled,
            first_outer=False,
        )
    if outer.tail:
        emit_outer_window(
            indent,
            "tail",
            str(outer.full_chunks * outer.chunk),
            outer.tail,
            variant.l0_tail,
            first_outer=False,
        )
    return accumulator


def _emit_full_window_cube_dag(
    context: EmissionContext,
    program_name: str,
    plan: CubeKernelPlan,
) -> str:
    """Emit a dependency-ordered, non-split cube DAG for one spatial region.

    The source spells the solver-owned outer schedule explicitly: output-tile
    subdivision, K-window accumulation, L1 intermediates, retained boundary
    panels, resident boundary lifetimes, and final drains. PyPTO chooses only
    each child L0 matmul realization.
    """

    writer = program_header(
        program_name,
        context.interface,
        context.graph,
        plan.work_units,
        kernel_name_hint=context.region_id + "_cube",
    )
    _emit_cube_dag_body(
        writer,
        3,
        context,
        plan,
        split_index="0",
        atomic_sink=False,
        split_sink=None,
    )
    emit_return(writer, context.interface)
    return writer.render()


def _emit_cube_dag_body(  # noqa: PLR0913
    writer: SourceWriter,
    indent: int,
    context: EmissionContext,
    plan: CubeKernelPlan,
    *,
    split_index: str,
    atomic_sink: bool,
    split_sink: CubeMatmulPlan | None,
) -> None:
    """Replay one spatial region and one optional split share of a cube DAG."""

    graph = context.graph
    lowered = context.lowered
    step = context.step
    io = context.interface
    if any(
        resident.use_count < 2 or resident.first_use >= resident.last_use
        for resident in plan.resident_boundaries
    ):
        raise SourceEmissionError(
            "cube resident-boundary lifetime must span at least two requests"
        )
    if tuple(matmul.op for matmul in plan.matmuls) != plan.execution_order:
        raise SourceEmissionError(
            "cube matmul requests do not preserve the selected execution order"
        )
    first_occurrences = tuple(dict.fromkeys(plan.execution_order))
    if first_occurrences != step.op_order:
        raise SourceEmissionError(
            "cube plan first-use order differs from the kernel-step order"
        )
    if tuple(step.solver_ops) != tuple(sorted(step.solver_ops)):
        # Solver identities need not be execution ordered, but they are the
        # complete membership set and therefore must remain canonical.
        raise SourceEmissionError("cube solver-op membership is not canonical")
    if step.sequential_tiles is None or len(step.sequential_tiles) != len(
        step.solver_ops
    ):
        raise SourceEmissionError("cube DAG omits per-operation sequential tiles")
    sequential_by_op = dict(zip(step.op_order, step.sequential_tiles, strict=True))

    m_partition = plan.m_partition
    n_partition = plan.n_partition
    validate_grid(step, plan.spatial_tiles, m_partition, n_partition)
    if m_partition.big != m_partition.small or n_partition.big != n_partition.small:
        raise SourceEmissionError(
            "full-window cube DAG source currently requires uniform spatial regions"
        )

    coordinates = emit_partition_indices(writer, indent, m_partition, n_partition)
    graph_ops = graph.op_map()
    local: dict[int, str] = {}
    producer_by_tensor: dict[int, int] = {}
    resident_values: dict[int, str] = {}
    stored_outputs: set[str] = set()

    for request_index, matmul in enumerate(plan.matmuls):
        operation = lowered.operation(matmul.op)
        if len(operation.inputs) != 2 or len(operation.outputs) != 1:
            raise SourceEmissionError(
                f"cube operation {matmul.op} is not a binary single-result matmul"
            )
        if operation.inputs != (matmul.lhs.tensor, matmul.rhs.tensor):
            raise SourceEmissionError(
                f"cube request {matmul.instance} operand identities are stale"
            )
        if operation.outputs != (matmul.output.tensor,):
            raise SourceEmissionError(
                f"cube request {matmul.instance} output identity is stale"
            )
        graph_op = graph_ops[operation.graph_op_id]
        if (
            graph_op.kind != "matmul"
            or graph_op.attributes.get("lhs_transposed")
            or graph_op.attributes.get("rhs_transposed")
        ):
            raise SourceEmissionError(
                "full-window cube DAG source requires non-transposed matmuls"
            )
        if matmul.instance != request_index:
            raise SourceEmissionError(
                "cube request instances are not dense and ordered"
            )
        if sequential_by_op[matmul.op] != matmul.k_loop.l1_window_k:
            raise SourceEmissionError(
                f"cube operation {matmul.op} sequential tile differs from its L1 window"
            )
        if not matmul.output_variants:
            raise SourceEmissionError("cube DAG request omits its output-tile variants")
        if matmul.accumulator_dtype != "fp32":
            raise SourceEmissionError("cube DAG source requires FP32 accumulation")
        if matmul.storage_dtype not in {"fp32", "fp16", "bf16"}:
            raise SourceEmissionError(
                f"cube DAG storage dtype {matmul.storage_dtype!r} is unsupported"
            )
        _validate_lowered_l0_capacity(context, matmul)

        lhs, lhs_row, lhs_col = _cube_dag_operand(
            writer,
            indent,
            context,
            matmul.lhs,
            matmul.lhs_producer,
            matmul.lhs_resident_boundary,
            matmul.instance,
            "lhs",
            coordinates.row,
            coordinates.col,
            local,
            producer_by_tensor,
            resident_values,
            split_index,
        )
        rhs, rhs_row, rhs_col = _cube_dag_operand(
            writer,
            indent,
            context,
            matmul.rhs,
            matmul.rhs_producer,
            matmul.rhs_resident_boundary,
            matmul.instance,
            "rhs",
            coordinates.row,
            coordinates.col,
            local,
            producer_by_tensor,
            resident_values,
            split_index,
        )

        if matmul.retained_panels.lhs:
            lhs = _emit_retained_panel(
                writer,
                indent,
                lhs,
                matmul.lhs,
                lhs_row,
                lhs_col,
                matmul.instance,
                "lhs",
            )
            lhs_row = lhs_col = "0"
        if matmul.retained_panels.rhs:
            rhs = _emit_retained_panel(
                writer,
                indent,
                rhs,
                matmul.rhs,
                rhs_row,
                rhs_col,
                matmul.instance,
                "rhs",
            )
            rhs_row = rhs_col = "0"

        output_row, output_col = _cube_tensor_region_offsets(
            matmul.output,
            coordinates.row,
            coordinates.col,
            split_index,
        )
        output_targets = {
            output_value: io.output_arguments[output_value]
            for output_value in io.output_values
            if solver_tensor_for_value(
                lowered, io.output_allocation_owners[output_value]
            )
            == matmul.output.tensor
        }
        if matmul.is_sink != bool(output_targets):
            raise SourceEmissionError(
                f"cube request {matmul.instance} sink identity differs from the region ABI"
            )
        state = ""
        if not matmul.is_sink:
            state = f"matmul_{matmul.instance}_l1"
            writer.line(
                indent,
                f"{state} = pl.create_l1("
                f"[{matmul.output.height}, {matmul.output.width}], "
                f"dtype={pypto_dtype(matmul.storage_dtype)})",
            )

        variants = {variant.shape: variant for variant in matmul.output_variants}
        variant_counts = {variant.shape: 0 for variant in matmul.output_variants}
        output_tiles_m, output_tiles_n = matmul.output_grid
        if (
            output_tiles_m <= 0
            or output_tiles_n <= 0
            or output_tiles_m != _ceil_div(matmul.output.height, matmul.output_tile[0])
            or output_tiles_n != _ceil_div(matmul.output.width, matmul.output_tile[1])
        ):
            raise SourceEmissionError(
                f"cube request {matmul.instance} output-tile grid is stale"
            )
        for tile_m in range(output_tiles_m):
            local_row = tile_m * matmul.output_tile[0]
            tile_height = min(matmul.output_tile[0], matmul.output.height - local_row)
            for tile_n in range(output_tiles_n):
                local_col = tile_n * matmul.output_tile[1]
                tile_width = min(matmul.output_tile[1], matmul.output.width - local_col)
                shape = (tile_height, tile_width)
                variant = variants.get(shape)
                if variant is None:
                    raise SourceEmissionError(
                        f"cube request {matmul.instance} omits output variant {shape}"
                    )
                variant_counts[shape] += 1
                tile_index = tile_m * output_tiles_n + tile_n
                if matmul is split_sink:
                    accumulator = _emit_split_cube_output_tile(
                        writer,
                        indent,
                        matmul,
                        variant,
                        lhs,
                        rhs,
                        lhs_row,
                        lhs_col,
                        rhs_row,
                        rhs_col,
                        local_row,
                        local_col,
                        tile_height,
                        tile_width,
                        tile_index,
                    )
                else:
                    accumulator = _emit_cube_dag_output_tile(
                        writer,
                        indent,
                        matmul,
                        lhs,
                        rhs,
                        lhs_row,
                        lhs_col,
                        rhs_row,
                        rhs_col,
                        local_row,
                        local_col,
                        tile_height,
                        tile_width,
                        tile_index,
                    )
                if matmul.is_sink:
                    atomic_suffix = ", atomic=pl.AtomicType.Add" if atomic_sink else ""
                    for output_value, output_argument in output_targets.items():
                        writer.line(
                            indent,
                            f"{output_argument} = pl.assemble("
                            f"{output_argument}, {accumulator}, "
                            f"[{_add_offset(output_row, local_row)}, "
                            f"{_add_offset(output_col, local_col)}]"
                            f"{atomic_suffix})",
                        )
                        stored_outputs.add(output_value)
                else:
                    next_state = f"matmul_{matmul.instance}_l1_{tile_index}"
                    writer.line(
                        indent,
                        f"{next_state} = pl.assemble({state}, {accumulator}, "
                        f"[{local_row}, {local_col}])",
                    )
                    state = next_state
        for shape, variant in variants.items():
            if variant_counts[shape] != variant.count:
                raise SourceEmissionError(
                    f"cube request {matmul.instance} output variant {shape} count is stale"
                )

        if not matmul.is_sink:
            local[matmul.output.tensor] = state
        producer_by_tensor[matmul.output.tensor] = matmul.instance

    if set(resident_values) != set(range(len(plan.resident_boundaries))):
        raise SourceEmissionError(
            "cube DAG did not materialize every resident boundary"
        )

    missing = set(io.output_values) - stored_outputs
    if missing:
        raise SourceEmissionError(
            "cube DAG does not drain region outputs " + ", ".join(sorted(missing))
        )


def _cube_dag_operand(  # noqa: PLR0913
    writer: SourceWriter,
    indent: int,
    context: EmissionContext,
    region,
    declared_producer: int,
    resident_boundary: int,
    consumer_instance: int,
    role: str,
    spatial_row: str,
    spatial_col: str,
    local: Mapping[int, str],
    producer_by_tensor: Mapping[int, int],
    resident_values: dict[int, str],
    split_index: str,
) -> tuple[str, str, str]:
    if region.tensor in local:
        actual_producer = producer_by_tensor[region.tensor]
        if declared_producer != actual_producer or actual_producer >= consumer_instance:
            raise SourceEmissionError(
                f"cube {role} producer does not precede its consumer"
            )
        if resident_boundary != -1:
            raise SourceEmissionError(
                f"cube {role} cannot be both produced and a resident boundary"
            )
        return local[region.tensor], "0", "0"
    if declared_producer != -1:
        raise SourceEmissionError(f"cube {role} producer result is unavailable")
    tensor = context.lowered.tensor(region.tensor)
    argument = context.interface.input_arguments.get(tensor.value_id)
    if argument is None:
        raise SourceEmissionError(
            f"cube external {role} tensor {tensor.value_id!r} is not a region input"
        )
    row_offset, col_offset = _cube_tensor_region_offsets(
        region,
        spatial_row,
        spatial_col,
        split_index,
    )
    if resident_boundary < 0:
        return argument, row_offset, col_offset
    plan = context.step.plan
    if not isinstance(plan, CubeKernelPlan):
        raise SourceEmissionError("cube operand does not carry a cube plan")
    if resident_boundary >= len(plan.resident_boundaries):
        raise SourceEmissionError(f"cube {role} resident boundary is out of range")
    resident = plan.resident_boundaries[resident_boundary]
    if resident_boundary not in resident_values:
        if resident.first_use != consumer_instance:
            raise SourceEmissionError(
                f"cube {role} resident boundary is used before materialization"
            )
        name = f"resident_{resident_boundary}_{role}"
        writer.line(
            indent,
            f"{name} = pl.slice({argument}, [{region.height}, {region.width}], "
            f"[{row_offset}, {col_offset}])",
        )
        resident_values[resident_boundary] = name
    elif resident.first_use == consumer_instance:
        raise SourceEmissionError(
            f"cube resident boundary {resident_boundary} is materialized twice"
        )
    return resident_values[resident_boundary], "0", "0"


def _cube_tensor_region_offsets(
    region: CubeTensorRegionPlan,
    spatial_row: str,
    spatial_col: str,
    split_index: str,
) -> tuple[str, str]:
    """Return offsets for one concrete per-spatial/per-split tensor region."""

    parallel_axes = sum(
        binding is CubeAxisBinding.PARALLEL_K
        for binding in (region.height_binding, region.width_binding)
    )
    if parallel_axes > 1:
        raise SourceEmissionError(
            "cube tensor region cannot bind both axes to parallel K"
        )
    parallel_extent = (
        region.height
        if region.height_binding is CubeAxisBinding.PARALLEL_K
        else region.width
    )
    parallel_offset = (
        "0" if parallel_axes == 0 else f"{split_index} * {parallel_extent}"
    )
    return _cube_region_offsets(
        region.height_binding,
        region.width_binding,
        spatial_row,
        spatial_col,
        parallel_offset,
    )


def _emit_retained_panel(
    writer: SourceWriter,
    indent: int,
    source: str,
    region: CubeTensorRegionPlan,
    row_offset: str,
    col_offset: str,
    instance: int,
    role: str,
) -> str:
    name = f"matmul_{instance}_{role}_retained"
    writer.line(
        indent,
        f"{name} = pl.slice({source}, [{region.height}, {region.width}], "
        f"[{row_offset}, {col_offset}])",
    )
    return name


def _emit_cube_dag_output_tile(  # noqa: PLR0913
    writer: SourceWriter,
    indent: int,
    matmul: CubeMatmulPlan,
    lhs: str,
    rhs: str,
    lhs_row: str,
    lhs_col: str,
    rhs_row: str,
    rhs_col: str,
    output_row: int,
    output_col: int,
    output_height: int,
    output_width: int,
    tile_index: int,
) -> str:
    loop = matmul.k_loop
    if (
        loop.chunk <= 0
        or loop.full_chunks <= 0
        or loop.full_chunks * loop.chunk + loop.tail != matmul.effective_contraction
        or loop.tail >= loop.chunk
        or loop.pipeline_stages not in {1, 2}
    ):
        raise SourceEmissionError(
            f"cube request {matmul.instance} K-window descriptor is stale"
        )
    prefix = f"matmul_{matmul.instance}_tile_{tile_index}"

    def emit_window(level: int, suffix: str, k_offset: str, k_extent: int) -> None:
        writer.line(
            level,
            f"{prefix}_lhs_{suffix} = pl.slice({lhs}, "
            f"[{output_height}, {k_extent}], "
            f"[{_add_offset(lhs_row, output_row)}, "
            f"{_add_expression(lhs_col, k_offset)}])",
        )
        writer.line(
            level,
            f"{prefix}_rhs_{suffix} = pl.slice({rhs}, "
            f"[{k_extent}, {output_width}], "
            f"[{_add_expression(rhs_row, k_offset)}, "
            f"{_add_offset(rhs_col, output_col)}])",
        )

    emit_window(indent, "init", "0", loop.chunk)
    accumulator = f"{prefix}_accumulator"
    writer.line(
        indent,
        f"{accumulator} = pl.matmul({prefix}_lhs_init, {prefix}_rhs_init, "
        "out_dtype=pl.FP32)",
    )
    if loop.full_chunks > 1:
        iterator = "pl.pipeline" if loop.pipeline_stages > 1 else "pl.range"
        stage = f", stage={loop.pipeline_stages}" if loop.pipeline_stages > 1 else ""
        writer.line(
            indent,
            f"for {prefix}_k in {iterator}(1, {loop.full_chunks}{stage}):",
        )
        emit_window(
            indent + 1,
            "rolled",
            f"{prefix}_k * {loop.chunk}",
            loop.chunk,
        )
        writer.line(
            indent + 1,
            f"{accumulator} = pl.matmul_acc({accumulator}, "
            f"{prefix}_lhs_rolled, {prefix}_rhs_rolled)",
        )
    if loop.tail:
        emit_window(
            indent,
            "tail",
            str(loop.full_chunks * loop.chunk),
            loop.tail,
        )
        writer.line(
            indent,
            f"{accumulator} = pl.matmul_acc({accumulator}, "
            f"{prefix}_lhs_tail, {prefix}_rhs_tail)",
        )
    return accumulator


def _cube_region_offsets(
    height_binding: CubeAxisBinding,
    width_binding: CubeAxisBinding,
    spatial_row: str,
    spatial_col: str,
    parallel_k: str = "0",
) -> tuple[str, str]:
    def coordinate(binding: CubeAxisBinding) -> str:
        if binding is CubeAxisBinding.SPATIAL_M:
            return spatial_row
        if binding is CubeAxisBinding.SPATIAL_N:
            return spatial_col
        if binding is CubeAxisBinding.PARALLEL_K:
            return parallel_k
        return "0"

    return coordinate(height_binding), coordinate(width_binding)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _add_offset(base: str, offset: int) -> str:
    if offset == 0:
        return base
    if base == "0":
        return str(offset)
    return f"{base} + {offset}"


def _add_expression(base: str, offset: str) -> str:
    if offset == "0":
        return base
    if base == "0":
        return offset
    return f"{base} + {offset}"


def _emit_cube_window(
    writer: SourceWriter,
    indent: int,
    lhs: str,
    rhs: str,
    row_offset: str,
    col_offset: str,
    output_tile: list[int],
    k_extent: int,
    k_offset: str,
    *,
    first: bool,
    lhs_transposed: bool,
    rhs_transposed: bool,
    suffix: str = "",
) -> None:
    m_extent, n_extent = output_tile
    lhs_shape = (k_extent, m_extent) if lhs_transposed else (m_extent, k_extent)
    lhs_offset = (k_offset, row_offset) if lhs_transposed else (row_offset, k_offset)
    rhs_shape = (n_extent, k_extent) if rhs_transposed else (k_extent, n_extent)
    rhs_offset = (col_offset, k_offset) if rhs_transposed else (k_offset, col_offset)
    writer.line(
        indent,
        f"lhs_mat_natural{suffix} = pl.tile.load({lhs}, "
        f"[{lhs_offset[0]}, {lhs_offset[1]}], "
        f"[{lhs_shape[0]}, {lhs_shape[1]}], target_memory=pl.Mem.Mat)",
    )
    writer.line(
        indent,
        f"rhs_mat_natural{suffix} = pl.tile.load({rhs}, "
        f"[{rhs_offset[0]}, {rhs_offset[1]}], "
        f"[{rhs_shape[0]}, {rhs_shape[1]}], target_memory=pl.Mem.Mat)",
    )
    lhs_mat = f"lhs_mat_natural{suffix}"
    rhs_mat = f"rhs_mat_natural{suffix}"
    if lhs_transposed:
        lhs_mat = f"lhs_mat{suffix}"
        writer.line(
            indent,
            f"{lhs_mat} = pl.tile.transpose_view(lhs_mat_natural{suffix})",
        )
    if rhs_transposed:
        rhs_mat = f"rhs_mat{suffix}"
        writer.line(
            indent,
            f"{rhs_mat} = pl.tile.transpose_view(rhs_mat_natural{suffix})",
        )
    writer.line(
        indent,
        f"lhs_left{suffix} = pl.tile.move({lhs_mat}, target_memory=pl.Mem.Left)",
    )
    writer.line(
        indent,
        f"rhs_right{suffix} = pl.tile.move({rhs_mat}, target_memory=pl.Mem.Right)",
    )
    if first:
        writer.line(
            indent,
            f"accumulator = pl.tile.matmul(lhs_left{suffix}, rhs_right{suffix})",
        )
    else:
        writer.line(
            indent,
            f"accumulator = pl.tile.matmul_acc(accumulator, lhs_left{suffix}, rhs_right{suffix})",
        )


def _argument_for_cube_tensor(context: EmissionContext, tensor: int) -> str:
    descriptor = context.lowered.tensor(tensor)
    value_id = (
        descriptor.alias_of if descriptor.alias_of is not None else descriptor.value_id
    )
    try:
        return context.interface.input_arguments[value_id]
    except KeyError as error:
        raise SourceEmissionError(
            f"cube external tensor {tensor} ({value_id}) is not a region input"
        ) from error


def _validate_l0_variant(
    matmul: CubeMatmulPlan,
    output_tile: list[int],
    chunk: int,
    tail: int,
    full_chunks: int,
) -> None:
    variants = matmul.output_variants
    if len(variants) != 1:
        raise SourceEmissionError("cube source v1 requires one output-shape variant")
    variant = variants[0]
    if variant.count != 1 or variant.shape != tuple(output_tile):
        raise SourceEmissionError("cube output variant does not match one region tile")
    expected = (output_tile[0], output_tile[1], chunk)
    if variant.l0_init.tile != expected:
        raise SourceEmissionError(
            "cube initial L0 tile differs from the emitted K window"
        )
    if full_chunks > 1:
        if variant.l0_rolled is None or variant.l0_rolled.tile != expected:
            raise SourceEmissionError(
                "cube rolled L0 tile differs from the emitted K window"
            )
    if tail:
        expected_tail = (output_tile[0], output_tile[1], tail)
        if variant.l0_tail is None or variant.l0_tail.tile != expected_tail:
            raise SourceEmissionError("cube tail L0 tile differs from the emitted tail")


def _validate_lowered_l0_capacity(
    context: EmissionContext,
    matmul: CubeMatmulPlan,
) -> None:
    """Reject plans whose outer K pipeline over-rotates L0 operands."""

    raw_config = context.problem.get("l0_matmul_config")
    if not isinstance(raw_config, Mapping):
        raise SourceEmissionError("cube source requires an L0 matmul target profile")
    l0a_capacity = raw_config.get("l0a_bytes")
    l0b_capacity = raw_config.get("l0b_bytes")
    box_align_m = raw_config.get("box_align_m", 1)
    box_align_n = raw_config.get("box_align_n", 1)
    if (
        not isinstance(l0a_capacity, int)
        or l0a_capacity <= 0
        or not isinstance(l0b_capacity, int)
        or l0b_capacity <= 0
        or not isinstance(box_align_m, int)
        or box_align_m <= 0
        or not isinstance(box_align_n, int)
        or box_align_n <= 0
    ):
        raise SourceEmissionError("cube source received invalid L0 target capacities")

    lhs_bytes = context.lowered.tensor(matmul.lhs.tensor).byte_width
    rhs_bytes = context.lowered.tensor(matmul.rhs.tensor).byte_width

    def align_up(value: int, alignment: int) -> int:
        return (value + alignment - 1) // alignment * alignment

    def validate_child(label: str, child: L0MatmulPlan, outer_depth: int) -> None:
        tile = child.tile
        depths = child.buffer_depths
        physical_m = align_up(tile[0], box_align_m)
        physical_n = align_up(tile[1], box_align_n)
        l0a_bytes = physical_m * tile[2] * lhs_bytes * max(depths[0], outer_depth)
        l0b_bytes = tile[2] * physical_n * rhs_bytes * max(depths[1], outer_depth)
        if l0a_bytes > l0a_capacity or l0b_bytes > l0b_capacity:
            raise SourceEmissionError(
                f"{label} exceeds lowered L0 operand capacity: "
                f"L0A {l0a_bytes}/{l0a_capacity} bytes, "
                f"L0B {l0b_bytes}/{l0b_capacity} bytes"
            )

    for variant in matmul.output_variants:
        label = f"cube output variant {variant.shape}"
        validate_child(f"{label} initial L0 plan", variant.l0_init, 1)
        if variant.l0_rolled is not None:
            validate_child(
                f"{label} outer K pipeline",
                variant.l0_rolled,
                matmul.k_loop.pipeline_stages,
            )
        if variant.l0_tail is not None:
            validate_child(f"{label} tail L0 plan", variant.l0_tail, 1)
