"""Strict decoding of a PTO-Fusebox solver solution."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, Literal, TypeVar, overload

from ..ir import SOLUTION_SCHEMA
from ..lowered import LoweredRegion, lowered_region
from ..solver import RegionSolveResult
from .schema import (
    AxisPartition,
    CubeAivZeroSeedThenAtomicPlan,
    CubeAxisBinding,
    CubeFinalDrainPlan,
    CubeFirstPartialThenAtomicPlan,
    CubeKLoopPlan,
    CubeKernelPlan,
    CubeMatmulPlan,
    CubeOperandRole,
    CubeOutputTileVariant,
    CubeResidentBoundaryPlan,
    CubeRetainedPanelPlan,
    CubeSpatialPolicy,
    CubeSplitMergePolicy,
    CubeTensorRegionPlan,
    KernelKind,
    KernelStep,
    L0KLoopPlan,
    L0MatmulPlan,
    L0OutputTarget,
    L0PhaseCostPlan,
    L0Stationarity,
    LaunchPlan,
    MixedAlgorithm,
    MixedCrossCoreProtocol,
    MixedFeatureRoundTripPlan,
    MixedEngine,
    MixedFifoPlan,
    MixedKernelPlan,
    MixedPipelineAxis,
    MixedPipelineMode,
    MixedStagePlan,
    MixedTransferDirection,
    MixedTransferPlan,
    MixedVectorSplit,
    ScheduleContractError,
    ScheduledRegion,
    VectorCoordinateTransform,
    VectorGeneratedPhaseWorkPlan,
    VectorInputLifetimePlan,
    VectorInputUsePlan,
    VectorKernelPlan,
    VectorLoopPlan,
    VectorP4WorkPlan,
    VectorP4RecipePlan,
    VectorP4SubstitutionPlan,
    VectorPhasePlan,
    VectorPhysicalFramePlan,
    VectorPrimitiveWorkPlan,
    VectorReductionSeedPlan,
    VectorReductionSplitKind,
    VectorReductionSplitPlan,
    VectorReplayPassKind,
    VectorReplayPassPlan,
    VectorReplayPhase,
    VectorSpatialPolicy,
    VectorSerialPhasePlan,
    VectorStreamKind,
    VectorTensorFramePlan,
    VectorWorkspaceFramePlan,
)

EnumT = TypeVar("EnumT", bound=Enum)


def scheduled_region(result: RegionSolveResult) -> ScheduledRegion:
    """Decode and validate one solved region without replanning it."""

    if result.status != "solved" or result.solution is None or result.problem is None:
        raise ScheduleContractError(f"region {result.region.id} is not solved")
    solution = result.solution
    if solution.get("schema_version") != SOLUTION_SCHEMA:
        raise ScheduleContractError(
            f"solution schema must be {SOLUTION_SCHEMA!r}, got "
            f"{solution.get('schema_version')!r}"
        )
    _expect_keys(solution, required={"schema_version", "steps"}, field="solution")
    raw_steps = _sequence(solution.get("steps"), "steps")
    if not raw_steps:
        raise ScheduleContractError("solution contains no kernel steps")

    lowered = lowered_region(result)
    graph_mapping = result.solver_op_to_graph
    steps: list[KernelStep] = []
    covered: set[int] = set()
    for index, raw_step in enumerate(raw_steps):
        item = _mapping(raw_step, f"steps[{index}]")
        _expect_keys(
            item,
            required={
                "kind",
                "ops",
                "op_order",
                "launch",
                "sequential_tiles",
                "plan",
                "latency_cycles",
            },
            field=f"steps[{index}]",
        )
        solver_ops = tuple(
            _bounded_int(op, f"steps[{index}].ops", len(graph_mapping))
            for op in _sequence(item.get("ops"), f"steps[{index}].ops")
        )
        if not solver_ops or len(set(solver_ops)) != len(solver_ops):
            raise ScheduleContractError(
                f"steps[{index}].ops must contain distinct operations"
            )
        if covered.intersection(solver_ops):
            raise ScheduleContractError(
                f"steps[{index}].ops overlaps an earlier kernel step"
            )
        covered.update(solver_ops)

        order = tuple(
            _bounded_int(op, f"steps[{index}].op_order", len(graph_mapping))
            for op in _sequence(item.get("op_order"), f"steps[{index}].op_order")
        )
        if set(order) != set(solver_ops) or len(order) != len(solver_ops):
            raise ScheduleContractError(
                f"steps[{index}].op_order is not a permutation of its operations"
            )
        raw_sequential = item.get("sequential_tiles")
        sequential_tiles: tuple[int, ...] | None
        if raw_sequential is None:
            sequential_tiles = None
        else:
            sequential_tiles = tuple(
                _nonnegative_int(value, f"steps[{index}].sequential_tiles")
                for value in _sequence(
                    raw_sequential, f"steps[{index}].sequential_tiles"
                )
            )
            if len(sequential_tiles) != len(order):
                raise ScheduleContractError(
                    f"steps[{index}].sequential_tiles must align with op_order"
                )
        launch = _parse_launch(item.get("launch"), field=f"steps[{index}].launch")
        kind = _enum(KernelKind, item.get("kind"), f"steps[{index}].kind")
        plan_value = item.get("plan")
        if kind is KernelKind.VECTOR:
            plan = _parse_vector_plan(
                plan_value,
                field=f"steps[{index}].plan",
                op_bound=len(graph_mapping),
                tensor_bound=len(result.solver_tensor_to_value),
            )
            _validate_vector_phase_links(
                plan,
                lowered=lowered,
                step_ops=solver_ops,
                step_order=order,
                field=f"steps[{index}].plan",
            )
            _validate_vector_launch_contract(
                plan,
                launch=launch,
                sequential_tiles=sequential_tiles,
                field=f"steps[{index}].plan",
            )
        elif kind is KernelKind.CUBE:
            plan = _parse_cube_plan(
                plan_value,
                field=f"steps[{index}].plan",
                op_bound=len(graph_mapping),
                tensor_bound=len(result.solver_tensor_to_value),
            )
            _validate_cube_contract(
                plan,
                lowered=lowered,
                step_ops=solver_ops,
                step_order=order,
                sequential_tiles=sequential_tiles,
                launch=launch,
                field=f"steps[{index}].plan",
            )
        else:
            plan = _parse_mixed_plan(
                plan_value,
                field=f"steps[{index}].plan",
                op_bound=len(graph_mapping),
                tensor_bound=len(result.solver_tensor_to_value),
            )
            _validate_mixed_contract(
                plan,
                lowered=lowered,
                step_ops=solver_ops,
                step_order=order,
                sequential_tiles=sequential_tiles,
                launch=launch,
                field=f"steps[{index}].plan",
            )
        latency = _finite_number(
            item.get("latency_cycles"), f"steps[{index}].latency_cycles"
        )
        steps.append(
            KernelStep(
                index=index,
                kind=kind,
                solver_ops=solver_ops,
                graph_ops=tuple(graph_mapping[op] for op in solver_ops),
                op_order=order,
                sequential_tiles=sequential_tiles,
                launch=launch,
                latency=latency,
                plan=plan,
            )
        )

    if covered != set(range(len(graph_mapping))):
        raise ScheduleContractError(
            "kernel steps do not cover every lowered solver operation"
        )
    return ScheduledRegion(
        region_id=result.region.id,
        tensor_values=result.solver_tensor_to_value,
        steps=tuple(steps),
    )


def _validate_vector_phase_links(
    plan: VectorKernelPlan,
    *,
    lowered: LoweredRegion,
    step_ops: tuple[int, ...],
    step_order: tuple[int, ...],
    field: str,
) -> None:
    if plan.kind in {
        VectorStreamKind.MATERIALIZED,
        VectorStreamKind.POINTWISE,
    }:
        body = plan.phase(VectorReplayPhase.BODY)
        if body.ops != step_order:
            raise ScheduleContractError(
                f"{field}.phases[body].ops does not preserve the selected "
                "operation order exactly"
            )
        for name in (
            VectorReplayPhase.STATS,
            VectorReplayPhase.APPLY,
            VectorReplayPhase.FINALIZE,
        ):
            if plan.phase(name).ops:
                raise ScheduleContractError(
                    f"{field}.phases[{name.value}].ops must be empty for "
                    f"{plan.kind.value!r} replay"
                )
        if body.loop is None:
            raise ScheduleContractError(
                f"{field}.phases[body] requires the authoritative body loop"
            )

    step_set = set(step_ops)
    step_positions = {op: position for position, op in enumerate(step_order)}
    producer_by_tensor: dict[int, int] = {}
    for operation in lowered.operations:
        for tensor in operation.outputs:
            producer_by_tensor[tensor] = operation.index
    for phase in plan.phases:
        phase_field = f"{field}.phases[{phase.name.value}]"
        if not set(phase.ops).issubset(step_set):
            raise ScheduleContractError(
                f"{phase_field}.ops references an operation outside its kernel step"
            )
        if tuple(sorted(phase.ops, key=step_positions.__getitem__)) != phase.ops:
            raise ScheduleContractError(
                f"{phase_field}.ops does not preserve the selected operation order"
            )
        positions = {op: position for position, op in enumerate(phase.ops)}
        touched_tensors: set[int] = set()
        for op in phase.ops:
            operation = lowered.operation(op)
            touched_tensors.update(operation.inputs)
            touched_tensors.update(operation.outputs)
        frame_tensors = {frame.tensor for frame in phase.tensor_frames}
        if frame_tensors != touched_tensors:
            raise ScheduleContractError(
                f"{phase_field}.tensor_frames do not cover exactly the phase tensors"
            )
        for frame in phase.tensor_frames:
            tensor = lowered.tensor(frame.tensor)
            if frame.logical[0] > tensor.height or frame.logical[1] > tensor.width:
                raise ScheduleContractError(
                    f"{phase_field} frame for tensor {frame.tensor} exceeds its extent"
                )
        frame_by_tensor = {frame.tensor: frame for frame in phase.tensor_frames}
        # PyPTO's tensor-to-tile lowering creates a workspace only for row
        # reductions. Column reductions lower directly to their tile op.
        expected_workspace_ops = (
            {op for op in phase.ops if lowered.operation(op).op_type == "Reduction"}
            if plan.physical_frame.reduced_axis == 1
            else set()
        )
        if {workspace.op for workspace in phase.workspaces} != expected_workspace_ops:
            raise ScheduleContractError(
                f"{phase_field}.workspaces do not cover exactly the phase reductions"
            )
        for workspace in phase.workspaces:
            operation = lowered.operation(workspace.op)
            if not operation.inputs or operation.inputs[0] != workspace.source_tensor:
                raise ScheduleContractError(
                    f"{phase_field} workspace {workspace.op} has the wrong source tensor"
                )
            source_frame = frame_by_tensor[workspace.source_tensor]
            # Row-reduction scratch pads its contiguous extent to at least 128
            # elements. It intentionally differs from a narrower source frame
            # while retaining the same logical valid shape.
            expected_physical = (
                source_frame.physical[0],
                max(128, source_frame.physical[1]),
            )
            if (
                workspace.logical != source_frame.logical
                or workspace.physical != expected_physical
            ):
                raise ScheduleContractError(
                    f"{phase_field} workspace {workspace.op} differs from its lowered scratch frame"
                )
        for lifetime in phase.input_lifetimes:
            if lifetime.tensor not in frame_tensors:
                raise ScheduleContractError(
                    f"{phase_field} lifetime tensor {lifetime.tensor} has no frame"
                )
            use_positions: list[int] = []
            for use in lifetime.uses:
                if use.op not in positions:
                    raise ScheduleContractError(
                        f"{phase_field} lifetime tensor {lifetime.tensor} has a use "
                        f"outside the phase"
                    )
                operation = lowered.operation(use.op)
                if use.arg >= len(operation.inputs):
                    raise ScheduleContractError(
                        f"{phase_field} use ({use.op}, {use.arg}) has no such operand"
                    )
                if operation.inputs[use.arg] != lifetime.tensor:
                    raise ScheduleContractError(
                        f"{phase_field} use ({use.op}, {use.arg}) does not reference "
                        f"tensor {lifetime.tensor}"
                    )
                use_positions.append(positions[use.op])
            if not use_positions:
                raise ScheduleContractError(
                    f"{phase_field} lifetime tensor {lifetime.tensor} has no uses"
                )
            if lifetime.first_use_step != min(
                use_positions
            ) or lifetime.last_use_step != max(use_positions):
                raise ScheduleContractError(
                    f"{phase_field} lifetime tensor {lifetime.tensor} has stale "
                    "first/last use positions"
                )
        expected_boundary_uses = {
            (op, argument)
            for op in phase.ops
            for argument, tensor in enumerate(lowered.operation(op).inputs)
            if producer_by_tensor.get(tensor) not in step_set
        }
        actual_boundary_uses = {
            (use.op, use.arg)
            for lifetime in phase.input_lifetimes
            for use in lifetime.uses
        }
        if actual_boundary_uses != expected_boundary_uses:
            raise ScheduleContractError(
                f"{phase_field} input lifetimes do not cover exactly the "
                "boundary-tensor uses"
            )
    _validate_vector_replay_pass_links(
        plan,
        lowered=lowered,
        step_ops=step_ops,
        step_order=step_order,
        field=field,
    )
    _validate_vector_p4_contract(plan, lowered=lowered, field=field)


def _validate_vector_replay_pass_links(
    plan: VectorKernelPlan,
    *,
    lowered: LoweredRegion,
    step_ops: tuple[int, ...],
    step_order: tuple[int, ...],
    field: str,
) -> None:
    if plan.kind is not VectorStreamKind.MULTI_PASS:
        return
    if plan.stream_passes != len(plan.replay_passes):
        raise ScheduleContractError(f"{field}.stream_passes differs from replay_passes")
    if plan.axis not in {1, 2} or plan.chunk <= 0 or plan.full_chunks <= 0:
        raise ScheduleContractError(f"{field}.replay_passes has invalid geometry")
    if plan.full_chunks * plan.chunk + plan.tail != plan.extent:
        raise ScheduleContractError(
            f"{field}.replay_passes do not cover the reduced extent"
        )

    order_position = {op: index for index, op in enumerate(step_order)}
    step_set = set(step_ops)
    producer_by_tensor: dict[int, int] = {}
    consumers_by_tensor: dict[int, set[int]] = {}
    for operation in lowered.operations:
        for tensor in operation.outputs:
            producer_by_tensor[tensor] = operation.index
        for tensor in operation.inputs:
            consumers_by_tensor.setdefault(tensor, set()).add(operation.index)
    prior_states: set[int] = set()
    covered_ops: set[int] = set()
    for replay in plan.replay_passes:
        replay_field = f"{field}.replay_passes[{replay.index}]"
        replay_set = set(replay.ops)
        if not replay_set.issubset(step_set):
            raise ScheduleContractError(
                f"{replay_field}.ops references an operation outside its kernel step"
            )
        if tuple(sorted(replay.ops, key=order_position.__getitem__)) != replay.ops:
            raise ScheduleContractError(
                f"{replay_field}.ops does not preserve the selected operation order"
            )
        covered_ops.update(replay_set)
        if not set(replay.state_inputs).issubset(prior_states):
            raise ScheduleContractError(
                f"{replay_field}.state_inputs are not produced by earlier passes"
            )
        expected_states = {
            output
            for op in replay.ops
            if lowered.operation(op).op_type == "Reduction"
            for output in lowered.operation(op).outputs
        }
        if set(replay.state_outputs) != expected_states:
            raise ScheduleContractError(
                f"{replay_field}.state_outputs do not match its reductions"
            )
        expected_outputs = {
            tensor
            for op in replay.ops
            for tensor in lowered.operation(op).outputs
            if tensor in lowered.required_outputs
            or any(
                consumer not in step_set
                for consumer in consumers_by_tensor.get(tensor, set())
            )
        }
        if set(replay.output_tensors) != expected_outputs:
            raise ScheduleContractError(
                f"{replay_field}.output_tensors do not match kernel boundaries"
            )
        if replay.kind is VectorReplayPassKind.REDUCTION and not expected_states:
            raise ScheduleContractError(
                f"{replay_field} is a reduction pass without a reduction"
            )
        if replay.kind is VectorReplayPassKind.APPLY and expected_states:
            raise ScheduleContractError(
                f"{replay_field} apply pass contains a reduction"
            )

        touched = {
            tensor
            for op in replay.ops
            for tensor in (
                *lowered.operation(op).inputs,
                *lowered.operation(op).outputs,
            )
        }
        if {frame.tensor for frame in replay.tensor_frames} != touched:
            raise ScheduleContractError(
                f"{replay_field}.tensor_frames do not cover exactly its tensors"
            )
        frame_by_tensor = {frame.tensor: frame for frame in replay.tensor_frames}
        for frame in replay.tensor_frames:
            tensor = lowered.tensor(frame.tensor)
            if frame.logical[0] > tensor.height or frame.logical[1] > tensor.width:
                raise ScheduleContractError(
                    f"{replay_field} frame for tensor {frame.tensor} exceeds its extent"
                )
        expected_workspace_ops = (
            {op for op in replay.ops if lowered.operation(op).op_type == "Reduction"}
            if plan.physical_frame.reduced_axis == 1
            else set()
        )
        if {workspace.op for workspace in replay.workspaces} != expected_workspace_ops:
            raise ScheduleContractError(
                f"{replay_field}.workspaces do not cover exactly its reductions"
            )
        for workspace in replay.workspaces:
            operation = lowered.operation(workspace.op)
            if not operation.inputs or operation.inputs[0] != workspace.source_tensor:
                raise ScheduleContractError(
                    f"{replay_field} workspace {workspace.op} has the wrong source tensor"
                )
            source_frame = frame_by_tensor[workspace.source_tensor]
            expected_physical = (
                source_frame.physical[0],
                max(128, source_frame.physical[1]),
            )
            if (
                workspace.logical != source_frame.logical
                or workspace.physical != expected_physical
            ):
                raise ScheduleContractError(
                    f"{replay_field} workspace {workspace.op} differs from its lowered scratch frame"
                )
        positions = {op: index for index, op in enumerate(replay.ops)}
        expected_boundary_uses: set[tuple[int, int]] = set()
        expected_state_inputs: set[int] = set()
        for op in replay.ops:
            operation = lowered.operation(op)
            for argument, tensor in enumerate(operation.inputs):
                producer = producer_by_tensor.get(tensor)
                if producer in replay_set:
                    continue
                if tensor in replay.state_inputs:
                    expected_state_inputs.add(tensor)
                    continue
                if producer in step_set:
                    raise ScheduleContractError(
                        f"{replay_field} has an undeclared cross-pass tensor {tensor}"
                    )
                expected_boundary_uses.add((op, argument))
        if set(replay.state_inputs) != expected_state_inputs:
            raise ScheduleContractError(
                f"{replay_field}.state_inputs do not match cross-pass uses"
            )
        actual_boundary_uses = {
            (use.op, use.arg)
            for lifetime in replay.input_lifetimes
            for use in lifetime.uses
        }
        if actual_boundary_uses != expected_boundary_uses:
            raise ScheduleContractError(
                f"{replay_field}.input_lifetimes do not match boundary uses"
            )
        for lifetime in replay.input_lifetimes:
            use_positions = [positions[use.op] for use in lifetime.uses]
            if (
                not use_positions
                or lifetime.first_use_step != min(use_positions)
                or lifetime.last_use_step != max(use_positions)
                or lifetime.use_count != len(use_positions)
            ):
                raise ScheduleContractError(
                    f"{replay_field} has stale input lifetime bounds"
                )

        if replay.kind is VectorReplayPassKind.REDUCTION:
            if (
                not replay.init.present
                or replay.init.chunk_index != 0
                or replay.init.extent != plan.chunk
                or replay.loop.first_chunk != 1
                or replay.loop.trip_count != plan.full_chunks - 1
            ):
                raise ScheduleContractError(
                    f"{replay_field} reduction loop differs from the stream geometry"
                )
        elif (
            replay.init.present
            or replay.loop.first_chunk != 0
            or replay.loop.trip_count != plan.full_chunks
        ):
            raise ScheduleContractError(
                f"{replay_field} apply loop differs from the stream geometry"
            )
        if replay.loop.pipeline_stages not in {1, 2}:
            raise ScheduleContractError(
                f"{replay_field} has an unsupported pipeline depth"
            )
        if replay.tail.present != (plan.tail > 0) or (
            replay.tail.present
            and (
                replay.tail.chunk_index != plan.full_chunks
                or replay.tail.extent != plan.tail
            )
        ):
            raise ScheduleContractError(
                f"{replay_field} tail differs from the stream geometry"
            )
        prior_states.update(replay.state_outputs)

    if covered_ops != step_set:
        raise ScheduleContractError(
            f"{field}.replay_passes do not cover every selected operation"
        )


def _validate_vector_launch_contract(
    plan: VectorKernelPlan,
    *,
    launch: LaunchPlan,
    sequential_tiles: tuple[int, ...] | None,
    field: str,
) -> None:
    if (
        launch.parts_m != plan.m_partition.parts
        or launch.parts_n != plan.n_partition.parts
    ):
        raise ScheduleContractError(f"{field} launch grid differs from its partitions")
    if launch.cores > plan.work_units:
        raise ScheduleContractError(f"{field} uses more cores than work units")
    if plan.kind in {
        VectorStreamKind.MATERIALIZED,
        VectorStreamKind.POINTWISE,
    } and (sequential_tiles is None or any(sequential_tiles)):
        raise ScheduleContractError(
            f"{field} materialized replay has nonzero sequential tiles"
        )


def _validate_vector_p4_contract(
    plan: VectorKernelPlan,
    *,
    lowered: LoweredRegion,
    field: str,
) -> None:
    expected_recipe = {
        VectorStreamKind.SOFTMAX_FLASH: "softmax_flash.v1",
        VectorStreamKind.LAYERNORM_WELFORD: "welford.v1",
    }.get(plan.kind)
    recipe = plan.p4_recipe
    if expected_recipe is None:
        if recipe is not None or plan.p4_work.generated:
            raise ScheduleContractError(
                f"{field} carries generated P4 work for {plan.kind.value!r}"
            )
        return
    if recipe is None or recipe.version != expected_recipe:
        raise ScheduleContractError(
            f"{field} omits the {expected_recipe!r} emission recipe"
        )
    if not plan.p4_work.generated:
        raise ScheduleContractError(
            f"{field}.p4_work must be generated for {expected_recipe!r}"
        )
    if not plan.p4_work.stats_init.generated or not plan.p4_work.stats_update.generated:
        raise ScheduleContractError(
            f"{field}.p4_work omits generated online-statistics phases"
        )
    expected_finalize = plan.kind is VectorStreamKind.LAYERNORM_WELFORD
    if plan.p4_work.finalize.generated != expected_finalize:
        raise ScheduleContractError(
            f"{field}.p4_work.finalize disagrees with {expected_recipe!r}"
        )

    stats = plan.phase(VectorReplayPhase.STATS)
    apply = plan.phase(VectorReplayPhase.APPLY)
    if recipe.input_tensor not in {frame.tensor for frame in stats.tensor_frames}:
        raise ScheduleContractError(
            f"{field}.p4_recipe input tensor is absent from the stats phase"
        )
    substitution_ops = tuple(item.op for item in recipe.apply_substitutions)
    if any(
        op not in stats.ops or lowered.operation(op).op_type != "Reduction"
        for op in substitution_ops
    ):
        raise ScheduleContractError(
            f"{field}.p4_recipe substitutions must be stats-phase reductions"
        )
    if any(op in apply.ops for op in substitution_ops):
        raise ScheduleContractError(
            f"{field}.p4_recipe substitutions must be cut from the apply replay"
        )
    first_reduction = lowered.operation(substitution_ops[0])
    if not first_reduction.inputs or first_reduction.inputs[0] != recipe.input_tensor:
        raise ScheduleContractError(
            f"{field}.p4_recipe input tensor differs from its first reduction"
        )


def _parse_launch(value: Any, *, field: str) -> LaunchPlan:
    item = _mapping(value, field)
    _expect_keys(item, required={"tile", "parts", "split", "cores"}, field=field)
    tile = _int_tuple(item.get("tile"), 3, f"{field}.tile")
    parts = _int_tuple(item.get("parts"), 2, f"{field}.parts")
    return LaunchPlan(
        tile_w=tile[0],
        tile_h=tile[1],
        tile_k=tile[2],
        parts_m=parts[0],
        parts_n=parts[1],
        split=_positive_int(item.get("split"), f"{field}.split"),
        cores=_positive_int(item.get("cores"), f"{field}.cores"),
    )


def _parse_vector_plan(
    value: Any,
    *,
    field: str,
    op_bound: int,
    tensor_bound: int,
) -> VectorKernelPlan:
    item = _mapping(value, field)
    required = {
        "kind",
        "coordinate_transform",
        "spatial_policy",
        "work_units",
        "m_partition",
        "n_partition",
        "full_peak_ub_bytes",
        "workspace_free_peak_ub_bytes",
        "chunk_peak_ub_bytes",
        "stream_band_count",
        "physical_frame",
        "axis",
        "free_tile",
        "free_tile_alloc",
        "extent",
        "chunk",
        "full_chunks",
        "tail",
        "stream_passes",
        "phases",
        "replay_passes",
        "tile",
        "strip",
        "strip_grid",
        "overlap_granted",
        "reduction_split",
        "p4_work",
        "p4_recipe",
    }
    _expect_keys(item, required=required, field=field)
    frame = _mapping(item.get("physical_frame"), f"{field}.physical_frame")
    _expect_keys(
        frame,
        required={
            "element_granule",
            "iteration_rows",
            "iteration_cols",
            "reduced_axis",
            "align_rows",
        },
        field=f"{field}.physical_frame",
    )
    phases = tuple(
        _parse_vector_phase(
            phase,
            field=f"{field}.phases[{index}]",
            op_bound=op_bound,
            tensor_bound=tensor_bound,
        )
        for index, phase in enumerate(_sequence(item.get("phases"), f"{field}.phases"))
    )
    expected_phases = tuple(VectorReplayPhase)
    if tuple(phase.name for phase in phases) != expected_phases:
        raise ScheduleContractError(
            f"{field}.phases must contain body, stats, apply, finalize in order"
        )
    reduction = _parse_vector_reduction_split(
        item.get("reduction_split"), field=f"{field}.reduction_split"
    )
    replay_passes = tuple(
        _parse_vector_replay_pass(
            replay,
            field=f"{field}.replay_passes[{index}]",
            op_bound=op_bound,
            tensor_bound=tensor_bound,
        )
        for index, replay in enumerate(
            _sequence(item.get("replay_passes"), f"{field}.replay_passes")
        )
    )
    if tuple(replay.index for replay in replay_passes) != tuple(
        range(len(replay_passes))
    ):
        raise ScheduleContractError(
            f"{field}.replay_passes must use contiguous ordered indices"
        )
    kind = _enum(VectorStreamKind, item.get("kind"), f"{field}.kind")
    if kind is VectorStreamKind.MULTI_PASS and not replay_passes:
        raise ScheduleContractError(f"{field}.replay_passes is empty for multi_pass")
    if kind is not VectorStreamKind.MULTI_PASS and replay_passes:
        raise ScheduleContractError(
            f"{field}.replay_passes is present for {kind.value!r}"
        )
    return VectorKernelPlan(
        kind=kind,
        coordinate_transform=_enum(
            VectorCoordinateTransform,
            item.get("coordinate_transform"),
            f"{field}.coordinate_transform",
        ),
        spatial_policy=_enum(
            VectorSpatialPolicy,
            item.get("spatial_policy"),
            f"{field}.spatial_policy",
        ),
        work_units=_positive_int(item.get("work_units"), f"{field}.work_units"),
        m_partition=_parse_axis_partition(
            item.get("m_partition"), field=f"{field}.m_partition"
        ),
        n_partition=_parse_axis_partition(
            item.get("n_partition"), field=f"{field}.n_partition"
        ),
        full_peak_ub_bytes=_nonnegative_int(
            item.get("full_peak_ub_bytes"), f"{field}.full_peak_ub_bytes"
        ),
        workspace_free_peak_ub_bytes=_nonnegative_int(
            item.get("workspace_free_peak_ub_bytes"),
            f"{field}.workspace_free_peak_ub_bytes",
        ),
        chunk_peak_ub_bytes=_nonnegative_int(
            item.get("chunk_peak_ub_bytes"), f"{field}.chunk_peak_ub_bytes"
        ),
        stream_band_count=_nonnegative_int(
            item.get("stream_band_count"), f"{field}.stream_band_count"
        ),
        physical_frame=VectorPhysicalFramePlan(
            element_granule=_positive_int(
                frame.get("element_granule"), f"{field}.physical_frame.element_granule"
            ),
            iteration_rows=_positive_int(
                frame.get("iteration_rows"), f"{field}.physical_frame.iteration_rows"
            ),
            iteration_cols=_positive_int(
                frame.get("iteration_cols"), f"{field}.physical_frame.iteration_cols"
            ),
            reduced_axis=_bounded_axis(
                frame.get("reduced_axis"), f"{field}.physical_frame.reduced_axis"
            ),
            align_rows=_bool(
                frame.get("align_rows"), f"{field}.physical_frame.align_rows"
            ),
        ),
        axis=_bounded_axis(item.get("axis"), f"{field}.axis"),
        free_tile=_nonnegative_int(item.get("free_tile"), f"{field}.free_tile"),
        free_tile_alloc=_nonnegative_int(
            item.get("free_tile_alloc"), f"{field}.free_tile_alloc"
        ),
        extent=_nonnegative_int(item.get("extent"), f"{field}.extent"),
        chunk=_nonnegative_int(item.get("chunk"), f"{field}.chunk"),
        full_chunks=_nonnegative_int(item.get("full_chunks"), f"{field}.full_chunks"),
        tail=_nonnegative_int(item.get("tail"), f"{field}.tail"),
        stream_passes=_positive_int(
            item.get("stream_passes"), f"{field}.stream_passes"
        ),
        phases=phases,
        replay_passes=replay_passes,
        tile=_int_tuple(item.get("tile"), 2, f"{field}.tile"),
        strip=_nonnegative_int_tuple(item.get("strip"), 2, f"{field}.strip"),
        strip_grid=_int_tuple(item.get("strip_grid"), 2, f"{field}.strip_grid"),
        overlap_granted=_bool(item.get("overlap_granted"), f"{field}.overlap_granted"),
        reduction_split=reduction,
        p4_work=_parse_p4_work(item.get("p4_work"), field=f"{field}.p4_work"),
        p4_recipe=_parse_p4_recipe(
            item.get("p4_recipe"),
            field=f"{field}.p4_recipe",
            op_bound=op_bound,
            tensor_bound=tensor_bound,
        ),
    )


def _parse_vector_replay_pass(
    value: Any,
    *,
    field: str,
    op_bound: int,
    tensor_bound: int,
) -> VectorReplayPassPlan:
    item = _mapping(value, field)
    _expect_keys(
        item,
        required={
            "index",
            "kind",
            "ops",
            "state_inputs",
            "state_outputs",
            "output_tensors",
            "input_lifetimes",
            "tensor_frames",
            "workspaces",
            "init",
            "loop",
            "tail",
        },
        field=field,
    )
    ops = tuple(
        _bounded_int(op, f"{field}.ops", op_bound)
        for op in _sequence(item.get("ops"), f"{field}.ops")
    )
    if len(ops) != len(set(ops)):
        raise ScheduleContractError(f"{field}.ops contains duplicate operations")

    def tensors(name: str) -> tuple[int, ...]:
        result = tuple(
            _bounded_int(tensor, f"{field}.{name}", tensor_bound)
            for tensor in _sequence(item.get(name), f"{field}.{name}")
        )
        if len(result) != len(set(result)):
            raise ScheduleContractError(f"{field}.{name} contains duplicates")
        return result

    lifetimes = tuple(
        _parse_vector_input_lifetime(
            lifetime,
            field=f"{field}.input_lifetimes[{index}]",
            op_bound=op_bound,
            tensor_bound=tensor_bound,
        )
        for index, lifetime in enumerate(
            _sequence(item.get("input_lifetimes"), f"{field}.input_lifetimes")
        )
    )
    frames = tuple(
        _parse_vector_tensor_frame(
            frame,
            field=f"{field}.tensor_frames[{index}]",
            tensor_bound=tensor_bound,
        )
        for index, frame in enumerate(
            _sequence(item.get("tensor_frames"), f"{field}.tensor_frames")
        )
    )
    workspaces = tuple(
        _parse_vector_workspace_frame(
            workspace,
            field=f"{field}.workspaces[{index}]",
            op_bound=op_bound,
            tensor_bound=tensor_bound,
        )
        for index, workspace in enumerate(
            _sequence(item.get("workspaces"), f"{field}.workspaces")
        )
    )
    loop = _parse_optional_loop(item.get("loop"), field=f"{field}.loop")
    init = _parse_optional_serial(item.get("init"), field=f"{field}.init")
    tail = _parse_optional_serial(item.get("tail"), field=f"{field}.tail")
    if loop is None or init is None or tail is None:
        raise ScheduleContractError(
            f"{field} requires concrete init, loop, and tail descriptors"
        )
    return VectorReplayPassPlan(
        index=_nonnegative_int(item.get("index"), f"{field}.index"),
        kind=_enum(VectorReplayPassKind, item.get("kind"), f"{field}.kind"),
        ops=ops,
        state_inputs=tensors("state_inputs"),
        state_outputs=tensors("state_outputs"),
        output_tensors=tensors("output_tensors"),
        input_lifetimes=lifetimes,
        tensor_frames=frames,
        workspaces=workspaces,
        loop=loop,
        init=init,
        tail=tail,
    )


def _parse_vector_phase(
    value: Any,
    *,
    field: str,
    op_bound: int,
    tensor_bound: int,
) -> VectorPhasePlan:
    item = _mapping(value, field)
    name = _enum(VectorReplayPhase, item.get("name"), f"{field}.name")
    required = {"name", "ops", "input_lifetimes", "tensor_frames", "workspaces"}
    expected_optional: set[str]
    if name is VectorReplayPhase.BODY:
        expected_optional = {"loop"}
    elif name is VectorReplayPhase.STATS:
        expected_optional = {"init", "loop", "tail"}
    elif name is VectorReplayPhase.APPLY:
        expected_optional = {"loop", "tail"}
    else:
        expected_optional = {"serial"}
    _expect_keys(item, required=required, optional=expected_optional, field=field)
    ops = tuple(
        _bounded_int(op, f"{field}.ops", op_bound)
        for op in _sequence(item.get("ops"), f"{field}.ops")
    )
    if len(ops) != len(set(ops)):
        raise ScheduleContractError(f"{field}.ops contains duplicate operations")
    lifetimes = tuple(
        _parse_vector_input_lifetime(
            lifetime,
            field=f"{field}.input_lifetimes[{index}]",
            op_bound=op_bound,
            tensor_bound=tensor_bound,
        )
        for index, lifetime in enumerate(
            _sequence(item.get("input_lifetimes"), f"{field}.input_lifetimes")
        )
    )
    if len({lifetime.tensor for lifetime in lifetimes}) != len(lifetimes):
        raise ScheduleContractError(f"{field} contains duplicate tensor lifetimes")
    frames = tuple(
        _parse_vector_tensor_frame(
            frame,
            field=f"{field}.tensor_frames[{index}]",
            tensor_bound=tensor_bound,
        )
        for index, frame in enumerate(
            _sequence(item.get("tensor_frames"), f"{field}.tensor_frames")
        )
    )
    if len({frame.tensor for frame in frames}) != len(frames):
        raise ScheduleContractError(f"{field} contains duplicate tensor frames")
    workspaces = tuple(
        _parse_vector_workspace_frame(
            workspace,
            field=f"{field}.workspaces[{index}]",
            op_bound=op_bound,
            tensor_bound=tensor_bound,
        )
        for index, workspace in enumerate(
            _sequence(item.get("workspaces"), f"{field}.workspaces")
        )
    )
    if len({workspace.op for workspace in workspaces}) != len(workspaces):
        raise ScheduleContractError(f"{field} contains duplicate op workspaces")
    loop = _parse_optional_loop(item.get("loop"), field=f"{field}.loop")
    init = _parse_optional_serial(item.get("init"), field=f"{field}.init")
    tail = _parse_optional_serial(item.get("tail"), field=f"{field}.tail")
    serial = _parse_optional_serial(item.get("serial"), field=f"{field}.serial")
    return VectorPhasePlan(
        name, ops, lifetimes, frames, workspaces, loop, init, tail, serial
    )


def _parse_vector_tensor_frame(
    value: Any,
    *,
    field: str,
    tensor_bound: int,
) -> VectorTensorFramePlan:
    item = _mapping(value, field)
    _expect_keys(item, required={"tensor", "logical", "physical"}, field=field)
    logical = _int_tuple(item.get("logical"), 2, f"{field}.logical")
    physical = _int_tuple(item.get("physical"), 2, f"{field}.physical")
    if physical[0] < logical[0] or physical[1] < logical[1]:
        raise ScheduleContractError(f"{field}.physical cannot shrink logical shape")
    return VectorTensorFramePlan(
        tensor=_bounded_int(item.get("tensor"), f"{field}.tensor", tensor_bound),
        logical=logical,
        physical=physical,
    )


def _parse_vector_workspace_frame(
    value: Any,
    *,
    field: str,
    op_bound: int,
    tensor_bound: int,
) -> VectorWorkspaceFramePlan:
    item = _mapping(value, field)
    _expect_keys(
        item,
        required={"op", "source_tensor", "logical", "physical"},
        field=field,
    )
    logical = _int_tuple(item.get("logical"), 2, f"{field}.logical")
    physical = _int_tuple(item.get("physical"), 2, f"{field}.physical")
    if physical[0] < logical[0] or physical[1] < logical[1]:
        raise ScheduleContractError(f"{field}.physical cannot shrink logical shape")
    return VectorWorkspaceFramePlan(
        op=_bounded_int(item.get("op"), f"{field}.op", op_bound),
        source_tensor=_bounded_int(
            item.get("source_tensor"), f"{field}.source_tensor", tensor_bound
        ),
        logical=logical,
        physical=physical,
    )


def _parse_vector_input_lifetime(
    value: Any,
    *,
    field: str,
    op_bound: int,
    tensor_bound: int,
) -> VectorInputLifetimePlan:
    item = _mapping(value, field)
    _expect_keys(
        item,
        required={
            "tensor",
            "first_use_step",
            "last_use_step",
            "use_count",
            "uses",
        },
        field=field,
    )
    uses = tuple(
        _parse_vector_input_use(
            use,
            field=f"{field}.uses[{index}]",
            op_bound=op_bound,
        )
        for index, use in enumerate(_sequence(item.get("uses"), f"{field}.uses"))
    )
    use_count = _positive_int(item.get("use_count"), f"{field}.use_count")
    if use_count != len(uses):
        raise ScheduleContractError(f"{field}.use_count does not match its uses")
    first = _nonnegative_int(item.get("first_use_step"), f"{field}.first_use_step")
    last = _nonnegative_int(item.get("last_use_step"), f"{field}.last_use_step")
    if first > last:
        raise ScheduleContractError(f"{field} has an inverted lifetime")
    return VectorInputLifetimePlan(
        tensor=_bounded_int(item.get("tensor"), f"{field}.tensor", tensor_bound),
        first_use_step=first,
        last_use_step=last,
        use_count=use_count,
        uses=uses,
    )


def _parse_vector_input_use(
    value: Any, *, field: str, op_bound: int
) -> VectorInputUsePlan:
    item = _mapping(value, field)
    _expect_keys(item, required={"op", "arg"}, field=field)
    return VectorInputUsePlan(
        op=_bounded_int(item.get("op"), f"{field}.op", op_bound),
        arg=_nonnegative_int(item.get("arg"), f"{field}.arg"),
    )


def _parse_vector_reduction_split(
    value: Any, *, field: str
) -> VectorReductionSplitPlan:
    item = _mapping(value, field)
    _expect_keys(
        item, required={"kind", "factor", "partial_extent", "seed"}, field=field
    )
    seed_item = _mapping(item.get("seed"), f"{field}.seed")
    _expect_keys(
        seed_item,
        required={"present", "work_units", "valid_rows", "valid_cols"},
        field=f"{field}.seed",
    )
    seed = VectorReductionSeedPlan(
        present=_bool(seed_item.get("present"), f"{field}.seed.present"),
        work_units=_nonnegative_int(
            seed_item.get("work_units"), f"{field}.seed.work_units"
        ),
        valid_rows=_nonnegative_int(
            seed_item.get("valid_rows"), f"{field}.seed.valid_rows"
        ),
        valid_cols=_nonnegative_int(
            seed_item.get("valid_cols"), f"{field}.seed.valid_cols"
        ),
    )
    return VectorReductionSplitPlan(
        kind=_enum(VectorReductionSplitKind, item.get("kind"), f"{field}.kind"),
        factor=_positive_int(item.get("factor"), f"{field}.factor"),
        partial_extent=_nonnegative_int(
            item.get("partial_extent"), f"{field}.partial_extent"
        ),
        seed=seed,
    )


def _parse_p4_work(value: Any, *, field: str) -> VectorP4WorkPlan:
    item = _mapping(value, field)
    _expect_keys(
        item,
        required={"generated", "stats_init", "stats_update", "finalize"},
        field=field,
    )
    return VectorP4WorkPlan(
        generated=_bool(item.get("generated"), f"{field}.generated"),
        stats_init=_parse_generated_work(
            item.get("stats_init"), field=f"{field}.stats_init"
        ),
        stats_update=_parse_generated_work(
            item.get("stats_update"), field=f"{field}.stats_update"
        ),
        finalize=_parse_generated_work(item.get("finalize"), field=f"{field}.finalize"),
    )


def _parse_p4_recipe(
    value: Any,
    *,
    field: str,
    op_bound: int,
    tensor_bound: int,
) -> VectorP4RecipePlan | None:
    if value is None:
        return None
    item = _mapping(value, field)
    _expect_keys(
        item,
        required={"version", "input_tensor", "state", "apply_substitutions"},
        field=field,
    )
    version = item.get("version")
    expected = {
        "softmax_flash.v1": (
            ("running_max", "running_sum"),
            ("running_max", "running_sum"),
        ),
        "welford.v1": (
            ("running_mean", "running_m2", "running_count"),
            ("mean", "variance"),
        ),
    }
    if version not in expected:
        raise ScheduleContractError(f"{field}.version is unsupported: {version!r}")
    state = tuple(
        _nonempty_string(name, f"{field}.state")
        for name in _sequence(item.get("state"), f"{field}.state")
    )
    bindings = tuple(
        _parse_p4_substitution(
            binding,
            field=f"{field}.apply_substitutions[{index}]",
            op_bound=op_bound,
        )
        for index, binding in enumerate(
            _sequence(
                item.get("apply_substitutions"),
                f"{field}.apply_substitutions",
            )
        )
    )
    expected_state, expected_values = expected[str(version)]
    if (
        state != expected_state
        or tuple(binding.value for binding in bindings) != expected_values
    ):
        raise ScheduleContractError(
            f"{field} does not match its versioned state contract"
        )
    if len({binding.op for binding in bindings}) != len(bindings):
        raise ScheduleContractError(f"{field} contains duplicate substitution ops")
    return VectorP4RecipePlan(
        version=str(version),
        input_tensor=_bounded_int(
            item.get("input_tensor"), f"{field}.input_tensor", tensor_bound
        ),
        state=state,
        apply_substitutions=bindings,
    )


def _parse_p4_substitution(
    value: Any, *, field: str, op_bound: int
) -> VectorP4SubstitutionPlan:
    item = _mapping(value, field)
    _expect_keys(item, required={"op", "value"}, field=field)
    return VectorP4SubstitutionPlan(
        op=_bounded_int(item.get("op"), f"{field}.op", op_bound),
        value=_nonempty_string(item.get("value"), f"{field}.value"),
    )


def _parse_generated_work(value: Any, *, field: str) -> VectorGeneratedPhaseWorkPlan:
    item = _mapping(value, field)
    _expect_keys(item, required={"generated", "primitives"}, field=field)
    primitives = tuple(
        _parse_primitive_work(primitive, field=f"{field}.primitives[{index}]")
        for index, primitive in enumerate(
            _sequence(item.get("primitives"), f"{field}.primitives")
        )
    )
    return VectorGeneratedPhaseWorkPlan(
        generated=_bool(item.get("generated"), f"{field}.generated"),
        primitives=primitives,
    )


def _parse_primitive_work(value: Any, *, field: str) -> VectorPrimitiveWorkPlan:
    item = _mapping(value, field)
    _expect_keys(item, required={"kind", "wide", "thin", "stream_starts"}, field=field)
    kind = item.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ScheduleContractError(f"{field}.kind must be a non-empty string")
    return VectorPrimitiveWorkPlan(
        kind=kind,
        wide=_nonnegative_int(item.get("wide"), f"{field}.wide"),
        thin=_nonnegative_int(item.get("thin"), f"{field}.thin"),
        stream_starts=_nonnegative_int(
            item.get("stream_starts"), f"{field}.stream_starts"
        ),
    )


def _parse_mixed_plan(
    value: Any,
    *,
    field: str,
    op_bound: int,
    tensor_bound: int,
) -> MixedKernelPlan:
    item = _mapping(value, field)
    required = {
        "emit_compatible",
        "source_codegen_ready",
        "algorithm",
        "protocol",
        "mode",
        "m_partition",
        "n_partition",
        "spatial_tiles",
        "split_k",
        "work_units",
        "group_capacity",
        "cube_window_k",
        "cube_stage_peak_l1_bytes",
        "vector_stage_kind",
        "vector_stage_peak_ub_bytes",
        "vector_split",
        "vector_lanes",
        "pipeline_axis",
        "pipeline_extent",
        "pipeline_chunk",
        "items_per_spatial_tile",
        "active_groups",
        "min_trips_per_group",
        "max_trips_per_group",
        "pipeline_stages",
        "requested_skew_depth",
        "model_overlap_granted",
        "overlap_implementable",
        "pipeline_fill_absorbed",
        "max_alternations",
        "output_engines_uniform",
        "protocol_producer_stages",
        "protocol_peer_stage",
        "protocol_sink_stage",
        "protocol_producer_bundle",
        "protocol_reply_bundle",
        "protocol_skew_compatible",
        "topology_stages",
        "stages",
        "transfers",
        "fifos",
        "feature_round_trip",
    }
    _expect_keys(item, required=required, field=field)
    raw_stages = _sequence(item.get("stages"), f"{field}.stages")
    stages = tuple(
        _parse_mixed_stage(
            stage,
            field=f"{field}.stages[{index}]",
            op_bound=op_bound,
            tensor_bound=tensor_bound,
        )
        for index, stage in enumerate(raw_stages)
    )
    topology_stages = tuple(
        _parse_mixed_stage(
            stage,
            field=f"{field}.topology_stages[{index}]",
            op_bound=op_bound,
            tensor_bound=tensor_bound,
            topology_only=True,
        )
        for index, stage in enumerate(
            _sequence(item.get("topology_stages"), f"{field}.topology_stages")
        )
    )
    transfers = tuple(
        _parse_mixed_transfer(
            transfer,
            field=f"{field}.transfers[{index}]",
            tensor_bound=tensor_bound,
            stage_bound=len(stages),
        )
        for index, transfer in enumerate(
            _sequence(item.get("transfers"), f"{field}.transfers")
        )
    )
    return MixedKernelPlan(
        emit_compatible=_bool(item.get("emit_compatible"), f"{field}.emit_compatible"),
        source_codegen_ready=_bool(
            item.get("source_codegen_ready"), f"{field}.source_codegen_ready"
        ),
        algorithm=_enum(MixedAlgorithm, item.get("algorithm"), f"{field}.algorithm"),
        protocol=_enum(
            MixedCrossCoreProtocol, item.get("protocol"), f"{field}.protocol"
        ),
        mode=_enum(MixedPipelineMode, item.get("mode"), f"{field}.mode"),
        m_partition=_parse_axis_partition(
            item.get("m_partition"), field=f"{field}.m_partition"
        ),
        n_partition=_parse_axis_partition(
            item.get("n_partition"), field=f"{field}.n_partition"
        ),
        spatial_tiles=_positive_int(
            item.get("spatial_tiles"), f"{field}.spatial_tiles"
        ),
        split_k=_positive_int(item.get("split_k"), f"{field}.split_k"),
        work_units=_positive_int(item.get("work_units"), f"{field}.work_units"),
        group_capacity=_positive_int(
            item.get("group_capacity"), f"{field}.group_capacity"
        ),
        cube_window_k=_nonnegative_int(
            item.get("cube_window_k"), f"{field}.cube_window_k"
        ),
        cube_stage_peak_l1_bytes=_nonnegative_int(
            item.get("cube_stage_peak_l1_bytes"),
            f"{field}.cube_stage_peak_l1_bytes",
        ),
        vector_stage_kind=_enum(
            VectorStreamKind,
            item.get("vector_stage_kind"),
            f"{field}.vector_stage_kind",
        ),
        vector_stage_peak_ub_bytes=_nonnegative_int(
            item.get("vector_stage_peak_ub_bytes"),
            f"{field}.vector_stage_peak_ub_bytes",
        ),
        vector_split=_enum(
            MixedVectorSplit, item.get("vector_split"), f"{field}.vector_split"
        ),
        vector_lanes=_positive_int(item.get("vector_lanes"), f"{field}.vector_lanes"),
        pipeline_axis=_enum(
            MixedPipelineAxis, item.get("pipeline_axis"), f"{field}.pipeline_axis"
        ),
        pipeline_extent=_positive_int(
            item.get("pipeline_extent"), f"{field}.pipeline_extent"
        ),
        pipeline_chunk=_positive_int(
            item.get("pipeline_chunk"), f"{field}.pipeline_chunk"
        ),
        items_per_spatial_tile=_positive_int(
            item.get("items_per_spatial_tile"), f"{field}.items_per_spatial_tile"
        ),
        active_groups=_positive_int(
            item.get("active_groups"), f"{field}.active_groups"
        ),
        min_trips_per_group=_positive_int(
            item.get("min_trips_per_group"), f"{field}.min_trips_per_group"
        ),
        max_trips_per_group=_positive_int(
            item.get("max_trips_per_group"), f"{field}.max_trips_per_group"
        ),
        pipeline_stages=_positive_int(
            item.get("pipeline_stages"), f"{field}.pipeline_stages"
        ),
        requested_skew_depth=_nonnegative_int(
            item.get("requested_skew_depth"), f"{field}.requested_skew_depth"
        ),
        model_overlap_granted=_bool(
            item.get("model_overlap_granted"), f"{field}.model_overlap_granted"
        ),
        overlap_implementable=_bool(
            item.get("overlap_implementable"), f"{field}.overlap_implementable"
        ),
        pipeline_fill_absorbed=_bool(
            item.get("pipeline_fill_absorbed"), f"{field}.pipeline_fill_absorbed"
        ),
        max_alternations=_nonnegative_int(
            item.get("max_alternations"), f"{field}.max_alternations"
        ),
        output_engines_uniform=_bool(
            item.get("output_engines_uniform"), f"{field}.output_engines_uniform"
        ),
        protocol_producer_stages=_mixed_indices(
            item.get("protocol_producer_stages"),
            len(stages),
            f"{field}.protocol_producer_stages",
        ),
        protocol_peer_stage=_nullable_mixed_index(
            item.get("protocol_peer_stage"), len(stages), f"{field}.protocol_peer_stage"
        ),
        protocol_sink_stage=_nullable_mixed_index(
            item.get("protocol_sink_stage"), len(stages), f"{field}.protocol_sink_stage"
        ),
        protocol_producer_bundle=_mixed_indices(
            item.get("protocol_producer_bundle"),
            len(transfers),
            f"{field}.protocol_producer_bundle",
        ),
        protocol_reply_bundle=_mixed_indices(
            item.get("protocol_reply_bundle"),
            len(transfers),
            f"{field}.protocol_reply_bundle",
        ),
        protocol_skew_compatible=_bool(
            item.get("protocol_skew_compatible"), f"{field}.protocol_skew_compatible"
        ),
        topology_stages=topology_stages,
        stages=stages,
        transfers=transfers,
        fifos=tuple(
            _parse_mixed_fifo(
                fifo,
                field=f"{field}.fifos[{index}]",
                tensor_bound=tensor_bound,
            )
            for index, fifo in enumerate(_sequence(item.get("fifos"), f"{field}.fifos"))
        ),
        feature_round_trip=_parse_mixed_feature_round_trip(
            item.get("feature_round_trip"), field=f"{field}.feature_round_trip"
        ),
    )


def _parse_mixed_stage(
    value: Any,
    *,
    field: str,
    op_bound: int,
    tensor_bound: int,
    topology_only: bool = False,
) -> MixedStagePlan:
    item = _mapping(value, field)
    if topology_only:
        _expect_keys(item, required={"engine", "ops"}, field=field)
        return MixedStagePlan(
            topology_stage=-1,
            engine=_enum(MixedEngine, item.get("engine"), f"{field}.engine"),
            ops=_mixed_indices(item.get("ops"), op_bound, f"{field}.ops"),
            valid_rows=0,
            valid_cols=0,
            cube_window_k=(),
            vector_stream=None,
        )
    _expect_keys(
        item,
        required={
            "topology_stage",
            "engine",
            "ops",
            "valid_rows",
            "valid_cols",
            "cube_window_k",
            "vector_stream",
        },
        field=field,
    )
    stream = item.get("vector_stream")
    return MixedStagePlan(
        topology_stage=_nonnegative_int(
            item.get("topology_stage"), f"{field}.topology_stage"
        ),
        engine=_enum(MixedEngine, item.get("engine"), f"{field}.engine"),
        ops=_mixed_indices(item.get("ops"), op_bound, f"{field}.ops"),
        valid_rows=_positive_int(item.get("valid_rows"), f"{field}.valid_rows"),
        valid_cols=_positive_int(item.get("valid_cols"), f"{field}.valid_cols"),
        cube_window_k=tuple(
            _positive_int(chunk, f"{field}.cube_window_k")
            for chunk in _sequence(item.get("cube_window_k"), f"{field}.cube_window_k")
        ),
        vector_stream=(
            None
            if stream is None
            else _parse_vector_plan(
                stream,
                field=f"{field}.vector_stream",
                op_bound=op_bound,
                tensor_bound=tensor_bound,
            )
        ),
    )


def _parse_mixed_transfer(
    value: Any,
    *,
    field: str,
    tensor_bound: int,
    stage_bound: int,
) -> MixedTransferPlan:
    item = _mapping(value, field)
    _expect_keys(
        item,
        required={
            "tensor",
            "producer_stage",
            "consumer_stage",
            "producer_engine",
            "consumer_engine",
        },
        field=field,
    )
    return MixedTransferPlan(
        tensor=_bounded_int(item.get("tensor"), f"{field}.tensor", tensor_bound),
        producer_stage=_bounded_int(
            item.get("producer_stage"), f"{field}.producer_stage", stage_bound
        ),
        consumer_stage=_bounded_int(
            item.get("consumer_stage"), f"{field}.consumer_stage", stage_bound
        ),
        producer_engine=_enum(
            MixedEngine, item.get("producer_engine"), f"{field}.producer_engine"
        ),
        consumer_engine=_enum(
            MixedEngine, item.get("consumer_engine"), f"{field}.consumer_engine"
        ),
    )


def _parse_mixed_fifo(value: Any, *, field: str, tensor_bound: int) -> MixedFifoPlan:
    item = _mapping(value, field)
    _expect_keys(
        item,
        required={
            "tensor",
            "direction",
            "wire_dtype",
            "spatial_m",
            "spatial_n",
            "valid_rows",
            "valid_cols",
            "slot_bytes",
            "slot_count",
            "reserved_bytes",
            "pipe_id",
            "bundle",
        },
        field=field,
    )
    return MixedFifoPlan(
        tensor=_bounded_int(item.get("tensor"), f"{field}.tensor", tensor_bound),
        direction=_enum(
            MixedTransferDirection, item.get("direction"), f"{field}.direction"
        ),
        wire_dtype=_dtype(item.get("wire_dtype"), f"{field}.wire_dtype"),
        spatial_m=_bool(item.get("spatial_m"), f"{field}.spatial_m"),
        spatial_n=_bool(item.get("spatial_n"), f"{field}.spatial_n"),
        valid_rows=_positive_int(item.get("valid_rows"), f"{field}.valid_rows"),
        valid_cols=_positive_int(item.get("valid_cols"), f"{field}.valid_cols"),
        slot_bytes=_positive_int(item.get("slot_bytes"), f"{field}.slot_bytes"),
        slot_count=_positive_int(item.get("slot_count"), f"{field}.slot_count"),
        reserved_bytes=_positive_int(
            item.get("reserved_bytes"), f"{field}.reserved_bytes"
        ),
        pipe_id=_nonnegative_int(item.get("pipe_id"), f"{field}.pipe_id"),
        bundle=_optional_index(item.get("bundle"), f"{field}.bundle"),
    )


def _parse_mixed_feature_round_trip(
    value: Any, *, field: str
) -> MixedFeatureRoundTripPlan | None:
    if value is None:
        return None
    item = _mapping(value, field)
    _expect_keys(
        item,
        required={
            "intermediate_extent",
            "intermediate_chunk",
            "intermediate_chunks",
            "output_extent",
            "producer_window_k",
            "persistent_accumulator_bytes",
            "first_chunk_initializes",
            "later_chunks_accumulate",
        },
        field=field,
    )
    return MixedFeatureRoundTripPlan(
        intermediate_extent=_positive_int(
            item.get("intermediate_extent"), f"{field}.intermediate_extent"
        ),
        intermediate_chunk=_positive_int(
            item.get("intermediate_chunk"), f"{field}.intermediate_chunk"
        ),
        intermediate_chunks=_positive_int(
            item.get("intermediate_chunks"), f"{field}.intermediate_chunks"
        ),
        output_extent=_positive_int(
            item.get("output_extent"), f"{field}.output_extent"
        ),
        producer_window_k=tuple(
            _positive_int(window, f"{field}.producer_window_k[{index}]")
            for index, window in enumerate(
                _sequence(
                    item.get("producer_window_k"),
                    f"{field}.producer_window_k",
                )
            )
        ),
        persistent_accumulator_bytes=_positive_int(
            item.get("persistent_accumulator_bytes"),
            f"{field}.persistent_accumulator_bytes",
        ),
        first_chunk_initializes=_bool(
            item.get("first_chunk_initializes"), f"{field}.first_chunk_initializes"
        ),
        later_chunks_accumulate=_bool(
            item.get("later_chunks_accumulate"), f"{field}.later_chunks_accumulate"
        ),
    )


def _mixed_indices(value: Any, bound: int, field: str) -> tuple[int, ...]:
    indices = tuple(
        _bounded_int(index, field, bound) for index in _sequence(value, field)
    )
    if len(set(indices)) != len(indices):
        raise ScheduleContractError(f"{field} contains duplicate indices")
    return indices


def _nullable_mixed_index(value: Any, bound: int, field: str) -> int | None:
    if value is None:
        return None
    return _bounded_int(value, field, bound)


def _validate_mixed_contract(  # noqa: PLR0913
    plan: MixedKernelPlan,
    *,
    lowered: LoweredRegion,
    step_ops: tuple[int, ...],
    step_order: tuple[int, ...],
    sequential_tiles: tuple[int, ...] | None,
    launch: LaunchPlan,
    field: str,
) -> None:
    if not plan.emit_compatible or not plan.source_codegen_ready:
        return
    fifo_owned_dual_role = False
    if (
        plan.protocol is MixedCrossCoreProtocol.ONE_WAY
        and len(plan.stages) == 2
        and len(plan.fifos) == 1
        and plan.m_partition.parts == 1
        and plan.n_partition.parts == 1
        and plan.stages[0].engine is MixedEngine.VECTOR
        and plan.stages[1].engine is MixedEngine.CUBE
        and len(plan.stages[1].ops) == 1
        and plan.fifos[0].direction is MixedTransferDirection.VECTOR_TO_CUBE
        and plan.fifos[0].spatial_m
        and plan.fifos[0].spatial_n
    ):
        sink = lowered.operations[plan.stages[1].ops[0]]
        fifo_owned_dual_role = sink.op_type == "MatMul" and sink.inputs == (
            plan.fifos[0].tensor,
            plan.fifos[0].tensor,
        )
    if plan.cube_stage_peak_l1_bytes <= 0 and not fifo_owned_dual_role:
        raise ScheduleContractError(
            f"{field}.cube_stage_peak_l1_bytes must be positive for source replay"
        )
    if plan.split_k != 1 or launch.split != 1:
        raise ScheduleContractError(f"{field} source-ready mixed replay cannot split K")
    if (
        launch.parts_m != plan.m_partition.parts
        or launch.parts_n != plan.n_partition.parts
        or plan.spatial_tiles != plan.m_partition.parts * plan.n_partition.parts
    ):
        raise ScheduleContractError(f"{field} launch grid differs from its partitions")
    if (
        plan.work_units != plan.spatial_tiles
        or plan.active_groups > plan.group_capacity
    ):
        raise ScheduleContractError(f"{field} has inconsistent mixed work-unit counts")
    expected_launch_cores = plan.active_groups * (1 + plan.vector_lanes)
    if launch.cores != expected_launch_cores:
        raise ScheduleContractError(
            f"{field} launch cores differ from its mixed group participation"
        )
    if plan.min_trips_per_group != plan.max_trips_per_group:
        raise ScheduleContractError(
            f"{field} source-ready mixed replay requires a uniform group loop"
        )
    loop_items = plan.pipeline_extent // plan.pipeline_chunk
    expected_loop_items = (
        plan.pipeline_work_items
        if plan.pipeline_axis is MixedPipelineAxis.SPATIAL_REGION
        else plan.items_per_spatial_tile
    )
    if (
        plan.pipeline_extent % plan.pipeline_chunk != 0
        or loop_items != expected_loop_items
    ):
        raise ScheduleContractError(
            f"{field} pipeline extent does not cover its work items"
        )
    if plan.active_groups * plan.max_trips_per_group != plan.pipeline_work_items:
        raise ScheduleContractError(
            f"{field} active groups do not cover its work items"
        )
    if len(plan.stages) != len(plan.topology_stages) or not plan.stages:
        raise ScheduleContractError(f"{field} stage descriptors are incomplete")
    phase_local_vector_pipeline = (
        plan.protocol is MixedCrossCoreProtocol.ONE_WAY
        and plan.stages[0].engine is MixedEngine.VECTOR
        and plan.stages[0].vector_stream is not None
        and plan.stages[0].vector_stream.kind is VectorStreamKind.SOFTMAX_FLASH
    )
    successor_overlap = plan.max_trips_per_group >= 2
    if plan.protocol is MixedCrossCoreProtocol.ONE_WAY:
        expected_overlap = successor_overlap and not phase_local_vector_pipeline
        expected_stages = 2 if expected_overlap else 1
        if (
            plan.model_overlap_granted != expected_overlap
            or plan.overlap_implementable != expected_overlap
            or plan.pipeline_fill_absorbed
            or plan.pipeline_stages != expected_stages
            or plan.requested_skew_depth != expected_stages - 1
        ):
            raise ScheduleContractError(
                f"{field} one-way pipeline depth differs from its successor loop"
            )
    if plan.protocol in {
        MixedCrossCoreProtocol.SINGLE_ROUND_TRIP_BUNDLE,
        MixedCrossCoreProtocol.BRANCHED_ROUND_TRIP_BUNDLE,
    }:
        expected_fill_absorbed = (
            successor_overlap
            and plan.algorithm is not MixedAlgorithm.FEATURE_CHUNK_ROUND_TRIP
        )
        expected_stages = 3 if successor_overlap else 1
        expected_skew = 2 if successor_overlap else 0
        if (
            plan.pipeline_stages != expected_stages
            or plan.requested_skew_depth != expected_skew
            or plan.model_overlap_granted != successor_overlap
            or plan.overlap_implementable != successor_overlap
            or plan.pipeline_fill_absorbed != expected_fill_absorbed
        ):
            raise ScheduleContractError(
                f"{field} round-trip pipeline differs from its successor loop"
            )
    if tuple(stage.topology_stage for stage in plan.stages) != tuple(
        range(len(plan.stages))
    ):
        raise ScheduleContractError(f"{field}.stages are not densely indexed")
    vector_stage_peaks: list[int] = []
    for index, (stage, topology) in enumerate(
        zip(plan.stages, plan.topology_stages, strict=True)
    ):
        stage_field = f"{field}.stages[{index}]"
        if stage.engine is not topology.engine or stage.ops != topology.ops:
            raise ScheduleContractError(
                f"{stage_field} differs from its immutable topology stage"
            )
        if not stage.ops:
            raise ScheduleContractError(f"{stage_field}.ops must not be empty")
        if stage.engine is MixedEngine.CUBE:
            if stage.vector_stream is not None or len(stage.cube_window_k) != len(
                stage.ops
            ):
                raise ScheduleContractError(
                    f"{stage_field} has an incomplete cube-window contract"
                )
            for op, window in zip(stage.ops, stage.cube_window_k, strict=True):
                if lowered.operation(op).op_type != "MatMul" or window <= 0:
                    raise ScheduleContractError(
                        f"{stage_field} does not describe cube matmul work"
                    )
        else:
            if stage.cube_window_k or stage.vector_stream is None:
                raise ScheduleContractError(
                    f"{stage_field} has an incomplete vector-stream contract"
                )
            if any(lowered.operation(op).op_type == "MatMul" for op in stage.ops):
                raise ScheduleContractError(
                    f"{stage_field} contains cube work in a vector stage"
                )
            if plan.algorithm is not MixedAlgorithm.FEATURE_CHUNK_ROUND_TRIP:
                _validate_vector_phase_links(
                    stage.vector_stream,
                    lowered=lowered,
                    step_ops=stage.ops,
                    step_order=stage.ops,
                    field=f"{stage_field}.vector_stream",
                )
            realized_peak = (
                stage.vector_stream.chunk_peak_ub_bytes
                if stage.vector_stream.kind is VectorStreamKind.SOFTMAX_FLASH
                else _mixed_materialized_source_peak(
                    stage.vector_stream, lowered=lowered
                )
            )
            vector_stage_peaks.append(realized_peak)
            if realized_peak > plan.vector_stage_peak_ub_bytes:
                raise ScheduleContractError(
                    f"{stage_field} Vec peak exceeds the mixed plan"
                )

    cube_windows = tuple(
        window
        for stage in plan.stages
        if stage.engine is MixedEngine.CUBE
        for window in stage.cube_window_k
    )
    uniform_cube_window = (
        cube_windows[0]
        if cube_windows and all(window == cube_windows[0] for window in cube_windows)
        else 0
    )
    if plan.cube_window_k != uniform_cube_window:
        raise ScheduleContractError(
            f"{field}.cube_window_k differs from its authoritative stage windows"
        )
    if (
        not vector_stage_peaks
        or max(vector_stage_peaks) != plan.vector_stage_peak_ub_bytes
    ):
        raise ScheduleContractError(
            f"{field} aggregate Vec peak differs from its vector stages"
        )
    if (
        next(
            stage.vector_stream.kind
            for stage in plan.stages
            if stage.vector_stream is not None
        )
        is not plan.vector_stage_kind
    ):
        raise ScheduleContractError(
            f"{field} first vector-stage kind differs from its compatibility summary"
        )
    flattened_ops = tuple(op for stage in plan.stages for op in stage.ops)
    if len(flattened_ops) != len(set(flattened_ops)) or set(flattened_ops) != set(
        step_ops
    ):
        raise ScheduleContractError(
            f"{field}.stages do not preserve the selected operation order"
        )
    if plan.protocol is MixedCrossCoreProtocol.BRANCHED_ROUND_TRIP_BUNDLE:
        order_position = {op: position for position, op in enumerate(step_order)}
        stage_by_op = {
            op: stage_index
            for stage_index, stage in enumerate(plan.stages)
            for op in stage.ops
        }
        producer_by_tensor = {
            tensor: op for op in step_ops for tensor in lowered.operation(op).outputs
        }
        for stage_index, stage in enumerate(plan.stages):
            if tuple(sorted(stage.ops, key=order_position.__getitem__)) != stage.ops:
                raise ScheduleContractError(
                    f"{field}.stages[{stage_index}].ops do not preserve the "
                    "selected operation order"
                )
            for op in stage.ops:
                for tensor in lowered.operation(op).inputs:
                    producer = producer_by_tensor.get(tensor)
                    if producer is None:
                        continue
                    producer_stage = stage_by_op[producer]
                    if producer_stage > stage_index:
                        raise ScheduleContractError(
                            f"{field}.stages contain a backward data dependency"
                        )
                    if (
                        producer_stage == stage_index
                        and order_position[producer] >= order_position[op]
                    ):
                        raise ScheduleContractError(
                            f"{field}.stages[{stage_index}].ops reorder a data "
                            "dependency"
                        )
    elif (
        plan.algorithm is not MixedAlgorithm.FEATURE_CHUNK_ROUND_TRIP
        and flattened_ops != step_order
    ):
        raise ScheduleContractError(
            f"{field}.stages do not preserve the selected operation order"
        )
    if sequential_tiles is None:
        raise ScheduleContractError(f"{field} omits per-operation sequential tiles")
    sequential_by_op = dict(zip(step_order, sequential_tiles, strict=True))
    for stage in plan.stages:
        if stage.engine is MixedEngine.CUBE:
            for op, window in zip(stage.ops, stage.cube_window_k, strict=True):
                if sequential_by_op[op] != window:
                    raise ScheduleContractError(
                        f"{field} cube window differs from sequential tile for op {op}"
                    )
        elif any(sequential_by_op[op] != 0 for op in stage.ops):
            raise ScheduleContractError(
                f"{field} vector operations must have zero sequential tiles"
            )

    if len(plan.transfers) != len(plan.fifos):
        raise ScheduleContractError(f"{field} FIFO records do not cover every transfer")
    producer_by_tensor: dict[int, int] = {}
    for operation in lowered.operations:
        for tensor in operation.outputs:
            producer_by_tensor[tensor] = operation.index
    for index, (transfer, fifo) in enumerate(
        zip(plan.transfers, plan.fifos, strict=True)
    ):
        transfer_field = f"{field}.transfers[{index}]"
        producer = plan.stages[transfer.producer_stage]
        consumer = plan.stages[transfer.consumer_stage]
        if (
            transfer.producer_stage >= transfer.consumer_stage
            or transfer.producer_engine is not producer.engine
            or transfer.consumer_engine is not consumer.engine
            or transfer.producer_engine is transfer.consumer_engine
        ):
            raise ScheduleContractError(
                f"{transfer_field} does not connect two ordered unlike stages"
            )
        if producer_by_tensor.get(transfer.tensor) not in producer.ops or not any(
            transfer.tensor in lowered.operation(op).inputs for op in consumer.ops
        ):
            raise ScheduleContractError(
                f"{transfer_field}.tensor is not the declared stage boundary"
            )
        direction = (
            MixedTransferDirection.CUBE_TO_VECTOR
            if producer.engine is MixedEngine.CUBE
            else MixedTransferDirection.VECTOR_TO_CUBE
        )
        tensor = lowered.tensor(fifo.tensor)
        wire_dtype = tensor.dtype.lower()
        if direction is MixedTransferDirection.CUBE_TO_VECTOR:
            producers = [
                lowered.operation(op)
                for op in producer.ops
                if lowered.operation(op).outputs == (transfer.tensor,)
            ]
            if len(producers) != 1:
                raise ScheduleContractError(
                    f"{transfer_field}.tensor has no unique producer"
                )
            producing_op = producers[0]
            if producing_op.op_type == "MatMul":
                operand_dtype = lowered.tensor(producing_op.inputs[0]).dtype.lower()
                wire_dtype = (
                    "fp32" if operand_dtype in {"fp32", "fp16", "bf16"} else "int32"
                )
        wire_byte_width = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "int32": 4,
            "int16": 2,
            "int8": 1,
            "bool": 1,
        }[wire_dtype]
        spatial_frame = plan.pipeline_axis is MixedPipelineAxis.SPATIAL_REGION
        expected_rows = plan.m_partition.big if fifo.spatial_m else tensor.height
        expected_cols = plan.n_partition.big if fifo.spatial_n else tensor.width
        if (
            direction is MixedTransferDirection.VECTOR_TO_CUBE
            and producer.vector_stream is not None
            and producer.vector_stream.kind is VectorStreamKind.SOFTMAX_FLASH
            and fifo.spatial_m
            and not fifo.spatial_n
            and producer.vector_stream.extent == tensor.width
        ):
            expected_cols = producer.vector_stream.chunk
        if (
            fifo.tensor != transfer.tensor
            or fifo.direction is not direction
            or fifo.wire_dtype != wire_dtype
            or fifo.pipe_id != index
            or (spatial_frame and fifo.valid_rows != expected_rows)
            or (spatial_frame and fifo.valid_cols != expected_cols)
            or fifo.slot_bytes != fifo.valid_rows * fifo.valid_cols * wire_byte_width
            or fifo.reserved_bytes != fifo.slot_bytes * fifo.slot_count
        ):
            raise ScheduleContractError(
                f"{field}.fifos[{index}] differs from its transfer geometry"
            )

    if plan.protocol is MixedCrossCoreProtocol.ONE_WAY:
        if (
            plan.mode is not MixedPipelineMode.ONE_WAY
            or len(plan.transfers) != 1
            or plan.max_alternations != 1
            or plan.protocol_producer_stages != (plan.transfers[0].producer_stage,)
            or plan.protocol_peer_stage != plan.transfers[0].consumer_stage
            or plan.protocol_producer_bundle != (0,)
            or plan.protocol_reply_bundle
            or plan.protocol_sink_stage is not None
            or plan.protocol_skew_compatible
            or plan.fifos[0].bundle != -1
        ):
            raise ScheduleContractError(f"{field} has an inconsistent one-way protocol")
    elif plan.protocol is MixedCrossCoreProtocol.SINGLE_ROUND_TRIP_BUNDLE:
        producer_bundle = plan.protocol_producer_bundle
        reply_bundle = plan.protocol_reply_bundle
        if (
            not producer_bundle
            or len(reply_bundle) != 1
            or sorted((*producer_bundle, *reply_bundle))
            != list(range(len(plan.transfers)))
        ):
            raise ScheduleContractError(
                f"{field} has an inconsistent single-round-trip protocol"
            )
        producer_stages = tuple(
            plan.transfers[index].producer_stage for index in producer_bundle
        )
        peer_stages = {
            plan.transfers[index].consumer_stage for index in producer_bundle
        }
        sink_stages = {plan.transfers[index].consumer_stage for index in reply_bundle}
        if (
            plan.mode is not MixedPipelineMode.SINGLE_ROUND_TRIP_SKEW
            or plan.protocol_producer_stages != producer_stages
            or peer_stages != {plan.protocol_peer_stage}
            or plan.transfers[reply_bundle[0]].producer_stage
            != plan.protocol_peer_stage
            or sink_stages != {plan.protocol_sink_stage}
            or not plan.protocol_skew_compatible
            or any(plan.fifos[index].bundle != 0 for index in producer_bundle)
            or any(plan.fifos[index].bundle != 1 for index in reply_bundle)
        ):
            raise ScheduleContractError(
                f"{field} has an inconsistent single-round-trip protocol"
            )
    elif plan.protocol is MixedCrossCoreProtocol.BRANCHED_ROUND_TRIP_BUNDLE:
        producer_bundle = plan.protocol_producer_bundle
        reply_bundle = plan.protocol_reply_bundle
        if (
            len(producer_bundle) < 2
            or len(reply_bundle) != len(producer_bundle)
            or sorted((*producer_bundle, *reply_bundle))
            != list(range(len(plan.transfers)))
            or plan.protocol_peer_stage is not None
            or plan.protocol_sink_stage is None
        ):
            raise ScheduleContractError(
                f"{field} has an incomplete branched-round-trip protocol"
            )
        producer_stages = tuple(
            sorted(plan.transfers[index].producer_stage for index in producer_bundle)
        )
        peer_stages = {
            plan.transfers[index].consumer_stage for index in producer_bundle
        }
        reply_producers = {
            plan.transfers[index].producer_stage for index in reply_bundle
        }
        reply_consumers = {
            plan.transfers[index].consumer_stage for index in reply_bundle
        }
        producer_engine = plan.stages[producer_stages[0]].engine
        peer_engine = plan.stages[next(iter(peer_stages))].engine
        incoming_per_peer = {
            peer: sum(
                plan.transfers[index].consumer_stage == peer
                for index in producer_bundle
            )
            for peer in peer_stages
        }
        replies_per_peer = {
            peer: sum(
                plan.transfers[index].producer_stage == peer for index in reply_bundle
            )
            for peer in peer_stages
        }
        if (
            plan.mode is not MixedPipelineMode.SINGLE_ROUND_TRIP_SKEW
            or plan.protocol_producer_stages != producer_stages
            or peer_stages != reply_producers
            or reply_consumers != {plan.protocol_sink_stage}
            or any(count != 1 for count in incoming_per_peer.values())
            or any(count != 1 for count in replies_per_peer.values())
            or any(
                plan.stages[index].engine is not producer_engine
                for index in producer_stages
            )
            or any(
                plan.stages[index].engine is not peer_engine for index in peer_stages
            )
            or producer_engine is peer_engine
            or plan.stages[plan.protocol_sink_stage].engine is not producer_engine
            or not plan.protocol_skew_compatible
            or any(plan.fifos[index].bundle != 0 for index in producer_bundle)
            or any(plan.fifos[index].bundle != 1 for index in reply_bundle)
        ):
            raise ScheduleContractError(
                f"{field} has an inconsistent branched-round-trip protocol"
            )
    elif plan.protocol is MixedCrossCoreProtocol.MULTI_ROUND_TRIP_SEQUENTIAL:
        expected_engines = (
            MixedEngine.CUBE,
            MixedEngine.VECTOR,
            MixedEngine.CUBE,
            MixedEngine.VECTOR,
        )
        if (
            plan.mode is not MixedPipelineMode.MULTI_ROUND_TRIP_SEQUENTIAL
            or plan.algorithm is not MixedAlgorithm.GENERIC
            or tuple(stage.engine for stage in plan.stages) != expected_engines
            or len(plan.transfers) != 3
            or any(
                transfer.producer_stage != index or transfer.consumer_stage != index + 1
                for index, transfer in enumerate(plan.transfers)
            )
            or tuple(fifo.direction for fifo in plan.fifos)
            != (
                MixedTransferDirection.CUBE_TO_VECTOR,
                MixedTransferDirection.VECTOR_TO_CUBE,
                MixedTransferDirection.CUBE_TO_VECTOR,
            )
            or plan.max_alternations != 3
            or plan.protocol_producer_stages
            or plan.protocol_peer_stage is not None
            or plan.protocol_sink_stage is not None
            or plan.protocol_producer_bundle
            or plan.protocol_reply_bundle
            or plan.protocol_skew_compatible
            or any(fifo.bundle != -1 for fifo in plan.fifos)
            or plan.pipeline_stages != 1
            or plan.requested_skew_depth != 0
            or plan.model_overlap_granted
            or plan.overlap_implementable
            or plan.pipeline_fill_absorbed
        ):
            raise ScheduleContractError(
                f"{field} has an inconsistent sequential multi-round-trip protocol"
            )
    else:
        raise ScheduleContractError(f"{field} source-ready protocol is unsupported")

    if plan.algorithm is MixedAlgorithm.FEATURE_CHUNK_ROUND_TRIP:
        feature = plan.feature_round_trip
        if (
            feature is None
            or feature.intermediate_chunks < 2
            or feature.intermediate_chunks * feature.intermediate_chunk
            != feature.intermediate_extent
            or plan.pipeline_axis is not MixedPipelineAxis.INTERMEDIATE_FEATURE_CHUNK
            or plan.pipeline_extent != feature.intermediate_extent
            or plan.pipeline_chunk != feature.intermediate_chunk
            or plan.items_per_spatial_tile != feature.intermediate_chunks
            or plan.pipeline_work_items
            != plan.spatial_tiles * feature.intermediate_chunks
            or plan.min_trips_per_group != feature.intermediate_chunks
            or plan.max_trips_per_group != feature.intermediate_chunks
            or len(feature.producer_window_k) != len(plan.protocol_producer_bundle)
            or not feature.first_chunk_initializes
            or not feature.later_chunks_accumulate
        ):
            raise ScheduleContractError(f"{field}.feature_round_trip is incomplete")
    elif plan.feature_round_trip is not None:
        raise ScheduleContractError(
            f"{field} generic plan carries feature-round-trip state"
        )


def _mixed_materialized_source_peak(
    plan: VectorKernelPlan, *, lowered: LoweredRegion
) -> int:
    """Return current-main PyPTO's physical mixed-stage Vec allocation bound."""

    if plan.kind is not VectorStreamKind.MATERIALIZED:
        return plan.full_peak_ub_bytes
    workspace_bytes = [
        workspace.physical[0]
        * workspace.physical[1]
        * lowered.tensor(workspace.source_tensor).byte_width
        for phase in plan.phases
        for workspace in phase.workspaces
    ]
    if not workspace_bytes:
        return plan.full_peak_ub_bytes
    return plan.workspace_free_peak_ub_bytes + sum(workspace_bytes)


def _parse_cube_plan(
    value: Any,
    *,
    field: str,
    op_bound: int,
    tensor_bound: int,
) -> CubeKernelPlan:
    item = _mapping(value, field)
    _expect_keys(
        item,
        required={
            "emit_compatible",
            "spatial_policy",
            "m_partition",
            "n_partition",
            "spatial_tiles",
            "split_k",
            "work_units",
            "peak_l1_bytes",
            "split_merge_policy",
            "first_partial_then_atomic",
            "aiv_zero_seed_then_atomic",
            "model_overlap_granted",
            "overlap_implementable",
            "execution_order",
            "resident_boundaries",
            "matmuls",
        },
        field=field,
    )
    residents = tuple(
        _parse_cube_resident(
            resident,
            field=f"{field}.resident_boundaries[{index}]",
            tensor_bound=tensor_bound,
        )
        for index, resident in enumerate(
            _sequence(item.get("resident_boundaries"), f"{field}.resident_boundaries")
        )
    )
    if len({resident.id for resident in residents}) != len(residents):
        raise ScheduleContractError(
            f"{field}.resident_boundaries must use unique canonical ids"
        )
    matmuls = tuple(
        _parse_cube_matmul(
            matmul,
            field=f"{field}.matmuls[{index}]",
            op_bound=op_bound,
            tensor_bound=tensor_bound,
            resident_bound=len(residents),
        )
        for index, matmul in enumerate(
            _sequence(item.get("matmuls"), f"{field}.matmuls")
        )
    )
    if tuple(matmul.instance for matmul in matmuls) != tuple(range(len(matmuls))):
        raise ScheduleContractError(f"{field}.matmuls must use dense ordered instances")
    split = _mapping(
        item.get("first_partial_then_atomic"), f"{field}.first_partial_then_atomic"
    )
    _expect_keys(
        split,
        required={
            "present",
            "first_work_units",
            "atomic_work_units",
            "synchronization_cycles",
        },
        field=f"{field}.first_partial_then_atomic",
    )
    zero_seed = _mapping(
        item.get("aiv_zero_seed_then_atomic"),
        f"{field}.aiv_zero_seed_then_atomic",
    )
    _expect_keys(
        zero_seed,
        required={
            "present",
            "seed_work_units",
            "atomic_work_units",
            "seed_bytes",
            "synchronization_cycles",
        },
        field=f"{field}.aiv_zero_seed_then_atomic",
    )
    result = CubeKernelPlan(
        emit_compatible=_bool(item.get("emit_compatible"), f"{field}.emit_compatible"),
        spatial_policy=_enum(
            CubeSpatialPolicy, item.get("spatial_policy"), f"{field}.spatial_policy"
        ),
        m_partition=_parse_axis_partition(
            item.get("m_partition"), field=f"{field}.m_partition"
        ),
        n_partition=_parse_axis_partition(
            item.get("n_partition"), field=f"{field}.n_partition"
        ),
        spatial_tiles=_positive_int(
            item.get("spatial_tiles"), f"{field}.spatial_tiles"
        ),
        split_k=_positive_int(item.get("split_k"), f"{field}.split_k"),
        work_units=_positive_int(item.get("work_units"), f"{field}.work_units"),
        peak_l1_bytes=_nonnegative_int(
            item.get("peak_l1_bytes"), f"{field}.peak_l1_bytes"
        ),
        split_merge_policy=_enum(
            CubeSplitMergePolicy,
            item.get("split_merge_policy"),
            f"{field}.split_merge_policy",
        ),
        first_partial_then_atomic=CubeFirstPartialThenAtomicPlan(
            present=_bool(
                split.get("present"), f"{field}.first_partial_then_atomic.present"
            ),
            first_work_units=_nonnegative_int(
                split.get("first_work_units"),
                f"{field}.first_partial_then_atomic.first_work_units",
            ),
            atomic_work_units=_nonnegative_int(
                split.get("atomic_work_units"),
                f"{field}.first_partial_then_atomic.atomic_work_units",
            ),
            synchronization_cycles=_finite_number(
                split.get("synchronization_cycles"),
                f"{field}.first_partial_then_atomic.synchronization_cycles",
            ),
        ),
        aiv_zero_seed_then_atomic=CubeAivZeroSeedThenAtomicPlan(
            present=_bool(
                zero_seed.get("present"),
                f"{field}.aiv_zero_seed_then_atomic.present",
            ),
            seed_work_units=_nonnegative_int(
                zero_seed.get("seed_work_units"),
                f"{field}.aiv_zero_seed_then_atomic.seed_work_units",
            ),
            atomic_work_units=_nonnegative_int(
                zero_seed.get("atomic_work_units"),
                f"{field}.aiv_zero_seed_then_atomic.atomic_work_units",
            ),
            seed_bytes=_nonnegative_int(
                zero_seed.get("seed_bytes"),
                f"{field}.aiv_zero_seed_then_atomic.seed_bytes",
            ),
            synchronization_cycles=_finite_number(
                zero_seed.get("synchronization_cycles"),
                f"{field}.aiv_zero_seed_then_atomic.synchronization_cycles",
            ),
        ),
        model_overlap_granted=_bool(
            item.get("model_overlap_granted"), f"{field}.model_overlap_granted"
        ),
        overlap_implementable=_bool(
            item.get("overlap_implementable"), f"{field}.overlap_implementable"
        ),
        execution_order=tuple(
            _bounded_int(op, f"{field}.execution_order", op_bound)
            for op in _sequence(item.get("execution_order"), f"{field}.execution_order")
        ),
        resident_boundaries=residents,
        matmuls=matmuls,
    )
    _validate_cube_links(result, field=field)
    return result


def _parse_cube_resident(
    value: Any, *, field: str, tensor_bound: int
) -> CubeResidentBoundaryPlan:
    item = _mapping(value, field)
    _expect_keys(
        item,
        required={
            "id",
            "region",
            "role",
            "first_use",
            "last_use",
            "use_count",
            "bytes",
        },
        field=field,
    )
    first = _nonnegative_int(item.get("first_use"), f"{field}.first_use")
    last = _nonnegative_int(item.get("last_use"), f"{field}.last_use")
    if first > last:
        raise ScheduleContractError(f"{field} has an inverted lifetime")
    return CubeResidentBoundaryPlan(
        id=_nonnegative_int(item.get("id"), f"{field}.id"),
        region=_parse_cube_region(
            item.get("region"), field=f"{field}.region", tensor_bound=tensor_bound
        ),
        role=_enum(CubeOperandRole, item.get("role"), f"{field}.role"),
        first_use=first,
        last_use=last,
        use_count=_positive_int(item.get("use_count"), f"{field}.use_count"),
        bytes=_positive_int(item.get("bytes"), f"{field}.bytes"),
    )


def _parse_cube_matmul(
    value: Any,
    *,
    field: str,
    op_bound: int,
    tensor_bound: int,
    resident_bound: int,
) -> CubeMatmulPlan:
    item = _mapping(value, field)
    _expect_keys(
        item,
        required={
            "instance",
            "op",
            "lhs_producer",
            "rhs_producer",
            "lhs_resident_boundary",
            "rhs_resident_boundary",
            "is_sink",
            "lhs_ephemeral",
            "rhs_ephemeral",
            "output_ephemeral",
            "contraction",
            "effective_contraction",
            "accumulator_dtype",
            "storage_dtype",
            "lhs",
            "rhs",
            "output",
            "k_loop",
            "output_tile",
            "output_grid",
            "output_variants",
            "retained_panels",
            "final_drain",
        },
        field=field,
    )
    retained = _mapping(item.get("retained_panels"), f"{field}.retained_panels")
    _expect_keys(
        retained,
        required={"lhs", "rhs", "lhs_bytes", "rhs_bytes"},
        field=f"{field}.retained_panels",
    )
    return CubeMatmulPlan(
        instance=_nonnegative_int(item.get("instance"), f"{field}.instance"),
        op=_bounded_int(item.get("op"), f"{field}.op", op_bound),
        lhs_producer=_optional_index(item.get("lhs_producer"), f"{field}.lhs_producer"),
        rhs_producer=_optional_index(item.get("rhs_producer"), f"{field}.rhs_producer"),
        lhs_resident_boundary=_optional_index(
            item.get("lhs_resident_boundary"),
            f"{field}.lhs_resident_boundary",
            bound=resident_bound,
        ),
        rhs_resident_boundary=_optional_index(
            item.get("rhs_resident_boundary"),
            f"{field}.rhs_resident_boundary",
            bound=resident_bound,
        ),
        is_sink=_bool(item.get("is_sink"), f"{field}.is_sink"),
        lhs_ephemeral=_bool(item.get("lhs_ephemeral"), f"{field}.lhs_ephemeral"),
        rhs_ephemeral=_bool(item.get("rhs_ephemeral"), f"{field}.rhs_ephemeral"),
        output_ephemeral=_bool(
            item.get("output_ephemeral"), f"{field}.output_ephemeral"
        ),
        contraction=_positive_int(item.get("contraction"), f"{field}.contraction"),
        effective_contraction=_positive_int(
            item.get("effective_contraction"), f"{field}.effective_contraction"
        ),
        accumulator_dtype=_dtype(
            item.get("accumulator_dtype"), f"{field}.accumulator_dtype"
        ),
        storage_dtype=_dtype(item.get("storage_dtype"), f"{field}.storage_dtype"),
        lhs=_parse_cube_region(
            item.get("lhs"), field=f"{field}.lhs", tensor_bound=tensor_bound
        ),
        rhs=_parse_cube_region(
            item.get("rhs"), field=f"{field}.rhs", tensor_bound=tensor_bound
        ),
        output=_parse_cube_region(
            item.get("output"), field=f"{field}.output", tensor_bound=tensor_bound
        ),
        k_loop=_parse_cube_k_loop(item.get("k_loop"), field=f"{field}.k_loop"),
        output_tile=_int_tuple(item.get("output_tile"), 2, f"{field}.output_tile"),
        output_grid=_int_tuple(item.get("output_grid"), 2, f"{field}.output_grid"),
        output_variants=tuple(
            _parse_cube_output_variant(
                variant, field=f"{field}.output_variants[{index}]"
            )
            for index, variant in enumerate(
                _sequence(item.get("output_variants"), f"{field}.output_variants")
            )
        ),
        retained_panels=CubeRetainedPanelPlan(
            lhs=_bool(retained.get("lhs"), f"{field}.retained_panels.lhs"),
            rhs=_bool(retained.get("rhs"), f"{field}.retained_panels.rhs"),
            lhs_bytes=_nonnegative_int(
                retained.get("lhs_bytes"), f"{field}.retained_panels.lhs_bytes"
            ),
            rhs_bytes=_nonnegative_int(
                retained.get("rhs_bytes"), f"{field}.retained_panels.rhs_bytes"
            ),
        ),
        final_drain=_parse_cube_final_drain(
            item.get("final_drain"), field=f"{field}.final_drain"
        ),
    )


def _parse_cube_region(
    value: Any, *, field: str, tensor_bound: int
) -> CubeTensorRegionPlan:
    item = _mapping(value, field)
    _expect_keys(
        item,
        required={"tensor", "height_binding", "width_binding", "height", "width"},
        field=field,
    )
    return CubeTensorRegionPlan(
        tensor=_bounded_int(item.get("tensor"), f"{field}.tensor", tensor_bound),
        height_binding=_enum(
            CubeAxisBinding, item.get("height_binding"), f"{field}.height_binding"
        ),
        width_binding=_enum(
            CubeAxisBinding, item.get("width_binding"), f"{field}.width_binding"
        ),
        height=_positive_int(item.get("height"), f"{field}.height"),
        width=_positive_int(item.get("width"), f"{field}.width"),
    )


def _parse_cube_k_loop(value: Any, *, field: str) -> CubeKLoopPlan:
    item = _mapping(value, field)
    _expect_keys(
        item,
        required={"l1_window_k", "chunk", "full_chunks", "tail", "pipeline_stages"},
        field=field,
    )
    return CubeKLoopPlan(
        l1_window_k=_positive_int(item.get("l1_window_k"), f"{field}.l1_window_k"),
        chunk=_positive_int(item.get("chunk"), f"{field}.chunk"),
        full_chunks=_nonnegative_int(item.get("full_chunks"), f"{field}.full_chunks"),
        tail=_nonnegative_int(item.get("tail"), f"{field}.tail"),
        pipeline_stages=_positive_int(
            item.get("pipeline_stages"), f"{field}.pipeline_stages"
        ),
    )


def _parse_cube_output_variant(value: Any, *, field: str) -> CubeOutputTileVariant:
    item = _mapping(value, field)
    _expect_keys(
        item,
        required={"shape", "count", "l0_init", "l0_rolled", "l0_tail"},
        field=field,
    )
    l0_init = _parse_l0_matmul(
        item.get("l0_init"), field=f"{field}.l0_init", required=True
    )
    if l0_init is None:
        raise AssertionError("required L0 plan parser returned None")
    return CubeOutputTileVariant(
        shape=_int_tuple(item.get("shape"), 2, f"{field}.shape"),
        count=_positive_int(item.get("count"), f"{field}.count"),
        l0_init=l0_init,
        l0_rolled=_parse_l0_matmul(
            item.get("l0_rolled"), field=f"{field}.l0_rolled", required=False
        ),
        l0_tail=_parse_l0_matmul(
            item.get("l0_tail"), field=f"{field}.l0_tail", required=False
        ),
    )


def _parse_l0_matmul(value: Any, *, field: str, required: bool) -> L0MatmulPlan | None:
    if value is None:
        if required:
            raise ScheduleContractError(f"{field} must contain an L0 plan")
        return None
    item = _mapping(value, field)
    _expect_keys(
        item,
        required={
            "tile",
            "stationarity",
            "output_stationary_holds_a",
            "buffer_depths",
            "output_target",
            "k_loop",
            "estimated_traffic_bytes",
            "estimated_cost_cycles",
            "padded_compute_volume",
            "phases",
        },
        field=field,
    )
    k_loop = _mapping(item.get("k_loop"), f"{field}.k_loop")
    _expect_keys(
        k_loop,
        required={"chunk", "full_chunks", "tail", "pipeline_stages"},
        field=f"{field}.k_loop",
    )
    phases = _mapping(item.get("phases"), f"{field}.phases")
    phase_names = {
        "load_cycles",
        "mad_cycles",
        "init_cycles",
        "rolled_cycles",
        "tail_cycles",
        "drain_cycles",
        "wall_cycles",
    }
    _expect_keys(phases, required=phase_names, field=f"{field}.phases")
    return L0MatmulPlan(
        tile=_int_tuple(item.get("tile"), 3, f"{field}.tile"),
        stationarity=_enum(
            L0Stationarity, item.get("stationarity"), f"{field}.stationarity"
        ),
        output_stationary_holds_a=_bool(
            item.get("output_stationary_holds_a"),
            f"{field}.output_stationary_holds_a",
        ),
        buffer_depths=_int_tuple(
            item.get("buffer_depths"), 3, f"{field}.buffer_depths"
        ),
        output_target=_enum(
            L0OutputTarget, item.get("output_target"), f"{field}.output_target"
        ),
        k_loop=L0KLoopPlan(
            chunk=_positive_int(k_loop.get("chunk"), f"{field}.k_loop.chunk"),
            full_chunks=_nonnegative_int(
                k_loop.get("full_chunks"), f"{field}.k_loop.full_chunks"
            ),
            tail=_nonnegative_int(k_loop.get("tail"), f"{field}.k_loop.tail"),
            pipeline_stages=_positive_int(
                k_loop.get("pipeline_stages"), f"{field}.k_loop.pipeline_stages"
            ),
        ),
        estimated_traffic_bytes=_nonnegative_int(
            item.get("estimated_traffic_bytes"), f"{field}.estimated_traffic_bytes"
        ),
        estimated_cost_cycles=_finite_number(
            item.get("estimated_cost_cycles"), f"{field}.estimated_cost_cycles"
        ),
        padded_compute_volume=_nonnegative_int(
            item.get("padded_compute_volume"), f"{field}.padded_compute_volume"
        ),
        phases=L0PhaseCostPlan(
            **{
                name: _finite_number(phases.get(name), f"{field}.phases.{name}")
                for name in phase_names
            }
        ),
    )


def _parse_cube_final_drain(value: Any, *, field: str) -> CubeFinalDrainPlan:
    item = _mapping(value, field)
    _expect_keys(
        item,
        required={
            "required",
            "target_l1",
            "atomic",
            "valid_rows",
            "valid_cols",
            "tile_count",
            "bytes",
            "cycles",
        },
        field=field,
    )
    return CubeFinalDrainPlan(
        required=_bool(item.get("required"), f"{field}.required"),
        target_l1=_bool(item.get("target_l1"), f"{field}.target_l1"),
        atomic=_bool(item.get("atomic"), f"{field}.atomic"),
        valid_rows=_nonnegative_int(item.get("valid_rows"), f"{field}.valid_rows"),
        valid_cols=_nonnegative_int(item.get("valid_cols"), f"{field}.valid_cols"),
        tile_count=_nonnegative_int(item.get("tile_count"), f"{field}.tile_count"),
        bytes=_nonnegative_int(item.get("bytes"), f"{field}.bytes"),
        cycles=_finite_number(item.get("cycles"), f"{field}.cycles"),
    )


def _validate_cube_links(plan: CubeKernelPlan, *, field: str) -> None:
    for matmul in plan.matmuls:
        if (
            matmul.lhs_producer >= matmul.instance
            or matmul.rhs_producer >= matmul.instance
        ):
            raise ScheduleContractError(
                f"{field}.matmuls[{matmul.instance}] has a non-prior producer"
            )
        for role, resident_id in (
            (CubeOperandRole.LHS, matmul.lhs_resident_boundary),
            (CubeOperandRole.RHS, matmul.rhs_resident_boundary),
        ):
            if resident_id < 0:
                continue
            resident = plan.resident_boundaries[resident_id]
            if resident.role is not role:
                raise ScheduleContractError(
                    f"{field}.matmuls[{matmul.instance}] links {role.value} to a "
                    f"{resident.role.value} resident boundary"
                )
            if not (resident.first_use <= matmul.instance <= resident.last_use):
                raise ScheduleContractError(
                    f"{field}.matmuls[{matmul.instance}] lies outside resident "
                    f"boundary {resident_id}'s lifetime"
                )


def _validate_cube_contract(
    plan: CubeKernelPlan,
    *,
    lowered: LoweredRegion,
    step_ops: tuple[int, ...],
    step_order: tuple[int, ...],
    sequential_tiles: tuple[int, ...] | None,
    launch: LaunchPlan,
    field: str,
) -> None:
    spatial_tiles = plan.m_partition.parts * plan.n_partition.parts
    if (
        launch.parts_m != plan.m_partition.parts
        or launch.parts_n != plan.n_partition.parts
    ):
        raise ScheduleContractError(f"{field} launch grid differs from its partitions")
    if launch.tile_h != plan.m_partition.big or launch.tile_w != plan.n_partition.big:
        raise ScheduleContractError(
            f"{field} launch tile differs from its spatial partitions"
        )
    if plan.spatial_tiles != spatial_tiles:
        raise ScheduleContractError(
            f"{field}.spatial_tiles differs from its partition grid"
        )
    if launch.split != plan.split_k:
        raise ScheduleContractError(f"{field}.split_k differs from the launch split")
    if plan.work_units != spatial_tiles * plan.split_k:
        raise ScheduleContractError(f"{field}.work_units differs from grid times split")
    if launch.cores > plan.work_units:
        raise ScheduleContractError(f"{field} uses more cores than work units")

    split = plan.first_partial_then_atomic
    zero_seed = plan.aiv_zero_seed_then_atomic
    if split.synchronization_cycles < 0:
        raise ScheduleContractError(
            f"{field}.first_partial_then_atomic has negative synchronization cost"
        )
    if zero_seed.synchronization_cycles < 0:
        raise ScheduleContractError(
            f"{field}.aiv_zero_seed_then_atomic has negative synchronization cost"
        )
    split_empty = (
        not split.present
        and split.first_work_units == 0
        and split.atomic_work_units == 0
        and split.synchronization_cycles == 0
    )
    zero_seed_empty = (
        not zero_seed.present
        and zero_seed.seed_work_units == 0
        and zero_seed.atomic_work_units == 0
        and zero_seed.seed_bytes == 0
        and zero_seed.synchronization_cycles == 0
    )
    if plan.split_k == 1:
        if (
            plan.split_merge_policy is not CubeSplitMergePolicy.NONE
            or not split_empty
            or not zero_seed_empty
        ):
            raise ScheduleContractError(f"{field} has split-merge work for split_k=1")
    elif plan.split_merge_policy is CubeSplitMergePolicy.FIRST_PARTIAL_THEN_ATOMIC:
        if (
            not split.present
            or split.first_work_units != spatial_tiles
            or split.atomic_work_units != spatial_tiles * (plan.split_k - 1)
            or not zero_seed_empty
        ):
            raise ScheduleContractError(
                f"{field} has an inconsistent FirstPartialThenAtomic descriptor"
            )
    elif plan.split_merge_policy is CubeSplitMergePolicy.AIV_ZERO_SEED_THEN_ATOMIC:
        sinks = tuple(matmul for matmul in plan.matmuls if matmul.is_sink)
        expected_seed_bytes = spatial_tiles * sum(
            matmul.final_drain.bytes for matmul in sinks
        )
        if (
            not split_empty
            or not zero_seed.present
            or zero_seed.seed_work_units != spatial_tiles
            or zero_seed.atomic_work_units != plan.work_units
            or len(sinks) != 1
            or zero_seed.seed_bytes != expected_seed_bytes
            or zero_seed.seed_bytes <= 0
        ):
            raise ScheduleContractError(
                f"{field} has an inconsistent AivZeroSeedThenAtomic descriptor"
            )
    else:
        raise ScheduleContractError(
            f"{field} has no supported merge policy for split_k>1"
        )

    step_set = set(step_ops)
    if any(matmul.op not in step_set for matmul in plan.matmuls):
        raise ScheduleContractError(
            f"{field}.matmuls references an op outside its step"
        )
    if plan.execution_order != tuple(matmul.op for matmul in plan.matmuls):
        raise ScheduleContractError(
            f"{field} execution order differs from the serialized request order"
        )

    resident_uses: list[list[tuple[int, CubeOperandRole]]] = [
        [] for _ in plan.resident_boundaries
    ]
    if sequential_tiles is None:
        raise ScheduleContractError(f"{field} omits cube sequential tiles")
    sequential_by_op = dict(zip(step_order, sequential_tiles, strict=True))
    for matmul in plan.matmuls:
        matmul_field = f"{field}.matmuls[{matmul.instance}]"
        operation = lowered.operation(matmul.op)
        if (
            operation.op_type != "MatMul"
            or len(operation.inputs) != 2
            or len(operation.outputs) != 1
        ):
            raise ScheduleContractError(
                f"{matmul_field} does not describe a lowered matmul"
            )
        if (
            matmul.lhs.tensor != operation.inputs[0]
            or matmul.rhs.tensor != operation.inputs[1]
            or matmul.output.tensor != operation.outputs[0]
        ):
            raise ScheduleContractError(
                f"{matmul_field} tensor regions differ from the lowered matmul operands"
            )
        lhs_tensor = lowered.tensor(operation.inputs[0])
        rhs_tensor = lowered.tensor(operation.inputs[1])
        output_tensor = lowered.tensor(operation.outputs[0])
        if (
            matmul.contraction != lhs_tensor.width
            or matmul.contraction != rhs_tensor.height
        ):
            raise ScheduleContractError(
                f"{matmul_field}.contraction differs from its operand tensors"
            )
        if (
            output_tensor.height != lhs_tensor.height
            or output_tensor.width != rhs_tensor.width
        ):
            raise ScheduleContractError(
                f"{matmul_field} output tensor geometry differs from its operands"
            )
        for name, region in (
            ("lhs", matmul.lhs),
            ("rhs", matmul.rhs),
            ("output", matmul.output),
        ):
            _validate_cube_region_geometry(
                region,
                lowered=lowered,
                m_extent=plan.m_partition.big,
                n_extent=plan.n_partition.big,
                split=plan.split_k,
                field=f"{matmul_field}.{name}",
            )
        if matmul.effective_contraction > matmul.contraction:
            raise ScheduleContractError(
                f"{matmul_field}.effective_contraction exceeds the full contraction"
            )
        if (
            matmul.lhs.width != matmul.effective_contraction
            or matmul.rhs.height != matmul.effective_contraction
            or matmul.output.height != matmul.lhs.height
            or matmul.output.width != matmul.rhs.width
        ):
            raise ScheduleContractError(
                f"{matmul_field} regions do not form its scheduled matmul"
            )
        loop = matmul.k_loop
        if sequential_by_op[matmul.op] != loop.l1_window_k:
            raise ScheduleContractError(
                f"{matmul_field}.k_loop differs from its common sequential tile"
            )
        if loop.full_chunks * loop.chunk + loop.tail != matmul.effective_contraction:
            raise ScheduleContractError(f"{matmul_field}.k_loop does not cover K")
        if loop.tail >= loop.chunk or loop.l1_window_k < loop.chunk:
            raise ScheduleContractError(
                f"{matmul_field}.k_loop has invalid chunk geometry"
            )
        if loop.pipeline_stages not in {1, 2}:
            raise ScheduleContractError(
                f"{matmul_field}.k_loop has unsupported pipeline depth"
            )

        for role, producer_index, operand in (
            (CubeOperandRole.LHS, matmul.lhs_producer, matmul.lhs),
            (CubeOperandRole.RHS, matmul.rhs_producer, matmul.rhs),
        ):
            if producer_index < 0:
                continue
            producer = plan.matmuls[producer_index]
            if not _cube_regions_address_same_value(producer.output, operand):
                raise ScheduleContractError(
                    f"{matmul_field} {role.value} region differs from producer "
                    f"{producer_index}'s output region"
                )

        for role, resident_index, operand in (
            (CubeOperandRole.LHS, matmul.lhs_resident_boundary, matmul.lhs),
            (CubeOperandRole.RHS, matmul.rhs_resident_boundary, matmul.rhs),
        ):
            if resident_index < 0:
                continue
            resident = plan.resident_boundaries[resident_index]
            if resident.region != operand:
                raise ScheduleContractError(
                    f"{matmul_field} {role.value} region differs from resident "
                    f"boundary {resident_index}"
                )
            resident_uses[resident_index].append((matmul.instance, role))

        _validate_cube_output_contract(
            matmul,
            lowered=lowered,
            split_k=plan.split_k,
            field=matmul_field,
        )

    for index, resident in enumerate(plan.resident_boundaries):
        uses = resident_uses[index]
        if len(uses) != resident.use_count:
            raise ScheduleContractError(
                f"{field}.resident_boundaries[{index}].use_count is stale"
            )
        instances = [instance for instance, _ in uses]
        if (
            not instances
            or resident.first_use != min(instances)
            or resident.last_use != max(instances)
        ):
            raise ScheduleContractError(
                f"{field}.resident_boundaries[{index}] has stale lifetime bounds"
            )
        expected_bytes = (
            resident.region.height
            * resident.region.width
            * lowered.tensor(resident.region.tensor).byte_width
        )
        if resident.bytes != expected_bytes:
            raise ScheduleContractError(
                f"{field}.resident_boundaries[{index}].bytes differs from its tensor region"
            )


def _validate_cube_region_geometry(
    region: CubeTensorRegionPlan,
    *,
    lowered: LoweredRegion,
    m_extent: int,
    n_extent: int,
    split: int,
    field: str,
) -> None:
    tensor = lowered.tensor(region.tensor)
    expected_height = _cube_binding_extent(
        region.height_binding,
        tensor.height,
        m_extent=m_extent,
        n_extent=n_extent,
        split=split,
        field=f"{field}.height",
    )
    expected_width = _cube_binding_extent(
        region.width_binding,
        tensor.width,
        m_extent=m_extent,
        n_extent=n_extent,
        split=split,
        field=f"{field}.width",
    )
    if region.height != expected_height or region.width != expected_width:
        raise ScheduleContractError(
            f"{field} extent differs from its tensor and axis bindings"
        )


def _cube_binding_extent(
    binding: CubeAxisBinding,
    full_extent: int,
    *,
    m_extent: int,
    n_extent: int,
    split: int,
    field: str,
) -> int:
    if binding in {CubeAxisBinding.FULL, CubeAxisBinding.SEQUENTIAL_K}:
        return full_extent
    if binding is CubeAxisBinding.SPATIAL_M:
        return min(full_extent, m_extent)
    if binding is CubeAxisBinding.SPATIAL_N:
        return min(full_extent, n_extent)
    if full_extent % split:
        raise ScheduleContractError(
            f"{field} cannot divide extent {full_extent} across split {split}"
        )
    return full_extent // split


def _validate_cube_output_contract(
    matmul: CubeMatmulPlan,
    *,
    lowered: LoweredRegion,
    split_k: int,
    field: str,
) -> None:
    tile_m, tile_n = matmul.output_tile
    expected_grid = (
        _ceil_div(matmul.output.height, tile_m),
        _ceil_div(matmul.output.width, tile_n),
    )
    if matmul.output_grid != expected_grid:
        raise ScheduleContractError(
            f"{field}.output_grid does not cover its output region"
        )

    expected_variants: dict[tuple[int, int], int] = {}
    full_m, tail_m = divmod(matmul.output.height, tile_m)
    full_n, tail_n = divmod(matmul.output.width, tile_n)
    for shape, count in (
        ((tile_m, tile_n), full_m * full_n),
        ((tail_m, tile_n), full_n),
        ((tile_m, tail_n), full_m),
        ((tail_m, tail_n), 1 if tail_m and tail_n else 0),
    ):
        if shape[0] and shape[1] and count:
            expected_variants[shape] = expected_variants.get(shape, 0) + count
    actual_variants = {
        variant.shape: variant.count for variant in matmul.output_variants
    }
    if (
        len(actual_variants) != len(matmul.output_variants)
        or actual_variants != expected_variants
    ):
        raise ScheduleContractError(
            f"{field}.output_variants do not partition the output grid"
        )

    chunked = matmul.k_loop.full_chunks >= 2
    init_k = matmul.k_loop.chunk if chunked else matmul.effective_contraction
    for index, variant in enumerate(matmul.output_variants):
        variant_field = f"{field}.output_variants[{index}]"
        _validate_l0_contract(
            variant.l0_init,
            expected_shape=variant.shape,
            expected_contraction=init_k,
            field=f"{variant_field}.l0_init",
        )
        if chunked:
            if variant.l0_rolled is None:
                raise ScheduleContractError(f"{variant_field}.l0_rolled is required")
            _validate_l0_contract(
                variant.l0_rolled,
                expected_shape=variant.shape,
                expected_contraction=matmul.k_loop.chunk,
                field=f"{variant_field}.l0_rolled",
            )
        elif variant.l0_rolled is not None:
            raise ScheduleContractError(f"{variant_field}.l0_rolled is unexpected")
        if chunked and matmul.k_loop.tail:
            if variant.l0_tail is None:
                raise ScheduleContractError(f"{variant_field}.l0_tail is required")
            _validate_l0_contract(
                variant.l0_tail,
                expected_shape=variant.shape,
                expected_contraction=matmul.k_loop.tail,
                field=f"{variant_field}.l0_tail",
            )
        elif variant.l0_tail is not None:
            raise ScheduleContractError(f"{variant_field}.l0_tail is unexpected")

    drain = matmul.final_drain
    expected_tiles = matmul.output_grid[0] * matmul.output_grid[1]
    expected_bytes = (
        matmul.output.height
        * matmul.output.width
        * lowered.tensor(matmul.output.tensor).byte_width
    )
    if (
        not drain.required
        or drain.target_l1 != (not matmul.is_sink)
        or drain.atomic != (matmul.is_sink and split_k > 1)
    ):
        raise ScheduleContractError(f"{field}.final_drain has inconsistent ownership")
    if (
        drain.valid_rows != tile_m
        or drain.valid_cols != tile_n
        or drain.tile_count != expected_tiles
        or drain.bytes != expected_bytes
        or drain.cycles < 0
    ):
        raise ScheduleContractError(f"{field}.final_drain differs from its output grid")

    retained = matmul.retained_panels
    expected_lhs_bytes = (
        matmul.lhs.height
        * matmul.lhs.width
        * lowered.tensor(matmul.lhs.tensor).byte_width
        if retained.lhs
        else 0
    )
    expected_rhs_bytes = (
        matmul.rhs.height
        * matmul.rhs.width
        * lowered.tensor(matmul.rhs.tensor).byte_width
        if retained.rhs
        else 0
    )
    if (
        retained.lhs_bytes != expected_lhs_bytes
        or retained.rhs_bytes != expected_rhs_bytes
    ):
        raise ScheduleContractError(f"{field}.retained_panels byte count is stale")


def _cube_regions_address_same_value(
    producer: CubeTensorRegionPlan,
    consumer: CubeTensorRegionPlan,
) -> bool:
    """Return whether two propagated regions name the same per-task L1 value."""

    def same_axis(left: CubeAxisBinding, right: CubeAxisBinding) -> bool:
        full_bindings = {CubeAxisBinding.FULL, CubeAxisBinding.SEQUENTIAL_K}
        return left is right or {left, right}.issubset(full_bindings)

    return (
        producer.tensor == consumer.tensor
        and producer.height == consumer.height
        and producer.width == consumer.width
        and same_axis(producer.height_binding, consumer.height_binding)
        and same_axis(producer.width_binding, consumer.width_binding)
    )


def _validate_l0_contract(
    plan: L0MatmulPlan,
    *,
    expected_shape: tuple[int, int],
    expected_contraction: int,
    field: str,
) -> None:
    if plan.tile[:2] != expected_shape:
        raise ScheduleContractError(f"{field}.tile differs from its output variant")
    if any(depth <= 0 for depth in plan.buffer_depths):
        raise ScheduleContractError(f"{field}.buffer_depths must be positive")
    loop = plan.k_loop
    if plan.tile[2] != loop.chunk:
        raise ScheduleContractError(f"{field}.tile K differs from its L0 loop chunk")
    if loop.full_chunks * loop.chunk + loop.tail != expected_contraction:
        raise ScheduleContractError(
            f"{field}.k_loop does not cover its requested contraction"
        )
    if loop.tail >= loop.chunk or loop.pipeline_stages not in {1, 2}:
        raise ScheduleContractError(f"{field}.k_loop has invalid geometry")
    if (
        plan.estimated_traffic_bytes < 0
        or plan.estimated_cost_cycles < 0
        or plan.padded_compute_volume < 0
    ):
        raise ScheduleContractError(f"{field} contains a negative cost quantity")


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _parse_axis_partition(value: Any, *, field: str) -> AxisPartition:
    item = _mapping(value, field)
    _expect_keys(item, required={"big", "small", "num_big", "parts"}, field=field)
    result = AxisPartition(
        big=_positive_int(item.get("big"), f"{field}.big"),
        small=_positive_int(item.get("small"), f"{field}.small"),
        num_big=_nonnegative_int(item.get("num_big"), f"{field}.num_big"),
        parts=_positive_int(item.get("parts"), f"{field}.parts"),
    )
    if result.small > result.big or result.num_big > result.parts:
        raise ScheduleContractError(f"{field} is not a valid balanced partition")
    return result


def _parse_optional_loop(value: Any, *, field: str) -> VectorLoopPlan | None:
    if value is None:
        return None
    item = _mapping(value, field)
    _expect_keys(
        item,
        required={"first_chunk", "trip_count", "pipeline_stages"},
        field=field,
    )
    return VectorLoopPlan(
        first_chunk=_nonnegative_int(item.get("first_chunk"), f"{field}.first_chunk"),
        trip_count=_nonnegative_int(item.get("trip_count"), f"{field}.trip_count"),
        pipeline_stages=_positive_int(
            item.get("pipeline_stages"), f"{field}.pipeline_stages"
        ),
    )


def _parse_optional_serial(value: Any, *, field: str) -> VectorSerialPhasePlan | None:
    if value is None:
        return None
    item = _mapping(value, field)
    _expect_keys(item, required={"present", "chunk_index", "extent"}, field=field)
    return VectorSerialPhasePlan(
        present=_bool(item.get("present"), f"{field}.present"),
        chunk_index=_nonnegative_int(item.get("chunk_index"), f"{field}.chunk_index"),
        extent=_nonnegative_int(item.get("extent"), f"{field}.extent"),
    )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScheduleContractError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ScheduleContractError(f"{field} must be an array")
    return value


def _expect_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    field: str,
    optional: set[str] | None = None,
) -> None:
    missing = required - value.keys()
    extras = value.keys() - required - (optional or set())
    if missing:
        raise ScheduleContractError(f"{field} omits fields {sorted(missing)}")
    if extras:
        raise ScheduleContractError(f"{field} contains unknown fields {sorted(extras)}")


def _enum(kind: type[EnumT], value: Any, field: str) -> EnumT:
    try:
        return kind(value)
    except (TypeError, ValueError) as error:
        choices = ", ".join(repr(item.value) for item in kind)
        raise ScheduleContractError(
            f"{field} must be one of {choices}, got {value!r}"
        ) from error


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ScheduleContractError(f"{field} must be a boolean")
    return value


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScheduleContractError(f"{field} must be a non-empty string")
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
        raise ScheduleContractError(f"{field} contains out-of-range index {result}")
    return result


def _optional_index(value: Any, field: str, *, bound: int | None = None) -> int:
    if value == -1:
        return -1
    result = _nonnegative_int(value, field)
    if bound is not None and result >= bound:
        raise ScheduleContractError(f"{field} contains out-of-range index {result}")
    return result


@overload
def _int_tuple(value: Any, size: Literal[2], field: str) -> tuple[int, int]: ...


@overload
def _int_tuple(value: Any, size: Literal[3], field: str) -> tuple[int, int, int]: ...


@overload
def _int_tuple(value: Any, size: int, field: str) -> tuple[int, ...]: ...


def _int_tuple(value: Any, size: int, field: str) -> tuple[int, ...]:
    items = _sequence(value, field)
    if len(items) != size:
        raise ScheduleContractError(f"{field} must contain exactly {size} integers")
    return tuple(_positive_int(item, field) for item in items)


@overload
def _nonnegative_int_tuple(
    value: Any, size: Literal[2], field: str
) -> tuple[int, int]: ...


@overload
def _nonnegative_int_tuple(value: Any, size: int, field: str) -> tuple[int, ...]: ...


def _nonnegative_int_tuple(value: Any, size: int, field: str) -> tuple[int, ...]:
    items = _sequence(value, field)
    if len(items) != size:
        raise ScheduleContractError(f"{field} must contain exactly {size} integers")
    return tuple(_nonnegative_int(item, field) for item in items)


def _bounded_axis(value: Any, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result > 2:
        raise ScheduleContractError(f"{field} must be 0, 1, or 2")
    return result


def _finite_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ScheduleContractError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ScheduleContractError(f"{field} must be a finite number")
    return result


def _dtype(value: Any, field: str) -> str:
    allowed = {"fp32", "fp16", "bf16", "int32", "int16", "int8", "bool"}
    if value not in allowed:
        raise ScheduleContractError(f"{field} has unsupported dtype {value!r}")
    return str(value)
