"""Immutable schedule types shared by validation and PyPTO source emission."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ScheduleContractError(ValueError):
    """Raised when solver JSON is incomplete or internally inconsistent."""


class KernelKind(Enum):
    VECTOR = "vector"
    CUBE = "cube"
    MIXED = "mixed"


class VectorStreamKind(Enum):
    MATERIALIZED = "materialized"
    POINTWISE = "pointwise"
    REDUCTION_FOLDED = "reduction_folded"
    REDUCTION_SPANNING = "reduction_spanning"
    SOFTMAX_FLASH = "softmax_flash"
    LAYERNORM_WELFORD = "layernorm_welford"
    MODEL_AHEAD_MULTI_REDUCTION = "model_ahead_multi_reduction"


class VectorCoordinateTransform(Enum):
    NONE = "none"
    SINGLETON_COLUMN_TO_ROW = "singleton_column_to_row"


class VectorReplayPhase(Enum):
    BODY = "body"
    STATS = "stats"
    APPLY = "apply"
    FINALIZE = "finalize"


class VectorReductionSplitKind(Enum):
    NONE = "none"
    COL_SUM_ATOMIC_ADD = "col_sum_atomic_add"


class CubeSpatialPolicy(Enum):
    UNIFORM = "uniform"
    CLAMPED_OVERLAP = "clamped_overlap"


class CubeSplitMergePolicy(Enum):
    NONE = "none"
    FIRST_PARTIAL_THEN_ATOMIC = "first_partial_then_atomic"


class CubeAxisBinding(Enum):
    FULL = "full"
    SPATIAL_M = "spatial_m"
    SPATIAL_N = "spatial_n"
    PARALLEL_K = "parallel_k"
    SEQUENTIAL_K = "sequential_k"


class CubeOperandRole(Enum):
    LHS = "lhs"
    RHS = "rhs"


class L0Stationarity(Enum):
    OUTPUT = "output"
    A = "a"
    B = "b"


class L0OutputTarget(Enum):
    ACC = "acc"
    L1 = "l1"
    GM = "gm"


@dataclass(frozen=True)
class AxisPartition:
    big: int
    small: int
    num_big: int
    parts: int


@dataclass(frozen=True)
class LaunchPlan:
    tile_w: int
    tile_h: int
    tile_k: int
    parts_m: int
    parts_n: int
    split: int
    cores: int


@dataclass(frozen=True)
class VectorPhysicalFramePlan:
    element_granule: int
    iteration_rows: int
    iteration_cols: int
    reduced_axis: int
    align_rows: bool


@dataclass(frozen=True)
class VectorLoopPlan:
    first_chunk: int
    trip_count: int
    pipeline_stages: int


@dataclass(frozen=True)
class VectorSerialPhasePlan:
    present: bool
    chunk_index: int
    extent: int


@dataclass(frozen=True)
class VectorInputUsePlan:
    op: int
    arg: int


@dataclass(frozen=True)
class VectorInputLifetimePlan:
    tensor: int
    first_use_step: int
    last_use_step: int
    use_count: int
    uses: tuple[VectorInputUsePlan, ...]


@dataclass(frozen=True)
class VectorTensorFramePlan:
    tensor: int
    logical: tuple[int, int]
    physical: tuple[int, int]


@dataclass(frozen=True)
class VectorWorkspaceFramePlan:
    op: int
    source_tensor: int
    logical: tuple[int, int]
    physical: tuple[int, int]


@dataclass(frozen=True)
class VectorPhasePlan:
    name: VectorReplayPhase
    ops: tuple[int, ...]
    input_lifetimes: tuple[VectorInputLifetimePlan, ...]
    tensor_frames: tuple[VectorTensorFramePlan, ...]
    workspaces: tuple[VectorWorkspaceFramePlan, ...]
    loop: VectorLoopPlan | None = None
    init: VectorSerialPhasePlan | None = None
    tail: VectorSerialPhasePlan | None = None
    serial: VectorSerialPhasePlan | None = None


@dataclass(frozen=True)
class VectorReductionSeedPlan:
    present: bool
    work_units: int
    valid_rows: int
    valid_cols: int


@dataclass(frozen=True)
class VectorReductionSplitPlan:
    kind: VectorReductionSplitKind
    factor: int
    partial_extent: int
    seed: VectorReductionSeedPlan


@dataclass(frozen=True)
class VectorPrimitiveWorkPlan:
    kind: str
    wide: int
    thin: int
    stream_starts: int


@dataclass(frozen=True)
class VectorGeneratedPhaseWorkPlan:
    generated: bool
    primitives: tuple[VectorPrimitiveWorkPlan, ...]


@dataclass(frozen=True)
class VectorP4WorkPlan:
    generated: bool
    stats_init: VectorGeneratedPhaseWorkPlan
    stats_update: VectorGeneratedPhaseWorkPlan
    finalize: VectorGeneratedPhaseWorkPlan


@dataclass(frozen=True)
class VectorP4SubstitutionPlan:
    op: int
    value: str


@dataclass(frozen=True)
class VectorP4RecipePlan:
    version: str
    input_tensor: int
    state: tuple[str, ...]
    apply_substitutions: tuple[VectorP4SubstitutionPlan, ...]


@dataclass(frozen=True)
class VectorKernelPlan:
    kind: VectorStreamKind
    coordinate_transform: VectorCoordinateTransform
    work_units: int
    m_partition: AxisPartition
    n_partition: AxisPartition
    full_peak_ub_bytes: int
    chunk_peak_ub_bytes: int
    stream_band_count: int
    physical_frame: VectorPhysicalFramePlan
    axis: int
    free_tile: int
    free_tile_alloc: int
    extent: int
    chunk: int
    full_chunks: int
    tail: int
    stream_passes: int
    phases: tuple[VectorPhasePlan, ...]
    tile: tuple[int, int]
    strip: tuple[int, int]
    strip_grid: tuple[int, int]
    overlap_granted: bool
    reduction_split: VectorReductionSplitPlan
    p4_work: VectorP4WorkPlan
    p4_recipe: VectorP4RecipePlan | None

    def phase(self, name: VectorReplayPhase) -> VectorPhasePlan:
        """Return the unique phase with ``name``."""

        return next(phase for phase in self.phases if phase.name is name)


@dataclass(frozen=True)
class CubeTensorRegionPlan:
    tensor: int
    height_binding: CubeAxisBinding
    width_binding: CubeAxisBinding
    height: int
    width: int


@dataclass(frozen=True)
class CubeKLoopPlan:
    l1_window_k: int
    chunk: int
    full_chunks: int
    tail: int
    pipeline_stages: int


@dataclass(frozen=True)
class L0KLoopPlan:
    chunk: int
    full_chunks: int
    tail: int
    pipeline_stages: int


@dataclass(frozen=True)
class L0PhaseCostPlan:
    load_cycles: float
    mad_cycles: float
    init_cycles: float
    rolled_cycles: float
    tail_cycles: float
    drain_cycles: float
    wall_cycles: float


@dataclass(frozen=True)
class L0MatmulPlan:
    tile: tuple[int, int, int]
    stationarity: L0Stationarity
    output_stationary_holds_a: bool
    buffer_depths: tuple[int, int, int]
    output_target: L0OutputTarget
    k_loop: L0KLoopPlan
    estimated_traffic_bytes: int
    estimated_cost_cycles: float
    padded_compute_volume: int
    phases: L0PhaseCostPlan


@dataclass(frozen=True)
class CubeOutputTileVariant:
    shape: tuple[int, int]
    count: int
    l0_init: L0MatmulPlan
    l0_rolled: L0MatmulPlan | None
    l0_tail: L0MatmulPlan | None


@dataclass(frozen=True)
class CubeRetainedPanelPlan:
    lhs: bool
    rhs: bool
    lhs_bytes: int
    rhs_bytes: int


@dataclass(frozen=True)
class CubeFinalDrainPlan:
    required: bool
    target_l1: bool
    atomic: bool
    valid_rows: int
    valid_cols: int
    tile_count: int
    bytes: int
    cycles: float


@dataclass(frozen=True)
class CubeResidentBoundaryPlan:
    id: int
    region: CubeTensorRegionPlan
    role: CubeOperandRole
    first_use: int
    last_use: int
    use_count: int
    bytes: int


@dataclass(frozen=True)
class CubeMatmulPlan:
    instance: int
    op: int
    lhs_producer: int
    rhs_producer: int
    lhs_resident_boundary: int
    rhs_resident_boundary: int
    is_sink: bool
    lhs_ephemeral: bool
    rhs_ephemeral: bool
    output_ephemeral: bool
    contraction: int
    effective_contraction: int
    accumulator_dtype: str
    storage_dtype: str
    lhs: CubeTensorRegionPlan
    rhs: CubeTensorRegionPlan
    output: CubeTensorRegionPlan
    k_loop: CubeKLoopPlan
    output_tile: tuple[int, int]
    output_grid: tuple[int, int]
    output_variants: tuple[CubeOutputTileVariant, ...]
    retained_panels: CubeRetainedPanelPlan
    final_drain: CubeFinalDrainPlan


@dataclass(frozen=True)
class CubeFirstPartialThenAtomicPlan:
    present: bool
    first_work_units: int
    atomic_work_units: int
    synchronization_cycles: float


@dataclass(frozen=True)
class CubeKernelPlan:
    emit_compatible: bool
    spatial_policy: CubeSpatialPolicy
    m_partition: AxisPartition
    n_partition: AxisPartition
    spatial_tiles: int
    split_k: int
    work_units: int
    peak_l1_bytes: int
    split_merge_policy: CubeSplitMergePolicy
    first_partial_then_atomic: CubeFirstPartialThenAtomicPlan
    model_overlap_granted: bool
    overlap_implementable: bool
    execution_order: tuple[int, ...]
    resident_boundaries: tuple[CubeResidentBoundaryPlan, ...]
    matmuls: tuple[CubeMatmulPlan, ...]


@dataclass(frozen=True)
class MixedKernelPlan:
    """Marker for a mixed schedule, which the homogeneous source backend rejects."""

    source_codegen_ready: bool


KernelPlan = VectorKernelPlan | CubeKernelPlan | MixedKernelPlan


@dataclass(frozen=True)
class KernelStep:
    index: int
    kind: KernelKind
    solver_ops: tuple[int, ...]
    graph_ops: tuple[str, ...]
    op_order: tuple[int, ...]
    sequential_tiles: tuple[int, ...] | None
    retained_tensors: tuple[int, ...]
    launch: LaunchPlan
    latency: float
    plan: KernelPlan

    @property
    def tile_w(self) -> int:
        return self.launch.tile_w

    @property
    def tile_h(self) -> int:
        return self.launch.tile_h

    @property
    def tile_k(self) -> int:
        return self.launch.tile_k

    @property
    def parts_m(self) -> int:
        return self.launch.parts_m

    @property
    def parts_n(self) -> int:
        return self.launch.parts_n

    @property
    def split(self) -> int:
        return self.launch.split

    @property
    def cores(self) -> int:
        return self.launch.cores


@dataclass(frozen=True)
class ScheduledRegion:
    region_id: str
    tensor_values: tuple[str, ...]
    steps: tuple[KernelStep, ...]
