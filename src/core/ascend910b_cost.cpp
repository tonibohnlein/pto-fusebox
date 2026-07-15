#include "core/ascend910b_cost.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <tuple>

#include "core/subgraph_structure.h"
#include "core/types.h"

// ============================================================================
// Utility
// ============================================================================

static std::vector<int64_t> all_divisors(int64_t n) {
  std::vector<int64_t> result;
  for (int64_t i = 1; i * i <= n; i++) {
    if (n % i == 0) {
      result.push_back(i);
      if (i != n / i)
        result.push_back(n / i);
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

// ============================================================================
// Grounded pto-isa machine model (Ascend 910B / A2A3)
// ============================================================================
// All costs are in CORE CYCLES, matching pto-isa's EstimateLinearCycles (cube)
// and EstimateBandwidthCycles (transfers), using the grounded A2/A3 coefficients
// (per-direction bandwidths, core clock, L0/vector-register sizes).
namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

// Legacy vector REDUCTION tree coefficients (pto-isa stub perf-sim,
// vec_tile_study/vec_reduce). A reduction is NOT a single slope*repeat op: it
// lowers to a barrier-separated tree. The stub's count-mode accounting is,
// however, rows-independent; PTO-ISA explicitly documents that simplification
// as inaccurate on hardware. Exact PyPTO TROWSUM/TROWMAX descriptors therefore
// use the fit-backend tables below, while descriptor-free research instances
// and unsupported dtypes retain this structural fallback.
constexpr double kVecRowReducePass = 45.0;   // per barrier-isolated count-mode vadd pass
constexpr double kVecRowReduceFinal = 51.0;  // K=1 base (the final cross-lane vcadd block)
constexpr double kVecColReduceSlope = 16.0;  // streamed count-mode vadd per row-pair
constexpr double kVecColReduceLevel = 30.0;  // per-level startup (log2(H) barriers)
constexpr double kVecExpandScalarHead = 11.0;
constexpr double kVecExpandScalarTail = 13.0;

// PTO-ISA A2/A3 formula backend, formula_params.csv @ d5eaf8e4. Each grounded
// entry predicts round(slope * valid_rows * valid_cols + bias) cycles and is
// validated by PTO-ISA's fit-backend tests against cycle profiling. AutoFuse
// emits additional DMA-aligned widths, so unsupported widths interpolate the
// two neighboring total-cycle predictions for the requested valid row count.
// Above the measured range, proportional continuation from the last anchor is
// deliberately monotone and introduces no new fitted coefficient.
struct AxisReductionFormula {
  int64_t cols;
  double slope;
  double bias;
};

constexpr std::array<AxisReductionFormula, 12> kRowMaxFp32{{
    {8, 0.875, 46},       {32, 0.2188, 47},   {64, 0.1094, 32},
    {96, 0.0937, 61},     {128, 0.0625, 48},  {144, 0.0692, 77},
    {176, 0.0558, 79},    {208, 0.0517, 100}, {240, 0.0448, 100},
    {272, 0.0485, 98},    {304, 0.0434, 110}, {336, 0.0418, 131},
}};

constexpr std::array<AxisReductionFormula, 12> kRowSumFp32{{
    {8, 0.875, 58},       {32, 0.2187, 59},   {64, 0.1094, 44},
    {96, 0.0938, 75},     {128, 0.0625, 61.5}, {144, 0.0698, 92},
    {176, 0.0575, 93},    {208, 0.0535, 99},  {240, 0.0469, 99},
    {272, 0.0484, 104},   {304, 0.0431, 104}, {336, 0.0419, 121},
}};

constexpr std::array<AxisReductionFormula, 20> kRowMaxFp16{{
    {32, 0.2187, 48},   {64, 0.1094, 43},   {96, 0.0729, 43},
    {128, 0.0547, 33},  {160, 0.0562, 62},  {208, 0.0433, 63},
    {240, 0.0388, 62},  {272, 0.0369, 74},  {304, 0.0323, 80},
    {336, 0.0302, 97},  {368, 0.0276, 93},  {400, 0.0279, 92},
    {432, 0.025, 101},  {464, 0.0232, 119}, {496, 0.0222, 119},
    {528, 0.0254, 111}, {560, 0.0241, 110}, {592, 0.0232, 127},
    {624, 0.022, 127},  {656, 0.0224, 125},
}};

constexpr std::array<AxisReductionFormula, 20> kRowSumFp16{{
    {32, 0.2188, 62},   {64, 0.1094, 57},   {96, 0.0729, 57},
    {128, 0.0547, 47},  {160, 0.0563, 78},  {208, 0.0436, 79},
    {240, 0.0391, 79},  {272, 0.0366, 94},  {304, 0.0323, 102},
    {336, 0.0298, 115}, {368, 0.0272, 111}, {400, 0.0275, 103},
    {432, 0.0255, 103}, {464, 0.0238, 121}, {496, 0.0229, 121},
    {528, 0.0249, 111}, {560, 0.0234, 106}, {592, 0.0219, 140},
    {624, 0.0208, 140}, {656, 0.0213, 126},
}};

constexpr std::array<AxisReductionFormula, 5> kColMaxFp32{{
    {8, 2.125, 24}, {32, 0.5391, 16}, {64, 0.2734, 17},
    {96, 0.3646, 2}, {128, 0.1571, 10},
}};

constexpr std::array<AxisReductionFormula, 6> kColMaxFp16{{
    {32, 0.5352, 16}, {64, 0.2695, 16}, {96, 0.1836, 19},
    {128, 0.1367, 17}, {160, 0.2187, 2}, {192, 0.1823, 2.3325},
}};

constexpr std::array<AxisReductionFormula, 13> kColSumFp32{{
    {8, 2.375, 29}, {16, 1.1958, 18.271}, {32, 0.6016, 15},
    {64, 0.3047, 15}, {96, 0.2187, 14}, {128, 0.1728, 8},
    {144, 0.1578, 27}, {176, 0.1314, 30}, {208, 0.1166, 36},
    {240, 0.1031, 36}, {272, 0.096, 35}, {304, 0.0876, 35},
    {336, 0.0833, 34},
}};

constexpr std::array<AxisReductionFormula, 21> kColSumFp16{{
    {16, 1.1916, 20.664}, {32, 0.5977, 15}, {64, 0.3008, 15},
    {96, 0.2044, 15}, {128, 0.1523, 15}, {160, 0.1312, 14},
    {208, 0.1029, 25}, {240, 0.09, 28}, {272, 0.0838, 27},
    {304, 0.0755, 30}, {336, 0.069, 30}, {368, 0.069, 30},
    {400, 0.0616, 29}, {432, 0.057, 36}, {464, 0.0536, 36},
    {496, 0.0507, 36}, {528, 0.0498, 35}, {560, 0.0474, 35},
    {592, 0.0453, 35}, {624, 0.0434, 35}, {656, 0.0431, 34},
}};

template <size_t N>
double InterpolateReductionCycles(const std::array<AxisReductionFormula, N> &table,
                                  int64_t valid_rows, int64_t valid_cols) {
  if (valid_rows <= 0 || valid_cols <= 0) return -1.0;
  auto at = [&](const AxisReductionFormula &entry) {
    return entry.slope * (double)valid_rows * (double)entry.cols + entry.bias;
  };
  const auto upper = std::lower_bound(
      table.begin(), table.end(), valid_cols,
      [](const AxisReductionFormula &entry, int64_t cols) { return entry.cols < cols; });
  if (upper == table.begin()) return std::round(at(*upper));
  if (upper == table.end()) {
    const auto &last = table.back();
    return std::round(at(last) * (double)valid_cols / (double)last.cols);
  }
  if (upper->cols == valid_cols) return std::round(at(*upper));
  const auto &lower = *(upper - 1);
  const double alpha = (double)(valid_cols - lower.cols) /
                       (double)(upper->cols - lower.cols);
  return std::round(at(lower) + alpha * (at(*upper) - at(lower)));
}

double GroundedRowReductionCyclesImpl(VectorPrimitiveFamily family, DType dtype,
                                      int64_t valid_rows, int64_t valid_cols) {
  const bool sum = family == VectorPrimitiveFamily::RowSum;
  const bool extrema = family == VectorPrimitiveFamily::RowExtrema;
  if (!sum && !extrema) return -1.0;
  if (dtype == DType::FP32)
    return sum ? InterpolateReductionCycles(kRowSumFp32, valid_rows, valid_cols)
               : InterpolateReductionCycles(kRowMaxFp32, valid_rows, valid_cols);
  if (dtype == DType::FP16)
    return sum ? InterpolateReductionCycles(kRowSumFp16, valid_rows, valid_cols)
               : InterpolateReductionCycles(kRowMaxFp16, valid_rows, valid_cols);
  return -1.0;
}

double GroundedColumnReductionCyclesImpl(VectorPrimitiveFamily family, DType dtype,
                                         int64_t valid_rows, int64_t valid_cols) {
  const bool sum = family == VectorPrimitiveFamily::ColSum;
  const bool extrema = family == VectorPrimitiveFamily::ColExtrema;
  if (!sum && !extrema) return -1.0;
  if (dtype == DType::FP32)
    return sum ? InterpolateReductionCycles(kColSumFp32, valid_rows, valid_cols)
               : InterpolateReductionCycles(kColMaxFp32, valid_rows, valid_cols);
  if (dtype == DType::FP16)
    return sum ? InterpolateReductionCycles(kColSumFp16, valid_rows, valid_cols)
               : InterpolateReductionCycles(kColMaxFp16, valid_rows, valid_cols);
  return -1.0;
}

// Count-mode dispatch floor: a binary-ALU vector op whose contiguous width is NOT repeat-aligned
// (cols % epr != 0) enters count-mask dispatch, paying a one-time ~16-cycle floor independent of
// the op/repeat (pto-isa cce_costmodel_core.hpp kCountModeFloorCycles; medians 12-18, std<3). This
// is the +16 unaligned-width penalty the vec_tile_study flagged as previously unmodeled.
constexpr double kVecCountModeFloor = 16.0;

constexpr uint8_t kVectorPhaseBody = 1u << 0;
constexpr uint8_t kVectorPhaseStats = 1u << 1;
constexpr uint8_t kVectorPhaseApply = 1u << 2;
constexpr uint8_t kVectorPhaseFinalize = 1u << 3;
constexpr std::array<uint8_t, 4> kVectorPhases{
    kVectorPhaseBody, kVectorPhaseStats, kVectorPhaseApply,
    kVectorPhaseFinalize};
constexpr uint8_t kSkipRetainedFromPrev = 1u << 0;
constexpr uint8_t kSkipRetainThese = 1u << 1;
constexpr uint8_t kSkipAnyRetained =
    kSkipRetainedFromPrev | kSkipRetainThese;

size_t VectorPhaseIndex(uint8_t phase) {
  for (size_t i = 0; i < kVectorPhases.size(); ++i)
    if (kVectorPhases[i] == phase) return i;
  return 0;
}

struct VectorPrimitiveGrounding {
  double slope;
  double fixed;
  bool binary_count_mode;
};

// pto-isa cce_costmodel_vector_compute.hpp, 910B3 calibrations.  This is the
// single coefficient table for both source-DAG pointwise ops and generated P4
// work.  Geometry-specific composite overhead is added by the caller.
inline VectorPrimitiveGrounding PrimitiveGrounding(VectorPrimitiveFamily family) {
  switch (family) {
    case VectorPrimitiveFamily::Add: return {2.0, 24.0, true};
    case VectorPrimitiveFamily::Mul: return {2.0, 25.0, true};
    case VectorPrimitiveFamily::Div: return {4.0, 30.0, true};
    case VectorPrimitiveFamily::Exp: return {2.0, 31.0, false};
    case VectorPrimitiveFamily::Log: return {2.0, 33.0, false};
    case VectorPrimitiveFamily::Abs: return {1.0, 29.0, false};
    case VectorPrimitiveFamily::Sqrt: return {2.0, 39.0, false};
    case VectorPrimitiveFamily::Rsqrt: return {1.0, 24.0, false};
    case VectorPrimitiveFamily::ScalarAdd: return {1.0, 31.0, false};
    case VectorPrimitiveFamily::ScalarMul: return {1.0, 26.0, false};
    case VectorPrimitiveFamily::ScalarMax: return {1.0, 23.0, false};
    case VectorPrimitiveFamily::ScalarMin: return {1.0, 30.0, false};
    case VectorPrimitiveFamily::RowSum:
    case VectorPrimitiveFamily::RowExtrema:
    case VectorPrimitiveFamily::ColSum:
    case VectorPrimitiveFamily::ColExtrema:
    case VectorPrimitiveFamily::Reduction: return {0.0, 0.0, false};
    case VectorPrimitiveFamily::Generic: return {0.0, 0.0, false};
  }
  return {0.0, 0.0, false};
}

struct VectorFrameShape {
  int64_t rows = 1;
  int64_t cols = 1;
};

// Shape one source op sees inside a planned frame.  A global size-one axis is
// a broadcast/folded axis and remains one; every other axis follows the emitted
// strip or reduced-axis chunk.  Taking the maximum over operands reproduces the
// pointwise/reduction work shape while excluding broadcast storage.
inline VectorFrameShape OpFrameShape(const Problem *p, const Op &op, int64_t frame_rows,
                                     int64_t frame_cols) {
  VectorFrameShape shape;
  auto include = [&](size_t tensor_id) {
    const Tensor &tensor = p->tensors[tensor_id];
    const int64_t rows = tensor.height == 1 ? 1 : std::min(tensor.height, frame_rows);
    const int64_t cols = tensor.width == 1 ? 1 : std::min(tensor.width, frame_cols);
    shape.rows = std::max(shape.rows, rows);
    shape.cols = std::max(shape.cols, cols);
  };
  include(op.output());
  for (size_t input : op.inputs) include(input);
  return shape;
}

inline bool HasGroundedVectorSemantics(const Op &op) {
  return op.vector_primitive != VectorPrimitiveFamily::Generic &&
         op.vector_geometry != VectorOpGeometry::Generic;
}

// Cost one source-DAG pointwise op at the exact valid frame emitted for one
// strip/chunk in one logical task.  In a reduction-layout group TROWEXPANDBIN
// consumes a col-major statistic: vbrcb (18) + pipe_barrier (1) precede the
// underlying binary op, so it starts an independent stream.  A pure pointwise
// group's row-major size-one operand uses the raw strided binary path instead.
inline double GroundedVectorOpCompute(const Problem *p, const Op &op, int64_t frame_rows,
                                      int64_t frame_cols, bool pw_stream_start,
                                      bool row_expand_composite) {
  const int64_t reg = p->vec_reg_bytes > 0 ? p->vec_reg_bytes : 256;
  const int64_t element_bytes = dtype_bytes(p->tensors[op.output()].dtype);
  const int64_t epr = std::max<int64_t>(1, reg / element_bytes);
  const VectorFrameShape shape = OpFrameShape(p, op, frame_rows, frame_cols);
  const bool expanded = op.vector_geometry == VectorOpGeometry::RowExpand ||
                        op.vector_geometry == VectorOpGeometry::ColExpand;
  const int64_t repeats = expanded
                              ? shape.rows * ((shape.cols + epr - 1) / epr)
                              : (shape.rows * shape.cols + epr - 1) / epr;

  VectorPrimitiveGrounding grounding = PrimitiveGrounding(op.vector_primitive);
  const bool row_expand = op.vector_geometry == VectorOpGeometry::RowExpand;
  const bool composite = row_expand && row_expand_composite;
  if (composite) grounding.fixed += 19.0;  // vbrcb + PIPE_V barrier
  double cycles = grounding.slope * (double)repeats;
  if (pw_stream_start || composite) cycles += grounding.fixed;

  if (grounding.binary_count_mode) {
    bool count_mode = shape.cols % epr != 0;
    if (row_expand) {
      const int64_t block_elems = std::max<int64_t>(1, 32 / element_bytes);
      count_mode = shape.cols / epr > shape.rows ||
                   (shape.cols + block_elems - 1) / block_elems > 255;
    }
    if (count_mode) cycles += kVecCountModeFloor;
  }
  return cycles;
}

inline double GroundedReductionCompute(const Problem *p, const Op &op, int reduced_axis,
                                       int64_t frame_rows, int64_t frame_cols) {
  const int64_t reg = p->vec_reg_bytes > 0 ? p->vec_reg_bytes : 256;
  const DType dtype =
      op.inputs.empty() ? p->tensors[op.output()].dtype : p->tensors[op.inputs[0]].dtype;
  const int64_t epr = std::max<int64_t>(1, reg / dtype_bytes(dtype));
  const VectorFrameShape shape = OpFrameShape(p, op, frame_rows, frame_cols);
  if (reduced_axis == 2) {
    const double grounded =
        GroundedColumnReductionCycles(op.vector_primitive, dtype, shape.rows, shape.cols);
    if (grounded >= 0.0) return grounded;
    return kVecColReduceSlope * (double)std::max<int64_t>(0, shape.rows - 1) +
           kVecColReduceLevel *
               (shape.rows > 1 ? std::log2((double)shape.rows) : 0.0);
  }
  const double grounded =
      GroundedRowReductionCycles(op.vector_primitive, dtype, shape.rows, shape.cols);
  if (grounded >= 0.0) return grounded;
  const int64_t passes = std::max<int64_t>(1, (shape.cols + epr - 1) / epr);
  return kVecRowReducePass * (double)(passes - 1) + kVecRowReduceFinal;
}

// P1/P2 emit one thin add/max after every non-initial statistics chunk.  The
// merge is absent from the source DAG, so exact source semantics must add it
// explicitly, just as P4 adds its generated online-stat work.
inline double GeneratedReductionMergeCompute(const Problem *p,
                                              const VectorStreamPlan &plan,
                                              int64_t iterations,
                                              int64_t element_bytes) {
  if (iterations <= 0) return 0.0;
  const int64_t reg = p->vec_reg_bytes > 0 ? p->vec_reg_bytes : 256;
  const int64_t epr =
      std::max<int64_t>(1, reg / std::max<int64_t>(1, element_bytes));
  const int64_t repeats =
      (std::max<int64_t>(1, plan.free_tile) + epr - 1) / epr;
  const auto grounding = PrimitiveGrounding(VectorPrimitiveFamily::Add);
  const bool count_mode = plan.axis == 1 || plan.free_tile % epr != 0;
  const double per_task = grounding.slope * (double)repeats + grounding.fixed +
                          (count_mode ? kVecCountModeFloor : 0.0);
  return per_task * (double)iterations *
         (double)std::max<int64_t>(1, plan.work_units);
}

// Grounded per-op VECTOR compute cycles (pto-isa perf-sim, vec_tile_study). Shared by the
// vector-only and the mixed cube+vector paths so a reduction costs the same in both.
//   Pointwise: slope*repeat + (head+tail IF this op starts a vector stream). Fix 3: the
//     perf-sim pays head+tail only when the VEC queue is empty (a back-to-back chain overlaps
//     its startup), so the caller passes pw_stream_start=true only for the first pointwise op
//     of a stream (chain start, or after a reduction/matmul barrier) -- not per op.
//   Descriptor-free reduction (Fix 1): the legacy REDUCED-AXIS stub tree. Exact
//     PyPTO reductions take GroundedReductionCompute at their emitted frame and
//     use the row-aware PTO-ISA fit model instead.
inline double VecOpCompute(const Problem *p, const Op &op, int reduced_axis,
                           bool pw_stream_start, bool row_expand_composite) {
  const int64_t reg = p->vec_reg_bytes > 0 ? p->vec_reg_bytes : 256;
  const int64_t epr = std::max<int64_t>(1, reg / dtype_bytes(p->tensors[op.output()].dtype));
  if (op.type == OpType::Reduction) {
    const int64_t W = (int64_t)p->tensors[op.inputs[0]].width;
    const int64_t H = (int64_t)p->tensors[op.inputs[0]].height;
    if (reduced_axis == 2)  // reduce height: pairwise vadd tree across rows
      return kVecColReduceSlope * (double)std::max<int64_t>(0, H - 1) +
             kVecColReduceLevel * (H > 1 ? std::log2((double)H) : 0.0);
    const int64_t K = std::max<int64_t>(1, W / epr);  // reduce width: ROWS-independent
    return kVecRowReducePass * (double)(K - 1) + kVecRowReduceFinal;
  }
  if (HasGroundedVectorSemantics(op)) {
    int64_t frame_rows = p->tensors[op.output()].height;
    int64_t frame_cols = p->tensors[op.output()].width;
    for (size_t input : op.inputs) {
      frame_rows = std::max(frame_rows, p->tensors[input].height);
      frame_cols = std::max(frame_cols, p->tensors[input].width);
    }
    return GroundedVectorOpCompute(p, op, frame_rows, frame_cols,
                                   pw_stream_start, row_expand_composite);
  }
  int64_t elems = (int64_t)p->tensors[op.output()].width * p->tensors[op.output()].height;
  int64_t width = (int64_t)p->tensors[op.output()].width;  // contiguous extent (count-mode axis)
  for (auto t : op.inputs) {
    elems = std::max(elems, (int64_t)p->tensors[t].width * p->tensors[t].height);
    width = std::max(width, (int64_t)p->tensors[t].width);
  }
  const int64_t repeat = (elems + epr - 1) / epr;
  // Per-op slope (vdiv=4, vrsqrt/vrelu=1) overrides the group default (~2) when the adapter set it.
  const double slope = op.vec_slope > 0.0 ? op.vec_slope : p->vec_slope_pw;
  // Per-op fixed (head+tail) — vadd 24 / vmul 25 / vexp 31 / vdiv 30 — overrides the group default
  // (~32) when the adapter set it. Charged ONCE per chain (the stream-start op), so it is the
  // stream-start op's own fixed. Exact-match to pto-isa's calibrated per-op fixed.
  const double fixed = op.vec_fixed > 0.0 ? op.vec_fixed : (p->vec_op_head + p->vec_op_tail);
  double cycles = slope * (double)repeat + (pw_stream_start ? fixed : 0.0);
  // A width not aligned to one SIMD repeat (cols % epr != 0) pays the count-mask dispatch floor.
  if (width % epr != 0) cycles += kVecCountModeFloor;
  return cycles;
}

// Map generated P4 primitive tallies into the same source-DAG coefficient
// table.  RowExpandSub adds the emitted vbrcb+barrier composite overhead.
inline VectorPrimitiveGrounding PrimitiveGrounding(VectorPrimitiveKind kind) {
  switch (kind) {
    case VectorPrimitiveKind::Add: return PrimitiveGrounding(VectorPrimitiveFamily::Add);
    case VectorPrimitiveKind::Mul: return PrimitiveGrounding(VectorPrimitiveFamily::Mul);
    case VectorPrimitiveKind::Div: return PrimitiveGrounding(VectorPrimitiveFamily::Div);
    case VectorPrimitiveKind::Exp: return PrimitiveGrounding(VectorPrimitiveFamily::Exp);
    case VectorPrimitiveKind::RowExpandSub: {
      auto grounding = PrimitiveGrounding(VectorPrimitiveFamily::Add);
      grounding.fixed += 19.0;
      return grounding;
    }
    case VectorPrimitiveKind::ScalarAdd: return PrimitiveGrounding(VectorPrimitiveFamily::ScalarAdd);
    case VectorPrimitiveKind::ScalarMul: return PrimitiveGrounding(VectorPrimitiveFamily::ScalarMul);
    case VectorPrimitiveKind::RowSum:
    case VectorPrimitiveKind::RowMax:
    case VectorPrimitiveKind::Count: return {0.0, 0.0, false};
  }
  return {0.0, 0.0, false};
}

// Total compute work across all logical tasks for one generated P4 phase.
// WaveComputeCycles consumes this total immediately afterwards. In particular,
// every task executes its own barrier-separated reduction tree; rows do not
// cooperate on a single tree merely because they share a wave.
inline double GeneratedP4PhaseCompute(const Problem *p, const VectorStreamPlan &plan,
                                      const VectorPhaseWorkPlan &phase, int64_t chunk_extent,
                                      int64_t iterations, DType dtype) {
  if (!phase.generated || chunk_extent <= 0 || iterations <= 0) return 0.0;
  const int64_t element_bytes = dtype_bytes(dtype);
  const int64_t reg = p->vec_reg_bytes > 0 ? p->vec_reg_bytes : 256;
  const int64_t epr = std::max<int64_t>(1, reg / std::max<int64_t>(1, element_bytes));
  const int64_t wide_repeats =
      (std::max<int64_t>(1, plan.free_tile) * chunk_extent + epr - 1) / epr;
  const int64_t row_expand_repeats =
      std::max<int64_t>(1, plan.free_tile) * ((chunk_extent + epr - 1) / epr);
  const int64_t thin_repeats = (std::max<int64_t>(1, plan.free_tile) + epr - 1) / epr;
  const bool wide_count_mode = chunk_extent % epr != 0;
  const int64_t block_elems = std::max<int64_t>(1, 32 / std::max<int64_t>(1, element_bytes));
  const bool row_expand_count_mode = chunk_extent / epr > std::max<int64_t>(1, plan.free_tile) ||
                                     (chunk_extent + block_elems - 1) / block_elems > 255;
  const bool thin_count_mode = 1 % epr != 0;

  double per_task = 0.0;
  for (size_t i = 0; i < static_cast<size_t>(VectorPrimitiveKind::Count); ++i) {
    const auto kind = static_cast<VectorPrimitiveKind>(i);
    const VectorPrimitiveWork &work = phase.primitives[i];
    if (kind == VectorPrimitiveKind::RowSum || kind == VectorPrimitiveKind::RowMax) {
      const VectorPrimitiveFamily family =
          kind == VectorPrimitiveKind::RowSum ? VectorPrimitiveFamily::RowSum
                                              : VectorPrimitiveFamily::RowExtrema;
      double reduction = GroundedRowReductionCycles(
          family, dtype, std::max<int64_t>(1, plan.free_tile), chunk_extent);
      if (reduction < 0.0) {
        const int64_t passes = std::max<int64_t>(1, (chunk_extent + epr - 1) / epr);
        reduction = kVecRowReducePass * (double)(passes - 1) + kVecRowReduceFinal;
      }
      per_task += (double)work.wide * reduction;
      continue;
    }
    const VectorPrimitiveGrounding grounding = PrimitiveGrounding(kind);
    const int64_t primitive_wide_repeats =
        kind == VectorPrimitiveKind::RowExpandSub ? row_expand_repeats : wide_repeats;
    per_task += grounding.slope *
                ((double)work.wide * (double)primitive_wide_repeats +
                 (double)work.thin * (double)thin_repeats);
    per_task += (double)work.stream_starts * grounding.fixed;
    if (grounding.binary_count_mode) {
      const bool primitive_wide_count_mode =
          kind == VectorPrimitiveKind::RowExpandSub ? row_expand_count_mode : wide_count_mode;
      if (primitive_wide_count_mode) per_task += (double)work.wide * kVecCountModeFloor;
      if (thin_count_mode) per_task += (double)work.thin * kVecCountModeFloor;
    }
  }
  return per_task * (double)iterations * (double)std::max<int64_t>(1, plan.work_units);
}

// Per-direction "cycles per byte" for a transfer: a byte costs
// (1/2^30)/bw_GiBps * freq_hz cycles (pto-isa EstimateBandwidthCycles).
struct ByteCost {
  double reload = 0.0;  // GM->L1   (cube operand reload)
  double store = 0.0;   // L0C->GM  (cube output writeback)
  double l0a = 0.0;     // L1->L0A  (lhs extract)
  double l0b = 0.0;     // L1->L0B  (rhs extract)
  double ub_in = 0.0;   // GM->UB   (vector load)
  double ub_out = 0.0;  // UB->GM   (vector store)
};

ByteCost MakeByteCost(const Problem* p) {
  // Per-direction cycles/byte: a byte costs freq / (2^30 * bw_GiBps) cycles
  // (pto-isa EstimateBandwidthCycles); bandwidths are GiB/s, per direction.
  auto cpb = [&](double bw_gibps) { return p->cube_freq_hz / (kGiB * bw_gibps); };
  return {cpb(p->bw_gm_l1), cpb(p->bw_l0c_gm), cpb(p->bw_l1_l0a),
          cpb(p->bw_l1_l0b), cpb(p->bw_gm_ub), cpb(p->bw_ub_gm)};
}

// Largest per-load K chunk that fits two ping-pong buffers in the L1 window and
// leaves at least two rolled iterations after a serial first accumulation
// initializes the carry. Returning 0 means the concrete emitter has no stage-2
// steady state and the outer GM/compute roofline must serialize.
int64_t CubePipelinedChunk(int64_t extent, int64_t window) {
  if (extent < 48 || window < 32) return 0;
  const int64_t limit = (std::min(window / 2, extent / 3) / 16) * 16;
  return limit >= 16 ? limit : 0;
}

// Preserve the pre-request-DAG lone-matmul window exactly. The displayed/emit
// window was a divisor of the full contraction, capped by the selected core's
// K share; choosing merely a divisor of K/S can change the K-loop while leaving
// the candidate cost unchanged.
int64_t CappedSinkWindow(int64_t output_k, int64_t l1_window, int64_t split) {
  split = std::max<int64_t>(1, split);
  const int64_t k_fractals = std::max<int64_t>(1, output_k / 16);
  const int64_t share_k = ((k_fractals + split - 1) / split) * 16;
  const int64_t cap = std::min({l1_window, share_k, output_k});
  for (int64_t candidate = (cap / 16) * 16; candidate >= 16; candidate -= 16) {
    if (output_k % candidate == 0) return candidate;
  }
  return 0;
}

// Cube MAC cost of one M x N x K matmul, in cycles. Grounded: the dtype-aware
// fractal count x cycles-per-repeat (pto-isa cce_costmodel_cube.hpp `mad`):
// kF = 32/dtype_bytes (fp32:8, fp16:16), cyc = 2 (fp32) else 1. cube_compute_cost
// (default 1) is a calibration multiplier.
double CubeMacCycles(const Problem* p, int64_t M, int64_t N, int64_t K, DType dt) {
  const int64_t kF = std::max<int64_t>(1, 32 / dtype_bytes(dt));
  const double repeats = (double)((M + 15) / 16) * (double)((N + 15) / 16) *
                         (double)((K + kF - 1) / kF);
  const double cyc = (dt == DType::FP32) ? 2.0 : 1.0;
  const double mult = (p->cube_compute_cost > 0) ? (double)p->cube_compute_cost : 1.0;
  return repeats * cyc * mult;
}

// L1->L0 operand extract for one M x N x K matmul, in cycles. The cube re-reads
// the lhs once per L0 N-block (l0_tile_n) and the rhs once per L0 M-block
// (l0_tile_m) — the same distribution-aware reuse as cube_operand_reload, one
// hierarchy level down. Double-buffering overlaps this with the MACs, so the
// caller takes max(MAC, extract). 0 when no L0 base tile.
double CubeExtractCycles(const Problem* p, const ByteCost& bc, int64_t M,
                         int64_t N, int64_t K, DType dt) {
  if (p->l0_tile_m <= 0 || p->l0_tile_n <= 0)
    return 0.0;
  const double db = (double)dtype_bytes(dt);
  const double MNK = (double)M * (double)N * (double)K;
  const double lhs_bytes = MNK / (double)p->l0_tile_n * db;
  const double rhs_bytes = MNK / (double)p->l0_tile_m * db;
  return lhs_bytes * bc.l0a + rhs_bytes * bc.l0b;
}

L0MatmulPlan DeriveL0MatmulPlan(const Problem* p, int64_t m, int64_t n, int64_t k, DType lhs_dtype,
                                DType rhs_dtype, DType output_dtype, bool accumulator_read,
                                L0OutputTarget output_target, L0PlanMemo* memo) {
  const L0PlanMemoKey key{m, n, k, lhs_dtype, rhs_dtype, output_dtype, accumulator_read, output_target};
  if (memo != nullptr) {
    const auto it = memo->find(key);
    if (it != memo->end()) return it->second;
  }
  L0MatmulConfig config = p->l0_matmul_config;
  config.m = m;
  config.n = n;
  config.k = k;
  config.bytes_a = dtype_bytes(lhs_dtype);
  config.bytes_b = dtype_bytes(rhs_dtype);
  config.bytes_c = dtype_bytes(output_dtype);
  config.accumulator_read = accumulator_read;
  config.output_target = output_target;
  // AutoTileMatmulL0's Mat-scratch placement currently forces an internal
  // producer to output-stationary. Keep the child plan on the same buildable
  // subset until operand-stationary scratch packing is implemented.
  if (output_target == L0OutputTarget::L1) {
    config.allow_a_stationary = false;
    config.allow_b_stationary = false;
  }
  L0MatmulPlan plan = choose_l0_matmul_plan(config);
  if (memo != nullptr) memo->emplace(key, plan);
  return plan;
}

// Wave-aware compute makespan. Uniform cube grids are equal-cost; a balanced
// vector grid totalizes U copies of its maximum valid region work, so this same
// equation prices the critical task rather than the average 11/10-row task.
// The hardware still sees one ready queue with no wave barriers; "waves" here
// is only the queue-makespan count ceil(U/C). The fullest wave-equivalent sets
// the wall, NOT an idealized total/C fractional division:
//
//   T_compute = ceil(U/C) * (W_total / U)
//
// vs the old W_total / min(U, C), which silently assumed work splits fractionally
// and so under-charged any U > C that is not a multiple of C (e.g. 32 units on 24
// cores: old = W/24, true = 2*(W/32) = W/16, a 33% miss). U = num_work_units; the
// effective parallelism is U/ceil(U/C), NOT min(U,C).
double WaveComputeCycles(double total_compute, int64_t num_work_units,
                         int64_t num_cores) {
  const int64_t units = std::max<int64_t>(1, num_work_units);
  const int64_t cores = std::max<int64_t>(1, num_cores);
  const int64_t waves = (units + cores - 1) / cores;
  return total_compute * (double)waves / (double)units;
}

// LPT makespan for a non-uniform parts_m x parts_n SpatialSchedule grid. The even
// split yields at most 4 distinct region shapes (m_ext, n_ext);
// `region_work(m_ext, n_ext)` gives one region's double-buffered cost. LPT-assign
// the P*Q regions across n_cores and return the busiest core's load. With
// parts == n_cores this is one wave -> the largest region's cost; the LPT also
// captures the +-1-fractal imbalance and multi-wave grids. The per-region work is
// supplied by the caller so a single-matmul sink and a chained group (sink region
// + backpropagated intermediate row-bands) share this distribution logic.
// `ksplit` > 1 applies split-K ON the grid: each region's K-contraction splits
// into ksplit equal partials, so the P*Q regions become P*Q*ksplit work units of
// work/ksplit each. This keeps split-K LPT-consistent with the grid (the equal-
// unit wave would optimistically ignore the +-1-fractal region imbalance).
template <typename RegionWork>
double LptMakespan(int64_t n_cores, const AxisPartition& pm, const AxisPartition& pn,
                   RegionWork region_work, int64_t ksplit = 1) {
  ksplit = std::max<int64_t>(1, ksplit);
  const int64_t m_sizes[2] = {pm.big, pm.small};
  const int64_t m_cnts[2] = {pm.num_big, pm.parts - pm.num_big};
  const int64_t n_sizes[2] = {pn.big, pn.small};
  const int64_t n_cnts[2] = {pn.num_big, pn.parts - pn.num_big};
  std::vector<double> regions;
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b) {
      const int64_t cnt = m_cnts[a] * n_cnts[b];
      if (cnt <= 0 || m_sizes[a] <= 0 || n_sizes[b] <= 0) continue;
      const double work = region_work(m_sizes[a], n_sizes[b]) / (double)ksplit;
      for (int64_t i = 0; i < cnt * ksplit; ++i) regions.push_back(work);
    }
  std::sort(regions.begin(), regions.end(), [](double x, double y) { return x > y; });
  std::vector<double> load(std::max<int64_t>(1, n_cores), 0.0);
  for (double w : regions) {  // longest-processing-time-first onto the least-loaded core
    size_t mn = 0;
    for (size_t c = 1; c < load.size(); ++c)
      if (load[c] < load[mn]) mn = c;
    load[mn] += w;
  }
  double mk = 0.0;
  for (double l : load) mk = std::max(mk, l);
  return mk;
}

// Role-aware cube split: region_work already describes ONE concrete split-K
// work unit, including any upstream recomputation induced by that K share. It
// must not be divided uniformly by ksplit (the historical left-band shortcut),
// because a general request DAG may contain Full-bound work that every split
// unit repeats and ParallelK-bound work whose M or N extent actually shrinks.
template <typename RegionWork>
double LptMakespanPerUnit(int64_t n_cores, const AxisPartition& pm, const AxisPartition& pn,
                          RegionWork region_work, int64_t ksplit) {
  ksplit = std::max<int64_t>(1, ksplit);
  const int64_t m_sizes[2] = {pm.big, pm.small};
  const int64_t m_counts[2] = {pm.num_big, pm.parts - pm.num_big};
  const int64_t n_sizes[2] = {pn.big, pn.small};
  const int64_t n_counts[2] = {pn.num_big, pn.parts - pn.num_big};
  std::vector<double> regions;
  for (int mi = 0; mi < 2; ++mi) {
    for (int ni = 0; ni < 2; ++ni) {
      const int64_t count = m_counts[mi] * n_counts[ni];
      if (count <= 0 || m_sizes[mi] <= 0 || n_sizes[ni] <= 0) continue;
      const double work = region_work(m_sizes[mi], n_sizes[ni], ksplit);
      for (int64_t i = 0; i < count * ksplit; ++i) regions.push_back(work);
    }
  }
  std::sort(regions.begin(), regions.end(), [](double lhs, double rhs) { return lhs > rhs; });
  std::vector<double> load(std::max<int64_t>(1, n_cores), 0.0);
  for (double work : regions) {
    size_t least = 0;
    for (size_t core = 1; core < load.size(); ++core) {
      if (load[core] < load[least]) least = core;
    }
    load[least] += work;
  }
  return *std::max_element(load.begin(), load.end());
}

}  // namespace

double GroundedRowReductionCycles(VectorPrimitiveFamily family, DType dtype,
                                  int64_t valid_rows, int64_t valid_cols) {
  return GroundedRowReductionCyclesImpl(family, dtype, valid_rows, valid_cols);
}

double GroundedColumnReductionCycles(VectorPrimitiveFamily family, DType dtype,
                                     int64_t valid_rows, int64_t valid_cols) {
  return GroundedColumnReductionCyclesImpl(family, dtype, valid_rows, valid_cols);
}

double GroundedVectorFillCycles(int64_t valid_rows, int64_t valid_cols) {
  if (valid_rows <= 0 || valid_cols <= 0) return -1.0;
  // TEXPANDS uses count-mode vector_dup(repeat=0). At a fresh seed-kernel
  // boundary PTO-ISA's vector queue is empty, so only its calibrated stream
  // head and tail remain; the valid extent is carried by SetVectorCount.
  return kVecExpandScalarHead + kVecExpandScalarTail;
}

// ============================================================================
// Factory
// ============================================================================

std::optional<Ascend910BCost> Ascend910BCost::create(const Problem &prob, const DAG &dag,
                                         std::vector<size_t> op_indices,
                                         bool allow_mixed) {
  if (op_indices.empty())
    return std::nullopt;
  allow_mixed = allow_mixed || prob.fuse_cube_vector;

  Ascend910BCost sg;
  sg.prob_ = &prob;
  sg.dag_ = &dag;
  sg.ops_ = std::move(op_indices);

  // P4 buildability is candidate-local. A homogeneous vector candidate must be
  // the exact pattern. A mixed candidate may contain cube stages around that
  // pattern only when the pattern is its complete vector op set; this is the
  // stage/cone form needed by QK->online-softmax without letting an unrelated
  // vector prefix/tail inherit the online algorithm.
  const FlatSet<size_t> candidate_ops(sg.ops_.begin(), sg.ops_.end());
  for (const P4Pattern &pattern : prob.p4_patterns) {
    if (pattern.kind == P4PatternKind::None) continue;
    bool exact = pattern.ops == candidate_ops;
    bool embedded_mixed_stage = false;
    if (!exact && allow_mixed) {
      bool has_cube = false;
      bool pattern_is_subset = true;
      bool covers_every_vector_op = true;
      for (size_t op : pattern.ops) {
        if (!candidate_ops.count(op)) {
          pattern_is_subset = false;
          break;
        }
      }
      if (pattern_is_subset) {
        for (size_t op : sg.ops_) {
          if (prob.ops[op].type == OpType::MatMul) {
            has_cube = true;
          } else if (!pattern.ops.count(op)) {
            covers_every_vector_op = false;
            break;
          }
        }
      }
      embedded_mixed_stage =
          has_cube && pattern_is_subset && covers_every_vector_op;
    }
    if (exact || embedded_mixed_stage) {
      sg.p4_pattern_kind_ = pattern.kind;
      sg.p4_apply_substitutions_ = pattern.apply_substitutions;
      break;
    }
  }

  const size_t num_tensors = prob.num_tensors();
  const size_t num_ops = prob.num_ops();

  std::vector<bool> is_in_sg(num_ops, false);
  std::vector<bool> is_produced(num_tensors, false);
  bool reduces_width = false, reduces_height = false;  // for reduced-axis homogeneity
  int64_t vector_min_dtype_bytes = INT64_MAX;
  int64_t vector_max_dtype_bytes = 0;
  bool all_vector_ops_grounded = true;

  for (auto i : sg.ops_) {
    const Op &candidate_op = prob.ops[i];
    is_in_sg[i] = true;
    const size_t output = candidate_op.output();
    is_produced[output] = true;
    const bool is_vector_op = candidate_op.type == OpType::Pointwise ||
                              candidate_op.type == OpType::Reduction;
    if (is_vector_op &&
        candidate_op.vector_capability == VectorOpCapability::Unsupported) {
      return std::nullopt;
    }
    // Mixed groups reuse the homogeneous vector stream derivation for their
    // AIV stage. Its iteration frame and DMA granule must therefore see only
    // tensors touched by VECTOR ops. A MatMul-produced crossing is included as
    // the vector consumer's input; unrelated Q/K/cube operands are not.
    if (is_vector_op) {
      const size_t t = output;
      sg.vector_iter_W_ = std::max(sg.vector_iter_W_, prob.tensors[t].width);
      sg.vector_iter_H_ = std::max(sg.vector_iter_H_, prob.tensors[t].height);
      vector_min_dtype_bytes = std::min(
          vector_min_dtype_bytes, (int64_t)dtype_bytes(prob.tensors[t].dtype));
      vector_max_dtype_bytes = std::max(
          vector_max_dtype_bytes, (int64_t)dtype_bytes(prob.tensors[t].dtype));
      for (size_t input : candidate_op.inputs) {
        vector_min_dtype_bytes =
            std::min(vector_min_dtype_bytes, (int64_t)dtype_bytes(prob.tensors[input].dtype));
        vector_max_dtype_bytes =
            std::max(vector_max_dtype_bytes, (int64_t)dtype_bytes(prob.tensors[input].dtype));
        sg.vector_iter_W_ = std::max(sg.vector_iter_W_, prob.tensors[input].width);
        sg.vector_iter_H_ = std::max(sg.vector_iter_H_, prob.tensors[input].height);
      }
    }
    if (is_vector_op) {
      if (HasGroundedVectorSemantics(candidate_op))
        sg.has_grounded_vector_semantics_ = true;
      else
        all_vector_ops_grounded = false;
    }
    if (is_vector_op) sg.has_vector_ = true;
    if (candidate_op.type == OpType::MatMul) {
      sg.has_matmul_ = true;
      int64_t Ki = prob.tensors[candidate_op.inputs[0]].width;
      sg.max_K_ = std::max(sg.max_K_, Ki);
    }
    if (candidate_op.type == OpType::Reduction) {
      sg.has_reduction_ = true;
      sg.reduction_count_++;
      // Reduced axis = the dim that collapses (input extent -> 1 in the output).
      size_t in0 = candidate_op.inputs[0], out = candidate_op.output();
      if (prob.tensors[out].width < prob.tensors[in0].width) {
        sg.reduced_axis_ = 1;  // width  (row reduction: [H,W] -> [H,1])
        sg.reduced_extent_ = std::max(sg.reduced_extent_, prob.tensors[in0].width);
        reduces_width = true;
      } else if (prob.tensors[out].height < prob.tensors[in0].height) {
        sg.reduced_axis_ = 2;  // height (col reduction: [H,W] -> [1,W])
        sg.reduced_extent_ = std::max(sg.reduced_extent_, prob.tensors[in0].height);
        reduces_height = true;
      }
    }
  }
  if (vector_min_dtype_bytes != INT64_MAX) {
    sg.vector_min_dtype_bytes_ = vector_min_dtype_bytes;
    sg.vector_max_dtype_bytes_ = std::max<int64_t>(1, vector_max_dtype_bytes);
    sg.vector_emit_granule_ =
        std::max<int64_t>(1, prob.vec_dma_align_bytes / vector_min_dtype_bytes);
  }

  // 910B defaults to unit-homogeneous subgraphs. Cube (MatMul) and vector
  // (Pointwise/Reduction) ops run on different cores and cross through GM, so
  // mixed admission requires an explicit model/runtime opt-in. Opaque ops
  // (gather/scatter/sort/transpose) remain singleton barriers and reductions on
  // different axes cannot share the unified grid.
  {
    bool has_cube = false, has_vector = false, has_opaque = false;
    for (auto i : sg.ops_) {
      switch (prob.ops[i].type) {
        case OpType::MatMul: has_cube = true; break;
        case OpType::Pointwise:
        case OpType::Reduction: has_vector = true; break;
        case OpType::Opaque: has_opaque = true; break;
      }
    }
    // The research Ascend910BMixed type and Problem::fuse_cube_vector both set
    // allow_mixed. They share the same plan and cost implementation; production
    // keeps the runtime policy false until the plan-driven emitter is complete.
    if (has_cube && has_vector && !allow_mixed)
      return std::nullopt;                                      // no cube↔vector fusion
    // Mixed alternation depth. Single-round-trip shapes are the 4 canonical
    // c→v / v→c / v→c→v / c→v→c (≤2 alternations). A deeper FIFO, including
    // full c→v→c→v attention, is represented but conservatively costed as
    // sequential; compiler/buildable mode rejects it until whole-FIFO skew is
    // available. This avoids both a fictional `max` and losing the topology in
    // analytic studies.
    if (has_cube && has_vector) {  // allow_mixed is true here (else returned above)
      std::vector<int> alt_depth(num_ops, 0);
      int max_alt = 0;
      for (size_t i : dag.topological_order()) {
        if (!is_in_sg[i]) continue;
        const bool i_cube = prob.ops[i].type == OpType::MatMul;
        int d = 0;
        for (auto t : prob.ops[i].inputs) {
          const int prod = dag.tensor_producer[t];
          if (prod < 0 || !is_in_sg[(size_t)prod]) continue;  // boundary input — no crossing
          const bool p_cube = prob.ops[(size_t)prod].type == OpType::MatMul;
          d = std::max(d, alt_depth[(size_t)prod] + (p_cube != i_cube ? 1 : 0));
        }
        alt_depth[i] = d;
        max_alt = std::max(max_alt, d);
      }
      sg.mixed_round_trip_depth_ = max_alt;
      if (max_alt > 2 && !prob.allow_model_ahead_mixed_multi_roundtrip)
        return std::nullopt;
    }
    if (has_opaque && sg.ops_.size() > 1) return std::nullopt;  // Opaque is a barrier
    // Reduced-axis homogeneity: a subgraph may not fuse reductions on DIFFERENT
    // axes (a width-reduction AND a height-reduction). A single unified tile is
    // coupled to the FULL extent on each reduced axis, so fusing both would force
    // the whole tensor into one tile on one core — no spatial parallelism, never
    // beneficial. The single reduced_axis_ also can't represent both (last wins),
    // so without this it would silently tile the un-forced reduced axis and break
    // that reduction. Force the partitioner to cut between them instead.
    if (reduces_width && reduces_height) return std::nullopt;
  }

  // Structural classification — boundary inputs/outputs, ephemerals, and sinks
  // — is computed ONCE by SubgraphStructure, the shared architecture-independent
  // layer. The cost model composes those facts and adds tiling/feasibility/cost
  // on top. (The execution-order DFS stays in the cost layer below: its roots
  // are refined by the epilogue detection, so it is not purely structural.)
  //
  // Rule recap: a tensor produced AND consumed inside is ephemeral (a live UB
  // band, normally zero DDR). Problem::required_outputs is the deliberate
  // exception: the same tensor is also a DDR boundary output. Other external
  // consumers remain the partition/solution ephemeral-gap concern.
  SubgraphStructure structure(prob, dag, sg.ops_);
  if (!structure.valid())
    return std::nullopt;  // empty op set, or no boundary output
  sg.boundary_inputs_  = structure.boundary_inputs();
  sg.boundary_outputs_ = structure.boundary_outputs();
  sg.ephemeral_        = structure.ephemeral();
  if (sg.has_reduction_) {
    for (size_t t : sg.boundary_outputs_) {
      const int64_t ext_r = sg.reduced_axis_ == 1 ? prob.tensors[t].width
                                                   : prob.tensors[t].height;
      if (ext_r > 1) {
        sg.reduction_spans_output_ = true;
        break;
      }
    }
  }
  // Local per-tensor ephemeral lookup the tiling code below indexes by tensor id.
  std::vector<bool> is_ephemeral(num_tensors, false);
  for (auto t : structure.ephemeral())
    is_ephemeral[t] = true;

  // Collect PW-produced ephemerals for the granule-fit check in is_valid_tiling
  // (PW has no k-loop, so its output slice must fit one (cfg.w, cfg.h) granule).
  for (auto i : sg.ops_) {
    size_t out_t = prob.ops[i].output();
    if (is_ephemeral[out_t] && prob.ops[i].type == OpType::Pointwise)
      sg.pw_produced_ephemerals_.push_back(out_t);
  }

  // Rules 2/3 (prologue-PW geometric condition):
  //   PW feeds a downstream MM's LHS (directly or through PW chain) →
  //     require cfg.w ≥ matmul.K
  //   PW feeds a downstream MM's RHS → require cfg.h ≥ matmul.K
  // One reverse-topological DP propagates the reachable matmul input role
  // through pointwise-only chains. The former per-pointwise forward BFS was
  // O(N^2) on a long chain and did entirely useless work for vector-only
  // candidates. Each in-subgraph edge is now inspected at most once.
  if (sg.has_matmul_) {
    std::vector<int64_t> reaches_lhs_k(num_ops, 0);
    std::vector<int64_t> reaches_rhs_k(num_ops, 0);
    const auto& topo = dag.topological_order();
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
      const size_t op_idx = *it;
      if (!is_in_sg[op_idx] || prob.ops[op_idx].type != OpType::Pointwise) continue;
      const size_t out_t = prob.ops[op_idx].output();
      for (size_t consumer : dag.tensor_consumers[out_t]) {
        if (!is_in_sg[consumer]) continue;
        const Op& consumer_op = prob.ops[consumer];
        if (consumer_op.type == OpType::MatMul) {
          const int64_t k = prob.tensors[consumer_op.inputs[0]].width;
          if (!consumer_op.inputs.empty() && consumer_op.inputs[0] == out_t)
            reaches_lhs_k[op_idx] = std::max(reaches_lhs_k[op_idx], k);
          if (consumer_op.inputs.size() > 1 && consumer_op.inputs[1] == out_t)
            reaches_rhs_k[op_idx] = std::max(reaches_rhs_k[op_idx], k);
        } else if (consumer_op.type == OpType::Pointwise) {
          reaches_lhs_k[op_idx] =
              std::max(reaches_lhs_k[op_idx], reaches_lhs_k[consumer]);
          reaches_rhs_k[op_idx] =
              std::max(reaches_rhs_k[op_idx], reaches_rhs_k[consumer]);
        }
      }
      sg.prologue_cfg_w_min_ =
          std::max(sg.prologue_cfg_w_min_, reaches_lhs_k[op_idx]);
      sg.prologue_cfg_h_min_ =
          std::max(sg.prologue_cfg_h_min_, reaches_rhs_k[op_idx]);
    }
  }

  if (sg.boundary_outputs_.empty())
    return std::nullopt;

  enum class SliceW : uint8_t { W_param, K_param };
  enum class SliceH : uint8_t { H_param, K_param };

  {
    // Structural sinks come from SubgraphStructure (an op whose output has no
    // in-subgraph consumer). The epilogue detection below may additionally
    // mark an internal MM as an effective sink — a cost-model refinement on top
    // of this structural set.
    sg.is_sink_op_vec_.assign(num_ops, false);
    std::vector<size_t> sink_ops = structure.sinks();
    for (auto s : sink_ops)
      sg.is_sink_op_vec_[s] = true;
    // sink_ops is non-empty: structure.valid() ⇒ ≥1 boundary output ⇒ ≥1 sink.
    size_t first_sink_out = prob.ops[sink_ops[0]].output();
    sg.out_W_ = prob.tensors[first_sink_out].width;
    sg.out_H_ = prob.tensors[first_sink_out].height;
    
    for (size_t si = 1; si < sink_ops.size(); si++) {
      size_t out = prob.ops[sink_ops[si]].output();
      if (prob.tensors[out].width != sg.out_W_ ||
          prob.tensors[out].height != sg.out_H_)
        return std::nullopt;
      if (prob.ops[sink_ops[si]].type == OpType::MatMul &&
          prob.ops[sink_ops[0]].type == OpType::MatMul) {
        if (prob.tensors[prob.ops[sink_ops[si]].inputs[0]].width !=
            prob.tensors[prob.ops[sink_ops[0]].inputs[0]].width)
          return std::nullopt;
      }
    }

    for (auto s : sink_ops) {
      if (prob.ops[s].type == OpType::Pointwise) {
        sg.has_pw_sink_ = true;
        break;
      }
    }
    // G6/S2 admission is deliberately narrower than "a reduction sink". The
    // emitted cross-core protocol is zero seed + atomic ADD, and replaying a
    // sliced cone is valid only when its sole reduction is the terminal col_sum
    // over M. Exact primitive metadata comes from the adapter; descriptor-free
    // research problems and row/max/min reductions stay at split=1. Requiring a
    // single structural sink also matches the emitter's serial multi-sink path.
    if (!sg.has_matmul_ && all_vector_ops_grounded && sink_ops.size() == 1 &&
        sg.reduction_count_ == 1) {
      const Op& sink = prob.ops[sink_ops.front()];
      if (sink.type == OpType::Reduction &&
          sink.vector_primitive == VectorPrimitiveFamily::ColSum &&
          sg.reduced_axis_ == 2) {
        sg.vector_reduction_split_kind_ =
            VectorReductionSplitKind::ColSumAtomicAdd;
      }
    }

    // Set output_K_ from the sink matmul (if any).
    //   MM-only sinks:  output_K_ = op_K(sink_mm) — standard temporal tiling.
    //   Mixed MM+PW sinks: output_K_ = op_K(sink_mm) — the MM sink determines
    //     K; has_pw_sink_ enforcement below ensures nk == 1.
    //   PW-only sinks:  output_K_ stays 1 — no temporal dimension.
    for (auto s : sink_ops) {
      if (prob.ops[s].type == OpType::MatMul) {
        sg.output_K_ = sg.op_K(s);
        sg.sink_mm_op_ = (int64_t)s;
        break;
      }
    }

    // Detect simple MM→PW epilogue pattern:
    //   All sinks are PW, and walking backward from PW sinks through PW-only
    //   chains reaches exactly one MM. That MM is the "effective sink" —
    //   its k-loop runs, accumulates into its ephemeral output, then the PW
    //   chain fires once on the completed tile. This is always valid.
    //
    //   When detected: output_K_ = effective sink MM's K, and we mark it in
    //   is_sink_op_vec_ so tiling propagation gives it FROM_NK inputs.
    if (sg.has_pw_sink_ && sg.has_matmul_ && sg.output_K_ == 1) {
      size_t found_mm = SIZE_MAX;
      bool valid = true;
      for (auto s : sink_ops) {
        if (prob.ops[s].type != OpType::Pointwise) {
          valid = false; break;  // mixed MM+PW sinks handled above
        }
        // BFS backward through PW-only chain from this PW sink
        std::vector<size_t> stack = {s};
        std::vector<bool> visited(num_ops, false);
        visited[s] = true;
        while (!stack.empty() && valid) {
          size_t op = stack.back(); stack.pop_back();
          for (auto t : prob.ops[op].inputs) {
            int prod = dag.tensor_producer[t];
            if (prod < 0 || !is_in_sg[(size_t)prod]) continue;
            if (visited[(size_t)prod]) continue;
            visited[(size_t)prod] = true;
            if (prob.ops[(size_t)prod].type == OpType::MatMul) {
              if (found_mm != SIZE_MAX && found_mm != (size_t)prod)
                valid = false;  // multiple MMs feed PW chain
              found_mm = (size_t)prod;
            } else {
              stack.push_back((size_t)prod);
            }
          }
        }
      }
      if (valid && found_mm != SIZE_MAX) {
        sg.has_simple_epilogue_ = true;
        sg.output_K_ = sg.op_K(found_mm);
        sg.sink_mm_op_ = static_cast<int64_t>(found_mm);
        sg.is_sink_op_vec_[found_mm] = true;
      }
    }

    std::vector<std::vector<size_t>> eph_consumers(num_tensors);
    for (auto i : sg.ops_)
      for (auto t : prob.ops[i].inputs)
        if (is_ephemeral[t])
          eph_consumers[t].push_back(i);

    struct RolePair { SliceW sw; SliceH sh; };
    std::vector<std::vector<RolePair>> eph_roles(num_tensors);
    std::vector<bool> eph_roles_computed(num_tensors, false);

    {
      std::vector<size_t> eph_order;
      for (auto op_idx : dag.topological_order())
        if (is_in_sg[op_idx])
          { size_t t = prob.ops[op_idx].output();
            if (is_ephemeral[t])
              eph_order.push_back(t); }

      for (int ei = (int)eph_order.size() - 1; ei >= 0; ei--) {
        size_t t = eph_order[ei];
        if (eph_roles_computed[t]) continue;

        for (auto cop : eph_consumers[t]) {
          const auto &op = prob.ops[cop];
          if (op.type == OpType::MatMul) {
            if (op.inputs[0] == t)
              eph_roles[t].push_back({SliceW::K_param, SliceH::H_param});
            else
              eph_roles[t].push_back({SliceW::W_param, SliceH::K_param});
          } else {
            size_t pw_out = op.output();
            if (sg.boundary_outputs_.count(pw_out)) {
              eph_roles[t].push_back({SliceW::W_param, SliceH::H_param});
            } else if (is_ephemeral[pw_out] && eph_roles_computed[pw_out]) {
              for (auto &r : eph_roles[pw_out]) eph_roles[t].push_back(r);
            } else {
              eph_roles[t].push_back({SliceW::W_param, SliceH::H_param});
            }
          }
        }
        eph_roles_computed[t] = true;
      }
    }

    std::set<int64_t> w_set, h_set, k_set;
    auto add_constraint = [&](size_t t, SliceW sw, SliceH sh) {
      int64_t W = prob.tensors[t].width;
      int64_t H = prob.tensors[t].height;
      if (sw == SliceW::W_param) w_set.insert(W); else k_set.insert(W);
      if (sh == SliceH::H_param) h_set.insert(H); else k_set.insert(H);
    };

    for (auto i : sg.ops_) {
      const auto &op = prob.ops[i];
      if (op.type == OpType::MatMul) {
        add_constraint(op.inputs[0], SliceW::K_param, SliceH::H_param);
        add_constraint(op.inputs[1], SliceW::W_param, SliceH::K_param);
        size_t out = op.output();
        if (sg.boundary_outputs_.count(out)) {
          add_constraint(out, SliceW::W_param, SliceH::H_param);
        } else if (is_ephemeral[out]) {
          for (auto &r : eph_roles[out]) add_constraint(out, r.sw, r.sh);
        }
      } else {
        size_t out = op.output();
        if (sg.boundary_outputs_.count(out)) {
          add_constraint(out, SliceW::W_param, SliceH::H_param);
          for (auto t : op.inputs) add_constraint(t, SliceW::W_param, SliceH::H_param);
        } else if (is_ephemeral[out]) {
          for (auto &r : eph_roles[out]) {
            add_constraint(out, r.sw, r.sh);
            for (auto t : op.inputs) add_constraint(t, r.sw, r.sh);
          }
          if (eph_roles[out].empty()) {
            add_constraint(out, SliceW::W_param, SliceH::H_param);
            for (auto t : op.inputs) add_constraint(t, SliceW::W_param, SliceH::H_param);
          }
        }
      }
    }

    sg.w_divides_.assign(w_set.begin(), w_set.end());
    sg.h_divides_.assign(h_set.begin(), h_set.end());
    sg.k_divides_.assign(k_set.begin(), k_set.end());
  }

  // Precompute reverse topo ops using DAG order
  const auto& topo = dag.topological_order();
  for (int ri = (int)topo.size() - 1; ri >= 0; ri--) {
    if (is_in_sg[topo[ri]]) {
      sg.reverse_topo_ops_.push_back(topo[ri]);
    }
  }

  {
    using TS = BoundaryTensorInfo::TileSource;
    struct TilePair { TS h; TS v; bool assigned = false; };
    std::vector<TilePair> tsrc(num_tensors);

    // Per-tensor set of distinct role signatures. Powers the multi-entry
    // boundary_tensor_info_ materialization:
    //   FULL signature (FIXED_1, FIXED_1) dominates — collapse to one full
    //     entry, drop partials.
    //   Multiple distinct partial signatures → one entry each, working set
    //     sums them. Capped at 2 partials per the row/col simplification.
    struct RoleSig { TS h; TS v; };
    std::vector<std::vector<RoleSig>> roles_per_tensor(num_tensors);
    auto push_role = [&](size_t t, TS h, TS v) {
      for (auto &r : roles_per_tensor[t])
        if (r.h == h && r.v == v) return;  // dedup identical signatures
      roles_per_tensor[t].push_back({h, v});
    };

    for (auto i : sg.ops_) {
      if (!sg.is_sink_op_vec_[i]) continue;
      { size_t t = prob.ops[i].output();
        tsrc[t] = {TS::FROM_NTW, TS::FROM_NTH, true};
        push_role(t, TS::FROM_NTW, TS::FROM_NTH); }
    }

    auto merge_source = [](TS existing, TS incoming) -> TS {
      if (existing == TS::FROM_NK || incoming == TS::FROM_NK) return TS::FROM_NK;
      return existing;
    };

    auto assign_or_check = [&](size_t t, TS new_h, TS new_v) {
      push_role(t, new_h, new_v);
      if (!tsrc[t].assigned) {
        tsrc[t] = {new_h, new_v, true};
      } else {
        // Merge for the tsrc-based tensor_tiling_ (used by op_scale in
        // compute_cost). Multi-role is tracked separately in
        // roles_per_tensor and materialized as distinct entries in
        // boundary_tensor_info_.
        tsrc[t].h = merge_source(tsrc[t].h, new_h);
        tsrc[t].v = merge_source(tsrc[t].v, new_v);
      }
    };

    for (auto op_idx : sg.reverse_topo_ops_) {
      const auto &op = prob.ops[op_idx];
      size_t out = op.output();

      if (!tsrc[out].assigned) tsrc[out] = {TS::FROM_NTW, TS::FROM_NTH, true};

      TS out_h = tsrc[out].h;
      TS out_v = tsrc[out].v;

      if (op.type == OpType::Pointwise) {
        // Broadcast-aware: an input with extent 1 on an axis the output tiles
        // is REUSED across all tiles, not split — so it is FIXED_1 on that axis,
        // not FROM_NT*. Without this a [1,N] broadcast input looks like it has
        // ntw/nth tiles but only one element, and is_valid_tiling rejects it
        // (derived tile count > tensor dim).
        for (auto t : op.inputs) {
          TS th = (prob.tensors[t].width == 1 && prob.tensors[out].width > 1)
                      ? TS::FIXED_1 : out_h;
          TS tv = (prob.tensors[t].height == 1 && prob.tensors[out].height > 1)
                      ? TS::FIXED_1 : out_v;
          assign_or_check(t, th, tv);
        }
      } else if (op.type == OpType::Reduction) {
        // A reduction has ONE input and collapses one axis: the input is read
        // FULL along the reduced axis (FIXED_1) and follows the output tiling
        // along the other. Must NOT take the matmul LHS/RHS path below, which
        // dereferences op.inputs[1] (a single-input reduction has none).
        for (auto t : op.inputs) {
          if (sg.reduced_axis_ == 2)
            assign_or_check(t, out_h, TS::FIXED_1);  // height reduced
          else
            assign_or_check(t, TS::FIXED_1, out_v);  // width reduced (default)
        }
      } else {
        size_t lhs = op.inputs[0], rhs = op.inputs[1];
        if (sg.is_sink_op_vec_[op_idx]) {
          assign_or_check(lhs, TS::FROM_NK, out_v);
          assign_or_check(rhs, out_h, TS::FROM_NK);
        } else {
          assign_or_check(lhs, TS::FIXED_1, out_v);
          assign_or_check(rhs, out_h, TS::FIXED_1);
        }
      }
    }

    sg.tensor_tiling_.resize(num_tensors);
    for (size_t t = 0; t < num_tensors; t++) {
      if (tsrc[t].assigned) {
        sg.tensor_tiling_[t] = {tsrc[t].h, tsrc[t].v};
      }
    }

    // Tensor-dim bounds (formerly min_*_dim_) are now checked per-entry in
    // is_valid_tiling rather than precomputed from the merged tsrc role.
    // Necessary for correctness under multi-role: each entry has its own
    // (h_source, v_source) signature, and each imposes its own bound.

    // Materialize boundary_tensor_info_ entries, one per distinct retained
    // role signature per tensor. Rules:
    //   full-role collapse: if any signature is FULL (FIXED_1, FIXED_1), it
    //     subsumes all partials — keep only the full entry. Full move covers
    //     any partial access.
    //   2-partial limit: if no full and >2 distinct partial signatures
    //     remain, reject the subgraph. Keeps the data structure bounded.
    std::vector<std::vector<int>> tensor_in_info(num_tensors);
    bool too_many_partials = false;
    auto ensure = [&](size_t t) -> const std::vector<int>& {
      if (!tensor_in_info[t].empty()) return tensor_in_info[t];

      // Fallback if no consumer pushed a role (shouldn't happen for tensors
      // we actually process, but be safe).
      if (roles_per_tensor[t].empty() && tsrc[t].assigned)
        push_role(t, tsrc[t].h, tsrc[t].v);

      auto &roles = roles_per_tensor[t];

      // full-role collapse: full role subsumes all others.
      bool has_full = false;
      for (auto &r : roles)
        if (r.h == TS::FIXED_1 && r.v == TS::FIXED_1) { has_full = true; break; }

      std::vector<RoleSig> retained;
      if (has_full) {
        retained.push_back({TS::FIXED_1, TS::FIXED_1});
      } else {
        retained = roles;
        if (retained.size() > 2) too_many_partials = true;
      }

      for (auto &r : retained) {
        BoundaryTensorInfo info;
        info.id = t;
        info.full_size = prob.tensors[t].width * prob.tensors[t].height;
        info.h_source = r.h;
        info.v_source = r.v;
        tensor_in_info[t].push_back((int)sg.boundary_tensor_info_.size());
        sg.boundary_tensor_info_.push_back(info);
      }
      return tensor_in_info[t];
    };

    for (auto t : sg.boundary_inputs_) {
      const auto &indices = ensure(t);
      if (is_produced[t])
        for (int idx : indices)
          sg.boundary_tensor_info_[idx].is_internally_produced = true;
    }

    for (auto t : sg.boundary_outputs_) {
      const auto &indices = ensure(t);
      // Eviction is per-tensor, priced at the producer's output tile size.
      // Flag exactly one entry to avoid N× eviction.
      if (!indices.empty())
        sg.boundary_tensor_info_[indices[0]].is_boundary_out = true;
      for (int idx : indices)
        sg.boundary_tensor_info_[idx].is_internally_produced = is_produced[t];
    }

    for (auto op_idx : sg.ops_) {
      const auto &op = prob.ops[op_idx];
      if (op.type == OpType::MatMul) {
        size_t out = op.output();
        if (sg.boundary_outputs_.count(out)) {
          const auto &indices = ensure(out);
          if (!indices.empty())
            sg.boundary_tensor_info_[indices[0]].is_mm_out = true;
        }
      }
    }

    for (auto op_idx : sg.ops_) {
      {
        size_t t = prob.ops[op_idx].output();
        if (sg.boundary_outputs_.count(t) && is_produced[t]) {
          const auto &indices = ensure(t);
          bool used_internally = false;
          for (auto cop : dag.tensor_consumers[t])
            if (is_in_sg[cop]) { used_internally = true; break; }
          if (used_internally)
            for (int idx : indices)
              sg.boundary_tensor_info_[idx].is_internally_produced = true;
        }
      }
    }

    if (too_many_partials) return std::nullopt;
  }

  sg.tensor_id_to_infos_.assign(num_tensors, std::vector<int>{});
  for (size_t idx = 0; idx < sg.boundary_tensor_info_.size(); idx++)
    sg.tensor_id_to_infos_[sg.boundary_tensor_info_[idx].id].push_back((int)idx);

  // Tile-size candidates: divisors of the role-required dims. There is no
  // super-native cap — cube tiles align to the 16-element fractal; vector tiles
  // have no alignment and no cap (a large vector tile is a per-core kernel
  // streamed in UB-chunks — the streaming fits memory, not a small tile).
  const bool matmul_910b = sg.has_matmul_;
  const int64_t cand_align = matmul_910b ? 16 : 1;
  auto valid_candidates = [&](const std::vector<int64_t> &dims) -> std::vector<int64_t> {
    if (dims.empty()) return {1};
    int64_t mx = *std::max_element(dims.begin(), dims.end());
    if (mx <= 0) return {1};
    auto divs = all_divisors(mx);
    std::vector<int64_t> result;
    for (auto c : divs) {
      if (cand_align > 1 && c % cand_align != 0) continue;  // fractal-aligned (cube)
      bool ok = true;
      for (auto v : dims) {
        if (c < v && v % c != 0) { ok = false; break; }
      }
      if (ok) result.push_back(c);
    }
    // Cube: pad a sub-16 dim UP to one 16-fractal (the cube is atomic at 16, so
    // a small-batch / GEMV output tiles to a single padded fractal — feasible,
    // not stuck at a sub-16 tile that fails the alignment check). Vector: 1.
    if (result.empty()) result.push_back(matmul_910b ? (int64_t)16 : (int64_t)1);
    return result;
  };

  // The spatial w/h candidates feed the grid via w_divides_ / h_divides_ directly
  // (gen_grid partitions the SINK output over divisors of the core count); the grid
  // handles the reduction's full-reduced-extent axis itself. Only the k candidates
  // are still materialized here.
  // PW-sink subgraphs: force k = output_K_ so nk == 1 (no temporal tiling).
  //   PW-only sinks:  output_K_ == 1 → k == 1 in solution.
  //   Mixed MM+PW sinks: output_K_ == op_K(mm) → k == K in solution
  //     (full reduction in one pass).
  // Exception: simple MM→PW epilogue — enumerate k candidates
  //   since the MM's k-loop completes before PW fires.
  if (sg.has_pw_sink_ && !sg.has_simple_epilogue_)
    sg.ks_cand_ = std::vector<int64_t>{std::max(sg.output_K_, (int64_t)1)};
  else
    sg.ks_cand_ = valid_candidates(sg.k_divides_);

  // 910B cube: the single-core k-tile is DERIVED per-op (greedy L1-strip fit in
  // derive_exec / cube_peak_l1), NOT searched — so the k search axis collapses
  // to a single sentinel (cfg.k = output_K_). This is design B: per-op k costs
  // no search (vs ~Dk^(m-1) for a searched per-op k). compute_cost overwrites
  // result.config.k with the derived per-core k for display/emit.
  if (matmul_910b)
    sg.ks_cand_ = std::vector<int64_t>{std::max(sg.output_K_, (int64_t)1)};

  // SpatialSchedule grid candidates: balanced ~C-region partitions the uniform
  // exact-divisor tiles cannot express (powers of two only yield power-of-two tile
  // counts, never a multiple of C). Used for BOTH the cube (C = cube cores) and the
  // vector (C = vector cores) paths. Each candidate is a (parts_m, parts_n,
  // split_k) TRIPLE: P*Q is a balanced 16-aligned spatial grid (a divisor of
  // {C, 2C}), bounded by each axis's 16-fractal cap; split_k is the parallel
  // contraction/reduction split. The WORK UNITS P*Q*S range freely -- filling all C
  // cores is a strong SOFT preference, but the cost (merge barrier vs streaming
  // gain) drives it (a small shape can be best at FEWER than C units when the split
  // merge outweighs recruiting idle cores). compute_cost evaluates each fixed
  // triple (no internal S sweep). PQ == 1 (the whole-output region) IS included
  // so the grid is self-sufficient on the 910B path: (1,1,S) is the pure
  // split-K / single-region fill the uniform whole-output tile used to provide.
  auto gen_grid = [&](int64_t C, int64_t maxP, int64_t maxQ, const std::vector<int64_t>& s_vals) {
    size_t cube_op_count = 0;
    for (size_t op_idx : sg.ops_) {
      if (prob.ops[op_idx].type == OpType::MatMul) ++cube_op_count;
    }
    const bool uniform_cube_only =
        matmul_910b &&
        ((cube_op_count > 1 && prob.require_uniform_cube_dag_grid) ||
         (sg.has_matmul_ && sg.has_vector_ && prob.require_buildable_mixed));
    std::set<int64_t> region_counts;  // balanced P*Q: divisors of {C, 2C} (incl. 1)
    for (int64_t R : {C, 2 * C})
      for (int64_t d = 1; d * d <= R; ++d)
        if (R % d == 0) {
          region_counts.insert(d);
          region_counts.insert(R / d);
        }
    for (int64_t PQ : region_counts) {
      for (int64_t P = 1; P <= PQ; ++P) {
        if (PQ % P != 0) continue;
        const int64_t Q = PQ / P;
        if (P > maxP || Q > maxQ) continue;
        if (uniform_cube_only && (sg.out_H_ % P != 0 || sg.out_W_ % Q != 0 || (sg.out_H_ / P) % 16 != 0 ||
                                  (sg.out_W_ / Q) % 16 != 0)) {
          continue;
        }
        for (int64_t S : s_vals) sg.grid_cand_.push_back({P, Q, S});
      }
    }
  };
  // Tile granularity: cube is 16x16-fractal aligned on both axes; vector has no
  // fractal constraint, so its free (row/height) axis tiles at 1 element and its
  // contiguous (width) axis at the 16-element DMA block. The finer row granule is
  // what lets a few-row reduction (softmax) tile enough regions to fill C.
  sg.grid_gran_h_ = matmul_910b ? 16 : 1;
  sg.grid_gran_w_ = 16;
  const int64_t Fm = (sg.out_H_ + sg.grid_gran_h_ - 1) / sg.grid_gran_h_;
  const int64_t Fn = (sg.out_W_ + sg.grid_gran_w_ - 1) / sg.grid_gran_w_;
  if (matmul_910b) {
    // Cube: the grid tiles the SINK output. Consumer-request propagation maps
    // those root M/N coordinates through arbitrary producer shapes, so internal
    // matmuls no longer need to share the sink's M dimension. split_k is legal
    // only with one matmul sink: multiple sinks need distinct split coordinates
    // and atomic targets, which are not represented by one SpatialTriple.
    const int64_t kfrac = std::max<int64_t>(1, sg.output_K_ / 16);
    size_t cube_sink_count = 0;
    for (size_t op_idx : sg.ops_) {
      if (sg.is_sink_op_vec_[op_idx] && prob.ops[op_idx].type == OpType::MatMul) ++cube_sink_count;
    }
    const std::vector<int64_t> split_values =
        cube_sink_count == 1 ? all_divisors(kfrac) : std::vector<int64_t>{1};
    // A mixed subgraph with a REDUCTION must still pin its reduced spatial axis.
    const int64_t pm = (sg.reduced_axis_ == 2) ? 1 : Fm;
    const int64_t pn = (sg.reduced_axis_ == 1) ? 1 : Fn;
    gen_grid(std::max<int64_t>(1, prob.num_cube_cores), pm, pn, split_values);
  } else if (sg.has_vector_) {
    // Vector: tile the output across the AIV cores. A reduced axis cannot be
    // spatially tiled (the whole row/col must be present to reduce), so its parts
    // pin to 1: a width reduction ([H,W]->[H,1]) tiles only height (parts_n = 1), a
    // height reduction only width (parts_m = 1); pointwise tiles both.
    const int64_t C = std::max<int64_t>(1, prob.num_vector_cores);
    const int64_t maxP = (sg.reduced_axis_ == 2) ? 1 : Fm;  // height reduced -> no M-split
    const int64_t maxQ = (sg.reduced_axis_ == 1) ? 1 : Fn;  // width  reduced -> no N-split
    // The triple's split_k is the REDUCED-AXIS (cross-core accumulation) split --
    // the vector analog of cube split-K, meaningful ONLY for a reduction SINK. It
    // lets P_spatial * S fill the cores when the non-reduced axis alone can't (a
    // softmax whose query rows are few). Pure pointwise has no axis to split -> S = 1.
    //
    // S ranges over the divisors of the reduced FRACTAL count (reduced_extent/16), capped by
    // the core budget -- MIRRORING the matmul split-K gate all_divisors(kfrac) at :870. This is
    // a FIDELITY constraint, not just a cap: it guarantees each core's reduced slice
    // IM/S = 16*(rcap/S) is 16-aligned, so the vector emit can realize the split EXACTLY
    // (disjoint reduced slices; padding the ragged tail slice would otherwise overlap the prior
    // slice -> atomic-add double-count, or read past the source tensor -> OOB DMA). Drawing S
    // from divisors of the CORE count (2*C) instead -- the previous behavior -- let the solver
    // cost splits S that do NOT divide the reduced extent (e.g. col_sum[128,256] costed S=6,
    // 6 ∤ 128); the emit cannot realize those and declines to a serial reduction, so the costed
    // parallelism was fictional. A non-16-aligned reduced axis is not cleanly splittable at all
    // (S=1): the emit runs it serial, which the cost model now prices honestly.
    std::vector<int64_t> s_vals = {1};
    if (sg.vector_reduction_split_kind_ != VectorReductionSplitKind::None &&
        sg.reduced_extent_ % 16 == 0) {
      const int64_t rcap = std::max<int64_t>(1, sg.reduced_extent_ / 16);
      s_vals.clear();
      for (int64_t s : all_divisors(rcap))
        if (s <= 2 * C) s_vals.push_back(s);
    }
    gen_grid(C, maxP, maxQ, s_vals);
  }

  // Depth-first topological execution order (the fixed pebbling order). Post-
  // order DFS from each sink over in-subgraph producers: an op is emitted only
  // after all its producers, and a branch is finished before its sibling
  // starts — minimizing simultaneously-live intermediate bands. Deterministic:
  // sinks and producers are visited in topological-position order so the same
  // subgraph always yields the same order (required for the cost cache).
  {
    auto by_topo = [&](size_t a, size_t b) {
      return dag.topo_position(a) < dag.topo_position(b);
    };
    auto sg_producers = [&](size_t op) {
      std::vector<size_t> preds;
      for (auto t : prob.ops[op].inputs) {
        int p = dag.tensor_producer[t];
        if (p >= 0 && is_in_sg[(size_t)p]) preds.push_back((size_t)p);
      }
      std::sort(preds.begin(), preds.end(), by_topo);
      preds.erase(std::unique(preds.begin(), preds.end()), preds.end());
      return preds;
    };
    std::vector<size_t> sinks;
    for (auto i : sg.ops_)
      if (sg.is_sink_op_vec_[i]) sinks.push_back(i);
    std::sort(sinks.begin(), sinks.end(), by_topo);

    std::vector<bool> visited(num_ops, false);
    std::vector<std::pair<size_t, bool>> stack;  // (op, expanded?)
    for (auto root : sinks) {
      if (visited[root]) continue;
      stack.push_back({root, false});
      while (!stack.empty()) {
        auto [op, expanded] = stack.back();
        stack.pop_back();
        if (expanded) {  // all producers already emitted
          sg.dfs_order_.push_back(op);
          continue;
        }
        if (visited[op]) continue;
        visited[op] = true;
        stack.push_back({op, true});  // emit after producers
        auto preds = sg_producers(op);
        // Push in reverse so the smallest-topo producer is processed first.
        for (auto it = preds.rbegin(); it != preds.rend(); ++it)
          if (!visited[*it]) stack.push_back({*it, false});
      }
    }
  }

  // Build the pure-cube recursive request DAG once. The sink owns SpatialM x
  // SpatialN. A matmul output request O[rows, cols] induces A[rows, K] and
  // B[K, cols]; only the designated sink's K is bound to the cross-core split.
  // Identical tensor-region requests are memoized, while distinct fan-out roles
  // become distinct schedule instances and are therefore explicitly recomputed.
  if (sg.has_matmul_ && !sg.has_vector_) {
    using B = CubeAxisBinding;
    using Key = std::tuple<size_t, int, int>;
    std::vector<size_t> cube_sinks;
    for (size_t op_idx : sg.ops_) {
      if (sg.is_sink_op_vec_[op_idx] && prob.ops[op_idx].type == OpType::MatMul) cube_sinks.push_back(op_idx);
    }
    std::sort(cube_sinks.begin(), cube_sinks.end(),
              [&](size_t a, size_t b) { return dag.topo_position(a) < dag.topo_position(b); });
    std::map<Key, int64_t> memo;
    auto request_tensor = [&](auto&& self, size_t tensor, B height_binding, B width_binding) -> int64_t {
      const Key key{tensor, static_cast<int>(height_binding), static_cast<int>(width_binding)};
      auto found = memo.find(key);
      if (found != memo.end()) return found->second;

      const int producer = dag.tensor_producer[tensor];
      if (producer < 0 || !is_in_sg[static_cast<size_t>(producer)]) return -1;
      const size_t op_idx = static_cast<size_t>(producer);
      const Op& op = prob.ops[op_idx];
      if (op.type != OpType::MatMul || op.inputs.size() != 2) return -1;

      // One split coordinate can belong to only one boundary sink. A multi-root
      // group is enumerated with S=1, so all of its contractions remain Full;
      // this also lets identical producer requests from two roots memoize.
      int64_t signed_op_idx = -1;
      if (op_idx <= static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        signed_op_idx = static_cast<int64_t>(op_idx);
      }
      const bool parallel_sink = cube_sinks.size() == 1 && signed_op_idx == sg.sink_mm_op_;
      const B produced_k_binding = parallel_sink ? B::ParallelK : B::Full;
      const B loaded_k_binding = parallel_sink ? B::ParallelK : B::SequentialK;
      const int64_t lhs_producer = self(self, op.inputs[0], height_binding, produced_k_binding);
      const int64_t rhs_producer = self(self, op.inputs[1], produced_k_binding, width_binding);

      CubeRequestNode node;
      node.op = op_idx;
      node.output.tensor = tensor;
      node.output.height_binding = height_binding;
      node.output.width_binding = width_binding;
      node.lhs.tensor = op.inputs[0];
      node.lhs.height_binding = height_binding;
      node.lhs.width_binding = loaded_k_binding;
      node.rhs.tensor = op.inputs[1];
      node.rhs.height_binding = loaded_k_binding;
      node.rhs.width_binding = width_binding;
      node.lhs_producer = lhs_producer;
      node.rhs_producer = rhs_producer;
      node.parallel_sink = parallel_sink;
      const int64_t node_idx = static_cast<int64_t>(sg.cube_request_nodes_.size());
      sg.cube_request_nodes_.push_back(node);
      memo.emplace(key, node_idx);
      if (parallel_sink) sg.cube_sink_request_node_ = node_idx;
      return node_idx;
    };

    for (size_t sink : cube_sinks) {
      const int64_t root = request_tensor(request_tensor, prob.ops[sink].output(), B::SpatialM, B::SpatialN);
      if (root >= 0) sg.cube_request_roots_.push_back(static_cast<size_t>(root));
    }
  }

  // Candidate-invariant pointwise liveness and phase cones.  This work belongs
  // in create(), not vector_stream_plan(): local search enumerates many tile
  // configurations for the same subgraph, while tensor lifetimes and semantic
  // P4 cut points do not depend on the tile.
  if (sg.has_vector_) {
    // Peak live tensor bands in the exact DFS replay order, plus one band for a
    // stage-2 prefetch.  A difference sweep computes the emitter's former
    // interval count in O(|ops|+|edges|+|tensors|), rather than its O(ops^2)
    // nested scan.
    std::vector<int> first(num_tensors, INT_MAX), last(num_tensors, -1);
    std::vector<int> vector_pos(num_ops, -1);
    for (int step = 0; step < (int)sg.dfs_order_.size(); ++step) {
      const size_t op_idx = sg.dfs_order_[(size_t)step];
      vector_pos[op_idx] = step;
      const Op& op = prob.ops[op_idx];
      if (op.type == OpType::MatMul) continue;
      const size_t out = op.output();
      first[out] = std::min(first[out], step);
      last[out] = std::max(last[out], step);
      for (size_t input : op.inputs) {
        first[input] = std::min(first[input], step);
        last[input] = std::max(last[input], step);
      }
    }
    std::vector<int> delta(sg.dfs_order_.size() + 1, 0);
    for (size_t t = 0; t < num_tensors; ++t) {
      if (last[t] < 0) continue;
      delta[(size_t)first[t]]++;
      delta[(size_t)last[t] + 1]--;
    }
    int live = 0, peak_live = 0;
    for (size_t step = 0; step < sg.dfs_order_.size(); ++step) {
      live += delta[step];
      peak_live = std::max(peak_live, live);
    }
    sg.vector_pipe_band_count_ = std::max<int64_t>(2, (int64_t)peak_live + 1);

    // Cache the candidate-invariant UB liveness topology. The tensor byte
    // contribution still depends on cfg/chunk, but producer/last-consumer
    // positions and transient membership do not. This replaces three local
    // vectors plus an interval-expansion loop in every vector_peak_ub() call.
    std::vector<bool> is_ub_band(num_tensors, false);
    for (size_t tensor : sg.ephemeral_) {
      const int producer = dag.tensor_producer[tensor];
      if (producer >= 0 && prob.ops[(size_t)producer].type == OpType::MatMul)
        continue;
      const int first_step =
          producer >= 0 && vector_pos[(size_t)producer] >= 0
              ? vector_pos[(size_t)producer]
              : 0;
      int last_consumer = -1;
      for (size_t consumer : dag.tensor_consumers[tensor]) {
        const int step = vector_pos[consumer];
        if (step >= 0 && prob.ops[consumer].type != OpType::MatMul)
          last_consumer = std::max(last_consumer, step);
      }
      if (last_consumer < 0) continue;
      sg.vector_ub_band_intervals_.push_back(
          {tensor, (size_t)first_step, (size_t)last_consumer + 1});
      is_ub_band[tensor] = true;
    }

    sg.vector_ub_transient_offsets_.reserve(sg.dfs_order_.size() + 1);
    sg.vector_ub_transient_offsets_.push_back(0);
    for (size_t op_idx : sg.dfs_order_) {
      const Op& op = prob.ops[op_idx];
      if (op.type != OpType::MatMul) {
        for (size_t input : op.inputs) {
          if (!is_ub_band[input])
            sg.vector_ub_transient_refs_.push_back(
                {input, kSkipAnyRetained});
        }
        // Reduction lowering owns a distinct work/layout tile even when the
        // source is already a live or retained band.
        if (op.type == OpType::Reduction && !op.inputs.empty())
          sg.vector_ub_transient_refs_.push_back({op.inputs[0], 0});
        const size_t output = op.output();
        if (!is_ub_band[output])
          sg.vector_ub_transient_refs_.push_back({output, kSkipRetainThese});
      }
      sg.vector_ub_transient_offsets_.push_back(
          sg.vector_ub_transient_refs_.size());
    }

    std::vector<uint8_t> vector_op_phase_mask(num_ops, 0);
    std::vector<uint8_t> vector_input_phase_mask(num_tensors, 0);
    for (size_t op : sg.ops_)
      if (prob.ops[op].type != OpType::MatMul)
        vector_op_phase_mask[op] |= kVectorPhaseBody;

    if (sg.has_reduction_) {
      FlatSet<size_t> reduction_ops;
      FlatSet<size_t> sink_ops;
      for (size_t op : sg.ops_) {
        if (prob.ops[op].type == OpType::Reduction) reduction_ops.insert(op);
        if (sg.is_sink_op_vec_[op]) sink_ops.insert(op);
      }
      FlatSet<size_t> substitutions = sg.p4_apply_substitutions_;
      if (substitutions.empty()) substitutions = reduction_ops;

      auto mark_cone = [&](const FlatSet<size_t>& roots, const FlatSet<size_t>& stops, uint8_t phase) {
        std::vector<size_t> stack(roots.begin(), roots.end());
        std::vector<bool> seen(num_ops, false);
        while (!stack.empty()) {
          const size_t op = stack.back();
          stack.pop_back();
          if (seen[op] || !is_in_sg[op] || stops.count(op)) continue;
          if (prob.ops[op].type == OpType::MatMul) continue;
          seen[op] = true;
          vector_op_phase_mask[op] |= phase;
          for (size_t input : prob.ops[op].inputs) {
            const int producer = dag.tensor_producer[input];
            if (producer >= 0 && is_in_sg[(size_t)producer]) stack.push_back((size_t)producer);
          }
        }
      };
      mark_cone(reduction_ops, {}, kVectorPhaseStats);
      if (sg.reduction_spans_output_)
        mark_cone(sink_ops, substitutions, kVectorPhaseApply);
      else
        mark_cone(sink_ops, substitutions, kVectorPhaseFinalize);
    }

    for (size_t op : sg.ops_) {
      if (prob.ops[op].type == OpType::MatMul) continue;
      const uint8_t phases = vector_op_phase_mask[op];
      for (size_t input : prob.ops[op].inputs) {
        const int producer = dag.tensor_producer[input];
        if (producer < 0 || !is_in_sg[(size_t)producer])
          vector_input_phase_mask[input] |= phases;
      }
    }

    for (size_t phase_idx = 0; phase_idx < kVectorPhases.size(); ++phase_idx) {
      const uint8_t phase = kVectorPhases[phase_idx];
      for (size_t op : sg.dfs_order_)
        if ((vector_op_phase_mask[op] & phase) != 0)
          sg.vector_phase_ops_[phase_idx].push_back(op);
      for (size_t tensor : sg.boundary_inputs_)
        if ((vector_input_phase_mask[tensor] & phase) != 0)
          sg.vector_phase_inputs_[phase_idx].push_back(tensor);
    }
  }

  // Build the mixed stage DAG once. Same-engine dependency edges form maximal
  // components; unlike-engine edges become explicit GM transfers. Every
  // candidate MixedSchedulePlan shares this immutable topology instead of
  // allocating and rescanning the op DAG in the local-search hot path.
  if (sg.has_matmul_ && sg.has_vector_) {
    auto topology = std::make_shared<MixedScheduleTopology>();
    std::vector<size_t> parent(num_ops, std::numeric_limits<size_t>::max());
    for (size_t op : sg.ops_) parent[op] = op;
    auto root = [&](size_t op) {
      while (parent[op] != op) op = parent[op];
      return op;
    };
    for (size_t consumer : sg.ops_) {
      const bool consumer_cube = prob.ops[consumer].type == OpType::MatMul;
      for (size_t tensor : prob.ops[consumer].inputs) {
        const int producer = dag.tensor_producer[tensor];
        if (producer < 0 || !is_in_sg[static_cast<size_t>(producer)]) continue;
        const size_t producer_op = static_cast<size_t>(producer);
        const bool producer_cube = prob.ops[producer_op].type == OpType::MatMul;
        if (producer_cube != consumer_cube) continue;
        const size_t producer_root = root(producer_op);
        const size_t consumer_root = root(consumer);
        if (producer_root != consumer_root) parent[consumer_root] = producer_root;
      }
    }

    std::map<size_t, size_t> root_to_stage;
    std::vector<size_t> op_to_stage(num_ops, std::numeric_limits<size_t>::max());
    for (size_t op : dag.topological_order()) {
      if (!is_in_sg[op]) continue;
      const size_t component = root(op);
      auto [it, inserted] = root_to_stage.emplace(component, topology->stages.size());
      if (inserted) {
        MixedStageTopology stage;
        stage.engine = prob.ops[op].type == OpType::MatMul
                           ? MixedEngine::Cube
                           : MixedEngine::Vector;
        topology->stages.push_back(std::move(stage));
      }
      op_to_stage[op] = it->second;
      topology->stages[it->second].ops.push_back(op);
    }

    std::set<std::tuple<size_t, size_t, size_t>> seen_transfers;
    for (size_t consumer : sg.ops_) {
      for (size_t tensor : prob.ops[consumer].inputs) {
        const int producer = dag.tensor_producer[tensor];
        if (producer < 0 || !is_in_sg[static_cast<size_t>(producer)]) continue;
        const size_t producer_op = static_cast<size_t>(producer);
        const size_t producer_stage = op_to_stage[producer_op];
        const size_t consumer_stage = op_to_stage[consumer];
        if (producer_stage == consumer_stage) continue;
        const auto key = std::make_tuple(tensor, producer_stage, consumer_stage);
        if (!seen_transfers.insert(key).second) continue;
        topology->transfers.push_back(
            {tensor, producer_stage, consumer_stage,
             topology->stages[producer_stage].engine,
             topology->stages[consumer_stage].engine});
      }
    }
    std::sort(topology->transfers.begin(), topology->transfers.end(),
              [](const MixedTransferTopology& lhs, const MixedTransferTopology& rhs) {
                return std::tie(lhs.producer_stage, lhs.consumer_stage, lhs.tensor) <
                       std::tie(rhs.producer_stage, rhs.consumer_stage, rhs.tensor);
              });
    topology->max_alternations = sg.mixed_round_trip_depth_;

    // Preserve the legacy sink-unit/fill classifier exactly, now as a
    // candidate-invariant topology property consumed by both cost and emit.
    bool saw_boundary_output = false;
    for (const auto& info : sg.boundary_tensor_info_) {
      if (info.is_boundary_out) {
        if (!saw_boundary_output) {
          topology->output_is_cube = info.is_mm_out;
          saw_boundary_output = true;
        } else if (topology->output_is_cube != info.is_mm_out) {
          topology->output_engines_uniform = false;
        }
      }
    }
    int64_t cube_to_vector = 0;
    int64_t vector_to_cube = 0;
    for (const MixedTransferTopology& transfer : topology->transfers) {
      if (transfer.producer_engine == MixedEngine::Cube) {
        ++cube_to_vector;
      } else {
        ++vector_to_cube;
      }
    }
    const bool one_way_chain =
        topology->stages.size() == 2 && topology->transfers.size() == 1 &&
        topology->transfers[0].producer_stage == 0 &&
        topology->transfers[0].consumer_stage == 1;
    const bool three_stage_chain =
        topology->stages.size() == 3 && topology->transfers.size() == 2 &&
        topology->transfers[0].producer_stage == 0 &&
        topology->transfers[0].consumer_stage == 1 &&
        topology->transfers[1].producer_stage == 1 &&
        topology->transfers[1].consumer_stage == 2;
    if (one_way_chain) {
      topology->mode = MixedPipelineMode::OneWay;
    } else if (three_stage_chain && cube_to_vector == 1 && vector_to_cube == 1 &&
               topology->max_alternations <= 2) {
      topology->mode = MixedPipelineMode::SingleRoundTripSkew;
    } else {
      topology->mode = MixedPipelineMode::MultiRoundTripSequential;
    }
    topology->emit_compatible =
        topology->output_engines_uniform &&
        (topology->mode == MixedPipelineMode::OneWay ||
         topology->mode == MixedPipelineMode::SingleRoundTripSkew);

    // Production increment 1 is deliberately narrower than the generic
    // infrastructure: one matmul followed by a linear, same-shape elementwise
    // epilogue.  This is the exact source algorithm replayed by AutoFuse.  In
    // particular, reductions/P4 and V->C/C->V->C remain analytic-only until
    // their stage-local plans and FIFO protocols are emitted.
    bool compiler_emit_compatible =
        one_way_chain && topology->stages[0].engine == MixedEngine::Cube &&
        topology->stages[1].engine == MixedEngine::Vector &&
        topology->stages[0].ops.size() == 1;
    if (compiler_emit_compatible) {
      const size_t matmul_op = topology->stages[0].ops.front();
      if (!prob.ops[matmul_op].mixed_emit_compatible) {
        compiler_emit_compatible = false;
      }
      // The tensor-level C->V emitter carries a K-window accumulator in the
      // matmul result value.  That is exact only when the result already has
      // the hardware accumulator dtype (FP32 for floating operands, INT32 for
      // integer operands).  A low-precision result needs a separate FP32/INT32
      // carry plus one final FIXPIPE narrowing before TPUSH; until that stage
      // contract is represented, reject the whole mixed topology rather than
      // silently falling back to one full-K slice.
      const Op& matmul = prob.ops[matmul_op];
      const bool has_binary_operands = matmul.inputs.size() == 2;
      const DType lhs_dtype = has_binary_operands
                                  ? prob.tensors[matmul.inputs[0]].dtype
                                  : DType::BOOL;
      const DType rhs_dtype = has_binary_operands
                                  ? prob.tensors[matmul.inputs[1]].dtype
                                  : DType::BOOL;
      // Keep compiler v0 on the floating accumulator surface. PTO also has an
      // INT8->INT32 matmul, but its successor needs an integer-family vector
      // capability table before we can price arbitrary elementwise epilogues.
      const bool compiler_cube_dtype = lhs_dtype == DType::FP32 ||
                                       lhs_dtype == DType::FP16 ||
                                       lhs_dtype == DType::BF16;
      if (!has_binary_operands || lhs_dtype != rhs_dtype || !compiler_cube_dtype ||
          prob.tensors[matmul.output()].dtype != cube_accumulator_dtype(lhs_dtype)) {
        compiler_emit_compatible = false;
      }
      const Tensor& reference = prob.tensors[prob.ops[matmul_op].output()];
      size_t previous = matmul_op;
      for (size_t vector_op : topology->stages[1].ops) {
        const Op& op = prob.ops[vector_op];
        const Tensor& output = prob.tensors[op.output()];
        int internal_inputs = 0;
        for (size_t tensor : op.inputs) {
          const int producer = dag.tensor_producer[tensor];
          if (producer >= 0 && is_in_sg[static_cast<size_t>(producer)]) {
            ++internal_inputs;
            if (static_cast<size_t>(producer) != previous) {
              compiler_emit_compatible = false;
            }
          }
          const Tensor& input = prob.tensors[tensor];
          const bool broadcastable =
              (input.height == 1 || input.height == reference.height) &&
              (input.width == 1 || input.width == reference.width);
          // Current emitted PTO primitives do not insert promotions/casts.
          // Every tensor operand and result therefore stays in the crossing
          // FP32 accumulator dtype throughout the epilogue.
          if (!broadcastable || input.dtype != output.dtype ||
              output.dtype != reference.dtype) {
            compiler_emit_compatible = false;
          }
        }
        if (op.type != OpType::Pointwise || !HasGroundedVectorSemantics(op) ||
            op.vector_capability != VectorOpCapability::Elementwise ||
            output.height != reference.height || output.width != reference.width ||
            internal_inputs != 1) {
          compiler_emit_compatible = false;
        }
        previous = vector_op;
      }
      for (size_t op : sg.ops_) {
        const size_t output = prob.ops[op].output();
        const bool is_final = op == previous;
        bool has_external_consumer = false;
        for (size_t consumer : dag.tensor_consumers[output]) {
          if (!is_in_sg[consumer]) has_external_consumer = true;
        }
        if ((!is_final && (sg.boundary_outputs_.count(output) ||
                           prob.required_outputs.count(output) ||
                           has_external_consumer)) ||
            (is_final && !sg.boundary_outputs_.count(output))) {
          compiler_emit_compatible = false;
        }
        int internal_consumers = 0;
        for (size_t consumer : dag.tensor_consumers[output]) {
          if (is_in_sg[consumer]) ++internal_consumers;
        }
        if ((!is_final && internal_consumers != 1) ||
            (is_final && internal_consumers != 0)) {
          compiler_emit_compatible = false;
        }
      }
    }
    topology->compiler_emit_compatible = compiler_emit_compatible;
    if (prob.require_buildable_mixed && !topology->compiler_emit_compatible) {
      return std::nullopt;
    }

    auto is_sink_unit = [&](size_t op) {
      return (prob.ops[op].type == OpType::MatMul) == topology->output_is_cube;
    };
    FlatSet<size_t> reaches_opposite;
    for (auto it = sg.reverse_topo_ops_.rbegin(); it != sg.reverse_topo_ops_.rend(); ++it) {
      const size_t op = *it;
      bool touches = !is_sink_unit(op);
      if (!touches) {
        for (size_t tensor : prob.ops[op].inputs) {
          const int producer = dag.tensor_producer[tensor];
          if (producer >= 0 && reaches_opposite.count(static_cast<size_t>(producer))) {
            touches = true;
            break;
          }
        }
      }
      if (touches) reaches_opposite.insert(op);
    }
    for (size_t op : sg.ops_) {
      if (is_sink_unit(op) && !reaches_opposite.count(op)) {
        topology->sink_runs_early_stage = true;
        break;
      }
    }
    sg.mixed_topology_ = std::move(topology);
  }

  return sg;
}

// ============================================================================
// Tiling validity (Replicates `evaluator.cpp` SHAPES_MISALIGNED EXACTLY)
// ============================================================================

bool Ascend910BCost::is_valid_tiling(const TileConfig &cfg) const {
  if (cfg.w <= 0 || cfg.h <= 0 || cfg.k <= 0)
    return false;

  // The tile is a DDR<->L1/UB panel. Cube tiles are 16-fractal aligned; vector
  // tiles have no alignment or cap (a large vector tile is a per-core kernel
  // streamed in UB-chunks — fits_on_chip checks the chunk). The output is NOT
  // bounded by L0c here — that L1->L0 sub-tiling is AutoTileMatmulL0's job.
  const bool matmul_910b = has_matmul_;
  if (matmul_910b) {
    if (cfg.w % 16 != 0 || cfg.h % 16 != 0) return false;  // 16-fractal aligned
  }

  // G6: a fixed vector reduced-axis split is legal only for the exact
  // terminal-col_sum protocol the emitter implements. Both axes must partition
  // without padding/overlap because atomic-add would double-count an overlap:
  // each reduced slice is DMA-granule aligned and each free-axis tile divides
  // the output exactly. split_k==0 is the legacy/ad-hoc form and stays serial
  // because it has no concrete emit descriptor to reconstruct.
  if (!matmul_910b && cfg.split_k > 1) {
    if (vector_reduction_split_kind_ == VectorReductionSplitKind::None)
      return false;
    if (reduced_extent_ % (cfg.split_k * vector_emit_granule_) != 0)
      return false;
    // cfg.w is the enumeration granule's physical maximum (16 even when the
    // logical output is only four columns). Reconstruct the element-balanced
    // logical region exactly as VectorStreamPlan does before testing overlap.
    const int64_t parts_n =
        cfg.parts_n > 0 ? cfg.parts_n
                        : std::max<int64_t>(1, out_W_ / std::max<int64_t>(1, cfg.w));
    const int64_t logical_w =
        std::min(partition_axis(out_W_, parts_n, /*granule=*/1).big,
                 vector_iter_W_);
    if (logical_w <= 0 || vector_iter_W_ % logical_w != 0)
      return false;
    // The seed is a real row-major TEXPANDS + assemble kernel. PTOAS requires
    // one row to span at least one DMA block; a thin FP32 [1,4] seed (16 B)
    // cannot lower even though the atomic body itself is valid.
    if (boundary_outputs_.empty() ||
        logical_w * dtype_bytes(prob_->tensors[*boundary_outputs_.begin()].dtype) <
            prob_->vec_dma_align_bytes)
      return false;
  }

  // Grid (SpatialSchedule) mode: w,h carry the PHYSICAL (max) region extent of a
  // non-uniform parts_m x parts_n partition, which need NOT evenly divide the
  // output -- so skip the exact-divisor check (the 16-alignment above + the L1
  // fit in fits_on_chip still apply). parts are clamped to the fractal count in
  // partition_axis, so no region is empty.
  //
  // NOTE (uniform tile, not on the live solver path): best_cost is GRID-ONLY on
  // the 910B -- the SpatialSchedule grid, including the (1,1) whole-output region,
  // covers every fill, so the solver never emits a parts_m == 0 cube/vector config.
  // The cfg.parts_m == 0 exact-divisor branch below is reached ONLY by a
  // directly-constructed TileConfig{w,h,k} (unit tests / ad-hoc API calls). Kept
  // for that path; not exercised by the partition/search/solution pipeline.
  if (cfg.parts_m == 0) {
    for (int64_t v : w_divides_)
      if (cfg.w < v && v % cfg.w != 0) return false;
    for (int64_t v : h_divides_)
      if (cfg.h < v && v % cfg.h != 0) return false;
  }
  // 910B cube tile spans the full contraction (k = max_K_, accumulated in L0c);
  // the per-op k-divisibility rule (temporal-tiling correctness) does not apply
  // here, and for a chained group max_K_ may exceed a smaller op's K.
  if ((!has_pw_sink_ || has_simple_epilogue_) && !matmul_910b) {
    for (int64_t v : k_divides_) {
      // cfg.k > op_K is physically undefined — there's nothing
      // to stream into the back half of the k-granule, and whatever gets
      // summed in would corrupt the accumulator.
      if (cfg.k > v) return false;
      if (cfg.k < v && v % cfg.k != 0) return false;
    }
  }
  // For PW-sink subgraphs k is irrelevant (nk is always 1): skip k
  // divisibility but enforce nk == 1 explicitly below.

  // Derived tile-count bounds: reject if ntw/nth/nk would exceed any tensor's
  // dimension in the corresponding direction. Without this, slice computation
  // produces zero-size slices (integer division W/h_tiles = 0 when h_tiles > W).
  int64_t ntw = std::max(out_W_ / cfg.w, (int64_t)1);
  int64_t nth = std::max(out_H_ / cfg.h, (int64_t)1);
  int64_t nk = has_matmul_ ? std::max(output_K_ / cfg.k, (int64_t)1) : 1;

  // PW-sink: no temporal tiling allowed, UNLESS simple MM→PW epilogue.
  if (has_pw_sink_ && !has_simple_epilogue_ && nk > 1) return false;

  // Rules 2/3: prologue-PW geometric condition. A PW that feeds
  // an MM's LHS (via PW-only chain) requires cfg.w ≥ matmul.K so a single
  // PW tile spans the full LHS K-axis; feeding RHS requires cfg.h ≥ matmul.K.
  // Applies at all nk — the constraint is geometric (PW tile shape vs
  // K-axis), not conditional on split-K. No prologue PW → thresholds are 0
  // and these checks are no-ops.
  if (cfg.w < prologue_cfg_w_min_) return false;
  if (cfg.h < prologue_cfg_h_min_) return false;

  // Per-entry tensor-dim bounds (multi-role). For each
  // distinct role signature of each boundary tensor, ensure the derived
  // tile count doesn't exceed the tensor's dim in that direction (otherwise
  // slice < 1). Replaces the old min_*_dim_ bounds which were computed from
  // the single merged role — incorrect when a tensor's roles disagree on
  // which axis is tiled how.
  // Divisibility isn't enforced here; the existing w_divides_ / h_divides_ /
  // k_divides_ checks cover the cfg-granule divisibility, and non-integer
  // slices are tolerated by compute_cost's floating-point accounting.
  for (const auto &info : boundary_tensor_info_) {
    int64_t ht = info.eval_h_tiles(ntw, nk);
    int64_t vt = info.eval_v_tiles(nth, nk);
    int64_t W = prob_->tensors[info.id].width;
    int64_t H = prob_->tensors[info.id].height;
    if (ht > W || vt > H) return false;
  }
  // Ephemerals aren't in boundary_tensor_info_; use tensor_tiling_ (single
  // merged role — ephemerals aren't materialized per-role since they're
  // zero-cost and don't affect WS/IO accounting).
  for (size_t t : ephemeral_) {
    const auto &tp = tensor_tiling_[t];
    BoundaryTensorInfo tmp;
    tmp.h_source = tp.h;
    tmp.v_source = tp.v;
    int64_t ht = tmp.eval_h_tiles(ntw, nk);
    int64_t vt = tmp.eval_v_tiles(nth, nk);
    int64_t W = prob_->tensors[t].width;
    int64_t H = prob_->tensors[t].height;
    if (ht > W || vt > H) return false;
  }

  // Granule-fit check on ephemerals. Every op in the subgraph runs at the
  // subgraph's (cfg.w, cfg.h) granule; the slice its producer writes per
  // execution must be representable within that granule:
  //   PW producer: slice ≤ (cfg.w, cfg.h) — except an admitted S2 cone, whose
  //                reduced M slice is M/S. PW has no independent k-loop.
  //   MM producer: slice ≤ native — MM's internal k-loop allows slices that
  //                exceed cfg as long as hardware-native-sized.
  // Mostly redundant with cfg ≤ native + role propagation, kept as a
  // defensive check against role-propagation surprises in long op chains.
  auto slice_for = [&](size_t t, int64_t &slice_w, int64_t &slice_h) {
    const auto &tp = tensor_tiling_[t];
    BoundaryTensorInfo tmp;
    tmp.h_source = tp.h;
    tmp.v_source = tp.v;
    int64_t ht = tmp.eval_h_tiles(ntw, nk);
    int64_t vt = tmp.eval_v_tiles(nth, nk);
    int64_t W = prob_->tensors[t].width;
    int64_t H = prob_->tensors[t].height;
    slice_w = W / std::max(ht, (int64_t)1);
    slice_h = H / std::max(vt, (int64_t)1);
  };
  const bool vector_reduction_split =
      !matmul_910b && cfg.split_k > 1 &&
      vector_reduction_split_kind_ != VectorReductionSplitKind::None;
  const int64_t pw_slice_h_limit =
      vector_reduction_split ? reduced_extent_ / cfg.split_k : cfg.h;
  for (size_t t : pw_produced_ephemerals_) {
    int64_t sw, sh;
    slice_for(t, sw, sh);
    // A terminal col_sum split replays its upstream pointwise cone once per
    // disjoint reduced-axis partial. Role propagation reports the unsplit M
    // extent because the sink output has M=1; compare the actual M/S slice the
    // emitter builds. Size-one broadcasts remain full and are not divided.
    if (vector_reduction_split && sh > 1) {
      if (sh % cfg.split_k != 0) return false;
      sh /= cfg.split_k;
    }
    if (sw > cfg.w || sh > pw_slice_h_limit) return false;
  }

  // Multi-role tensors are now modeled explicitly via multi-entry
  // boundary_tensor_info_; divisibility checks above cover
  // shape constraints across all role orientations, and the 2-partial limit
  // is enforced at Ascend910BCost::create. No symbolic-propagation conflict check
  // is needed — the former slow path rejected exactly the multi-role configs
  // the multi-role accounting now accepts.
  return true;
}


// Greedy per-op k + red-blue pebble peak over the fixed execution order.
// See the header for the model. Intermediate bands are k-independent residents;
// each matmul's boundary operand strip is sized by a per-op k derived to fit the
// headroom the bands leave. Peak = max over steps of (live bands + this step's
// operand strip). Strips count BOUNDARY operands only — an intermediate operand
// is already a live band (the same boundary/ephemeral split the roofline uses).
int64_t Ascend910BCost::cube_binding_extent(CubeAxisBinding binding, int64_t full_extent, int64_t m_extent,
                                            int64_t n_extent, int64_t split) const {
  switch (binding) {
    case CubeAxisBinding::Full:
    case CubeAxisBinding::SequentialK:
      return full_extent;
    case CubeAxisBinding::SpatialM:
      return std::min(full_extent, m_extent);
    case CubeAxisBinding::SpatialN:
      return std::min(full_extent, n_extent);
    case CubeAxisBinding::ParallelK:
      if (split <= 0 || full_extent % split != 0) return 0;
      return full_extent / split;
  }
  return 0;
}

int64_t Ascend910BCost::derive_exec(const TileConfig& cfg, int64_t sink_K_eff,
                                    const FlatSet<size_t>& retained_from_prev,
                                    const FlatSet<size_t>& retain_these,
                                    std::vector<int64_t>* pernode_k_out) const {
  // Full L1/Mat budget -- double-buffering does NOT reserve half. The two
  // ping-pong buffers are not "L1 + a spare half"; together they ARE the L1, so
  // the operand strip that fits is the full pool. Double-buffering is realized in
  // the emit by streaming each seq-K strip as >=2 sub-strips (load s+1 while
  // computing s) -- it HALVES the per-load k, not the resident operand. Reserving
  // half here would double-count the prefetch buffer and wrongly reject tiles
  // whose operand genuinely fits.
  const double l1 = (double)prob_->l1_capacity;

  if (!cube_request_nodes_.empty()) {
    if (sink_K_eff <= 0 || output_K_ % sink_K_eff != 0) return INT64_MAX;
    const int64_t split = output_K_ / sink_K_eff;
    const int64_t m_extent = std::min(cfg.h, out_H_);
    const int64_t n_extent = std::min(cfg.w, out_W_);
    const size_t node_count = cube_request_nodes_.size();

    // One liveness interval per memoized tensor-region request. Different
    // fan-out roles are distinct nodes; identical roles share one producer and
    // remain live through their last requesting consumer.
    std::vector<int64_t> last_use(node_count, -1);
    for (size_t consumer = 0; consumer < node_count; ++consumer) {
      const CubeRequestNode& node = cube_request_nodes_[consumer];
      if (node.lhs_producer >= 0) {
        last_use[static_cast<size_t>(node.lhs_producer)] = std::max<int64_t>(
            last_use[static_cast<size_t>(node.lhs_producer)], static_cast<int64_t>(consumer));
      }
      if (node.rhs_producer >= 0) {
        last_use[static_cast<size_t>(node.rhs_producer)] = std::max<int64_t>(
            last_use[static_cast<size_t>(node.rhs_producer)], static_cast<int64_t>(consumer));
      }
    }

    // Difference/prefix liveness sweep. Expanding every [producer,last_use]
    // interval here would make this candidate-hot derivation O(nodes^2).
    std::vector<int64_t> band_delta(node_count + 1, 0);
    for (size_t producer = 0; producer < node_count; ++producer) {
      if (last_use[producer] < 0) continue;
      const CubeRequest& request = cube_request_nodes_[producer].output;
      const Tensor& tensor = prob_->tensors[request.tensor];
      const int64_t h = cube_binding_extent(request.height_binding, tensor.height, m_extent, n_extent, split);
      const int64_t w = cube_binding_extent(request.width_binding, tensor.width, m_extent, n_extent, split);
      if (h <= 0 || w <= 0) return INT64_MAX;
      const int64_t bytes = h * w * dtype_bytes(tensor.dtype);
      band_delta[producer] += bytes;
      const size_t after_last = static_cast<size_t>(last_use[producer]) + 1;
      if (after_last < band_delta.size()) band_delta[after_last] -= bytes;
    }
    std::vector<int64_t> band_at(node_count, 0);
    int64_t live_bytes = 0;
    for (size_t step = 0; step < node_count; ++step) {
      live_bytes += band_delta[step];
      band_at[step] = live_bytes;
    }

    int64_t base = 0;
    for (size_t tensor : retained_from_prev) base += prob_->tensors[tensor].size_bytes();
    for (size_t tensor : retain_these) {
      if (!retained_from_prev.count(tensor)) base += prob_->tensors[tensor].size_bytes();
    }

    if (pernode_k_out) pernode_k_out->assign(node_count, 0);
    int64_t peak = 0;
    for (size_t step = 0; step < node_count; ++step) {
      const CubeRequestNode& node = cube_request_nodes_[step];
      const Op& op = prob_->ops[node.op];
      const Tensor& output = prob_->tensors[node.output.tensor];
      const int64_t h =
          cube_binding_extent(node.output.height_binding, output.height, m_extent, n_extent, split);
      const int64_t w =
          cube_binding_extent(node.output.width_binding, output.width, m_extent, n_extent, split);
      if (h <= 0 || w <= 0) return INT64_MAX;
      const int64_t K_eff = node.parallel_sink ? sink_K_eff : op_K(node.op);
      const int64_t lhs_b = node.lhs_producer >= 0 ? 0 : dtype_bytes(prob_->tensors[node.lhs.tensor].dtype);
      const int64_t rhs_b = node.rhs_producer >= 0 ? 0 : dtype_bytes(prob_->tensors[node.rhs.tensor].dtype);
      const int64_t per_unit = lhs_b * h + rhs_b * w;
      // Output/L0C subtiles are outer to the K-window loop, so the running C
      // never leaves L0C and consumes no L1 band. A2/A3 has no Mat->Acc path;
      // modeling a full-region L1 accumulator here would describe an
      // unemittable algorithm. Internal completed outputs are already in
      // band_at[] over their producer/consumer lifetime.
      const int64_t bands = base + band_at[step];
      if (bands > static_cast<int64_t>(l1)) return INT64_MAX;

      int64_t kk = K_eff;
      if (per_unit > 0) {
        const double headroom = l1 - static_cast<double>(bands);
        if (headroom <= 0) return INT64_MAX;
        int64_t max_kk = static_cast<int64_t>(headroom / static_cast<double>(per_unit));
        max_kk = (max_kk / 16) * 16;
        if (max_kk < 16) return INT64_MAX;
        max_kk = std::min(max_kk, K_eff);
        kk = 0;
        for (int64_t d = std::min(max_kk, K_eff); d >= 16; d -= 16) {
          if (K_eff % d == 0) {
            kk = d;
            break;
          }
        }
        if (kk == 0) return INT64_MAX;
      }
      if (pernode_k_out) (*pernode_k_out)[step] = kk;
      peak = std::max(peak, bands + per_unit * kk);
    }
    return peak;
  }

  const auto &order = dfs_order_;

  // Position of each op in the execution order (-1 = not in this subgraph).
  std::vector<int> pos(prob_->num_ops(), -1);
  for (int i = 0; i < (int)order.size(); ++i) pos[order[i]] = i;

  // Live intermediate-band bytes per step. The L1/Mat working set of a cube tile
  // has two charged contributors; the peak below sums only these:
  //
  //   (1) EPHEMERAL intermediates ARE charged. A fused intermediate (T in
  //       C=(A@B)@D) must become fully L1-resident for its consumer matmul to
  //       read it as an operand, so it occupies a [full_width, M-band h] band. We
  //       charge it across the whole interval [producer .. last consumer].
  //       Charging it AT the producer step too is deliberate and conservative:
  //         - T's band routinely EXCEEDS L0c (cube_capacity, 128KB) — e.g. a
  //           [256,256] FP32 band is 256KB > 128KB — so it cannot sit wholly in
  //           the L0c accumulator; it spills into L1 as it is produced.
  //         - Even when it fits L0c, at the producer->consumer transition T has
  //           materialised in L1 while the producer's operand strips may still be
  //           resident, so the two coexist.
  //       "Ephemeral" means no DDR round-trip — NOT zero memory.
  //
  //   (2) The boundary OUTPUT is NEVER charged to L1. On the 910B the L0c
  //       accumulator drains DIRECTLY back to DDR (write-back from L0c, no L1
  //       staging of the result), so a tile's own output never needs an L1 slot.
  //       (An ephemeral output is the consumer's input, charged via (1); a
  //       boundary output goes straight to DDR.) L0c sizing is the L0 sub-tiling
  //       level's job (AutoTileMatmulL0), not this DDR<->L1 model.
  //
  // Bands are k-INDEPENDENT; the per-step boundary-operand strip (sized by the
  // greedy k, below) is added on top of the live bands at each step.
  std::vector<int64_t> band_at(order.size(), 0);
  for (size_t t : ephemeral_) {
    int pr = dag_->tensor_producer[t];
    int prod = (pr >= 0 && pos[pr] >= 0) ? pos[pr] : 0;
    // An ephemeral is an L1 OPERAND band only while an in-subgraph MATMUL reads it
    // (the consumer needs it L1-resident). One consumed only by a VECTOR op (a
    // cube->vector crossing) is NOT an L1 band — it drains from L0c to the GM ring.
    // Homogeneous cube: every ephemeral has a matmul consumer, so last_mm == last
    // consumer and this is unchanged; it only differs inside a mixed group.
    int last_mm = -1;
    for (auto c : dag_->tensor_consumers[t])
      if (pos[c] >= 0 && prob_->ops[c].type == OpType::MatMul)
        last_mm = std::max(last_mm, pos[c]);
    if (last_mm < 0) continue;  // no in-subgraph matmul consumer -> not an L1 band
    int64_t h = std::min(cfg.h, prob_->tensors[t].height);
    int64_t bytes = prob_->tensors[t].width * h * dtype_bytes(prob_->tensors[t].dtype);
    for (int s = prod; s <= last_mm; ++s) band_at[s] += bytes;
  }

  // Retained (coupling-layer) tensors are full-resident across the subgraph.
  // Disabled for 910B (cross-subgraph data routes DDR); handled defensively.
  int64_t base = 0;
  for (auto t : retained_from_prev) base += prob_->tensors[t].size_bytes();
  for (auto t : retain_these)
    if (!retained_from_prev.count(t)) base += prob_->tensors[t].size_bytes();

  if (pernode_k_out) pernode_k_out->assign(prob_->num_ops(), 0);

  int64_t peak = 0;
  for (int s = 0; s < (int)order.size(); ++s) {
    const size_t opi = order[s];
    const Op &op = prob_->ops[opi];
    const int64_t bands = base + band_at[s];
    // A VECTOR op does not pressure L1 — it runs on the vector unit (UB). Skip it
    // from the cube L1 sweep. (Homogeneous cube has no vector ops, so this is never
    // taken there; it only matters inside a mixed group.)
    if (op.type != OpType::MatMul)
      continue;
    const size_t lhs = op.inputs[0], rhs = op.inputs[1];
    const int64_t M_o = prob_->tensors[op.output()].height;
    const int64_t N_o = prob_->tensors[op.output()].width;
    const int64_t h = std::min(cfg.h, M_o);
    const int64_t w = std::min(cfg.w, N_o);
    const int64_t K_eff = is_sink_op_vec_[opi] ? sink_K_eff : op_K(opi);
    // Per unit of k, the boundary operand strip costs lhs_bytes*h (LHS [k,h]) +
    // rhs_bytes*w (RHS [w,k]); an intermediate operand contributes 0 here (it is
    // a band). per_unit==0 => both operands are bands, no DDR strip to size.
    const int64_t lhs_b = ephemeral_.count(lhs) ? 0 : dtype_bytes(prob_->tensors[lhs].dtype);
    const int64_t rhs_b = ephemeral_.count(rhs) ? 0 : dtype_bytes(prob_->tensors[rhs].dtype);
    const int64_t per_unit = lhs_b * h + rhs_b * w;
    int64_t kk;
    if (per_unit == 0) {
      kk = K_eff;  // no boundary strip; full K in one accumulation
    } else {
      const double headroom = l1 - (double)bands;
      if (headroom <= 0) return INT64_MAX;  // bands alone overflow L1
      int64_t max_kk = (int64_t)(headroom / (double)per_unit);
      max_kk = (max_kk / 16) * 16;          // 16-fractal aligned
      if (max_kk < 16) return INT64_MAX;     // not even one fractal strip fits
      if (max_kk > K_eff) max_kk = K_eff;
      kk = 0;  // largest 16-aligned divisor of K_eff not exceeding max_kk
      for (int64_t d = std::min(max_kk, K_eff); d >= 16; d -= 16)
        if (K_eff % d == 0) { kk = d; break; }
      if (kk == 0) return INT64_MAX;
    }
    if (pernode_k_out) (*pernode_k_out)[opi] = kk;
    peak = std::max(peak, bands + per_unit * kk);
  }
  return peak;
}

int64_t Ascend910BCost::cube_peak_l1(const TileConfig &cfg,
                               std::vector<int64_t> *perop_k) const {
  if (cube_request_nodes_.empty()) return derive_exec(cfg, output_K_, {}, {}, perop_k);
  std::vector<int64_t> pernode_k;
  const int64_t peak = derive_exec(cfg, output_K_, {}, {}, perop_k ? &pernode_k : nullptr);
  if (perop_k) {
    perop_k->assign(prob_->num_ops(), 0);
    for (size_t node_idx = 0; node_idx < cube_request_nodes_.size(); ++node_idx) {
      const size_t op = cube_request_nodes_[node_idx].op;
      const int64_t k = pernode_k[node_idx];
      if ((*perop_k)[op] == 0 || k < (*perop_k)[op]) (*perop_k)[op] = k;
    }
  }
  return peak;
}

CubeSchedulePlan Ascend910BCost::cube_schedule_plan(
    const TileConfig &cfg,
    const FlatSet<size_t> &retained_from_prev,
    const FlatSet<size_t> &retain_these,
    int64_t parallel_split) const {
  return derive_cube_schedule_plan(cfg, retained_from_prev, retain_these, parallel_split, nullptr);
}

CubeSchedulePlan Ascend910BCost::derive_cube_schedule_plan(
    const TileConfig &cfg,
    const FlatSet<size_t> &retained_from_prev,
    const FlatSet<size_t> &retain_these,
    int64_t parallel_split,
    L0PlanMemo *l0_memo) const {
  CubeSchedulePlan plan;
  plan.config = cfg;
  const int64_t split = std::max<int64_t>(1, parallel_split);
  if (!has_matmul_ || has_vector_ || cube_request_nodes_.empty() || !is_valid_tiling(cfg) ||
      output_K_ % split != 0) {
    return plan;
  }

  const bool lone_matmul = cube_request_nodes_.size() == 1;
  std::vector<int64_t> pernode_window_k;
  // The pre-plan lone path derived its L1 window against the full contraction,
  // then capped that full-K divisor to the selected share. Keep that exact
  // reconstruction; general request DAGs derive directly at K/S because their
  // upstream request shapes themselves depend on S.
  const int64_t derive_sink_k = lone_matmul ? output_K_ : output_K_ / split;
  const int64_t peak = derive_exec(cfg, derive_sink_k, retained_from_prev, retain_these, &pernode_window_k);
  if (peak == INT64_MAX) return plan;

  const int64_t parts_m = cfg.parts_m > 0
                              ? cfg.parts_m
                              : std::max<int64_t>(1, out_H_ / std::max<int64_t>(1, cfg.h));
  const int64_t parts_n = cfg.parts_n > 0
                              ? cfg.parts_n
                              : std::max<int64_t>(1, out_W_ / std::max<int64_t>(1, cfg.w));
  plan.feasible = true;
  plan.emit_compatible = true;
  plan.m_partition = partition_axis(out_H_, parts_m, grid_gran_h_);
  plan.n_partition = partition_axis(out_W_, parts_n, grid_gran_w_);
  plan.spatial_tiles = parts_m * parts_n;
  plan.split_k = split;
  plan.work_units = plan.spatial_tiles * split;
  plan.peak_l1_bytes = peak;
  plan.seed_required = split > 1;
  plan.model_overlap_granted = true;
  plan.overlap_implementable = true;
  const int64_t m_extent = std::min(cfg.h, out_H_);
  const int64_t n_extent = std::min(cfg.w, out_W_);
  if (plan.seed_required && !boundary_outputs_.empty()) {
    const DType output_dtype = prob_->tensors[*boundary_outputs_.begin()].dtype;
    const int64_t bytes = dtype_bytes(output_dtype);
    const int64_t seed_h = std::min(
        m_extent, std::max<int64_t>(1, prob_->vec_capacity / std::max<int64_t>(1, n_extent * bytes)));
    plan.seed.present = true;
    plan.seed.work_units = ((out_H_ + seed_h - 1) / seed_h) * parts_n;
    plan.seed.valid_rows = seed_h;
    plan.seed.valid_cols = n_extent;
    plan.seed.bytes = plan.seed.work_units * seed_h * n_extent * bytes;
  }

  std::vector<int64_t> last_use(cube_request_nodes_.size(), -1);
  FlatSet<size_t> roots(cube_request_roots_.begin(), cube_request_roots_.end());
  for (size_t consumer = 0; consumer < cube_request_nodes_.size(); ++consumer) {
    const CubeRequestNode& node = cube_request_nodes_[consumer];
    if (node.lhs_producer >= 0) {
      last_use[static_cast<size_t>(node.lhs_producer)] =
          std::max<int64_t>(last_use[static_cast<size_t>(node.lhs_producer)], static_cast<int64_t>(consumer));
    }
    if (node.rhs_producer >= 0) {
      last_use[static_cast<size_t>(node.rhs_producer)] =
          std::max<int64_t>(last_use[static_cast<size_t>(node.rhs_producer)], static_cast<int64_t>(consumer));
    }
  }

  auto concrete_region = [&](const CubeRequest& request) {
    CubeTensorRegionPlan region;
    region.tensor = request.tensor;
    region.height_binding = request.height_binding;
    region.width_binding = request.width_binding;
    const Tensor& tensor = prob_->tensors[request.tensor];
    region.height = cube_binding_extent(request.height_binding, tensor.height, m_extent, n_extent, split);
    region.width = cube_binding_extent(request.width_binding, tensor.width, m_extent, n_extent, split);
    return region;
  };

  for (size_t node_idx = 0; node_idx < cube_request_nodes_.size(); ++node_idx) {
    const CubeRequestNode& node = cube_request_nodes_[node_idx];
    const Op& op = prob_->ops[node.op];

    CubeMatmulSchedule mm;
    mm.instance = node_idx;
    mm.op = node.op;
    mm.lhs_producer = node.lhs_producer;
    mm.rhs_producer = node.rhs_producer;
    mm.is_sink = roots.count(node_idx) != 0;
    mm.lhs_ephemeral = node.lhs_producer >= 0;
    mm.rhs_ephemeral = node.rhs_producer >= 0;
    mm.output_ephemeral = last_use[node_idx] >= 0;
    mm.contraction = op_K(node.op);
    mm.effective_contraction = node.parallel_sink ? mm.contraction / split : mm.contraction;
    mm.output = concrete_region(node.output);
    mm.lhs = concrete_region(node.lhs);
    mm.rhs = concrete_region(node.rhs);

    int64_t window = node_idx < pernode_window_k.size() && pernode_window_k[node_idx] > 0
                         ? pernode_window_k[node_idx]
                         : mm.effective_contraction;
    if (lone_matmul && node.parallel_sink) {
      window = CappedSinkWindow(mm.contraction, window, split);
      if (window == 0) return CubeSchedulePlan{};
    }
    window = std::min(window, mm.effective_contraction);
    mm.k_loop.l1_window_k = window;

    const bool loads_boundary = !mm.lhs_ephemeral || !mm.rhs_ephemeral;
    const int64_t db_chunk = loads_boundary
                                 ? CubePipelinedChunk(mm.effective_contraction, window)
                                 : 0;
    if (db_chunk > 0) {
      mm.k_loop.chunk = db_chunk;
      mm.k_loop.pipeline_stages = 2;
    } else {
      mm.k_loop.chunk = std::max<int64_t>(16, std::min(window, mm.effective_contraction));
      mm.k_loop.pipeline_stages = 1;
    }
    mm.k_loop.full_chunks = mm.effective_contraction / mm.k_loop.chunk;
    mm.k_loop.tail = mm.effective_contraction -
                     mm.k_loop.full_chunks * mm.k_loop.chunk;
    if (loads_boundary && mm.k_loop.pipeline_stages != 2) {
      plan.model_overlap_granted = false;
      plan.overlap_implementable = false;
    }

    const DType lhs_dtype = prob_->tensors[mm.lhs.tensor].dtype;
    const DType rhs_dtype = prob_->tensors[mm.rhs.tensor].dtype;
    const DType output_dtype = prob_->tensors[mm.output.tensor].dtype;
    mm.accumulator_dtype = cube_accumulator_dtype(lhs_dtype);
    mm.storage_dtype = output_dtype;
    const bool has_chunked_carry = mm.k_loop.full_chunks >= 2;
    const int64_t init_k = has_chunked_carry ? mm.k_loop.chunk : mm.effective_contraction;
    auto derive_variant = [&](int64_t tile_m, int64_t tile_n) {
      CubeOutputTileVariant variant;
      variant.height = tile_m;
      variant.width = tile_n;
      variant.l0_init =
          DeriveL0MatmulPlan(prob_, tile_m, tile_n, init_k, lhs_dtype, rhs_dtype, mm.accumulator_dtype,
                             /*accumulator_read=*/false, L0OutputTarget::Acc, l0_memo);
      if (has_chunked_carry) {
        variant.l0_rolled =
            DeriveL0MatmulPlan(prob_, tile_m, tile_n, mm.k_loop.chunk, lhs_dtype, rhs_dtype,
                               mm.accumulator_dtype, /*accumulator_read=*/true, L0OutputTarget::Acc, l0_memo);
        if (mm.k_loop.tail > 0) {
          variant.l0_tail =
              DeriveL0MatmulPlan(prob_, tile_m, tile_n, mm.k_loop.tail, lhs_dtype, rhs_dtype,
                                 mm.accumulator_dtype, /*accumulator_read=*/true, L0OutputTarget::Acc, l0_memo);
        }
      }
      return variant;
    };

    // Find one output tile that every K phase can keep wholly in L0C. This is
    // the legal PTO nesting: output tile outer, GM->L1 K windows inner. Repeating
    // the shared chooser to a fixed point handles a tail phase whose operand
    // capacity requires a smaller M/N tile than the first phase.
    int64_t tile_m = mm.output.height;
    int64_t tile_n = mm.output.width;
    for (int iteration = 0; iteration < 8; ++iteration) {
      const CubeOutputTileVariant probe = derive_variant(tile_m, tile_n);
      if (!probe.l0_init.feasible || (has_chunked_carry && !probe.l0_rolled.feasible) ||
          (mm.k_loop.tail > 0 && !probe.l0_tail.feasible)) {
        plan.emit_compatible = false;
        break;
      }
      int64_t next_m = std::min(tile_m, probe.l0_init.m);
      int64_t next_n = std::min(tile_n, probe.l0_init.n);
      if (probe.l0_rolled.feasible) {
        next_m = std::min(next_m, probe.l0_rolled.m);
        next_n = std::min(next_n, probe.l0_rolled.n);
      }
      if (probe.l0_tail.feasible) {
        next_m = std::min(next_m, probe.l0_tail.m);
        next_n = std::min(next_n, probe.l0_tail.n);
      }
      if (next_m == tile_m && next_n == tile_n) break;
      tile_m = next_m;
      tile_n = next_n;
    }
    if (tile_m <= 0 || tile_n <= 0) plan.emit_compatible = false;
    mm.output_tile_m = tile_m;
    mm.output_tile_n = tile_n;
    mm.output_tiles_m = (mm.output.height + tile_m - 1) / tile_m;
    mm.output_tiles_n = (mm.output.width + tile_n - 1) / tile_n;

    const int64_t full_m = mm.output.height / tile_m;
    const int64_t full_n = mm.output.width / tile_n;
    const int64_t tail_m = mm.output.height % tile_m;
    const int64_t tail_n = mm.output.width % tile_n;
    auto add_variant = [&](int64_t h, int64_t w, int64_t count) {
      if (h <= 0 || w <= 0 || count <= 0) return;
      CubeOutputTileVariant variant = derive_variant(h, w);
      variant.count = count;
      if (!variant.l0_init.feasible || variant.l0_init.m != h || variant.l0_init.n != w ||
          (has_chunked_carry &&
           (!variant.l0_rolled.feasible || variant.l0_rolled.m != h || variant.l0_rolled.n != w)) ||
          (mm.k_loop.tail > 0 &&
           (!variant.l0_tail.feasible || variant.l0_tail.m != h || variant.l0_tail.n != w))) {
        plan.emit_compatible = false;
      }
      mm.output_variants.push_back(std::move(variant));
    };
    add_variant(tile_m, tile_n, full_m * full_n);
    add_variant(tail_m, tile_n, full_n);
    add_variant(tile_m, tail_n, full_m);
    add_variant(tail_m, tail_n, 1);

    // Every output tile has one and only one post-K-loop drain. Internal
    // request values drain Acc->Mat and live in L1; roots drain Acc->GM (atomic
    // for split-K). No phase may perform a Mat->Acc reload.
    mm.final_drain.required = true;
    mm.final_drain.target_l1 = !mm.is_sink;
    mm.final_drain.atomic = mm.is_sink && split > 1;
    mm.final_drain.valid_rows = tile_m;
    mm.final_drain.valid_cols = tile_n;
    mm.final_drain.bytes = mm.output.height * mm.output.width * dtype_bytes(output_dtype);
    // PTO A2/A3's on-chip chain path is Acc(fp32)->Mat(bf16/fp16). It has no
    // Mat->Acc reload and no same-type fp32 Acc->Mat instruction. A boundary
    // root may still drain fp32 Acc directly to GM.
    if (prob_->use_hierarchical_cube_cost && mm.final_drain.target_l1 && output_dtype != DType::BF16 &&
        output_dtype != DType::FP16) {
      plan.emit_compatible = false;
    }
    L0MatmulConfig drain_config = prob_->l0_matmul_config;
    drain_config.bytes_c = dtype_bytes(output_dtype);
    const L0OutputTarget drain_target = mm.final_drain.target_l1 ? L0OutputTarget::L1 : L0OutputTarget::GM;
    for (const CubeOutputTileVariant& variant : mm.output_variants) {
      mm.final_drain.tile_count += variant.count;
      mm.final_drain.cycles +=
          static_cast<double>(variant.count) *
          estimate_l0_output_drain_cycles(variant.height, variant.width, drain_config, drain_target);
    }
    if (mm.output_variants.empty() || mm.final_drain.tile_count != mm.output_tiles_m * mm.output_tiles_n ||
        !std::isfinite(mm.final_drain.cycles)) {
      plan.emit_compatible = false;
    }
    plan.matmuls.push_back(mm);
    plan.execution_order.push_back(node.op);
  }

  if (cube_sink_request_node_ >= 0 && static_cast<size_t>(cube_sink_request_node_) < plan.matmuls.size()) {
    plan.config.k = plan.matmuls[static_cast<size_t>(cube_sink_request_node_)].k_loop.l1_window_k;
  }
  return plan;
}

// Vector (UB) pebble peak — see the header. Same interval-overlap sweep as the
// cube, over the UB pool: live ephemeral bands + the transient boundary tiles of
// the running op. The matmul band bug transposed to vector — softmax's e=[W,h]
// row band is ephemeral and was uncounted by the old static boundary-only sum.
int64_t Ascend910BCost::vector_peak_ub(const TileConfig &cfg,
                                 const FlatSet<size_t> &retained_from_prev,
                                 const FlatSet<size_t> &retain_these,
                                 int64_t reduce_chunk, int stream_axis) const {
  const auto &order = dfs_order_;

  // Tile footprint. The reduced axis of a reduction is READ FULL (FIXED_1 role), so
  // size it from the TENSOR's own extent — NOT the thin output-derived cfg on that
  // axis. On the live grid path cfg carries the collapsed OUTPUT extent
  // (out_H_/out_W_ = 1 for a bare reduction sink), so `min(cfg,dim)` would
  // under-count the full-axis read by the whole reduced extent (R0 — this is what
  // made a streamed bare reduction look "materialized"). A post-reduction tensor
  // ([·,1]) has extent 1 on that axis, so it stays thin either way; the NON-reduced
  // (free) axis stays cfg-tiled. reduce_chunk caps the reduced axis when streaming.
  // A pure-pointwise stream axis (no reduction) is a TILED axis, so it keeps the cfg
  // bound. This is the coupling the header contract promises, now honored on every path.
  const int ax = stream_axis ? stream_axis : reduced_axis_;

  // Emit granule (elements). The AutoFuse emit allocates DMA-block-aligned tiles: the CONTIGUOUS
  // (width) axis is always padded to `emit_gran`, and the ROW (height) axis is padded too for a
  // REDUCTION (its tile is col-major) — see auto_fuse_pass.cpp emit_strip. Feasibility must count
  // that padded footprint, else a thin free axis (e.g. an M-tile of 3 -> 8 for fp32, ~2.7x) is
  // under-counted and an over-UB group looks materializable (it then overflows AllocateMemoryAddr).
  // The whole tile chain shares the padded extent, so the emit uses the group's SMALLEST dtype
  // (largest element granule). create() caches it once per candidate subgraph; do not rescan the
  // op DAG inside every peak query/binary-search probe.
  const int64_t emit_gran = vector_emit_granule_;
  auto align_up = [](int64_t x, int64_t g) -> int64_t { return g <= 1 ? x : ((x + g - 1) / g) * g; };

  auto tile_bytes = [&](size_t t) -> int64_t {
    int64_t tw = std::min(cfg.w, prob_->tensors[t].width);
    int64_t th = std::min(cfg.h, prob_->tensors[t].height);
    if (has_reduction_ && ax == 1)       // reduced axis = width: read full, chunk-capped
      tw = std::min(prob_->tensors[t].width, reduce_chunk);
    else if (has_reduction_ && ax == 2)  // reduced axis = height: read full, chunk-capped
      th = std::min(prob_->tensors[t].height, reduce_chunk);
    else if (ax == 1) tw = std::min(tw, reduce_chunk);   // pure-pointwise stream axis (cfg-tiled)
    else if (ax == 2) th = std::min(th, reduce_chunk);
    // Pad to the emit's ALLOCATED shape: width always; height only for a reduction (col-major).
    const int64_t tw_al = align_up(tw, emit_gran);
    const int64_t th_al = has_reduction_ ? align_up(th, emit_gran) : th;
    return tw_al * th_al * dtype_bytes(prob_->tensors[t].dtype);
  };

  // Ephemeral vector bands use candidate-invariant intervals built once in
  // create(). Only their cfg/chunk-dependent byte sizes are replayed here.
  constexpr size_t kInlineVectorOps = 64;
  std::array<int64_t, kInlineVectorOps + 1> inline_band_delta;
  std::vector<int64_t> heap_band_delta;
  int64_t* band_delta = inline_band_delta.data();
  if (order.size() > kInlineVectorOps) {
    heap_band_delta.assign(order.size() + 1, 0);
    band_delta = heap_band_delta.data();
  } else {
    std::fill_n(band_delta, order.size() + 1, 0);
  }
  for (const VectorUBBandInterval& interval : vector_ub_band_intervals_) {
    const int64_t bytes = tile_bytes(interval.tensor);
    band_delta[interval.first] += bytes;
    band_delta[interval.after_last] -= bytes;
  }

  // Retained (coupling) tensors resident across the subgraph (disabled on 910B).
  int64_t base = 0;
  for (auto t : retained_from_prev) base += prob_->tensors[t].size_bytes();
  for (auto t : retain_these)
    if (!retained_from_prev.count(t)) base += prob_->tensors[t].size_bytes();

  int64_t peak = 0;
  int64_t live_bands = 0;
  for (int s = 0; s < (int)order.size(); ++s) {
    live_bands += band_delta[(size_t)s];
    const Op &op = prob_->ops[order[s]];
    if (op.type == OpType::MatMul) continue;  // cube op: not in the UB (vector) sweep
    int64_t transient = 0;
    const size_t begin = vector_ub_transient_offsets_[(size_t)s];
    const size_t end = vector_ub_transient_offsets_[(size_t)s + 1];
    for (size_t ref_idx = begin; ref_idx < end; ++ref_idx) {
      const VectorUBTransientRef& ref = vector_ub_transient_refs_[ref_idx];
      if ((ref.skip_mask & kSkipRetainedFromPrev) != 0 &&
          retained_from_prev.count(ref.tensor))
        continue;
      if ((ref.skip_mask & kSkipRetainThese) != 0 &&
          retain_these.count(ref.tensor))
        continue;
      transient += tile_bytes(ref.tensor);
    }
    peak = std::max(peak, base + live_bands + transient);
  }
  return peak;
}

// Derive the single-core UB stream — the analog of the matmul per-op seq-k.
// Materialize when the whole tile fits UB; otherwise stream the largest
// UB-fitting chunk. Besides geometry, record the loop trips/stages required by
// the emitted P1/P2/P4 algorithm. Candidate costing consumes this derived value
// immediately; final-solution consumers re-derive it from the winning config.
// Peak values stay in the local plan so compute_cost does not repeat the
// O(|ops|+|edges|) pebbling sweep.
VectorStreamPlan Ascend910BCost::vector_stream_plan(
    const TileConfig &cfg, const FlatSet<size_t> &retained_from_prev,
    const FlatSet<size_t> &retain_these) const {
  VectorStreamPlan plan;
  const int64_t budget = (int64_t)prob_->vec_capacity;
  const int64_t output_dtb = boundary_outputs_.empty()
                                 ? vector_min_dtype_bytes_
                                 : dtype_bytes(prob_->tensors[*boundary_outputs_.begin()].dtype);
  const int64_t vreg = prob_->vec_reg_bytes > 0 ? prob_->vec_reg_bytes : 256;
  auto align_up = [](int64_t x, int64_t g) { return g <= 1 ? x : ((x + g - 1) / g) * g; };
  auto planned_tile_bytes = [&](size_t tensor_id, int64_t tile_h, int64_t tile_w,
                                int64_t reduce_chunk, int stream_axis) {
    const Tensor& tensor = prob_->tensors[tensor_id];
    int64_t tw = std::min(tile_w, tensor.width);
    int64_t th = std::min(tile_h, tensor.height);
    const int axis = stream_axis ? stream_axis : reduced_axis_;
    if (has_reduction_ && axis == 1)
      tw = std::min(tensor.width, reduce_chunk);
    else if (has_reduction_ && axis == 2)
      th = std::min(tensor.height, reduce_chunk);
    else if (axis == 1)
      tw = std::min(tw, reduce_chunk);
    else if (axis == 2)
      th = std::min(th, reduce_chunk);
    const int64_t tw_alloc = align_up(tw, vector_emit_granule_);
    const int64_t th_alloc = has_reduction_ ? align_up(th, vector_emit_granule_) : th;
    return tw_alloc * th_alloc * dtype_bytes(tensor.dtype);
  };

  // The candidate grid is a partition of the logical output, not a consequence
  // of the physical UB/DMA shape. Keep its exact region count in the plan so an
  // aligned working buffer can never silently turn (for example) 12 logical
  // 11-row regions into 8 physical 16-row launches.
  const int64_t parts_m =
      cfg.parts_m > 0 ? cfg.parts_m
                      : std::max<int64_t>(1, out_H_ / std::max<int64_t>(1, cfg.h));
  const int64_t parts_n =
      cfg.parts_n > 0 ? cfg.parts_n
                      : std::max<int64_t>(1, out_W_ / std::max<int64_t>(1, cfg.w));
  // Candidate generation uses a DMA-sized granule only to cap/enumerate useful
  // region counts. Once a count is chosen, vector ownership is element-
  // balanced; the UB allocation supplies the actual DMA alignment. Otherwise
  // 128 / 6 would become six logical 32-wide regions (192 elements of replay)
  // merely because the candidate grid was enumerated in 16-element blocks.
  plan.m_partition = partition_axis(out_H_, parts_m, /*granule=*/1);
  plan.n_partition = partition_axis(out_W_, parts_n, /*granule=*/1);
  plan.work_units = plan.m_partition.parts * plan.n_partition.parts;

  // The generic emitter's iteration frame is the maximum input/output shape,
  // with a reduction axis pinned to its full extent.  Record it even for a
  // materialized tile: its internal strip loop is part of the algorithm too.
  plan.tile_h =
      reduced_axis_ == 2 ? vector_iter_H_ : std::min(plan.m_partition.big, vector_iter_H_);
  plan.tile_w =
      reduced_axis_ == 1 ? vector_iter_W_ : std::min(plan.n_partition.big, vector_iter_W_);
  TileConfig logical_cfg = cfg;
  logical_cfg.h = plan.tile_h;
  logical_cfg.w = plan.tile_w;
  plan.full_peak_ub_bytes = vector_peak_ub(logical_cfg, retained_from_prev, retain_these);
  const bool materializes = budget <= 0 || plan.full_peak_ub_bytes <= budget;

  // Materialized reductions and all pointwise groups use the same solver-owned
  // uniform strip scheduler the emitter used to rediscover.  It first exposes
  // enough row strips for a steady-state loop, then adds rows/width chunks until
  // the conservative live-band + prefetch footprint fits UB.  Multi-sink replay
  // remains serial; an over-UB multi-sink candidate must be cut.
  if (materializes || !has_reduction_) {
    if (!materializes && boundary_outputs_.size() > 1) return plan;
    plan.kind = materializes ? VectorStreamKind::Materialized : VectorStreamKind::Pointwise;
    plan.stream_passes = 1;
    plan.stream_band_count = vector_pipe_band_count_;
    const bool has_col_reduction = reduced_axis_ == 2;
    auto strip_peak = [&](int64_t sh, int64_t sw) {
      TileConfig strip_cfg = logical_cfg;
      strip_cfg.h = sh;
      strip_cfg.w = sw;
      const int64_t source_peak =
          vector_peak_ub(strip_cfg, retained_from_prev, retain_these);
      int64_t next_iteration_inputs = 0;
      for (size_t tensor : boundary_inputs_) {
        next_iteration_inputs += planned_tile_bytes(
            tensor, sh, sw, reduced_extent_ > 0 ? reduced_extent_ : std::max(sh, sw),
            /*stream_axis=*/0);
      }
      return source_peak + next_iteration_inputs;
    };
    auto strip_fits = [&](int64_t sh, int64_t sw) { return budget <= 0 || strip_peak(sh, sw) <= budget; };

    int64_t row_strips = 1;
    int64_t width_strips = 1;
    int64_t strip_w = plan.tile_w;
    bool serial_full_tile = false;
    if (!has_col_reduction && boundary_outputs_.size() == 1) {
      for (int64_t count : {8, 4, 2}) {
        if (count > plan.tile_h || plan.tile_h % count != 0) continue;
        if (has_reduction_ && (plan.tile_h / count) % vector_emit_granule_ != 0) continue;
        row_strips = count;
        break;
      }
      while (row_strips < plan.tile_h &&
             !strip_fits((plan.tile_h + row_strips - 1) / row_strips, plan.tile_w))
        row_strips = std::min<int64_t>(row_strips * 2, plan.tile_h);
      const int64_t strip_h = (plan.tile_h + row_strips - 1) / row_strips;
      if (!strip_fits(strip_h, plan.tile_w)) {
        // The +prefetch band is needed only for a pipeline.  If the exact
        // pebble peak proved the whole tile materializes, fall back to one
        // serial body instead of rejecting a valid short reduction tile.
        if (materializes) {
          row_strips = 1;
          serial_full_tile = true;
        } else {
          if (has_reduction_) return plan;
          int64_t lo = 1;
          int64_t hi =
              std::max<int64_t>(1, align_up(plan.tile_w, vector_emit_granule_) /
                                       vector_emit_granule_);
          int64_t best_granules = 0;
          while (lo <= hi) {
            const int64_t mid = lo + (hi - lo) / 2;
            const int64_t candidate_w = mid * vector_emit_granule_;
            if (strip_fits(strip_h, candidate_w)) {
              best_granules = mid;
              lo = mid + 1;
            } else {
              hi = mid - 1;
            }
          }
          if (best_granules == 0) return plan;
          strip_w = best_granules * vector_emit_granule_;
          width_strips = (plan.tile_w + strip_w - 1) / strip_w;
          if (!strip_fits(strip_h, strip_w)) return plan;
        }
      }
    }

    plan.feasible = true;
    plan.row_strips = row_strips;
    plan.width_strips = width_strips;
    plan.strip_h = (plan.tile_h + row_strips - 1) / row_strips;
    plan.strip_w = strip_w;
    const int64_t trips = row_strips * width_strips;
    plan.chunk_peak_ub_bytes =
        serial_full_tile || trips == 1 ? plan.full_peak_ub_bytes : strip_peak(plan.strip_h, plan.strip_w);
    const int64_t strip_bytes = plan.strip_h * plan.strip_w * output_dtb;
    const bool can_pipeline = trips >= 2 && strip_bytes >= vreg;
    plan.body = {0, trips, can_pipeline ? 2 : 1};
    plan.overlap_granted = plan.body.pipeline_stages == 2;

    // Materialized S2 protocol. Record it in the winning plan only after both
    // exact-partition proofs hold; emission consumes these values instead of
    // independently deciding that a costed split is buildable. The main launch
    // contains spatial work_units * factor partials; the zero seed remains one
    // disjoint launch per spatial region.
    if (materializes && cfg.split_k > 1 &&
        vector_reduction_split_kind_ != VectorReductionSplitKind::None &&
        reduced_extent_ % (cfg.split_k * vector_emit_granule_) == 0 &&
        vector_iter_W_ % std::max<int64_t>(1, plan.tile_w) == 0 &&
        !boundary_outputs_.empty() &&
        plan.tile_w * dtype_bytes(prob_->tensors[*boundary_outputs_.begin()].dtype) >=
            prob_->vec_dma_align_bytes) {
      plan.reduction_split_kind = vector_reduction_split_kind_;
      plan.reduction_split_factor = cfg.split_k;
      plan.reduction_partial_extent = reduced_extent_ / cfg.split_k;
      plan.reduction_seed = {/*present=*/true, plan.work_units,
                             /*valid_rows=*/out_H_,
                             /*valid_cols=*/plan.tile_w};
    }

    if (!materializes) {
      plan.axis = width_strips > 1 ? 1 : 2;
      plan.extent = plan.axis == 1 ? plan.tile_w : plan.tile_h;
      plan.chunk = plan.axis == 1 ? plan.strip_w : plan.strip_h;
      plan.full_chunks = plan.extent / plan.chunk;
      plan.tail = plan.extent - plan.full_chunks * plan.chunk;
    }
    return plan;
  }

  // UB-overflow reduction: the emitted online algorithm owns extra accumulator
  // and assemble bands absent from the source DAG, so size its reduced-axis
  // chunk from that concrete scratch contract.
  if (p4_pattern_kind_ == P4PatternKind::SoftmaxFlash)
    plan.kind = VectorStreamKind::SoftmaxFlash;
  else if (p4_pattern_kind_ == P4PatternKind::LayerNormWelford)
    plan.kind = VectorStreamKind::LayerNormWelford;
  else if (reduction_count_ > 1)
    plan.kind = VectorStreamKind::ModelAheadMultiReduction;
  else
    plan.kind =
        reduction_spans_output_ ? VectorStreamKind::ReductionSpanning : VectorStreamKind::ReductionFolded;
  plan.p4_work = make_vector_p4_work_plan(p4_pattern_kind_);
  plan.axis = reduced_axis_;
  plan.extent = reduced_extent_;
  const int64_t extra_bands =
      (plan.kind == VectorStreamKind::SoftmaxFlash || plan.kind == VectorStreamKind::LayerNormWelford ||
       plan.kind == VectorStreamKind::ModelAheadMultiReduction)
          ? 6
          : (plan.kind == VectorStreamKind::ReductionSpanning ? 5 : 2);
  plan.free_tile = reduced_axis_ == 1 ? plan.tile_h : plan.tile_w;
  plan.free_tile_alloc = align_up(plan.free_tile, vector_emit_granule_);
  plan.stream_passes = reduction_spans_output_ ? 2 : 1;

  // A stage-2 loop has two copies of every per-chunk source-DAG transient;
  // persistent accumulator/assemble/online-stat bands stay single-buffered.
  // Derive once with serial bands, then re-size if the resulting trip counts
  // enable either rolled phase.  Shrinking a pipelined chunk cannot disable its
  // trip-count guard, so two iterations reach the fixed point.
  plan.stream_band_count = (int64_t)ops_.size() + extra_bands;
  int64_t best = 0, best_peak = 0;
  int stats_stages = 1, apply_stages = 1;
  for (int iteration = 0; iteration < 2; ++iteration) {
    const int64_t bytes_per_element =
        std::max<int64_t>(1, plan.stream_band_count * plan.free_tile_alloc *
                                vector_max_dtype_bytes_);
    const int64_t cap = budget / bytes_per_element;
    best = std::max<int64_t>(vector_emit_granule_,
                             (cap / vector_emit_granule_) * vector_emit_granule_);
    best = std::min(best, plan.extent);
    if (best <= 0) return plan;
    const int64_t full_chunks = plan.extent / best;
    const int64_t chunk_bytes = plan.free_tile_alloc * best * vector_min_dtype_bytes_;
    auto stages_for = [&](int64_t trips) { return trips >= 2 && chunk_bytes >= vreg ? 2 : 1; };
    stats_stages = stages_for(std::max<int64_t>(0, full_chunks - 1));
    apply_stages = reduction_spans_output_ ? stages_for(full_chunks) : 1;
    const bool has_pipeline = stats_stages == 2 || apply_stages == 2;

    auto exact_chunk_peak = [&](int64_t chunk) {
      const int64_t aligned_chunk = align_up(chunk, vector_emit_granule_);
      const int64_t source_peak = vector_peak_ub(
          logical_cfg, retained_from_prev, retain_these, chunk, plan.axis);
      // These are emitter-generated accumulator/assemble/online-stat bands,
      // separate from the source-DAG lifetime replay above. Their values use
      // the widest participating dtype; source tensors retain their individual
      // dtypes in vector_peak_ub.
      const int64_t generated_scratch =
          extra_bands * plan.free_tile_alloc * aligned_chunk *
          vector_max_dtype_bytes_;
      int64_t next_iteration_inputs = 0;
      if (has_pipeline) {
        const int64_t tile_h = plan.axis == 1 ? plan.free_tile_alloc : aligned_chunk;
        const int64_t tile_w = plan.axis == 1 ? aligned_chunk : plan.free_tile_alloc;
        for (size_t tensor : boundary_inputs_)
          next_iteration_inputs +=
              planned_tile_bytes(tensor, tile_h, tile_w, chunk, plan.axis);
      }
      return source_peak + generated_scratch + next_iteration_inputs;
    };

    best_peak = exact_chunk_peak(best);
    if (best_peak > budget) {
      int64_t lo = 1;
      int64_t hi = std::max<int64_t>(1, best / vector_emit_granule_);
      int64_t best_granules = 0;
      while (lo <= hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        const int64_t candidate = mid * vector_emit_granule_;
        if (exact_chunk_peak(candidate) <= budget) {
          best_granules = mid;
          lo = mid + 1;
        } else {
          hi = mid - 1;
        }
      }
      if (best_granules == 0) return plan;
      best = best_granules * vector_emit_granule_;
      best_peak = exact_chunk_peak(best);
    }
    const int64_t required_bands =
        has_pipeline ? 2 * (int64_t)ops_.size() + extra_bands : plan.stream_band_count;
    if (required_bands == plan.stream_band_count) break;
    plan.stream_band_count = required_bands;
  }
  plan.feasible = true;
  plan.chunk = best;
  plan.chunk_peak_ub_bytes = best_peak;
  plan.full_chunks = plan.extent / best;
  plan.tail = plan.extent - plan.full_chunks * best;
  const int64_t stats_trips = std::max<int64_t>(0, plan.full_chunks - 1);
  plan.stats_init = {true, 0, plan.chunk};
  plan.stats = {1, stats_trips, stats_stages};
  plan.stats_tail = {plan.tail > 0, plan.full_chunks, plan.tail};
  if (reduction_spans_output_) {
    plan.apply = {0, plan.full_chunks, apply_stages};
    plan.apply_tail = {plan.tail > 0, plan.full_chunks, plan.tail};
  }
  const bool has_serial_finalize =
      plan.kind == VectorStreamKind::ReductionFolded || plan.kind == VectorStreamKind::LayerNormWelford;
  plan.finalize = {has_serial_finalize, 0, has_serial_finalize ? 1 : 0};
  const bool stats_pipeline = plan.stats.pipeline_stages >= 2;
  const bool apply_pipeline =
      plan.stream_passes == 1 || plan.apply.pipeline_stages >= 2;
  plan.overlap_granted = stats_pipeline && apply_pipeline;
  return plan;
}

Ascend910BCost::VecStream Ascend910BCost::vector_stream(
    const TileConfig &cfg, const FlatSet<size_t> &retained_from_prev,
    const FlatSet<size_t> &retain_these) const {
  const VectorStreamPlan plan =
      vector_stream_plan(cfg, retained_from_prev, retain_these);
  if (!plan.feasible) return {0, 0};
  if (!plan.streamed()) return {0, INT64_MAX};
  return {plan.axis, plan.chunk};
}

// 910B per-core two-pool feasibility (byte-based). Each core runs one tile, so
// the per-tile slice footprint IS the per-core footprint. Fork on cube/vector.
bool Ascend910BCost::fits_on_chip(const TileConfig &cfg,
                            const FlatSet<size_t> &retained_from_prev,
                            const FlatSet<size_t> &retain_these) const {
  // Mixed cube+vector group (only Ascend910BMixed builds one): needs BOTH pools.
  // Shared here so is_feasible() and the mixed compute_cost agree. The base model
  // never admits a mixed group, so it never reaches this branch.
  if (has_matmul_ && has_vector_)
    return mixed_fits_on_chip(cfg, retained_from_prev, retain_these);

  const bool cube = has_matmul_;
  // Cube: dynamic peak L1 over the execution order (live intermediate bands +
  // per-op-k-sized operand strips), NOT a static sum of all operand strips. The
  // sum both over-counted co-resident inputs and under-counted intermediate
  // bands that physically sit in L1/Mat between matmuls; derive_exec is the
  // red-blue pebble peak that the cost model's ephemeral-on-chip assumption
  // implies. Infeasible iff no per-op k assignment keeps the peak under L1.
  if (cube) {
    const int64_t split = cfg.parts_m > 0 && cfg.split_k > 0 ? cfg.split_k : 1;
    const int64_t derive_sink_k = cube_request_nodes_.size() == 1 ? output_K_ : output_K_ / split;
    return output_K_ % split == 0 &&
           derive_exec(cfg, derive_sink_k, retained_from_prev, retain_these, nullptr) != INT64_MAX;
  }

  // Vector: feasible iff the subgraph materializes OR streams to fit UB.
  return vector_stream(cfg, retained_from_prev, retain_these).chunk > 0;
}

bool Ascend910BCost::mixed_fits_on_chip(const TileConfig &cfg,
                                  const FlatSet<size_t> &retained_from_prev,
                                  const FlatSet<size_t> &retain_these) const {
  // Two-pool feasibility for a mixed cube+vector kernel — REUSE the homogeneous
  // single-core streams, now that both are affinity-aware (each skips the other
  // unit's ops, and treats a cube↔vector crossing as ring-streamed, not a resident
  // band):
  //   * CUBE stage — derive_exec sweeps the L1 operand bands and derives each
  //     matmul's per-op SEQ-K (slicing the contraction to fit L1). A held
  //     cube→cube intermediate is an L1 band; a crossing matmul output drains
  //     L0c→ring (not an L1 band); vector ops are skipped. L0c output sizing is
  //     deferred to AutoTileMatmulL0, as in the homogeneous cube.
  //   * VECTOR stage — vector_stream streams the [w,h] tile through UB in chunks
  //     (down to a min-chunk; free for a pointwise, recompute-costed only for a
  //     reduction). A held vector→vector intermediate is a UB band; a crossing tile
  //     popped from the ring is a transient; cube ops are skipped.
  // The crossing's DDR roundtrip is paid in compute_cost, not in feasibility.
  if (derive_exec(cfg, output_K_, retained_from_prev, retain_these, nullptr) == INT64_MAX) {
    return false;
  }

  // Analytic mixed studies may still use the homogeneous streaming planner.
  // Production v0 is stricter: its AIV body materializes one explicit row-half
  // and ExpandMixedKernel reserves eight full crossing slots in Vec memory.
  // Admitting a merely streamable full tile here would price an algorithm the
  // emitter does not build and can overflow MemoryReuse after expansion.
  if (!prob_->require_buildable_mixed) {
    return vector_stream(cfg, retained_from_prev, retain_these).chunk > 0;
  }
  if (!mixed_topology_ || !mixed_topology_->compiler_emit_compatible ||
      mixed_topology_->transfers.size() != 1) {
    return false;
  }
  const int64_t parts_m = cfg.parts_m > 0
                              ? cfg.parts_m
                              : std::max<int64_t>(1, out_H_ / std::max<int64_t>(1, cfg.h));
  const int64_t parts_n = cfg.parts_n > 0
                              ? cfg.parts_n
                              : std::max<int64_t>(1, out_W_ / std::max<int64_t>(1, cfg.w));
  const AxisPartition mp = partition_axis(out_H_, parts_m, grid_gran_h_);
  const AxisPartition np = partition_axis(out_W_, parts_n, grid_gran_w_);
  if (mp.num_big != 0 || np.num_big != 0 || mp.big < 2 || mp.big % 2 != 0) {
    return false;
  }

  TileConfig lane_cfg = cfg;
  lane_cfg.h = mp.big / 2;
  lane_cfg.w = np.big;
  lane_cfg.parts_m = 0;
  lane_cfg.parts_n = 0;
  lane_cfg.split_k = 1;
  const VectorStreamPlan lane_plan =
      vector_stream_plan(lane_cfg, retained_from_prev, retain_these);
  if (!lane_plan.feasible || lane_plan.kind != VectorStreamKind::Materialized) {
    return false;
  }

  const MixedTransferTopology& transfer = mixed_topology_->transfers.front();
  const int64_t slot_bytes =
      mp.big * np.big * dtype_bytes(prob_->tensors[transfer.tensor].dtype);
  constexpr int64_t kMixedFifoSlots = 8;
  const int64_t fifo_reserved = slot_bytes * kMixedFifoSlots;
  return fifo_reserved <= prob_->vec_capacity &&
         lane_plan.full_peak_ub_bytes <= prob_->vec_capacity - fifo_reserved;
}

bool Ascend910BCost::is_feasible(const TileConfig &cfg,
                           const FlatSet<size_t> &retained_from_prev,
                           const FlatSet<size_t> &retain_these) const {
  return is_valid_tiling(cfg) &&
         fits_on_chip(cfg, retained_from_prev, retain_these);
}

// ============================================================================
// Matmul operand reload (BYTES) — shared by the cube cost and the mixed cost
// ============================================================================

double Ascend910BCost::cube_operand_reload(const TileConfig &cfg,
                                           bool matmul_at_output_grid,
                                           double *lhs_bytes_out,
                                           double *rhs_bytes_out) const {
  // Build the in-subgraph produced/consumed sets (produced ⇒ on-chip ephemeral
  // operand, never reloaded from DDR; consumed ⇒ the op is an intermediate).
  FlatSet<size_t> produced, consumed;
  for (auto i : ops_) {
    produced.insert(prob_->ops[i].output());
    for (auto t : prob_->ops[i].inputs) consumed.insert(t);
  }
  // Distribution-aware reload: the left operand reloads with the N-tiling (1/w),
  // the right operand with the M-tiling (1/h); deduped per (tensor, role,
  // boundary?) so a shared operand in the same role is charged once. A consumed
  // (chained-intermediate) matmul tiles its output full-width (w_i = N_i) UNLESS
  // matmul_at_output_grid forces the output grid — the mixed feed-forward case,
  // where the matmul output is consumed elementwise by the vector stage and so is
  // tiled at cfg.w like a boundary output.
  double reload = 0.0, lhs_bytes = 0.0, rhs_bytes = 0.0;  // lhs->L0A, rhs->L0B (MTE1 split)
  std::set<std::tuple<size_t, int, bool>> counted;  // (tensor, 0=LHS/1=RHS, is_boundary_op)
  for (auto i : ops_) {
    if (prob_->ops[i].type != OpType::MatMul) continue;
    const size_t lhs = prob_->ops[i].inputs[0];
    const size_t rhs = prob_->ops[i].inputs[1];
    const size_t o   = prob_->ops[i].output();
    const double N_i = (double)prob_->tensors[o].width;
    const double M_i = (double)prob_->tensors[o].height;
    const double K_i = (double)prob_->tensors[lhs].width;       // contraction
    const bool is_boundary_op = matmul_at_output_grid || !consumed.count(o);
    const double w_i = is_boundary_op ? std::min((double)cfg.w, N_i) : N_i;
    const double h_i = std::min((double)cfg.h, M_i);            // shared M-band
    if (!produced.count(lhs) && counted.emplace(lhs, 0, is_boundary_op).second) {
      const double b = M_i * N_i * K_i / w_i * dtype_bytes(prob_->tensors[lhs].dtype);
      reload += b; lhs_bytes += b;
    }
    if (!produced.count(rhs) && counted.emplace(rhs, 1, is_boundary_op).second) {
      const double b = M_i * N_i * K_i / h_i * dtype_bytes(prob_->tensors[rhs].dtype);
      reload += b; rhs_bytes += b;
    }
  }
  if (lhs_bytes_out) *lhs_bytes_out = lhs_bytes;
  if (rhs_bytes_out) *rhs_bytes_out = rhs_bytes;
  return reload;
}

double Ascend910BCost::cube_request_reload(const TileConfig& cfg, int64_t split, double* lhs_bytes_out,
                                           double* rhs_bytes_out) const {
  split = std::max<int64_t>(1, split);
  if (cube_request_nodes_.empty() || output_K_ % split != 0) {
    if (lhs_bytes_out) *lhs_bytes_out = 0.0;
    if (rhs_bytes_out) *rhs_bytes_out = 0.0;
    return 0.0;
  }

  const int64_t parts_m =
      cfg.parts_m > 0 ? cfg.parts_m : std::max<int64_t>(1, out_H_ / std::max<int64_t>(1, cfg.h));
  const int64_t parts_n =
      cfg.parts_n > 0 ? cfg.parts_n : std::max<int64_t>(1, out_W_ / std::max<int64_t>(1, cfg.w));
  const AxisPartition pm = partition_axis(out_H_, parts_m, grid_gran_h_);
  const AxisPartition pn = partition_axis(out_W_, parts_n, grid_gran_w_);
  const int64_t m_sizes[2] = {pm.big, pm.small};
  const int64_t n_sizes[2] = {pn.big, pn.small};
  const int64_t m_counts[2] = {pm.num_big, pm.parts - pm.num_big};
  const int64_t n_counts[2] = {pn.num_big, pn.parts - pn.num_big};

  double lhs_total = 0.0;
  double rhs_total = 0.0;
  for (int mi = 0; mi < 2; ++mi) {
    for (int ni = 0; ni < 2; ++ni) {
      const int64_t region_count = m_counts[mi] * n_counts[ni];
      if (region_count <= 0) continue;
      std::set<std::tuple<size_t, int, int, int>> counted;
      double lhs_unit = 0.0;
      double rhs_unit = 0.0;
      auto add_request = [&](const CubeRequest& request, int port, double& bytes) {
        const auto key = std::make_tuple(request.tensor, port, static_cast<int>(request.height_binding),
                                         static_cast<int>(request.width_binding));
        if (!counted.insert(key).second) return;
        const Tensor& tensor = prob_->tensors[request.tensor];
        const int64_t h =
            cube_binding_extent(request.height_binding, tensor.height, m_sizes[mi], n_sizes[ni], split);
        const int64_t w =
            cube_binding_extent(request.width_binding, tensor.width, m_sizes[mi], n_sizes[ni], split);
        if (h > 0 && w > 0) bytes += static_cast<double>(h * w * dtype_bytes(tensor.dtype));
      };
      for (const CubeRequestNode& node : cube_request_nodes_) {
        if (node.lhs_producer < 0) add_request(node.lhs, 0, lhs_unit);
        if (node.rhs_producer < 0) add_request(node.rhs, 1, rhs_unit);
      }
      const double copies = static_cast<double>(region_count * split);
      lhs_total += lhs_unit * copies;
      rhs_total += rhs_unit * copies;
    }
  }
  if (lhs_bytes_out) *lhs_bytes_out = lhs_total;
  if (rhs_bytes_out) *rhs_bytes_out = rhs_total;
  return lhs_total + rhs_total;
}

// ============================================================================
// Cost computation
// ============================================================================

CostResult Ascend910BCost::compute_cost(const TileConfig &cfg,
                                  const FlatSet<size_t> &retained_from_prev,
                                  const FlatSet<size_t> &retain_these) const {
  return compute_cost_impl(cfg, retained_from_prev, retain_these, nullptr);
}

CostResult Ascend910BCost::compute_cost_impl(const TileConfig &cfg,
                                  const FlatSet<size_t> &retained_from_prev,
                                  const FlatSet<size_t> &retain_these,
                                  L0PlanMemo *l0_memo) const {
  if (has_matmul_ && has_vector_) {
    return compute_mixed_cost(cfg, retained_from_prev, retain_these);
  }
  CostResult result;
  result.config = cfg;
  VectorStreamPlan vector_stream;
  std::vector<int64_t> cube_window_k;
  int64_t cube_seed_fill_rounds = 0;

  if (!is_valid_tiling(cfg)) return result;
  if (has_matmul_) {
    if (has_vector_) {
      if (!fits_on_chip(cfg, retained_from_prev, retain_these)) return result;
    } else {
      const int64_t feasibility_split = cfg.parts_m > 0 && cfg.split_k > 0 ? cfg.split_k : 1;
      const int64_t derive_sink_k =
          cube_request_nodes_.size() == 1 ? output_K_ : output_K_ / feasibility_split;
      if (output_K_ % feasibility_split != 0 ||
          derive_exec(cfg, derive_sink_k, retained_from_prev, retain_these, &cube_window_k) == INT64_MAX) {
        return result;
      }
    }
  } else {
    vector_stream = vector_stream_plan(cfg, retained_from_prev, retain_these);
    if (!vector_stream.feasible) return result;
    // A fixed S>1 vector candidate is feasible only when UB planning produced
    // the exact cross-core algorithm. This removes streamed-col_sum duplicates
    // and prevents any future coarse-grid admission from degrading to split=1.
    if (cfg.split_k > 1 &&
        vector_stream.reduction_split_kind ==
            VectorReductionSplitKind::None)
      return result;
  }
  result.feasible = true;

  const ByteCost bc = MakeByteCost(prob_);  // per-direction cycles/byte (grounded)
  // Grid (SpatialSchedule) mode: the spatial region count is parts_m x parts_n
  // exactly. Cube w/h carry its fractal-balanced physical region; vector
  // ownership/work units come from VectorStreamPlan's element-balanced grid.
  // Uniform mode is the ad-hoc exact-divisor API path.
  const int num_tw = (cfg.parts_n > 0) ? (int)cfg.parts_n : std::max((int)(out_W_ / cfg.w), 1);
  const int num_th = (cfg.parts_m > 0) ? (int)cfg.parts_m : std::max((int)(out_H_ / cfg.h), 1);
  const int num_tiles =
      has_matmul_ ? num_tw * num_th : static_cast<int>(vector_stream.work_units);
  result.num_spatial_tiles = num_tiles;
  result.num_k_passes = has_matmul_ ? std::max((int)(output_K_ / cfg.k), 1) : 1;

  // 910B parallel-core roofline — the only cost model (the competition
  // single-context model was removed). Compute parallelizes across the unit's
  // cores (spatial tiles + split-K = independent work units); DDR traffic divides
  // across each core's own GM pipe (MTE2/FixPipe) up to the aggregate HBM ceiling.
  const int n_cores = has_matmul_ ? prob_->num_cube_cores : prob_->num_vector_cores;
  {
    // Per-direction realized parallel GM-pipe count. Each core has its own MTE2
    // (GM->L1/UB) and FixPipe (L0C/UB->GM), so a direction's DDR traffic divides
    // across `active` cores' pipes up to the aggregate HBM ceiling:
    //   par(active, peak) = min(active, hbm_aggregate_gibps / per_core_peak)
    // and that direction's cycles = bytes * cyc_per_byte / par. Exactly pto-isa
    // BwEff (effective bw = min(active*peak, hbm)). hbm<=0 => uncapped (pure
    // per-core divide; the cores never saturate a finite HBM).
    const double hbm = prob_->hbm_aggregate_gibps;
    auto par = [&](double active, double peak_gibps) {
      const double cap = (hbm > 0.0 && peak_gibps > 0.0)
                             ? hbm / peak_gibps : std::numeric_limits<double>::infinity();
      return std::max(1.0, std::min(active, cap));
    };
    if (has_matmul_) {
      // --- Request-DAG cube cost -------------------------------------------
      // Every work unit recursively produces exactly the tensor regions its
      // sinks request. This handles left chains, produced RHS operands, trees,
      // and memoized fan-out with the same symbolic roles derive_exec uses.
      double out_store = 0.0;
      for (const auto& info : boundary_tensor_info_)
        if (info.is_boundary_out)
          out_store += (double)info.full_size * dtype_bytes(prob_->tensors[info.id].dtype);
      const AxisPartition g_pm = partition_axis(out_H_, std::max<int64_t>(1, cfg.parts_m), grid_gran_h_);
      const AxisPartition g_pn = partition_axis(out_W_, std::max<int64_t>(1, cfg.parts_n), grid_gran_w_);
      const int64_t l0m = std::max<int64_t>(1, prob_->l0_tile_m);
      const int64_t l0n = std::max<int64_t>(1, prob_->l0_tile_n);
      const bool lone_matmul = cube_request_nodes_.size() == 1;
      const int64_t configured_split = cfg.parts_m > 0 && cfg.split_k > 0 ? cfg.split_k : 1;

      auto request_pipes = [&](int64_t m_ext, int64_t n_ext, int64_t split, bool phase_d) {
        double work = 0.0;
        double mac_pipe = 0.0;
        double extract_pipe = 0.0;
        for (const CubeRequestNode& node : cube_request_nodes_) {
          const Tensor& output = prob_->tensors[node.output.tensor];
          const int64_t m =
              cube_binding_extent(node.output.height_binding, output.height, m_ext, n_ext, split);
          const int64_t n = cube_binding_extent(node.output.width_binding, output.width, m_ext, n_ext, split);
          const int64_t k = node.parallel_sink ? op_K(node.op) / split : op_K(node.op);
          // Cube MAC/extract precision follows the operand dtype. The
          // accumulator/output is commonly FP32 for BF16/FP16 inputs and must
          // not make the Matrix/MTE1 work look like an FP32-input GEMM.
          const DType dtype = prob_->tensors[prob_->ops[node.op].inputs[0]].dtype;
          const double mac = CubeMacCycles(prob_, m, n, k, dtype);
          const double extract = CubeExtractCycles(prob_, bc, m, n, k, dtype);
          mac_pipe += mac;
          extract_pipe += extract;
          if (!phase_d) continue;
          const int64_t L =
              std::max<int64_t>(1, ((m + l0m - 1) / l0m) * ((n + l0n - 1) / l0n) * ((k + 63) / 64));
          work +=
              (mac + extract + static_cast<double>(L - 1) * std::max(mac, extract)) / static_cast<double>(L);
        }
        return phase_d ? work : std::max(mac_pipe, extract_pipe);
      };

      // Double-buffer floor: the max(compute, ddr) overlap is only real when the
      // operand reload can ping-pong, i.e. the per-core contraction is halvable
      // into >=2 seq-K sub-strips (>= 32 = two K-fractals; the emit's implicit
      // halving needs that). A tiny contraction can't overlap -> reload and
      // compute SERIALIZE (compute + ddr).
      auto db_roofline = [&](double comp, double dram, bool overlap) {
        return overlap ? std::max(comp, dram) : comp + dram;
      };

      // A global max(compute, DDR) is legal only if every request instance that
      // reads a boundary operand has a concrete stage-2 rolled K loop. The old
      // K/S>=32 scalar gate could grant overlap to a one-chunk sink in a chain.
      // Reuse the feasibility derivation for the configured split. Ad-hoc
      // non-grid callers can still sweep S and pay one O(nodes) derivation for
      // each alternative; no CubeSchedulePlan enters the enumeration hot path.
      auto overlap_implementable = [&](int64_t split) {
        std::vector<int64_t> derived_windows;
        const std::vector<int64_t>* windows = &cube_window_k;
        if (split != configured_split) {
          const int64_t derive_sink_k = lone_matmul ? output_K_ : output_K_ / split;
          if (derive_exec(cfg, derive_sink_k, retained_from_prev, retain_these, &derived_windows) ==
              INT64_MAX) {
            return false;
          }
          windows = &derived_windows;
        }
        for (size_t node_idx = 0; node_idx < cube_request_nodes_.size(); ++node_idx) {
          const CubeRequestNode& node = cube_request_nodes_[node_idx];
          if (node.lhs_producer >= 0 && node.rhs_producer >= 0) continue;
          const int64_t extent = node.parallel_sink ? op_K(node.op) / split : op_K(node.op);
          int64_t window =
              node_idx < windows->size() && (*windows)[node_idx] > 0 ? (*windows)[node_idx] : extent;
          if (lone_matmul && node.parallel_sink) {
            window = CappedSinkWindow(op_K(node.op), window, split);
          }
          window = std::min(window, extent);
          if (CubePipelinedChunk(extent, window) == 0) return false;
        }
        return true;
      };
      // Sink split-K: split the sink contraction into S per-tile partials to
      // recruit idle cores. The output write-back grows to S partials (L0C->GM via
      // FixPipe, SetAtomicAdd — each core writes independently, no merge barrier),
      // but that store pipe OVERLAPS the operand feed, so split-K only pays for the
      // writes once S*store exceeds the feed (max, not sum).
      //
      // S is a FIRST-CLASS design axis (like w,h): more cores cut compute (and,
      // until HBM saturates, the per-core-divided operand feed) while the S output
      // writes grow the store pipe. ENUMERATE S and take the min. Useful range:
      // S <= kfrac (>=1 fractal/partial); the wave model + the growing store pipe
      // reject the excessive splits (the old ceil(n_cores/num_tiles) core-fill bound
      // is gone — a split that overfills a wave can still cut compute). output_K_ is
      // the sink matmul's contraction (NOT max_K_); only S | kfrac is emittable.
      const int64_t kfrac = std::max<int64_t>(1, output_K_ / 16);
      // Evaluate one split factor S (>=1). S=1 is the spatial-only roofline; S>=2
      // splits the sink contraction into S equal 16-aligned partials (each owns
      // output_K_/S), writing the output S times (S atomic-add partials).
      struct SplitEval {
        double lat, compute, ddr, active, l1l0;
        int64_t seed_fill_rounds = 0;
      };
      auto eval_S = [&](int64_t S) -> SplitEval {
        S = std::max<int64_t>(1, S);
        const int64_t unitsS = (int64_t)num_tiles * S;
        const double activeS = (double)std::min<int64_t>(unitsS, n_cores);

        // Buildable cube candidates use the same hierarchical descriptor as
        // emission. Output/L0C tiles are outer to GM->L1 K windows; each child
        // L0 plan keeps Acc resident, rolled boundary loads overlap only inside
        // an actual stage-2 loop, and init/tail/final drain remain serial.
        const bool uniform_grid = g_pm.num_big == 0 && g_pn.num_big == 0;
        // A ragged spatial grid is emitted by the lone-matmul ceil-and-clamp
        // fallback only for split=1.  Combining that overlapping clamp with
        // split-K atomic partials would give more than one owner to edge output
        // elements, so it is not a buildable hierarchical schedule.  Do not
        // let the older analytic roofline rank that fictional configuration.
        if (prob_->use_hierarchical_cube_cost && !uniform_grid && S > 1) {
          return {std::numeric_limits<double>::infinity(), 0.0, 0.0, activeS, 0.0, 0};
        }
        if (prob_->use_hierarchical_cube_cost && uniform_grid) {
          const CubeSchedulePlan schedule =
              derive_cube_schedule_plan(cfg, retained_from_prev, retain_these, S, l0_memo);
          if (!schedule.feasible || !schedule.emit_compatible) {
            return {std::numeric_limits<double>::infinity(), 0.0, 0.0, activeS, 0.0, 0};
          }
          const double gm_read_scale = activeS / par(activeS, prob_->bw_gm_l1);
          const double gm_write_scale = activeS / par(activeS, prob_->bw_l0c_gm);
          double unit_wall = 0.0;
          double unit_compute = 0.0;
          double unit_ddr = 0.0;
          double unit_l1l0 = 0.0;
          for (const CubeMatmulSchedule& mm : schedule.matmuls) {
            const int64_t init_k = mm.k_loop.full_chunks >= 2 ? mm.k_loop.chunk : mm.effective_contraction;
            auto feed_cycles = [&](const CubeOutputTileVariant& variant, int64_t k_extent) {
              double bytes = 0.0;
              if (!mm.lhs_ephemeral) {
                bytes += static_cast<double>(variant.height * k_extent *
                                             dtype_bytes(prob_->tensors[mm.lhs.tensor].dtype));
              }
              if (!mm.rhs_ephemeral) {
                bytes += static_cast<double>(k_extent * variant.width *
                                             dtype_bytes(prob_->tensors[mm.rhs.tensor].dtype));
              }
              return bytes * bc.reload * gm_read_scale;
            };
            L0MatmulConfig drain_config = prob_->l0_matmul_config;
            drain_config.bytes_c = dtype_bytes(prob_->tensors[mm.output.tensor].dtype);
            const L0OutputTarget drain_target =
                mm.final_drain.target_l1 ? L0OutputTarget::L1 : L0OutputTarget::GM;
            for (const CubeOutputTileVariant& variant : mm.output_variants) {
              const double count = static_cast<double>(variant.count);
              const double init_inner = variant.l0_init.phases.wall_cycles;
              const double init_feed = feed_cycles(variant, init_k);
              double tile_wall = init_inner + init_feed;
              double tile_compute = init_inner;
              double tile_ddr = init_feed;
              double tile_l1l0 = variant.l0_init.phases.load_cycles;

              const int64_t rolled = std::max<int64_t>(0, mm.k_loop.full_chunks - 1);
              if (rolled > 0) {
                const double inner = variant.l0_rolled.phases.wall_cycles;
                const double feed = feed_cycles(variant, mm.k_loop.chunk);
                tile_wall += mm.k_loop.pipeline_stages >= 2 && rolled >= 2
                                 ? feed + inner + static_cast<double>(rolled - 1) * std::max(feed, inner)
                                 : static_cast<double>(rolled) * (feed + inner);
                tile_compute += static_cast<double>(rolled) * inner;
                tile_ddr += static_cast<double>(rolled) * feed;
                tile_l1l0 += static_cast<double>(rolled) * variant.l0_rolled.phases.load_cycles;
              }
              if (mm.k_loop.tail > 0) {
                const double inner = variant.l0_tail.phases.wall_cycles;
                const double feed = feed_cycles(variant, mm.k_loop.tail);
                tile_wall += inner + feed;
                tile_compute += inner;
                tile_ddr += feed;
                tile_l1l0 += variant.l0_tail.phases.load_cycles;
              }

              double drain =
                  estimate_l0_output_drain_cycles(variant.height, variant.width, drain_config, drain_target);
              if (!mm.final_drain.target_l1) drain *= gm_write_scale;
              tile_wall += drain;
              if (mm.final_drain.target_l1) {
                tile_compute += drain;
              } else {
                tile_ddr += drain;
              }
              unit_wall += count * tile_wall;
              unit_compute += count * tile_compute;
              unit_ddr += count * tile_ddr;
              unit_l1l0 += count * tile_l1l0;
            }
          }
          const int64_t waves = (unitsS + n_cores - 1) / n_cores;
          double latency = static_cast<double>(waves) * unit_wall;
          double compute = static_cast<double>(waves) * unit_compute;
          double ddr = static_cast<double>(waves) * unit_ddr;
          double l1l0 = static_cast<double>(waves) * unit_l1l0;
          int64_t seed_rounds = 0;
          if (schedule.seed.present) {
            const double seed_active =
                static_cast<double>(std::min<int64_t>(schedule.seed.work_units, prob_->num_vector_cores));
            const double seed_compute_work =
                static_cast<double>(schedule.seed.work_units) *
                GroundedVectorFillCycles(schedule.seed.valid_rows, schedule.seed.valid_cols);
            const double seed_compute =
                WaveComputeCycles(seed_compute_work, schedule.seed.work_units, prob_->num_vector_cores);
            const double seed_store =
                static_cast<double>(schedule.seed.bytes) * bc.ub_out / par(seed_active, prob_->bw_ub_gm);
            latency += seed_compute + seed_store +
                       static_cast<double>(schedule.seed.work_units * prob_->per_task_overhead_cycles);
            compute += seed_compute;
            ddr += seed_store;
            seed_rounds = (schedule.seed.work_units + prob_->num_vector_cores - 1) / prob_->num_vector_cores;
          }
          return {latency, compute, ddr, activeS, l1l0, seed_rounds};
        }

        double computeS = 0.0;
        if (cfg.parts_m > 0) {
          computeS = lone_matmul ? LptMakespan(
                                       n_cores, g_pm, g_pn,
                                       [&](int64_t m, int64_t n) {
                                         return request_pipes(m, n, /*split=*/1,
                                                              /*phase_d=*/true);
                                       },
                                       S)
                                 : LptMakespanPerUnit(
                                       n_cores, g_pm, g_pn,
                                       [&](int64_t m, int64_t n, int64_t split) {
                                         return request_pipes(m, n, split,
                                                              /*phase_d=*/true);
                                       },
                                       S);
        } else {
          const int64_t work_split = lone_matmul ? 1 : S;
          const int64_t copies = lone_matmul ? num_tiles : unitsS;
          computeS =
              WaveComputeCycles(request_pipes(std::min(cfg.h, out_H_), std::min(cfg.w, out_W_), work_split,
                                              /*phase_d=*/false) *
                                    static_cast<double>(copies),
                                unitsS, n_cores);
        }
        double reload_lhs = 0.0;
        double reload_rhs = 0.0;
        const double reload =
            lone_matmul ? cube_operand_reload(cfg, /*matmul_at_output_grid=*/false, &reload_lhs, &reload_rhs)
                        : cube_request_reload(cfg, S, &reload_lhs, &reload_rhs);
        // DDR is two SEPARATE, concurrent pipes: the operand feed (MTE2, GM->L1)
        // and the output write-back (FixPipe, L0C->GM; S atomic-add partials for a
        // split). They are distinct hardware, and pto-isa scores GM reads and GM
        // writes as independent aggregate groups, so they OVERLAP -- the cube DDR
        // term is max(feed, writes), NOT their sum. Each is per-core-divided +
        // HBM-capped at its own peak (S=1 => a single output store).
        const double feed   = reload * bc.reload / par(activeS, prob_->bw_gm_l1);
        const double writes = (double)S * out_store * bc.store / par(activeS, prob_->bw_l0c_gm);
        const double ddrS = std::max(feed, writes);
        // Double-buffer floor: K/S < 32 (< 2 K-fractals) can't ping-pong -> serialize.
        const double latS = db_roofline(computeS, ddrS, overlap_implementable(S));
        return {latS, computeS, ddrS, activeS, reload_lhs * bc.l0a + reload_rhs * bc.l0b, 0};
      };

      // S source: a SpatialSchedule TRIPLE fixes S (cfg.split_k from the (P,Q,S)
      // enumeration) and is evaluated as-is; an ad-hoc non-grid tile (split_k==0,
      // e.g. a directly-constructed TileConfig) sweeps S over the valid K-fractal
      // divisors and adopts a split only if it STRICTLY beats the spatial-only S=1.
      // Split-K is MODEL-AHEAD of the AutoFuse emit: gate on the buildable flag so a
      // buildable-mode harness never selects an unemittable split (default true = analytic).
      int64_t chosen_S = 1;
      SplitEval chosen = eval_S(1);
      if (prob_->allow_model_ahead_split_k) {
        if (cfg.split_k >= 1) {
          chosen_S = cfg.split_k;
          chosen = eval_S(chosen_S);
        } else {
          for (int64_t S : all_divisors(kfrac)) {
            if (S < 2) continue;
            const SplitEval e = eval_S(S);
            if (e.lat < chosen.lat) { chosen = e; chosen_S = S; }
          }
        }
      }
      result.uses_model_ahead_split_k = (chosen_S > 1);

      result.latency = chosen.lat;
      result.parallel_split = (int)chosen_S;
      result.cores_used = (int)chosen.active;
      result.compute_bound = chosen.compute >= chosen.ddr;
      result.ddr_traffic = chosen.ddr;
      result.l1l0_extract = chosen.l1l0;
      cube_seed_fill_rounds = chosen.seed_fill_rounds;
      // Displayed per-core k: the greedy single-core L1-fit k (derive_exec, a
      // divisor of output_K_), capped for a split by the per-core fractal share
      // ceil(kfrac/S)*16 -- the largest divisor of output_K_ not exceeding both.
      std::vector<int64_t> chosen_windows = cube_window_k;
      if (chosen_S != configured_split) {
        const int64_t derive_sink_k = lone_matmul ? output_K_ : output_K_ / chosen_S;
        derive_exec(cfg, derive_sink_k, retained_from_prev, retain_these, &chosen_windows);
      }
      const int64_t l1_k = (cube_sink_request_node_ >= 0 &&
                            static_cast<size_t>(cube_sink_request_node_) < chosen_windows.size() &&
                            chosen_windows[static_cast<size_t>(cube_sink_request_node_)] > 0)
                               ? chosen_windows[static_cast<size_t>(cube_sink_request_node_)]
                               : output_K_;
      if (chosen_S > 1) {
        const int64_t share_k = ((kfrac + chosen_S - 1) / chosen_S) * 16;
        int64_t per_core_k = 16;
        for (int64_t d = std::min({l1_k, share_k, output_K_}); d >= 16; d -= 16) {
          if (output_K_ % d == 0) { per_core_k = d; break; }
        }
        result.config.k = per_core_k;
      } else {
        result.config.k = l1_k > 0 ? l1_k : cfg.k;
      }
    } else {
      const double eff = (double)std::min<int64_t>(num_tiles, n_cores);
      const double vbudget = (double)prob_->vec_capacity;
      const bool reduction_streams = has_reduction_ && vector_stream.streamed();
      if (reduction_streams && reduction_count_ > 1 && !prob_->allow_model_ahead_multi_reduction_stream &&
          p4_pattern_kind_ == P4PatternKind::None) {
        result.feasible = false;
        return result;
      }

      const int64_t vreg = prob_->vec_reg_bytes > 0 ? prob_->vec_reg_bytes : 256;
      const DType vector_dtype = prob_->tensors[*boundary_outputs_.begin()].dtype;
      const int64_t dtb = dtype_bytes(vector_dtype);
      const int64_t dma_width =
          (!reduction_streams && vector_stream.strip_w > 0) ? vector_stream.strip_w : cfg.w;
      const double dma_pen = std::max(1.0, (double)vreg / std::max(1.0, (double)dma_width * (double)dtb));

      // Compute is classified once into body/stats/apply/finalize cones.  An op
      // present in both stats and apply (softmax's sub/exp) is intentionally
      // charged twice: the emitter recomputes it after the statistics barrier.
      auto phase_compute = [&](uint8_t phase, int64_t covered_extent, int64_t chunks,
                               int64_t override_rows = 0,
                               int64_t override_cols = 0,
                               int64_t override_work_units = 0) {
        double cycles = 0.0;
        bool prev_pw = false;
        bool prev_pw_grounded = false;
        int64_t frame_rows = vector_stream.strip_h;
        int64_t frame_cols = vector_stream.strip_w;
        int64_t frame_iterations = vector_stream.body.trip_count;
        if (reduction_streams) {
          const int64_t chunk_extent =
              covered_extent > 0 ? covered_extent / std::max<int64_t>(1, chunks) : 1;
          frame_rows = reduced_axis_ == 1 ? vector_stream.free_tile : chunk_extent;
          frame_cols = reduced_axis_ == 1 ? chunk_extent : vector_stream.free_tile;
          frame_iterations = std::max<int64_t>(1, chunks);
        }
        if (override_rows > 0) frame_rows = override_rows;
        if (override_cols > 0) frame_cols = override_cols;
        const int64_t frame_work_units =
            override_work_units > 0 ? override_work_units
                                    : vector_stream.work_units;
        for (size_t i : vector_phase_ops_[VectorPhaseIndex(phase)]) {
          const Op& op = prob_->ops[i];
          const bool pw = op.type != OpType::Reduction;
          const bool grounded = pw && HasGroundedVectorSemantics(op);
          const bool stream_start = pw && (!prev_pw || prev_pw_grounded != grounded);
          if (op.type == OpType::Reduction && has_grounded_vector_semantics_) {
            cycles += (double)frame_work_units *
                      (double)frame_iterations *
                      GroundedReductionCompute(prob_, op, reduced_axis_,
                                               frame_rows, frame_cols);
            prev_pw = false;
            prev_pw_grounded = false;
            continue;
          }
          if (grounded) {
            // Exact source semantics are paid once per emitted strip/chunk in
            // every logical task.  This is deliberately not cached in
            // CostResult: the primitive descriptor is candidate-invariant and
            // the three scalar frame values come from this stack-local plan.
            cycles += (double)frame_work_units * (double)frame_iterations *
                      GroundedVectorOpCompute(prob_, op, frame_rows, frame_cols,
                                              stream_start, has_reduction_);
            prev_pw = true;
            prev_pw_grounded = true;
            continue;
          }
          double scale = 1.0;
          if (reduction_streams) {
            int64_t op_extent = prob_->tensors[op.output()].width;
            if (reduced_axis_ == 2) op_extent = prob_->tensors[op.output()].height;
            for (size_t input : op.inputs) {
              const Tensor& tensor = prob_->tensors[input];
              op_extent = std::max(op_extent, reduced_axis_ == 1 ? tensor.width : tensor.height);
            }
            scale = op_extent > 1
                        ? (double)covered_extent / (double)std::max<int64_t>(1, vector_stream.extent)
                        : (double)chunks;
          }
          cycles += scale * VecOpCompute(prob_, op, reduced_axis_, stream_start,
                                         has_reduction_);
          prev_pw = pw;
          prev_pw_grounded = false;
        }
        return cycles;
      };

      // Exact per-input phase traffic.  Inputs participate only in the cones
      // that consume them; an apply-only scale/bias is no longer doubled merely
      // because x is read by both stats and apply.  A size-1 reduced axis is a
      // broadcast and is reloaded once per emitted chunk, while a spanning input
      // contributes only the covered reduced-axis extent.
      auto streamed_tensor_bytes = [&](size_t tensor_id, int64_t covered_extent, int64_t chunks) {
        const Tensor& tensor = prob_->tensors[tensor_id];
        const int64_t free_regions = reduced_axis_ == 1 ? vector_stream.m_partition.parts
                                                        : vector_stream.n_partition.parts;
        const int64_t tensor_free = reduced_axis_ == 1 ? tensor.height : tensor.width;
        const int64_t tensor_reduced = reduced_axis_ == 1 ? tensor.width : tensor.height;
        // Every SPMD block executes the same maximum static body. Ragged regions
        // clamp their base and overlap the preceding region; they do not shrink
        // the load/store. Price that emitted traffic rather than the logical
        // union (for example 48*3 rows, not 128 rows).
        const int64_t free_total =
            tensor_free == 1
                ? free_regions
                : free_regions * std::min<int64_t>(tensor_free, vector_stream.free_tile);
        const int64_t reduced_total = tensor_reduced == 1 ? chunks : covered_extent;
        return (double)free_total * (double)reduced_total *
               (double)dtype_bytes(tensor.dtype);
      };
      auto body_tensor_bytes = [&](size_t tensor_id) {
        const Tensor& tensor = prob_->tensors[tensor_id];
        const int64_t strip_m = tensor.height == 1 ? 1 : vector_stream.strip_h;
        const int64_t strip_n = tensor.width == 1 ? 1 : vector_stream.strip_w;
        return (double)vector_stream.work_units * (double)vector_stream.row_strips *
               (double)vector_stream.width_strips * (double)strip_m *
               (double)strip_n * (double)dtype_bytes(tensor.dtype);
      };
      auto input_cycles = [&](uint8_t phase, int64_t covered_extent, int64_t chunks) {
        double cycles = 0.0;
        for (size_t tensor : vector_phase_inputs_[VectorPhaseIndex(phase)]) {
          if (retained_from_prev.count(tensor)) continue;
          const double bytes = reduction_streams ? streamed_tensor_bytes(tensor, covered_extent, chunks)
                                                 : body_tensor_bytes(tensor);
          cycles += bytes * bc.ub_in * dma_pen;
        }
        return cycles;
      };
      auto output_cycles = [&](int64_t covered_extent, int64_t chunks) {
        double cycles = 0.0;
        for (size_t tensor : boundary_outputs_) {
          if (retain_these.count(tensor)) continue;
          const double bytes = reduction_streams ? streamed_tensor_bytes(tensor, covered_extent, chunks)
                                                 : body_tensor_bytes(tensor);
          cycles += bytes * bc.ub_out * dma_pen;
        }
        return cycles;
      };
      auto ddr_phase = [&](double active, double in_cycles, double out_cycles) {
        return in_cycles / par(active, prob_->bw_gm_ub) + out_cycles / par(active, prob_->bw_ub_gm);
      };
      auto phase_roofline = [](int stages, double compute, double ddr) {
        return stages == 2 ? std::max(compute, ddr) : compute + ddr;
      };

      double lat = 0.0;
      double compute_total_mk = 0.0;
      double ddr_total = 0.0;
      auto record_phase = [&](uint8_t traffic_phase, int64_t covered_extent, int64_t chunks,
                              int stages, double out_cycles, double total_compute_work) {
        if (chunks <= 0 && covered_extent <= 0 && total_compute_work <= 0.0 && out_cycles <= 0.0) return;
        const double compute = WaveComputeCycles(total_compute_work, num_tiles, n_cores);
        const double ddr =
            ddr_phase(eff, input_cycles(traffic_phase, covered_extent, chunks), out_cycles);
        lat += phase_roofline(stages, compute, ddr);
        compute_total_mk += compute;
        ddr_total += ddr;
      };
      auto add_phase = [&](uint8_t phase, int64_t covered_extent, int64_t chunks, int stages,
                           double out_cycles, double extra_compute = 0.0) {
        record_phase(phase, covered_extent, chunks, stages, out_cycles,
                     phase_compute(phase, covered_extent, chunks) + extra_compute);
      };
      auto add_generated_phase = [&](uint8_t traffic_phase, const VectorPhaseWorkPlan& work,
                                     int64_t chunk_extent, int64_t iterations, int stages,
                                     double out_cycles) {
        record_phase(traffic_phase, chunk_extent * iterations, iterations, stages, out_cycles,
                     GeneratedP4PhaseCompute(prob_, vector_stream, work, chunk_extent,
                                             iterations, vector_dtype));
      };

      const double io_in = input_cycles(kVectorPhaseBody, 0, 1);
      const double io_out = output_cycles(0, 1);
      auto ddr_io = [&](double active, double io_out_eff) { return ddr_phase(active, io_in, io_out_eff); };
      auto rfl = [&](double comp, double dram) {
        return phase_roofline(vector_stream.body.pipeline_stages, comp, dram);
      };

      if (!reduction_streams) {
        add_phase(kVectorPhaseBody, 0, 1, vector_stream.body.pipeline_stages, io_out);
      } else {
        if (vector_stream.p4_work.generated) {
          add_generated_phase(kVectorPhaseStats, vector_stream.p4_work.stats_init,
                              vector_stream.stats_init.extent, 1, 1, 0.0);
          add_generated_phase(kVectorPhaseStats, vector_stream.p4_work.stats_update,
                              vector_stream.chunk, vector_stream.stats.trip_count,
                              vector_stream.stats.pipeline_stages, 0.0);
          if (vector_stream.stats_tail.present)
            add_generated_phase(kVectorPhaseStats, vector_stream.p4_work.stats_update,
                                vector_stream.stats_tail.extent, 1, 1, 0.0);
          if (vector_stream.finalize.present)
            add_generated_phase(kVectorPhaseFinalize, vector_stream.p4_work.finalize,
                                /*chunk_extent=*/1, /*iterations=*/1, 1, 0.0);
        } else {
          if (has_grounded_vector_semantics_) {
            add_phase(kVectorPhaseStats, vector_stream.stats_init.extent, 1, 1, 0.0);
            add_phase(kVectorPhaseStats,
                      vector_stream.stats.trip_count * vector_stream.chunk,
                      vector_stream.stats.trip_count,
                      vector_stream.stats.pipeline_stages, 0.0,
                      GeneratedReductionMergeCompute(
                          prob_, vector_stream, vector_stream.stats.trip_count, dtb));
            if (vector_stream.stats_tail.present)
              add_phase(kVectorPhaseStats, vector_stream.stats_tail.extent, 1, 1,
                        0.0, GeneratedReductionMergeCompute(
                                 prob_, vector_stream, 1, dtb));
          } else {
            const double startup =
                (double)reduction_count_ *
                (prob_->vec_op_head + prob_->vec_op_tail);
            add_phase(kVectorPhaseStats, vector_stream.stats_init.extent, 1, 1,
                      0.0, startup);
            add_phase(kVectorPhaseStats,
                      vector_stream.stats.trip_count * vector_stream.chunk,
                      vector_stream.stats.trip_count,
                      vector_stream.stats.pipeline_stages, 0.0,
                      startup * vector_stream.stats.trip_count);
            if (vector_stream.stats_tail.present)
              add_phase(kVectorPhaseStats, vector_stream.stats_tail.extent, 1,
                        1, 0.0, startup);
          }
        }
        if (vector_stream.stream_passes == 2) {
          add_phase(kVectorPhaseApply, vector_stream.apply.trip_count * vector_stream.chunk,
                    vector_stream.apply.trip_count, vector_stream.apply.pipeline_stages,
                    output_cycles(vector_stream.apply.trip_count * vector_stream.chunk,
                                  vector_stream.apply.trip_count));
          if (vector_stream.apply_tail.present)
            add_phase(kVectorPhaseApply, vector_stream.apply_tail.extent, 1, 1,
                      output_cycles(vector_stream.apply_tail.extent, 1));
        } else {
          add_phase(kVectorPhaseFinalize, 0, 1, 1, output_cycles(0, 1));
        }
      }
      result.parallel_split = 1;
      result.cores_used = (int)eff;
      result.compute_bound = compute_total_mk >= ddr_total;
      result.ddr_traffic = ddr_total;
      // Reduced-axis (cross-core) split — the vector analog of the cube split-K.
      // SINK-ONLY: the S per-core partials reduce across cores through DDR (fine
      // for a boundary reduction output; an internal reduction split is a subgraph
      // cut, not an in-subgraph split). Mirrors the cube (§4.2): ENUMERATE S and
      // take the min. The partials accumulate via SetAtomicAdd (910B always has it).
      // The reduced partial is thin ([H,1] for a width-reduction, [1,W] for a
      // height-reduction) -> red_dim.
      //
      // C2 (device-grounded): the split is emittable ONLY when the reduction
      // MATERIALIZES its reduced band in UB. A STREAMED reduction (reduced band >>
      // UB) lowers to a single-core chunk-accumulation loop parallelized over the
      // FREE axis (parts_n) alone — the emit's stream path returns BEFORE the S2
      // atomic-add split (auto_fuse_pass.cpp:1553), so the cross-core split never
      // fires. Costing S there is fictional and INVERTS the argmin (device probe:
      // split-heavy/occ=1 costed cheapest but ran slowest; device-best fills cores
      // via parts_n). `vector_peak_ub` now couples the reduced axis to its full
      // extent internally (R0), so the raw-cfg call detects streaming correctly here
      // AND at the feasibility/compute sites — no manual coupling needed.
      //
      // G6: materialization alone is not sufficient. The winning
      // VectorStreamPlan must carry the exact terminal-col_sum atomic-add
      // protocol. Row reductions, max/min, multiple reductions/sinks, and
      // ragged free grids never enter this block, so they cannot receive
      // parallelism the emitter will silently discard.
      const bool reduction_materializes =
          vbudget <= 0.0 /* no UB model -> legacy */ || !vector_stream.streamed();
      if (reduction_materializes &&
          vector_stream.reduction_split_kind ==
              VectorReductionSplitKind::ColSumAtomicAdd) {
        const double red_dim = (double)(reduced_axis_ == 1 ? out_H_ : out_W_);
        struct RS { double lat, eff, ddr, compute; };
        auto eval_reduce_S = [&](int64_t S) -> RS {
          const double effS = (double)std::min<int64_t>(num_tiles * S, n_cores);
          // Replay exactly the body each partial task emits: one
          // [reduced_extent/S, free_tile] source/pointwise/reduction cone. This
          // preserves per-invocation startup and the col-reduction tree instead
          // of fractionally dividing one full-height invocation by S.
          const double split_compute = phase_compute(
              kVectorPhaseBody, 0, 1,
              vector_stream.reduction_partial_extent,
              vector_stream.tile_w, num_tiles * S);
          const double compS =
              WaveComputeCycles(split_compute, num_tiles * S, n_cores);
          // The thin reduced output is written S times (S atomic-add partials):
          // extra UB->GM store traffic FOLDED into the roofline, NOT an additive
          // merge. red_dim is the thin partial ([H,1] / [1,W]); charge it the same
          // DMA-shape penalty as the base store.
          const double io_out_S = io_out + (double)(S - 1) * red_dim * (double)dtb * bc.ub_out * dma_pen;
          const double ddrS = ddr_io(effS, io_out_S);
          const double streamS = rfl(compS, ddrS);
          return {streamS, effS, ddrS, compS};
        };
        auto apply_S = [&](int64_t S) {
          const RS e = eval_reduce_S(S);
          const VectorReductionSeedPlan& seed = vector_stream.reduction_seed;
          const double seed_active =
              (double)std::min<int64_t>(seed.work_units, n_cores);
          const double seed_compute_work =
              (double)seed.work_units *
              GroundedVectorFillCycles(seed.valid_rows, seed.valid_cols);
          const double seed_compute =
              WaveComputeCycles(seed_compute_work, seed.work_units, n_cores);
          const double seed_store_cycles =
              (double)seed.work_units * (double)seed.valid_rows *
              (double)seed.valid_cols * (double)dtb * bc.ub_out * dma_pen;
          const double seed_ddr =
              seed_store_cycles / par(seed_active, prob_->bw_ub_gm);
          // The seed launch completes before the atomic body is runnable.
          // TEXPANDS also precedes its dependent store, so this phase is a
          // serial compute+DDR sum rather than another roofline maximum.
          lat = e.lat + seed_compute + seed_ddr;
          result.parallel_split = (int)S;
          result.cores_used = (int)e.eff;
          result.compute_bound = e.compute + seed_compute >= e.ddr + seed_ddr;
          result.ddr_traffic = e.ddr + seed_ddr;
        };
        // SpatialSchedule owns one concrete split factor. Ad-hoc cfg.split_k=0
        // has no reconstructable emit descriptor and therefore stays serial.
        apply_S(vector_stream.reduction_split_factor);
      }
      // C3 — per-task host launch overhead. The kernel_fill term below is per-WAVE
      // (rounds = ceil(num_tiles/cores)), so it is FLAT for num_tiles <= cores and the model ties
      // plans the device ranks by task count (argmin lands on the most-tasks / device-slowest plan).
      // Charge each launched work unit (num_tiles spatial x the reduced-axis split) a small grounded
      // overhead so best_cost separates them toward fewer tasks. Self-gates: negligible vs a big
      // kernel's compute, comparable-to-compute for small ones. Vector-only (device-grounded here).
      const int64_t seed_tasks = vector_stream.reduction_seed.present
                                     ? vector_stream.reduction_seed.work_units
                                     : 0;
      result.latency =
          lat + ((double)num_tiles * (double)std::max(1, result.parallel_split) +
                 (double)seed_tasks) *
                    (double)prob_->per_task_overhead_cycles;
    }
    // Per-kernel pipeline fill — the DUAL of the eff core-fill incentive. A
    // tiling produces launched tasks; each core runs ceil(tasks/n_cores)
    // of them in sequence, paying one fill per pass. eff penalizes too FEW tiles
    // (under-filled cores); this penalizes too MANY (over-tiling), so the optimum
    // sits at ~one kernel per core. Vector reduced-axis split launches S body
    // tasks per spatial region, so it contributes real additional fill waves.
    // kernel_fill_cost==0 => no fill term.
    if (prob_->kernel_fill_cost > 0) {
      const int64_t body_tasks =
          num_tiles * std::max<int64_t>(1, result.parallel_split);
      const int64_t rounds = (body_tasks + n_cores - 1) / n_cores;
      const int64_t seed_rounds = vector_stream.reduction_seed.present
                                      ? (vector_stream.reduction_seed.work_units + n_cores - 1) /
                                            n_cores
                                      : 0;
      result.latency +=
          (double)(rounds + seed_rounds + cube_seed_fill_rounds) * (double)prob_->kernel_fill_cost;
    }
  }

  return result;
}

// ============================================================================
// Ascend910BMixed — the third subgraph type: a fused cube+vector mixed kernel.
// Cube-only and vector-only groups delegate to the (unchanged) base cost; only
// the mixed type is added here.
// ============================================================================

MixedSchedulePlan Ascend910BCost::derive_mixed_schedule_plan(
    const TileConfig &cfg,
    const FlatSet<size_t> &retained_from_prev,
    const FlatSet<size_t> &retain_these,
    int64_t parallel_split) const {
  MixedSchedulePlan plan;
  plan.config = cfg;
  if (!(has_matmul_ && has_vector_) || !mixed_topology_) return plan;
  plan.mode = mixed_topology_->mode;
  plan.emit_compatible = prob_->require_buildable_mixed
                             ? mixed_topology_->compiler_emit_compatible
                             : mixed_topology_->emit_compatible;

  const int64_t split = std::max<int64_t>(1, parallel_split);
  if (!is_valid_tiling(cfg) ||
      output_K_ % split != 0 || !fits_on_chip(cfg, retained_from_prev, retain_these)) {
    return plan;
  }

  const int64_t parts_m = cfg.parts_m > 0
                              ? cfg.parts_m
                              : std::max<int64_t>(1, out_H_ / std::max<int64_t>(1, cfg.h));
  const int64_t parts_n = cfg.parts_n > 0
                              ? cfg.parts_n
                              : std::max<int64_t>(1, out_W_ / std::max<int64_t>(1, cfg.w));
  plan.feasible = true;
  plan.m_partition = partition_axis(out_H_, parts_m, grid_gran_h_);
  plan.n_partition = partition_axis(out_W_, parts_n, grid_gran_w_);
  plan.spatial_tiles = parts_m * parts_n;
  plan.split_k = split;
  plan.work_units = plan.spatial_tiles * split;
  plan.group_capacity = std::max<int64_t>(
      1, std::min<int64_t>(prob_->num_cube_cores, prob_->num_vector_cores / 2));

  // The buildable C->V epilogue is split across the two buddy AIVs by rows.
  // Cube grids are 16-row aligned, so every uniform candidate has an exact
  // half-height vector shard.  Analytic shapes that do not satisfy that
  // invariant stay single-lane rather than receiving a fictional /2.
  bool row_split_legal = true;
  for (const MixedStageTopology& stage : mixed_topology_->stages) {
    if (stage.engine != MixedEngine::Vector) continue;
    for (size_t op_idx : stage.ops) {
      const Op& op = prob_->ops[op_idx];
      if (op.type == OpType::Reduction && !op.inputs.empty() &&
          prob_->tensors[op.inputs.front()].height >
              prob_->tensors[op.output()].height) {
        row_split_legal = false;
      }
    }
  }
  if (plan.emit_compatible && row_split_legal && plan.m_partition.num_big == 0 &&
      plan.m_partition.big >= 2 && plan.m_partition.big % 2 == 0) {
    plan.vector_split = MixedVectorSplit::Rows;
    plan.vector_lanes = 2;
  }

  // This reproduces the legacy assignment exactly: use every available group
  // before issuing a second item to any group. It intentionally exposes the
  // current fidelity gap -- two global tiles on two groups are not two
  // successor items in one group's inner pipeline.
  plan.loop.axis = MixedPipelineAxis::SpatialRegion;
  plan.loop.extent = plan.spatial_tiles;
  plan.loop.chunk = 1;
  plan.loop.items_per_spatial_tile = 1;
  plan.loop.work_items = plan.spatial_tiles;
  plan.loop.active_groups = std::min(plan.spatial_tiles, plan.group_capacity);
  plan.loop.min_trips_per_group =
      plan.loop.work_items / std::max<int64_t>(1, plan.loop.active_groups);
  plan.loop.max_trips_per_group =
      (plan.loop.work_items + std::max<int64_t>(1, plan.loop.active_groups) - 1) /
      std::max<int64_t>(1, plan.loop.active_groups);
  // Each split engine executes a sequential item loop.  The one-way FIFO, not
  // a generic software-pipeline tag around nested matmul work, decouples AIC
  // item k+1 from AIV item k.  This avoids multiplying AutoTileL0 buffer depth.
  plan.loop.pipeline_stages = 1;
  plan.loop.requested_skew_depth = 0;

  plan.overlap_implementable =
      plan.emit_compatible && plan.loop.min_trips_per_group >= 2 &&
      plan.loop.min_trips_per_group == plan.loop.max_trips_per_group;
  plan.model_overlap_granted = plan.overlap_implementable;
  plan.pipeline_fill_absorbed =
      mixed_topology_->sink_runs_early_stage && plan.model_overlap_granted;

  // Automatic 910B one-direction pipes use eight GM slots.  Increment 1 has
  // one same-shaped crossing, so the fixed slot is the full cube result tile;
  // valid bytes and transferred slot bytes are therefore identical.
  if (mixed_topology_->compiler_emit_compatible) {
    for (const MixedTransferTopology& transfer : mixed_topology_->transfers) {
      const Tensor& tensor = prob_->tensors[transfer.tensor];
      const int64_t rows = plan.m_partition.big;
      const int64_t cols = plan.n_partition.big;
      const int64_t slot_count = mixed_topology_->transfers.size() == 1 ? 8 : 4;
      const int64_t slot_bytes = rows * cols * dtype_bytes(tensor.dtype);
      plan.fifos.push_back(
          {transfer.tensor,
           transfer.producer_engine == MixedEngine::Cube
               ? MixedTransferDirection::CubeToVector
               : MixedTransferDirection::VectorToCube,
           rows, cols, slot_bytes, slot_count, slot_bytes * slot_count});
    }
  }
  return plan;
}

MixedSchedulePlan Ascend910BCost::mixed_schedule_plan(
    const TileConfig &cfg,
    const FlatSet<size_t> &retained_from_prev,
    const FlatSet<size_t> &retain_these,
    int64_t parallel_split) const {
  MixedSchedulePlan plan = derive_mixed_schedule_plan(
      cfg, retained_from_prev, retain_these, parallel_split);
  if (plan.feasible) {
    std::vector<int64_t> pernode_k;
    derive_exec(cfg, output_K_, retained_from_prev, retain_these, &pernode_k);
    if (sink_mm_op_ >= 0 && static_cast<size_t>(sink_mm_op_) < pernode_k.size() &&
        pernode_k[static_cast<size_t>(sink_mm_op_)] > 0) {
      plan.cube_window_k = pernode_k[static_cast<size_t>(sink_mm_op_)];
      plan.config.k = plan.cube_window_k;
    }
    if (mixed_topology_ && mixed_topology_->compiler_emit_compatible &&
        plan.vector_split == MixedVectorSplit::Rows) {
      TileConfig lane_cfg = cfg;
      lane_cfg.h = plan.m_partition.big / std::max<int64_t>(1, plan.vector_lanes);
      lane_cfg.w = plan.n_partition.big;
      lane_cfg.parts_m = 0;
      lane_cfg.parts_n = 0;
      lane_cfg.split_k = 1;
      const VectorStreamPlan vector_stage =
          vector_stream_plan(lane_cfg, retained_from_prev, retain_these);
      plan.vector_stage_kind = vector_stage.kind;
      plan.vector_stage_peak_ub_bytes = vector_stage.full_peak_ub_bytes;
    }
  }
  plan.topology = mixed_topology_;
  return plan;
}

CostResult Ascend910BCost::compute_mixed_cost(
    const TileConfig &cfg,
    const FlatSet<size_t> &retained_from_prev,
    const FlatSet<size_t> &retain_these) const {
  // ===== MIXED cube+vector kernel — 910B DDR-streamed, latency hidden =====
  // Cube ops run on the cube pool and vector ops on the vector pool CONCURRENTLY.
  // A cube↔vector intermediate cannot stay on chip (910B has no direct Acc→Vec
  // pipe), so it ROUND-TRIPS DDR — written by the producing unit, read by the
  // consuming unit (2x). That traffic is unavoidable, but the two stages OVERLAP
  // (the cube streams output tiles into DDR while the vector cores consume
  // already-written tiles), so the roundtrip LATENCY is hidden behind the slower
  // stage. The kernel tiles for UNITS (1 cube + 2 vector):
  //   lat = fill + max(cube_compute/eff_units, vector_compute/(2·eff_units), ddr·inv_B)
  // Compare the SEPARATED two-kernel cost (fusion saves the overlap + one fill;
  // the DDR total is the same on 910B). A future 950 makes the handoff direct
  // (the crossing intermediate avoids DDR) — same formula, cheaper ddr term.
  CostResult result;
  result.config = cfg;
  const MixedSchedulePlan schedule =
      derive_mixed_schedule_plan(cfg, retained_from_prev, retain_these, 1);
  // Mixed kernels need BOTH on-chip pools (L1/L0c for the cube stage, UB for the
  // vector stage) — fits_on_chip dispatches to mixed_fits_on_chip here. A large
  // shared tile that overflows UB is infeasible to fuse even when the separate
  // kernels each fit their one pool.
  if (!schedule.feasible) return result;
  result.feasible = true;
  const ByteCost bc = MakeByteCost(prob_);  // per-direction cycles/byte (grounded)
  // Grid mode (parts_m/parts_n > 0) fixes the tile count directly: cfg.w/cfg.h are
  // the big-region EXTENTS, not exact divisors, so out_W_/cfg.w would mis-floor the
  // count. Mirrors the base cube path (Ascend910BCost::compute_cost); uniform/ad-hoc
  // tiles (parts_* == 0, a directly-built TileConfig) fall back to the divide.
  const int num_tiles = static_cast<int>(schedule.spatial_tiles);
  result.num_spatial_tiles = num_tiles;
  result.num_k_passes = std::max((int)(output_K_ / cfg.k), 1);

  // cube_mac / cube_extract: the cube stage is hierarchical and double-buffered
  // (same as the homogeneous cube path) — its time is max(MACs, L1->L0 extract).
  double cube_mac = 0.0, cube_extract = 0.0, vector_compute = 0.0;
  bool prev_pw = false;  // Fix 3: pointwise startup once per stream (matmul/reduction break it)
  for (auto i : ops_) {
    const auto& op = prob_->ops[i];
    if (op.type == OpType::MatMul) {
      prev_pw = false;  // a cube->vector crossing syncs -> next vector op restarts its stream
      const size_t o = op.output();
      const int64_t Mo = prob_->tensors[o].height;
      const int64_t No = prob_->tensors[o].width;
      const int64_t Ko = prob_->tensors[op.inputs[0]].width;  // contraction
      const DType dt = prob_->tensors[op.inputs[0]].dtype;
      cube_mac += CubeMacCycles(prob_, Mo, No, Ko, dt);
      cube_extract += CubeExtractCycles(prob_, bc, Mo, No, Ko, dt);
    } else {  // Pointwise / Reduction — grounded per-op compute (reductions: axis-aware
      // tree, Fix 1; pointwise startup once per stream, Fix 3). Shared with the vector-only
      // path via VecOpCompute for one consistent cost.
      const bool pw = op.type != OpType::Reduction;
      vector_compute += VecOpCompute(prob_, op, reduced_axis_,
                                     /*pw_stream_start=*/pw && !prev_pw,
                                     has_reduction_);
      prev_pw = pw;
    }
  }
  // Fix 2 (UB-overflow streaming surcharge — mirror the homogeneous vector path). A fused
  // flash-attention kernel (matmul -> softmax) keeps the reduced (keys) axis resident and
  // PINNED, but it usually overflows UB, so the schedule streams that axis in chunks ONLINE
  // (pto_macro_fa_softmax): each element is touched once, so compute is NOT multiplied by
  // #reductions+1 — the only surcharge is a thin per-chunk correction (re-paid vector startup +
  // an O(ROWS) running max/sum rescale). Same formula as the vector-only branch.
  if (has_reduction_) {
    const double budget = (double)prob_->vec_capacity;
    const double peak = (double)vector_peak_ub(cfg, retained_from_prev, retain_these);
    if (budget > 0.0 && peak > budget) {
      const double nchunks = std::ceil(peak / budget);
      vector_compute += nchunks * (double)reduction_count_ * (prob_->vec_op_head + prob_->vec_op_tail);
    }
  }

  // Tiling for UNITS. With the fixed 1:2 cube:vector ratio, the atomic resource
  // is a UNIT = 1 cube + 2 vector cores (the 950 mix-cluster; on 910B the same
  // ratio + the GM ring). Regions tile over n_units = num_cube_cores; WITHIN a
  // unit the cube and its 2 vector cores are pipeline STAGES (not a finer output
  // grid), so the stage times divide by 1 and 2 cores per unit.
  const double n_units = (double)prob_->num_cube_cores;  // 1:2 physical capacity
  const double eff_units = (double)std::max<int64_t>(1, schedule.loop.active_groups);

  // DDR traffic. CRITICAL: fusion on 910B does NOT reduce DDR — the matmul still
  // reloads its operands and the intermediate still round-trips DDR (only 950's
  // direct pipe removes the roundtrip). The GM ring is FOUR independent, per-unit
  // HBM ports that OVERLAP, so we accumulate bytes PER PORT and take the MAX over
  // ports (grounded by mixed_contention: each read port caps at par() = hbm/peak,
  // matching the sim to 0%; mixed_ddr_bound: the single-core GM is subsumed into
  // the per-unit critical path). The four ports:
  //   gm_l1  : GM->L1 cube reads    — matmul operand reload (a) + vec->cube crossing reads
  //   gm_ub  : GM->UB vector reads  — vector boundary inputs (c) + cube->vec crossing reads
  //   l0c_gm : L0C->GM cube writes  — cube->vec crossing writes + boundary out iff sink is MatMul
  //   ub_gm  : UB->GM vector writes — vec->cube crossing writes + boundary out iff sink is vector
  // A cube operand is already in (a)/gm_l1, so it is excluded from the vector
  // boundary-input reads to avoid double counting; a same-unit ephemeral is free
  // (not crossing). A crossing ephemeral splits into one WRITE (by the producer's
  // unit) + one READ (by the consuming unit) — the 2x roundtrip, per-port.
  double gm_l1_bytes  = cube_operand_reload(cfg, /*matmul_at_output_grid=*/true);  // (a)
  double gm_ub_bytes  = 0.0, l0c_gm_bytes = 0.0, ub_gm_bytes = 0.0;
  FlatSet<size_t> cube_operands;
  for (auto i : ops_)
    if (prob_->ops[i].type == OpType::MatMul)
      for (auto t : prob_->ops[i].inputs) cube_operands.insert(t);
  for (const auto& info : boundary_tensor_info_) {                              // (c)
    const Tensor& tensor = prob_->tensors[info.id];
    const double bytes = (double)info.full_size * dtype_bytes(tensor.dtype);
    if (!info.is_internally_produced && !cube_operands.count(info.id)) {
      double request_multiplicity = 1.0;
      if (prob_->require_buildable_mixed) {
        const bool broadcasts_rows = tensor.height == 1 && out_H_ > 1;
        const bool broadcasts_cols = tensor.width == 1 && out_W_ > 1;
        if (broadcasts_rows && broadcasts_cols) {
          request_multiplicity =
              (double)schedule.spatial_tiles * (double)schedule.vector_lanes;
        } else if (broadcasts_rows) {
          // Both UP_DOWN lanes independently pop/load the same singleton row.
          request_multiplicity =
              (double)schedule.m_partition.parts * (double)schedule.vector_lanes;
        } else if (broadcasts_cols) {
          // Rows are disjoint across lanes; only N-region replay duplicates it.
          request_multiplicity = (double)schedule.n_partition.parts;
        }
      }
      gm_ub_bytes += bytes * request_multiplicity;
    }
    if (info.is_boundary_out)  // store direction is the sink unit: MatMul -> L0C->GM, vector -> UB->GM
      (info.is_mm_out ? l0c_gm_bytes : ub_gm_bytes) += bytes;
  }
  FlatSet<size_t> in_sg(ops_.begin(), ops_.end());                             // (b)
  for (auto t : ephemeral_) {
    const int prod = dag_->tensor_producer[t];
    const bool prod_cube = prod >= 0 && prob_->ops[(size_t)prod].type == OpType::MatMul;
    bool crosses = false;
    for (auto c : dag_->tensor_consumers[t])
      if (in_sg.count(c) && (prob_->ops[c].type == OpType::MatMul) != prod_cube) {
        crosses = true; break;
      }
    if (crosses) {
      const double bytes = (double)(prob_->tensors[t].width * prob_->tensors[t].height) *
                           dtype_bytes(prob_->tensors[t].dtype);
      // Roundtrip = one WRITE by the producer's unit + one READ by the consumer's.
      if (prod_cube) { l0c_gm_bytes += bytes; gm_ub_bytes += bytes; }  // cube writes L0C->GM, vector reads GM->UB
      else           { ub_gm_bytes  += bytes; gm_l1_bytes += bytes; }  // vector writes UB->GM, cube reads GM->L1
    }
  }

  // Per-port HBM cap: each port's traffic DIVIDES across active units up to the
  // aggregate HBM ceiling (par = max(1, min(active, hbm/peak))). Duplicated from
  // the base model (see Ascend910BCost::compute_cost) — the mixed branch has no
  // access to the base's local lambda. hbm<=0 => uncapped per-unit divide.
  const double hbm = prob_->hbm_aggregate_gibps;
  auto par = [&](double active, double peak_gibps) {
    const double cap = (hbm > 0.0 && peak_gibps > 0.0)
                           ? hbm / peak_gibps : std::numeric_limits<double>::infinity();
    return std::max(1.0, std::min(active, cap));
  };
  // Each port -> cycles at its own cyc/byte (GM->L1 reload, GM->UB ub_in, L0C->GM store,
  // UB->GM ub_out) and its own per-core peak, divided by par(). The two emitted AIV lanes
  // own independent vector pipes; buildable C->V therefore distributes vector traffic over
  // active_groups*2, while the analytic legacy path retains its historical per-unit count.
  // The four direction classes OVERLAP,
  // so ddr_lat = MAX over them (grounded: mixed_contention / mixed_ddr_bound). The VECTOR
  // ports + stage are split-K-INVARIANT (a sink split-K recruits CUBE cores only); the
  // cube ports, cube_stage, and ddr are recomputed per split factor S in eval_S below.
  const double active_vector_pipes =
      prob_->require_buildable_mixed
          ? eff_units * (double)std::max<int64_t>(1, schedule.vector_lanes)
          : eff_units;
  const double gm_ub_lat =
      gm_ub_bytes * bc.ub_in / par(active_vector_pipes, prob_->bw_gm_ub);
  const double ub_gm_lat =
      ub_gm_bytes * bc.ub_out / par(active_vector_pipes, prob_->bw_ub_gm);
  // Compute distribution: the LPT makespan over the non-uniform grid regions (the BUSIEST
  // unit), NOT the flat total/eff_units average, which under-predicts an imbalanced grid's
  // biggest region (up to ~2x at one region/unit, the few-tile decode corner). The CUBE region
  // work is recomputed PER REGION — max(Σ MAC, Σ extract) at the region extent — so it captures
  // the fractal/extract padding non-linearity (the per-region ceil) and lets extract vs MAC
  // dominate region-to-region, matching the mixed cube stage's own max(mac,extract) model. The
  // VECTOR region work is an output-area fraction (a documented vector-region approximation;
  // VecOpCompute is op-tensor-coupled, so a region-aware helper is a separate follow-up).
  // Uniform tiles (parts==0; ad-hoc/test) use the wave-aware total; eff_units/eff_cube (active
  // cores) still set the HBM par() + count.
  const double base_cube_work = std::max(cube_mac, cube_extract);  // MACs vs extract (total)
  const bool grid_mode = cfg.parts_m > 0;
  const AxisPartition g_pm = partition_axis(out_H_, std::max<int64_t>(1, cfg.parts_m), grid_gran_h_);
  const AxisPartition g_pn = partition_axis(out_W_, std::max<int64_t>(1, cfg.parts_n), grid_gran_w_);
  const double out_area = (double)std::max<int64_t>(1, out_H_ * out_W_);
  auto cube_region_work = [&](int64_t m_ext, int64_t n_ext) {
    double rmac = 0.0, rext = 0.0;  // per-region recompute: fractal padding + local extract/MAC max
    for (auto i : ops_) {
      const auto& op = prob_->ops[i];
      if (op.type != OpType::MatMul) continue;
      const int64_t Ko = prob_->tensors[op.inputs[0]].width;
      const DType dt = prob_->tensors[op.inputs[0]].dtype;
      rmac += CubeMacCycles(prob_, m_ext, n_ext, Ko, dt);
      rext += CubeExtractCycles(prob_, bc, m_ext, n_ext, Ko, dt);
    }
    return std::max(rmac, rext);
  };
  auto buildable_cube_region_work = [&](int64_t k_ext) {
    // Compiler mixed v0 contains exactly one matmul.  Re-evaluate its grounded
    // MAC-vs-L1->L0 wall for the concrete K phase so peeled init/tail work is
    // not accidentally included in the rolled phase roofline.
    if (!mixed_topology_ || mixed_topology_->stages.empty() ||
        mixed_topology_->stages[0].ops.size() != 1 || k_ext <= 0) {
      return 0.0;
    }
    const Op& op = prob_->ops[mixed_topology_->stages[0].ops.front()];
    const DType dt = prob_->tensors[op.inputs.front()].dtype;
    const int64_t m_ext = schedule.m_partition.big;
    const int64_t n_ext = schedule.n_partition.big;
    return std::max(CubeMacCycles(prob_, m_ext, n_ext, k_ext, dt),
                    CubeExtractCycles(prob_, bc, m_ext, n_ext, k_ext, dt));
  };
  auto vec_region_work = [&](int64_t m_ext, int64_t n_ext) {
    return vector_compute * ((double)(m_ext * n_ext) / out_area);  // area-fraction (approximation)
  };
  double buildable_vector_tile = 0.0;
  if (prob_->require_buildable_mixed) {
    // The emitted UP_DOWN body runs the complete pointwise chain once on each
    // half tile.  Price that exact valid frame: fixed startup is paid once per
    // lane/item, never area-scaled away or divided after the fact.
    const int64_t lane_rows =
        schedule.m_partition.big / std::max<int64_t>(1, schedule.vector_lanes);
    const int64_t tile_cols = schedule.n_partition.big;
    bool stream_start = true;
    for (size_t op_idx : mixed_topology_->stages[1].ops) {
      const Op& op = prob_->ops[op_idx];
      buildable_vector_tile += GroundedVectorOpCompute(
          prob_, op, lane_rows, tile_cols, stream_start,
          /*row_expand_composite=*/false);
      stream_start = false;
    }
  }
  // Vector stage runs on 2 cores per unit (split-K-invariant: a sink split-K recruits CUBE
  // cores only). Makespan over the grid, then halved across the unit's 2 AIV cores.
  const double vector_lanes = (double)std::max<int64_t>(1, schedule.vector_lanes);
  const double vec_stage = prob_->require_buildable_mixed
      ? buildable_vector_tile * (double)schedule.loop.max_trips_per_group
      : (grid_mode
             ? LptMakespan((int64_t)eff_units, g_pm, g_pn, vec_region_work)
             : WaveComputeCycles(vector_compute, num_tiles, (int64_t)eff_units)) /
            vector_lanes;
  const double one_vec_tile = prob_->require_buildable_mixed
      ? buildable_vector_tile
      : vector_compute / (vector_lanes * (double)num_tiles);

  // Pipeline wall (grounded EXACTLY by mixed_tile_study, the shape sweep). The cube and
  // vector units OVERLAP. In a 2-stage kernel (c->v or v->c) the output unit runs ONLY the
  // output op, so the wall is the SYMMETRIC cross-term: each unit's full stage plus ONE tile
  // of the OTHER (the un-overlapped fill/drain end) — max(cube_stage+one_vec_tile,
  // vec_stage+one_cube_tile, ddr_lat). This matches the sim to the decimal (c->v, v->c), folds
  // the fill INTO the max (so it is absorbed when DDR- or compute-bound by the other unit), and
  // charges an imbalanced fusion only one TINY non-bottleneck tile (a matmul + small epilogue is
  // NOT over-charged a full cube tile). A 3-stage kernel (v->c->v, c->v->c) has the output unit
  // busy from t=0 (it runs an earlier stage), so there is no idle end — plain max.
  // Detection: the sink unit is the producing unit of the boundary output (output_is_cube =
  // it is a MatMul). The fill is ABSORBED (3-stage, plain max) iff the sink unit can run at
  // t=0 — i.e. it has an "early stage" op whose whole input cone is same-unit + boundary,
  // independent of the opposite unit (the v-prologue of v->c->v, the first matmul of c->v->c).
  // Otherwise EVERY sink-unit op transitively waits on the opposite unit, so the sink idles
  // one opposite tile before starting and the fill ADDS (2-stage). Counting sink-unit ops is
  // NOT sufficient: a same-unit tail (c->v->v) has >1 sink op yet still idles at t=0. We flag
  // each op whose input cone touches the opposite unit (single forward-topo pass, producers
  // before consumers via reverse_topo_ops_), then look for a sink-unit op that does not.
  const bool output_is_cube = mixed_topology_->output_is_cube;
  const bool two_stage = !mixed_topology_->sink_runs_early_stage;
  // Fill is absorbed only when the shape is 3-stage AND there is a successor tile to skew
  // against (num_tiles >= 2). A single-tile 3-stage kernel pays the cross-term (A2), so the
  // diagnostic flag must reflect the ACTUAL wall, not just the structural shape.
  result.pipeline_fill_absorbed = schedule.pipeline_fill_absorbed;

  // Sink split-K (cube-sink, SINGLE matmul only). The sink matmul may split its
  // contraction across idle CUBE cores — atomic-add partials with NO merge barrier, so
  // the cores stay independent (exactly the base cube split-K; the vector prologue
  // overlaps orthogonally). Restricted to a single-matmul cube sink (v->c, v->v->c) so
  // ONLY the sink is ever split — never a mid-kernel matmul. split-K recruits cube cores
  // (eff_cube = min(num_tiles*S, n_units)) and grows the output write-back to S atomic
  // partials on L0C->GM; the vector stage/ports are untouched. S=1 == the spatial-only
  // cost, so a non-splittable kernel is unchanged and a split is taken only when it wins.
  int num_matmuls = 0;
  for (auto i : ops_) if (prob_->ops[i].type == OpType::MatMul) num_matmuls++;
  // Cube-sink split-K is MODEL-AHEAD of the emit (§12): gate on the buildable flag so a
  // buildable-mode harness never selects an unemittable mixed split (default true = analytic).
  const bool can_split = output_is_cube && num_matmuls == 1 &&
                         prob_->allow_model_ahead_split_k && !prob_->require_buildable_mixed;
  const double sink_store_bytes = l0c_gm_bytes;  // single-matmul cube sink: L0C->GM == the sink store
  std::vector<int64_t> buildable_cube_windows;
  if (prob_->require_buildable_mixed) {
    derive_exec(cfg, output_K_, retained_from_prev, retain_these,
                &buildable_cube_windows);
  }
  const int64_t buildable_cube_window =
      sink_mm_op_ >= 0 && static_cast<size_t>(sink_mm_op_) < buildable_cube_windows.size()
          ? buildable_cube_windows[static_cast<size_t>(sink_mm_op_)]
          : output_K_;
  struct MixEval { double wall, ddr, max_stage, eff_cube; };
  auto eval_S = [&](int64_t S) -> MixEval {
    const double eff_cube =
        std::min((double)schedule.loop.active_groups * (double)S, n_units);
    // Cube compute makespan (busiest core), split-K aware (LptMakespan ksplit = S; the wave arm
    // divides num_tiles*S units). Uniform-grid LptMakespan reduces exactly to WaveComputeCycles.
    const double cube_stage = grid_mode
        ? LptMakespan((int64_t)eff_cube, g_pm, g_pn, cube_region_work, S)
        : WaveComputeCycles(base_cube_work, num_tiles * S, (int64_t)eff_cube);
    const double one_cube_tile = (base_cube_work / (double)num_tiles) / std::min((double)S, n_units);
    const double gm_l1_lat  = gm_l1_bytes * bc.reload / par(eff_cube, prob_->bw_gm_l1);
    const double l0c_gm_lat = (l0c_gm_bytes + (double)(S - 1) * sink_store_bytes) * bc.store
                              / par(eff_cube, prob_->bw_l0c_gm);
    const double ddr = std::max({gm_l1_lat, gm_ub_lat, l0c_gm_lat, ub_gm_lat});
    // Analytic topologies retain the historical homogeneous two-fractal proxy
    // until they acquire stage-local cube plans. Buildable v0 is priced below
    // from its exact init/rolled/tail K phases instead of this global roofline.
    const double cube_dram = std::max(gm_l1_lat, l0c_gm_lat);
    const bool cube_db = output_K_ <= 1 ||
                         (output_K_ / std::max<int64_t>(1, S)) >= 32;
    const double cube_wall = cube_db ? cube_stage : cube_stage + cube_dram;
    // Buildable C->V pricing follows the actual lowered item order.  Ordinary
    // GM->L1 operand feed may overlap the K-window cube work, but TPUSH waits
    // for the crossing write before the AIC advances.  On AIV, TPOP, the
    // pointwise chain, and the final store are likewise ordered.  Cross-engine
    // FIFO execution can still overlap complete successor items: equal T>=2
    // trips use max(T*C+V, T*V+C); one trip is the serial C+V fill/drain.
    if (prob_->require_buildable_mixed) {
      const int64_t trips = schedule.loop.max_trips_per_group;
      const int64_t full_k = std::max<int64_t>(1, output_K_);
      const int64_t k_window =
          std::clamp(buildable_cube_window, int64_t{1}, full_k);
      const int64_t full_chunks = full_k / k_window;
      const int64_t tail_k = full_k - full_chunks * k_window;
      const double item_feed = gm_l1_lat / static_cast<double>(trips);
      auto feed_for_k = [&](int64_t k_extent) {
        return item_feed * static_cast<double>(k_extent) /
               static_cast<double>(full_k);
      };
      auto serial_k_phase = [&](int64_t k_extent) {
        return buildable_cube_region_work(k_extent) + feed_for_k(k_extent);
      };

      double item_cube = 0.0;
      // BuildTileMatmul falls back to one full-K operation when fewer than two
      // full windows exist or a tail cannot form a 16-wide cube fractal.
      if (full_chunks < 2 || (tail_k > 0 && tail_k % 16 != 0)) {
        item_cube = serial_k_phase(full_k);
      } else {
        // Serial first window seeds the accumulator.
        item_cube = serial_k_phase(k_window);

        // Remaining full windows are the only stage-2 region.  The loop is
        // pipelined exactly when at least two rolled iterations exist; model
        // its fill/drain rather than granting max() to init or tail.
        const int64_t rolled = full_chunks - 1;
        const double rolled_compute = buildable_cube_region_work(k_window);
        const double rolled_feed = feed_for_k(k_window);
        if (rolled >= 2) {
          item_cube += rolled_compute + rolled_feed +
                       static_cast<double>(rolled - 1) *
                           std::max(rolled_compute, rolled_feed);
        } else {
          item_cube += static_cast<double>(rolled) *
                       (rolled_compute + rolled_feed);
        }
        if (tail_k > 0) item_cube += serial_k_phase(tail_k);
      }

      // TPUSH is blocking after the final accumulator drain for every item.
      item_cube += l0c_gm_lat / static_cast<double>(trips);
      const double cube_phase = static_cast<double>(trips) * item_cube;
      const double vector_phase = gm_ub_lat + vec_stage + ub_gm_lat;
      const double wall = schedule.overlap_implementable
                              ? std::max(cube_phase + vector_phase / (double)trips,
                                         vector_phase + cube_phase / (double)trips)
                              : cube_phase + vector_phase;
      return {wall, std::max({gm_l1_lat, gm_ub_lat, l0c_gm_lat, ub_gm_lat}),
              std::max(cube_phase, vector_phase), eff_cube};
    }

    // Analytic/research topologies retain the broader mixed-study formula.
    // A single-tile kernel has nothing to skew against (mixed_tile_study NT=1:
    // overlap_factor 0.00).  A three-stage fill is absorbed only when the
    // per-group schedule itself has successor items.
    const bool overlap_ok = schedule.overlap_implementable && !two_stage;
    double wall = 0.0;
    if (!schedule.emit_compatible) {
      // Multi-message/multi-round-trip FIFO patterns are demoted by the
      // current PyPTO pass. Price the serial stage sum, never the skewed max.
      // pto-isa shows real serialization can be 1.0-1.34x worse because it also
      // disrupts intra-unit pipelines; the sum is therefore a conservative
      // lower bound pending a topology-specific serial calibration.
      wall = std::max(cube_wall + vec_stage, ddr);
    } else {
      wall = overlap_ok
          ? std::max({cube_wall, vec_stage, ddr})
          : std::max({cube_wall + one_vec_tile, vec_stage + one_cube_tile, ddr});
    }
    return {wall, ddr, std::max(cube_stage, vec_stage), eff_cube};
  };
  MixEval best = eval_S(1);
  int64_t best_S = 1;
  if (can_split) {
    const int64_t kfrac = std::max<int64_t>(1, output_K_ / 16);
    for (int64_t S : all_divisors(kfrac)) {
      if (S < 2 || output_K_ / S < 32) continue;  // need >=2 K-fractals/partial to ping-pong
      const MixEval e = eval_S(S);
      if (e.wall < best.wall - 1e-9) { best = e; best_S = S; }
    }
  }
  result.latency        = best.wall;
  result.ddr_traffic    = best.ddr;
  result.compute_bound  = best.max_stage >= best.ddr;
  result.parallel_split = (int)best_S;
  result.uses_model_ahead_split_k = (best_S > 1);
  result.cores_used =
      (int)(best.eff_cube + vector_lanes * eff_units);

  // AutoFuse emits one mixed launch with active_groups blocks; successor
  // spatial items live in the per-group inner loop.  Kernel fill is therefore
  // paid once, while the grounded per-task term follows the actual launch
  // blocks rather than the number of logical regions.
  if (prob_->kernel_fill_cost > 0) {
    result.latency += (double)prob_->kernel_fill_cost;
  }
  result.latency += (double)schedule.loop.active_groups *
                    (double)prob_->per_task_overhead_cycles;

  // Emitted per-core cube k: the single-core seq-k derived for the sink matmul
  // (same derivation as the homogeneous cube), so the lowered kernel and the
  // visualised tile carry the real contraction chunk, not the candidate cfg.k.
  std::vector<int64_t> pk = std::move(buildable_cube_windows);
  if (pk.empty()) {
    derive_exec(cfg, output_K_, retained_from_prev, retain_these, &pk);
  }
  if (sink_mm_op_ >= 0 && (size_t)sink_mm_op_ < pk.size() && pk[sink_mm_op_] > 0) {
    result.config.k = pk[sink_mm_op_];
    result.num_k_passes = static_cast<int>(
        (output_K_ + result.config.k - 1) / result.config.k);
  }
  return result;
}

CostResult Ascend910BMixed::compute_cost(
    const TileConfig &cfg,
    const FlatSet<size_t> &retained_from_prev,
    const FlatSet<size_t> &retain_these) const {
  return Ascend910BCost::compute_cost(cfg, retained_from_prev, retain_these);
}

// ============================================================================
// Granularity enumeration
// ============================================================================

CostResult Ascend910BCost::best_cost(const FlatSet<size_t>& retained_from_prev,
                                     const FlatSet<size_t>& retain_these) const {
  CostResult best;
  L0PlanMemo l0_memo;
  auto consider = [&](const TileConfig& cfg) {
    auto r = has_matmul_ && !has_vector_ && prob_->use_hierarchical_cube_cost
                 ? compute_cost_impl(cfg, retained_from_prev, retain_these, &l0_memo)
                 : compute_cost(cfg, retained_from_prev, retain_these);
    if (!r.feasible) return;
    bool take;
    if (best.latency == std::numeric_limits<double>::infinity()) {
      take = true;
    } else {
      // Lexicographic tiebreak among equal-latency tiles:
      //   1. fewer split-K partials — a balanced spatial grid that fills the
      //      cores at the same latency beats a K-split (less merge traffic,
      //      no atomic-add serialization, simpler emit). split-K only wins
      //      when it is STRICTLY faster (caught by the latency test above).
      //   2. lower DDR traffic   — matmul reuse (less reload); flat for PW
      //   3. more cores used     — fill the unit's cores (parallelism)
      //   4. EVENLY-DIVIDING tile — a grid whose +-1-fractal region extents do
      //      NOT evenly divide the output (e.g. h=1376 on a 4096 axis) the tiling
      //      emit cannot realize cleanly; at equal latency prefer a tile whose
      //      extents evenly divide the output. An imbalanced grid is used only
      //      where it is strictly faster (power-of-two / few-row fills).
      //   5. larger tile area    — best vectorization / least per-tile
      //      overhead (avoids the degenerate 1xN / 16x16 picks)
      //   6. larger k            — fewer L1 passes
      const double tol = 1e-6 * std::max(1.0, best.latency);
      const double dtol = 1e-9 * std::max(1.0, best.ddr_traffic);
      const double etol = 1e-9 * std::max(1.0, best.l1l0_extract);
      const long long ra = (long long)r.config.w * r.config.h;
      const long long ba = (long long)best.config.w * best.config.h;
      // Emit-friendliness: a tile whose extents EVENLY DIVIDE the output (every
      // region identical) lowers cleanly; a grid's +-1-block extents (e.g.
      // h=1366 on a 4096 axis) the tiling emit can't realize. So at equal
      // latency prefer a dividing tile -- the imbalanced grid is used only when
      // it is strictly faster (the power-of-two / few-row fills that have no
      // dividing C-tiling). Uniform tiles always divide; a grid sometimes does.
      const bool r_div = (out_W_ % std::max<int64_t>(1, r.config.w) == 0) &&
                         (out_H_ % std::max<int64_t>(1, r.config.h) == 0);
      const bool b_div = (out_W_ % std::max<int64_t>(1, best.config.w) == 0) &&
                         (out_H_ % std::max<int64_t>(1, best.config.h) == 0);
      if (r.latency < best.latency - tol) {
        take = true;
      } else if (r.latency > best.latency + tol) {
        take = false;
      } else if (r.parallel_split != best.parallel_split) {
        take = r.parallel_split < best.parallel_split;
      } else if (r.ddr_traffic < best.ddr_traffic - dtol) {
        take = true;
      } else if (r.ddr_traffic > best.ddr_traffic + dtol) {
        take = false;
      } else if (std::abs(r.l1l0_extract - best.l1l0_extract) > etol) {
        // Lower L1->L0 extract (MTE1) wins: the GM reload is port-symmetric and
        // ties transposes, but the L0A/L0B ports are not, so the TALL tile (large
        // h, less slow-L0B traffic) is faster. Perf-sim-driven (pto-isa
        // gml1_decision: removes a ~4% aspect regret); device eval pending.
        take = r.l1l0_extract < best.l1l0_extract;
      } else if (r.cores_used != best.cores_used) {
        take = r.cores_used > best.cores_used;
      } else if (r_div != b_div) {
        take = r_div;  // emit-friendly evenly-dividing tile beats an imbalanced grid
      } else if (ra != ba) {
        take = ra > ba;
      } else {
        take = r.config.k > best.config.k;
      }
    }
    if (take) best = r;
  };

  // GRID-ONLY (910B, cube AND vector). Feasibility is via derive_exec /
  // vector_stream (the cube seq-k fits L1, the vector streams UB, regardless of the
  // spatial tile), so the SpatialSchedule grid -- including the (1,1) whole-output
  // region -- covers every fill; uniform exact-divisor tiles are redundant.
  // SpatialSchedule grids: each (parts_m, parts_n) lands a balanced region count
  // the uniform tiles can't (e.g. exactly n_cores). w,h carry the physical (max)
  // region extent so fits_on_chip / reload are unchanged.
  // One config per (parts_m, parts_n, split_k) triple. The single-core seq-k is
  // NOT enumerated: grid_k is just a structurally-valid placeholder (largest k
  // divisor). fits_on_chip -> derive_exec then DERIVES the greedy L1-fitting
  // per-op k and returns infeasible (INT64_MAX) iff NO such k exists -- so a
  // triple is only accepted when a memory-fitting k EXISTS, and that derived k is
  // what compute_cost writes back to result.config.k for the emit. The parallel
  // split is the triple's split_k, evaluated by compute_cost as a fixed S.
  const int64_t grid_k = ks_cand_.empty() ? std::max<int64_t>(output_K_, 1) : ks_cand_.back();
  for (const auto &g : grid_cand_) {
    const AxisPartition pm = partition_axis(out_H_, g.parts_m, grid_gran_h_);
    const AxisPartition pn = partition_axis(out_W_, g.parts_n, grid_gran_w_);
    consider(TileConfig{pn.big, pm.big, grid_k, pm.parts, pn.parts, g.split_k});
  }
  return best;
}

// Same grid as best_cost, but COLLECT every feasible (config, cost) instead of the argmin — the
// candidate set the solver chose from. For the cost-vs-wall-time validation (dump modeled costs;
// force one for the device emit). Not on the hot path.
std::vector<std::pair<TileConfig, CostResult>> Ascend910BCost::enumerate_plans() const {
  std::vector<std::pair<TileConfig, CostResult>> out;
  L0PlanMemo l0_memo;
  const int64_t grid_k = ks_cand_.empty() ? std::max<int64_t>(output_K_, 1) : ks_cand_.back();
  for (const auto &g : grid_cand_) {
    const AxisPartition pm = partition_axis(out_H_, g.parts_m, grid_gran_h_);
    const AxisPartition pn = partition_axis(out_W_, g.parts_n, grid_gran_w_);
    const TileConfig cfg{pn.big, pm.big, grid_k, pm.parts, pn.parts, g.split_k};
    const CostResult r = has_matmul_ && !has_vector_ && prob_->use_hierarchical_cube_cost
                             ? compute_cost_impl(cfg, {}, {}, &l0_memo)
                             : compute_cost(cfg, {}, {});
    if (!r.feasible) continue;
    out.emplace_back(cfg, r);
  }
  return out;
}
