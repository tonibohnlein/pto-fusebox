#pragma once

#include <cstdint>
#include <string>

// Backend-parameterized L0 schedule selected once and shared by the cost model
// and PyPTO's AutoTileMatmulL0 lowering.  PTO Fusebox owns only the pure
// planning math; the compiler remains the sole owner of the IR rewrite.
enum class L0Stationarity { Output, A, B };
// A2/A3 can drain Acc to GM or L1, but it cannot reload Mat(L1) into Acc.
// Acc therefore means that a surrounding K-window schedule keeps C resident
// in L0C and this child invocation performs no drain.
enum class L0OutputTarget { Acc, GM, L1 };

struct L0MatmulConfig {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;

  int64_t l0a_bytes = 64 * 1024;
  int64_t l0b_bytes = 64 * 1024;
  int64_t l0c_bytes = 128 * 1024;
  int64_t bytes_a = 2;
  int64_t bytes_b = 2;
  int64_t bytes_c = 4;

  int64_t min_m = 16;
  int64_t min_n = 16;
  int64_t min_k = 16;
  int64_t align_m = 16;
  int64_t align_n = 16;
  int64_t align_k = 16;

  bool allow_a_stationary = false;
  bool allow_b_stationary = false;
  bool allow_double_buffer_c = false;
  bool accumulator_read = false;
  bool allow_padding = false;
  bool allow_k_boundary = false;
  L0OutputTarget output_target = L0OutputTarget::GM;

  double bw_l0a = 129.7;
  double bw_l0b = 85.4;
  double bw_drain = 118.0;
  // PTO-ISA PipeKey::L0C_TO_L1, expressed in bytes/cycle.
  double bw_l0c_l1 = 74.3;
  double drain_fixed_cycles = 164.0;
  double drain_row_cycles = 4.45;
  double drain_penalty_cycles = 2.6;
  int64_t drain_c0_bytes = 32;
  int64_t mad_head_cycles = 21;
  int64_t mad_k_fractal_bytes = 32;
};

struct L0MatmulPhaseCost {
  double load_cycles = 0.0;
  double mad_cycles = 0.0;
  // Phase-local wall terms for one output tile. The first K block fills the
  // L1->L0/Cube pipeline, subsequent full blocks form the rolled steady
  // state, and a partial K block plus the final output drain are serial.
  double init_cycles = 0.0;
  double rolled_cycles = 0.0;
  double tail_cycles = 0.0;
  double drain_cycles = 0.0;
  double wall_cycles = 0.0;
};

struct L0KLoopPlan {
  int64_t chunk = 0;
  int64_t full_chunks = 0;
  int64_t tail = 0;
  int pipeline_stages = 1;
};

struct L0MatmulPlan {
  bool feasible = false;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  L0Stationarity stationarity = L0Stationarity::Output;
  bool output_stationary_holds_a = true;
  int64_t buffer_depth_a = 2;
  int64_t buffer_depth_b = 2;
  int64_t buffer_depth_c = 1;
  L0OutputTarget output_target = L0OutputTarget::GM;
  int64_t estimated_traffic_bytes = 0;
  int64_t estimated_cost_cycles = 0;
  int64_t padded_compute_volume = 0;
  L0KLoopPlan k_loop;
  L0MatmulPhaseCost phases;
  std::string diagnostic;
};

L0MatmulPlan choose_l0_matmul_plan(const L0MatmulConfig& config);

double estimate_l0_output_drain_cycles(int64_t m, int64_t n, const L0MatmulConfig& config,
                                       L0OutputTarget target);
