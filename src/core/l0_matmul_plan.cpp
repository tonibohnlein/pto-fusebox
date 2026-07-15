#include "core/l0_matmul_plan.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <sstream>
#include <tuple>
#include <vector>

namespace {

int64_t ceil_div(int64_t value, int64_t divisor) { return (value + divisor - 1) / divisor; }

int64_t align_up(int64_t value, int64_t alignment) { return ceil_div(value, alignment) * alignment; }

int64_t align_down(int64_t value, int64_t alignment) { return value / alignment * alignment; }

struct Regime {
  L0Stationarity stationarity = L0Stationarity::Output;
  bool double_buffer_c = false;
};

struct Candidate {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t traffic_bytes = 0;
  int64_t wall_cycles = 0;
  int64_t padded_compute = 0;
  L0KLoopPlan k_loop;
  L0MatmulPhaseCost phases;
  Regime regime;
};

std::pair<int64_t, int64_t> operand_depths(L0Stationarity stationarity) {
  if (stationarity == L0Stationarity::A) return {1, 2};
  if (stationarity == L0Stationarity::B) return {2, 1};
  return {2, 2};
}

bool output_stationary_holds_a(int64_t m, int64_t n, const L0MatmulConfig& c) {
  const double held_a = static_cast<double>(c.bytes_a * c.m * c.k) / c.bw_l0a +
                        static_cast<double>(c.bytes_b * c.k * c.n * ceil_div(c.m, m)) / c.bw_l0b;
  const double held_b = static_cast<double>(c.bytes_a * c.m * c.k * ceil_div(c.n, n)) / c.bw_l0a +
                        static_cast<double>(c.bytes_b * c.k * c.n) / c.bw_l0b;
  return held_a <= held_b;
}

double load_cycles(int64_t m, int64_t n, int64_t k, const L0MatmulConfig& c, const Regime& r) {
  const double held_a = static_cast<double>(c.bytes_a * c.m * c.k) / c.bw_l0a +
                        static_cast<double>(c.bytes_b * c.k * c.n * ceil_div(c.m, m)) / c.bw_l0b;
  const double held_b = static_cast<double>(c.bytes_a * c.m * c.k * ceil_div(c.n, n)) / c.bw_l0a +
                        static_cast<double>(c.bytes_b * c.k * c.n) / c.bw_l0b;
  if (r.stationarity == L0Stationarity::A) return held_a;
  if (r.stationarity == L0Stationarity::B) return held_b;
  if (k >= c.k) return output_stationary_holds_a(m, n, c) ? held_a : held_b;
  return static_cast<double>(c.bytes_a * c.m * c.k * ceil_div(c.n, n)) / c.bw_l0a +
         static_cast<double>(c.bytes_b * c.k * c.n * ceil_div(c.m, m)) / c.bw_l0b;
}

int64_t mad_cycles(int64_t m, int64_t n, int64_t k, const L0MatmulConfig& c) {
  const int64_t k_fractal = std::max<int64_t>(1, c.mad_k_fractal_bytes / c.bytes_a);
  const int64_t cycles_per_repeat = std::max<int64_t>(1, c.bytes_a / 2);
  const int64_t full = c.k / k;
  const int64_t tail = c.k - full * k;
  const int64_t blocks = full + (tail > 0 ? 1 : 0);
  const int64_t fractals = full * ceil_div(k, k_fractal) + (tail > 0 ? ceil_div(tail, k_fractal) : 0);
  const int64_t per_tile = blocks * c.mad_head_cycles +
                           cycles_per_repeat * ceil_div(m, c.align_m) * fractals * ceil_div(n, c.align_n);
  return ceil_div(c.m, m) * ceil_div(c.n, n) * per_tile;
}

double one_block_load_cycles(int64_t m, int64_t n, int64_t k, const L0MatmulConfig& c) {
  return static_cast<double>(c.bytes_a * m * k) / c.bw_l0a +
         static_cast<double>(c.bytes_b * k * n) / c.bw_l0b;
}

double one_block_mad_cycles(int64_t m, int64_t n, int64_t k, const L0MatmulConfig& c) {
  const int64_t k_fractal = std::max<int64_t>(1, c.mad_k_fractal_bytes / c.bytes_a);
  const int64_t cycles_per_repeat = std::max<int64_t>(1, c.bytes_a / 2);
  return static_cast<double>(c.mad_head_cycles) +
         static_cast<double>(cycles_per_repeat * ceil_div(m, c.align_m) * ceil_div(k, k_fractal) *
                             ceil_div(n, c.align_n));
}

int64_t odd_part(int64_t value) {
  while ((value & 1) == 0) value >>= 1;
  return value;
}

double drain_cycles(int64_t m, int64_t n, const L0MatmulConfig& c) {
  if (c.output_target == L0OutputTarget::Acc) return 0.0;
  if (c.output_target == L0OutputTarget::L1) {
    return static_cast<double>(c.bytes_c * m * n) / c.bw_l0c_l1;
  }
  const int64_t drains = ceil_div(c.m, m) * ceil_div(c.n, n);
  const int64_t n0 = std::max<int64_t>(1, c.drain_c0_bytes / c.bytes_c);
  const int64_t n1 = ceil_div(n, n0);
  const double throughput = static_cast<double>(c.bytes_c * n) / c.bw_drain;
  const double per_row = std::max(c.drain_row_cycles, throughput) +
                         c.drain_penalty_cycles * static_cast<double>(odd_part(n1) - 1);
  return static_cast<double>(drains) * (c.drain_fixed_cycles + static_cast<double>(m) * per_row);
}

int64_t traffic_bytes(int64_t m, int64_t n, int64_t k, const L0MatmulConfig& c, const Regime& r) {
  int64_t lhs = 0;
  int64_t rhs = 0;
  if (r.stationarity == L0Stationarity::A ||
      (r.stationarity == L0Stationarity::Output && k >= c.k && output_stationary_holds_a(m, n, c))) {
    lhs = c.bytes_a * c.m * c.k;
    rhs = c.bytes_b * c.k * c.n * ceil_div(c.m, m);
  } else if (r.stationarity == L0Stationarity::B || (r.stationarity == L0Stationarity::Output && k >= c.k)) {
    lhs = c.bytes_a * c.m * c.k * ceil_div(c.n, n);
    rhs = c.bytes_b * c.k * c.n;
  } else {
    lhs = c.bytes_a * c.m * c.k * ceil_div(c.n, n);
    rhs = c.bytes_b * c.k * c.n * ceil_div(c.m, m);
  }
  const int64_t c_traffic =
      c.output_target == L0OutputTarget::Acc ? 0 : (c.accumulator_read ? 2 : 1) * c.bytes_c * c.m * c.n;
  return lhs + rhs + c_traffic;
}

std::vector<int64_t> legal_k_values(int64_t m, int64_t n, const L0MatmulConfig& c, int64_t a_budget,
                                    int64_t b_budget) {
  const int64_t capacity = std::min(a_budget / m, b_budget / n);
  const int64_t problem = c.allow_padding ? std::max(align_up(c.k, c.align_k), c.min_k) : c.k;
  const int64_t upper = align_down(std::min(capacity, problem), c.align_k);
  const bool peel = c.allow_k_boundary && c.k % c.align_k == 0;
  std::vector<int64_t> values;
  for (int64_t k = c.min_k; k <= upper; k += c.align_k) {
    if (!c.allow_padding && !peel && c.k % k != 0) continue;
    values.push_back(k);
  }
  return values;
}

bool better(const Candidate& a, const Candidate& b, const L0MatmulConfig& c) {
  if (a.wall_cycles != b.wall_cycles) return a.wall_cycles < b.wall_cycles;
  if (a.padded_compute != b.padded_compute) return a.padded_compute < b.padded_compute;
  if (ceil_div(c.k, a.k) != ceil_div(c.k, b.k)) return ceil_div(c.k, a.k) < ceil_div(c.k, b.k);
  if (a.phases.load_cycles != b.phases.load_cycles) return a.phases.load_cycles < b.phases.load_cycles;
  if (a.m * a.n != b.m * b.n) return a.m * a.n > b.m * b.n;
  return a.k > b.k;
}

std::optional<Candidate> best_in_regime(const L0MatmulConfig& c, const Regime& r, bool require_2d,
                                        bool require_full_k) {
  const auto [depth_a, depth_b] = operand_depths(r.stationarity);
  const int64_t a_budget = c.l0a_bytes / (c.bytes_a * depth_a);
  const int64_t b_budget = c.l0b_bytes / (c.bytes_b * depth_b);
  const int64_t c_budget = c.l0c_bytes / (c.bytes_c * (r.double_buffer_c ? 2 : 1));
  const int64_t m_upper = c.allow_padding ? align_up(c.m, c.align_m) : c.m;
  const int64_t n_upper = c.allow_padding ? align_up(c.n, c.align_n) : c.n;
  std::optional<Candidate> best;
  for (int64_t m = c.min_m; m <= m_upper; m += c.align_m) {
    if (m * c.min_n > c_budget) break;
    if (require_2d && ceil_div(c.m, m) < 2) continue;
    const int64_t n_max = std::min(n_upper, c_budget / m);
    for (int64_t n = c.min_n; n <= n_max; n += c.align_n) {
      if (require_2d && ceil_div(c.n, n) < 2) continue;
      for (int64_t k : legal_k_values(m, n, c, a_budget, b_budget)) {
        if (require_full_k && k != c.k) continue;
        Candidate candidate;
        candidate.m = m;
        candidate.n = n;
        candidate.k = k;
        candidate.regime = r;
        candidate.traffic_bytes = traffic_bytes(m, n, k, c, r);
        candidate.padded_compute = ceil_div(c.m, m) * m * ceil_div(c.n, n) * n * ceil_div(c.k, k) * k;
        candidate.phases.load_cycles = load_cycles(m, n, k, c, r);
        candidate.phases.mad_cycles = static_cast<double>(mad_cycles(m, n, k, c));
        candidate.phases.drain_cycles = drain_cycles(m, n, c);
        const double aggregate_compute = std::max(candidate.phases.load_cycles, candidate.phases.mad_cycles);
        const double selection_wall = r.double_buffer_c
                                          ? std::max(aggregate_compute, candidate.phases.drain_cycles) +
                                                std::min(aggregate_compute, candidate.phases.drain_cycles) /
                                                    static_cast<double>(ceil_div(c.m, m) * ceil_div(c.n, n))
                                          : aggregate_compute + candidate.phases.drain_cycles;
        candidate.k_loop.chunk = k;
        candidate.k_loop.full_chunks = c.k / k;
        candidate.k_loop.tail = c.k - candidate.k_loop.full_chunks * k;
        candidate.k_loop.pipeline_stages = candidate.k_loop.full_chunks >= 2 ? 2 : 1;
        const bool one_output_tile = ceil_div(c.m, m) == 1 && ceil_div(c.n, n) == 1;
        if (one_output_tile) {
          const double full_load = one_block_load_cycles(c.m, c.n, k, c);
          const double full_mad = one_block_mad_cycles(c.m, c.n, k, c);
          if (candidate.k_loop.full_chunks > 0) {
            candidate.phases.init_cycles = full_load + full_mad;
            if (candidate.k_loop.full_chunks >= 2) {
              candidate.phases.rolled_cycles =
                  static_cast<double>(candidate.k_loop.full_chunks - 1) * std::max(full_load, full_mad);
            }
          }
          if (candidate.k_loop.tail > 0) {
            candidate.phases.tail_cycles = one_block_load_cycles(c.m, c.n, candidate.k_loop.tail, c) +
                                           one_block_mad_cycles(c.m, c.n, candidate.k_loop.tail, c);
          }
          candidate.phases.wall_cycles = candidate.phases.init_cycles + candidate.phases.rolled_cycles +
                                         candidate.phases.tail_cycles + candidate.phases.drain_cycles;
        } else {
          // Existing full M/N-grid chooser: its nested output loops
          // overlap aggregate L1->L0 traffic and Cube work. The
          // CubeSchedulePlan fixed point shrinks every child to the
          // one-output-tile path above, whose phase decomposition is
          // the authoritative fused-kernel cost.
          candidate.phases.wall_cycles = selection_wall;
        }
        if (!std::isfinite(candidate.phases.wall_cycles) || !std::isfinite(selection_wall) ||
            selection_wall > 9007199254740992.0) {
          continue;
        }
        // Preserve the device-grounded chooser ordering. The explicit
        // phase wall is consumed by CubeSchedulePlan's hierarchy; using
        // it to re-select baseK would over-favour 16-wide blocks because
        // the current PTO fit has no per-iteration event/synchronization
        // term. Geometry calibration and phase composition are separate.
        candidate.wall_cycles = static_cast<int64_t>(std::llround(selection_wall));
        if (!best || better(candidate, *best, c)) best = candidate;
      }
    }
  }
  return best;
}

std::string validate(const L0MatmulConfig& c) {
  if (c.m <= 0 || c.n <= 0 || c.k <= 0) return "M, N, K must all be positive";
  if (c.l0a_bytes <= 0 || c.l0b_bytes <= 0 || c.l0c_bytes <= 0) return "L0 capacities must be positive";
  if (c.bytes_a <= 0 || c.bytes_b <= 0 || c.bytes_c <= 0) return "element byte sizes must be positive";
  if (c.min_m <= 0 || c.min_n <= 0 || c.min_k <= 0) return "minimum tile dimensions must be positive";
  if (c.align_m <= 0 || c.align_n <= 0 || c.align_k <= 0) return "tile alignments must be positive";
  if (c.bw_l0a <= 0 || c.bw_l0b <= 0 || c.bw_drain <= 0 || c.bw_l0c_l1 <= 0) {
    return "roofline bandwidths must be positive";
  }
  if (c.drain_fixed_cycles < 0 || c.drain_row_cycles < 0 || c.drain_penalty_cycles < 0 ||
      c.drain_c0_bytes <= 0) {
    return "drain parameters must be non-negative and drain_c0_bytes positive";
  }
  if (!c.allow_padding && c.m < c.min_m) {
    return "allow_padding=false but M=" + std::to_string(c.m) + " is below the cube minimum tile dimension " +
           std::to_string(c.min_m);
  }
  if (!c.allow_padding && c.n < c.min_n) {
    return "allow_padding=false but N=" + std::to_string(c.n) + " is below the cube minimum tile dimension " +
           std::to_string(c.min_n);
  }
  if (!c.allow_padding && c.k < c.min_k) {
    return "allow_padding=false but K=" + std::to_string(c.k) + " is below the cube minimum tile dimension " +
           std::to_string(c.min_k);
  }
  return {};
}

}  // namespace

L0MatmulPlan choose_l0_matmul_plan(const L0MatmulConfig& config) {
  L0MatmulPlan result;
  result.diagnostic = validate(config);
  if (!result.diagnostic.empty()) return result;

  const Regime baseline;
  auto best = best_in_regime(config, baseline, false, false);
  if (!best) {
    result.diagnostic = "no legal L0 matmul tile for the supplied shape and capacities";
    return result;
  }

  const bool tiled = best->m != config.m || best->n != config.n || best->k != config.k;
  if (tiled) {
    std::vector<L0Stationarity> stationarities{L0Stationarity::Output};
    if (config.allow_a_stationary) stationarities.push_back(L0Stationarity::A);
    if (config.allow_b_stationary) stationarities.push_back(L0Stationarity::B);
    for (L0Stationarity stationarity : stationarities) {
      const bool output = stationarity == L0Stationarity::Output;
      const int max_dbc = config.allow_double_buffer_c ? 1 : 0;
      for (int dbc = 0; dbc <= max_dbc; ++dbc) {
        if (output && dbc == 0) continue;
        const Regime regime{stationarity, dbc == 1};
        auto candidate = best_in_regime(config, regime, dbc == 1, !output || dbc == 1);
        if (candidate && candidate->wall_cycles < best->wall_cycles) best = candidate;
      }
    }
  }

  result.feasible = true;
  result.m = best->m;
  result.n = best->n;
  result.k = best->k;
  result.stationarity = best->regime.stationarity;
  result.output_stationary_holds_a = output_stationary_holds_a(best->m, best->n, config);
  const auto [depth_a, depth_b] = operand_depths(best->regime.stationarity);
  result.buffer_depth_a = depth_a;
  result.buffer_depth_b = depth_b;
  result.buffer_depth_c = best->regime.double_buffer_c ? 2 : 1;
  result.output_target = config.output_target;
  result.estimated_traffic_bytes = best->traffic_bytes;
  result.estimated_cost_cycles = best->wall_cycles;
  result.padded_compute_volume = best->padded_compute;
  result.k_loop = best->k_loop;
  result.phases = best->phases;
  if (config.m < config.min_m || config.n < config.min_n || config.k < config.min_k) {
    std::ostringstream message;
    message << "matmul shape is below the cube minimum; padded plan is (" << result.m << ", " << result.n
            << ", " << result.k << ")";
    result.diagnostic = message.str();
  } else {
    result.diagnostic.clear();
  }
  return result;
}

double estimate_l0_output_drain_cycles(int64_t m, int64_t n, const L0MatmulConfig& config,
                                       L0OutputTarget target) {
  if (m <= 0 || n <= 0) return std::numeric_limits<double>::infinity();
  L0MatmulConfig concrete = config;
  concrete.m = m;
  concrete.n = n;
  concrete.output_target = target;
  return drain_cycles(m, n, concrete);
}
