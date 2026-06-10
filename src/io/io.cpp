#include "io/io.h"
#include "core/cost.h"
#include "solution/solution.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

Problem read_problem(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error: cannot open '" << filename << "'\n";
        std::exit(1);
    }

    json j;
    try {
        j = json::parse(f);
    } catch (const json::parse_error& e) {
        std::cerr << "Error: failed to parse '" << filename << "': " << e.what() << "\n";
        std::exit(1);
    }

    // Validate required top-level keys exist before accessing them.
    for (const char* key : {"widths", "heights", "inputs", "outputs",
                             "base_costs", "op_types",
                             "fast_memory_capacity", "slow_memory_bandwidth",
                             "native_granularity"}) {
        if (!j.contains(key)) {
            std::cerr << "Error: missing required field '" << key
                      << "' in '" << filename << "'\n";
            std::exit(1);
        }
    }

    Problem p;

    // --- Tensors ---
    auto& widths  = j["widths"];
    auto& heights = j["heights"];
    if (widths.size() != heights.size()) {
        std::cerr << "Error: widths and heights arrays have different lengths\n";
        std::exit(1);
    }
    for (size_t i = 0; i < widths.size(); i++)
        p.tensors.push_back({widths[i].get<int64_t>(), heights[i].get<int64_t>()});

    const size_t num_tensors = p.num_tensors();

    // Optional per-tensor dtype (default FP32) — used by the 910B byte-based
    // two-pool working set. Accepts "FP32"/"FP16"/"BF16"/"INT32"/"INT16"/
    // "INT8"/"BOOL".
    if (j.contains("dtypes")) {
        auto& dts = j["dtypes"];
        for (size_t i = 0; i < dts.size() && i < num_tensors; i++) {
            const auto& s = dts[i].get_ref<const std::string&>();
            if (s == "FP16") p.tensors[i].dtype = DType::FP16;
            else if (s == "BF16") p.tensors[i].dtype = DType::BF16;
            else if (s == "INT32") p.tensors[i].dtype = DType::INT32;
            else if (s == "INT16") p.tensors[i].dtype = DType::INT16;
            else if (s == "INT8") p.tensors[i].dtype = DType::INT8;
            else if (s == "BOOL") p.tensors[i].dtype = DType::BOOL;
            else p.tensors[i].dtype = DType::FP32;
        }
    }

    // --- Ops ---
    auto& inputs     = j["inputs"];
    auto& outputs    = j["outputs"];
    auto& base_costs = j["base_costs"];
    auto& op_types   = j["op_types"];

    const size_t num_ops = op_types.size();
    if (inputs.size() != num_ops || outputs.size() != num_ops ||
        base_costs.size() != num_ops) {
        std::cerr << "Error: inputs/outputs/base_costs/op_types arrays have "
                     "inconsistent lengths\n";
        std::exit(1);
    }

    for (size_t i = 0; i < num_ops; i++) {
        Op op;
        const auto& type_str = op_types[i].get_ref<const std::string&>();
        if (type_str == "MatMul") {
            op.type = OpType::MatMul;
        } else if (type_str == "Pointwise") {
            op.type = OpType::Pointwise;
        } else if (type_str == "Reduction") {
            op.type = OpType::Reduction;
        } else if (type_str == "Opaque") {
            op.type = OpType::Opaque;
        } else {
            std::cerr << "Error: unknown op type '" << type_str
                      << "' for op " << i << "\n";
            std::exit(1);
        }

        for (auto& t : inputs[i]) {
            size_t idx = t.get<size_t>();
            if (idx >= num_tensors) {
                std::cerr << "Error: op " << i << " input tensor index " << idx
                          << " out of range (num_tensors=" << num_tensors << ")\n";
                std::exit(1);
            }
            op.inputs.push_back(idx);
        }
        {
            size_t idx = outputs[i][0].get<size_t>();
            if (idx >= num_tensors) {
                std::cerr << "Error: op " << i << " output tensor index " << idx
                          << " out of range (num_tensors=" << num_tensors << ")\n";
                std::exit(1);
            }
            op.outputs.push_back(idx);
        }
        op.base_cost = base_costs[i].get<int64_t>();
        p.ops.push_back(std::move(op));
    }

    // --- Tensor integrity checks ---
    // 1. Each tensor must have at most one producing op.
    // 2. Warn about isolated tensors (no producer AND no consumer).
    {
        std::vector<int> producer_op(num_tensors, -1);
        std::vector<bool> is_consumed(num_tensors, false);

        for (size_t i = 0; i < num_ops; i++) {
            { size_t t = p.ops[i].output();
                if (producer_op[t] >= 0) {
                    std::cerr << "Error: tensor " << t << " produced by both op "
                              << producer_op[t] << " and op " << i << "\n";
                    std::exit(1);
                }
                producer_op[t] = (int)i;
            }
            for (auto t : p.ops[i].inputs)
                is_consumed[t] = true;
        }

        for (size_t t = 0; t < num_tensors; t++) {
            if (producer_op[t] < 0 && !is_consumed[t]) {
                std::cerr << "Warning: tensor " << t << " ("
                          << p.tensors[t].width << "x" << p.tensors[t].height
                          << ") is isolated — not produced or consumed by any op\n";
            }
        }
    }

    // --- Hardware parameters ---
    p.fast_memory_capacity  = j["fast_memory_capacity"].get<int64_t>();
    p.slow_memory_bandwidth = j["slow_memory_bandwidth"].get<int64_t>();

    auto& ng = j["native_granularity"];
    if (!ng.is_array() || ng.size() < 2) {
        std::cerr << "Error: native_granularity must be a 2-element array [w, h]\n";
        std::exit(1);
    }
    p.native_w = ng[0].get<int64_t>();
    p.native_h = ng[1].get<int64_t>();

    if (p.fast_memory_capacity <= 0 || p.slow_memory_bandwidth <= 0 ||
        p.native_w <= 0 || p.native_h <= 0) {
        std::cerr << "Error: hardware parameters must be positive\n";
        std::exit(1);
    }

    // Optional 910B parallel-core fields. Absent => defaults (1/1/0) keep the
    // single-context competition behavior. Present => the parallel roofline +
    // unit-homogeneity constraint activate (cube 24 / vector 48 cores, etc.).
    if (j.contains("num_cube_cores"))   p.num_cube_cores   = j["num_cube_cores"].get<int>();
    if (j.contains("num_vector_cores")) p.num_vector_cores = j["num_vector_cores"].get<int>();
    if (j.contains("cube_capacity"))    p.cube_capacity    = j["cube_capacity"].get<int64_t>();
    if (j.contains("vec_capacity"))     p.vec_capacity     = j["vec_capacity"].get<int64_t>();
    if (j.contains("l1_capacity"))      p.l1_capacity      = j["l1_capacity"].get<int64_t>();
    if (j.contains("double_buffer"))    p.double_buffer    = j["double_buffer"].get<bool>();

    // -------------------------------------------------------------------------
    // Precompute retainable_tensors.
    //
    // A tensor is retainable across subgraph boundaries if ALL of:
    //   1. Its full size fits in fast memory.
    //      (A tensor larger than capacity can never be legally pinned.)
    //   2. It has at least one consuming op.
    //      (Graph outputs have no consumers — they are evicted at the end and
    //      never read by a later subgraph, so retaining them is pointless.)
    //
    // Graph inputs with a single consumer ARE included. Although in a simple
    // linear schedule they are read once and discarded, recomputation-based
    // strategies (e.g. diamond graphs like Example 3B) may place the same
    // graph input in two separate subgraphs. Retaining it after the first
    // subgraph eliminates the redundant reload in the second.
    //
    // The ordering layer is responsible for deciding whether retention is
    // actually beneficial for a given schedule; this set is the permissive
    // upper bound of candidates.
    // -------------------------------------------------------------------------
    // 910B: NO cross-subgraph retention. Each subgraph runs across the cores;
    // data crossing a subgraph boundary (incl. the cube<->vector handoff) routes
    // through DDR/GM, and there is no single shared fast memory that persists a
    // tensor across subgraph executions (L1/L0c and UB are per-core, transient).
    // So retainable_tensors stays empty — the per-core working set never pins a
    // cross-subgraph tensor. (Competition single-core keeps the original logic.)
    if (p.num_cube_cores <= 1 && p.num_vector_cores <= 1) {
        std::vector<size_t> consumer_count(num_tensors, 0);
        for (auto& op : p.ops)
            for (auto t : op.inputs)
                consumer_count[t]++;

        for (size_t i = 0; i < num_tensors; i++) {
            if (p.tensors[i].size() > p.fast_memory_capacity) continue; // rule 1
            if (consumer_count[i] == 0)                        continue; // rule 2
            p.retainable_tensors.insert(i);
        }
    }

    return p;
}

void write_solution(const std::string& filename, const Solution& sol) {
    json j;
    j["subgraphs"]         = json::array();
    j["granularities"]     = json::array();
    j["tensors_to_retain"] = json::array();
    j["traversal_orders"]  = json::array();
    j["subgraph_latencies"] = json::array();

    for (size_t i = 0; i < sol.num_steps(); i++) {
        const auto& step = sol.step(i);
        const auto& cfg  = step.config;

        j["subgraphs"].push_back(step.subgraph.ops());
        j["granularities"].push_back({cfg.w, cfg.h, cfg.k});
        j["tensors_to_retain"].push_back(
            std::vector<size_t>(step.retain_these.begin(), step.retain_these.end()));

        if (cfg.snake != SnakeDir::None) {
            int ntw = (int)(step.subgraph.output_width()  / cfg.w);
            int nth = (int)(step.subgraph.output_height() / cfg.h);
            if (ntw * nth > 1) {
                // Only emit an explicit traversal order when there is more than
                // one tile. A single-tile grid has nothing to permute; null
                // (raster default) is equivalent and cleaner in the output.
                auto order = make_traversal(ntw, nth, cfg.snake);
                j["traversal_orders"].push_back(
                    std::vector<int64_t>(order.begin(), order.end()));
            } else {
                j["traversal_orders"].push_back(nullptr);
            }
        } else {
            j["traversal_orders"].push_back(nullptr);
        }

        j["subgraph_latencies"].push_back(sol.step_latency(i));
    }

    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error: cannot write '" << filename << "'\n";
        std::exit(1);
    }
    f << j.dump(2) << "\n";
    if (!f) {
        std::cerr << "Error: write failed for '" << filename << "'\n";
        std::exit(1);
    }
}