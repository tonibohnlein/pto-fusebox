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
        std::cerr << "Error: cannot open " << filename << "\n";
        std::exit(1);
    }
    json j = json::parse(f);

    Problem p;
    auto& widths  = j["widths"];
    auto& heights = j["heights"];
    for (size_t i = 0; i < widths.size(); i++)
        p.tensors.push_back({widths[i].get<int64_t>(), heights[i].get<int64_t>()});

    auto& inputs     = j["inputs"];
    auto& outputs    = j["outputs"];
    auto& base_costs = j["base_costs"];
    auto& op_types   = j["op_types"];
    for (size_t i = 0; i < op_types.size(); i++) {
        Op op;
        op.type = (op_types[i].get<std::string>() == "MatMul")
                  ? OpType::MatMul : OpType::Pointwise;
        for (auto& t : inputs[i])  op.inputs.push_back(t.get<size_t>());
        for (auto& t : outputs[i]) op.outputs.push_back(t.get<size_t>());
        op.base_cost = base_costs[i].get<int64_t>();
        p.ops.push_back(std::move(op));
    }

    p.fast_memory_capacity  = j["fast_memory_capacity"].get<int64_t>();
    p.slow_memory_bandwidth = j["slow_memory_bandwidth"].get<int64_t>();
    auto& ng = j["native_granularity"];
    p.native_w = ng[0].get<int64_t>();
    p.native_h = ng[1].get<int64_t>();

    // -------------------------------------------------------------------------
    // Precompute retainable_tensors.
    //
    // A tensor is worth retaining across subgraph boundaries only if:
    //   1. It fits entirely in fast memory.
    //   2. It has at least one consumer (graph outputs are evicted, never reused).
    //   3. It is NOT a single-consumer graph input.
    //
    // Rationale for rule 3: a graph input with exactly one consumer is read once
    // and never needed again — retaining it wastes fast memory capacity with no
    // benefit.  A graph input with multiple consumers, on the other hand, can
    // save repeated slow-memory loads and is worth keeping resident.
    //
    // Produced tensors (op outputs) always remain candidates because they may be
    // consumed by multiple downstream subgraphs or kept for recomputation paths.
    // -------------------------------------------------------------------------
    {
        const size_t nt = p.num_tensors();

        std::vector<size_t> consumer_count(nt, 0);
        std::vector<bool>   has_producer(nt, false);

        for (auto& op : p.ops) {
            for (auto t : op.inputs)  consumer_count[t]++;
            for (auto t : op.outputs) has_producer[t] = true;
        }

        for (size_t i = 0; i < nt; i++) {
            // Rule 1: must fit in fast memory
            if (p.tensors[i].size() > p.fast_memory_capacity)
                continue;

            // Rule 2: must have at least one consumer
            if (consumer_count[i] == 0)
                continue;

            // Rule 3: single-consumer graph inputs are never worth retaining
            bool is_graph_input = !has_producer[i];
            if (is_graph_input && consumer_count[i] <= 1)
                continue;

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
            auto order = make_traversal(ntw, nth, cfg.snake);
            j["traversal_orders"].push_back(
                std::vector<int64_t>(order.begin(), order.end()));
        } else {
            j["traversal_orders"].push_back(nullptr);
        }

        j["subgraph_latencies"].push_back(sol.step_latency(i));
    }

    std::ofstream f(filename);
    f << j.dump(2) << "\n";
}