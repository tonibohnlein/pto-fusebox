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
    auto& widths = j["widths"];
    auto& heights = j["heights"];
    for (size_t i = 0; i < widths.size(); i++)
        p.tensors.push_back({widths[i].get<int64_t>(), heights[i].get<int64_t>()});

    auto& inputs = j["inputs"];
    auto& outputs = j["outputs"];
    auto& base_costs = j["base_costs"];
    auto& op_types = j["op_types"];
    for (size_t i = 0; i < op_types.size(); i++) {
        Op op;
        op.type = (op_types[i].get<std::string>() == "MatMul")
                  ? OpType::MatMul : OpType::Pointwise;
        for (auto& t : inputs[i]) op.inputs.push_back(t.get<size_t>());
        for (auto& t : outputs[i]) op.outputs.push_back(t.get<size_t>());
        op.base_cost = base_costs[i].get<int64_t>();
        p.ops.push_back(std::move(op));
    }

    p.fast_memory_capacity = j["fast_memory_capacity"].get<int64_t>();
    p.slow_memory_bandwidth = j["slow_memory_bandwidth"].get<int64_t>();
    auto& ng = j["native_granularity"];
    p.native_w = ng[0].get<int64_t>();
    p.native_h = ng[1].get<int64_t>();

    // Precompute which tensors can be retained (full size fits in fast memory)
    for (size_t i = 0; i < p.tensors.size(); i++)
        if (p.tensors[i].width * p.tensors[i].height <= p.fast_memory_capacity)
            p.retainable_tensors.insert(i);

    return p;
}

void write_solution(const std::string& filename, const Solution& sol) {
    json j;
    j["subgraphs"] = json::array();
    j["granularities"] = json::array();
    j["tensors_to_retain"] = json::array();
    j["traversal_orders"] = json::array();
    j["subgraph_latencies"] = json::array();

    for (size_t i = 0; i < sol.num_steps(); i++) {
        const auto& step = sol.step(i);
        const auto& cfg = step.config;

        j["subgraphs"].push_back(step.subgraph.ops());
        j["granularities"].push_back({cfg.w, cfg.h, cfg.k});
        j["tensors_to_retain"].push_back(
            std::vector<size_t>(step.retain_these.begin(), step.retain_these.end()));

        if (cfg.snake != SnakeDir::None) {
            int ntw = (int)(step.subgraph.output_width() / cfg.w);
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
