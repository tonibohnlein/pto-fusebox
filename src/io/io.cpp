#include "io/io.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cctype>
#include <map>

struct JsonParser {
    std::string src;
    size_t pos = 0;
    void skip_ws() { while (pos < src.size() && std::isspace(src[pos])) pos++; }
    char peek() { skip_ws(); return pos < src.size() ? src[pos] : 0; }
    char next() { skip_ws(); return pos < src.size() ? src[pos++] : 0; }
    void expect(char c) { if (next() != c) throw std::runtime_error(std::string("Expected '") + c + "'"); }
    std::string parse_string() {
        expect('"');
        std::string r;
        while (pos < src.size() && src[pos] != '"') r += src[pos++];
        pos++;
        return r;
    }
    double parse_number() {
        skip_ws();
        size_t start = pos;
        if (pos < src.size() && src[pos] == '-') pos++;
        while (pos < src.size() && (std::isdigit(src[pos]) || src[pos] == '.' || src[pos] == 'e' || src[pos] == 'E')) pos++;
        return std::stod(src.substr(start, pos - start));
    }
    std::vector<int64_t> parse_int_array() {
        std::vector<int64_t> r;
        expect('[');
        if (peek() != ']') { r.push_back((int64_t)parse_number()); while (peek() == ',') { next(); r.push_back((int64_t)parse_number()); } }
        expect(']');
        return r;
    }
    std::vector<std::vector<int64_t>> parse_2d_int_array() {
        std::vector<std::vector<int64_t>> r;
        expect('[');
        if (peek() != ']') { r.push_back(parse_int_array()); while (peek() == ',') { next(); r.push_back(parse_int_array()); } }
        expect(']');
        return r;
    }
    std::vector<std::string> parse_string_array() {
        std::vector<std::string> r;
        expect('[');
        if (peek() != ']') { r.push_back(parse_string()); while (peek() == ',') { next(); r.push_back(parse_string()); } }
        expect(']');
        return r;
    }
};

Problem read_problem(const std::string& filename) {
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot open " + filename);
    std::stringstream ss;
    ss << f.rdbuf();
    JsonParser p;
    p.src = ss.str();
    p.expect('{');

    Problem prob;
    prob.fast_memory_capacity = 0;
    prob.slow_memory_bandwidth = 1;
    prob.native_w = 128;
    prob.native_h = 128;

    std::vector<int64_t> widths, heights, base_costs;
    std::vector<std::vector<int64_t>> inputs, outputs;
    std::vector<std::string> op_types;

    while (p.peek() != '}') {
        std::string key = p.parse_string();
        p.expect(':');
        if (key == "widths") widths = p.parse_int_array();
        else if (key == "heights") heights = p.parse_int_array();
        else if (key == "inputs") inputs = p.parse_2d_int_array();
        else if (key == "outputs") outputs = p.parse_2d_int_array();
        else if (key == "base_costs") base_costs = p.parse_int_array();
        else if (key == "op_types") op_types = p.parse_string_array();
        else if (key == "fast_memory_capacity") prob.fast_memory_capacity = (int64_t)p.parse_number();
        else if (key == "slow_memory_bandwidth") prob.slow_memory_bandwidth = (int64_t)p.parse_number();
        else if (key == "native_granularity") {
            auto g = p.parse_int_array();
            if (g.size() >= 2) { prob.native_w = g[0]; prob.native_h = g[1]; }
        }
        if (p.peek() == ',') p.next();
    }
    p.expect('}');

    size_t n_tensors = widths.size();
    prob.tensors.resize(n_tensors);
    for (size_t i = 0; i < n_tensors; i++) {
        prob.tensors[i].width = widths[i];
        prob.tensors[i].height = heights[i];
    }

    size_t n_ops = op_types.size();
    prob.ops.resize(n_ops);
    for (size_t i = 0; i < n_ops; i++) {
        prob.ops[i].type = (op_types[i] == "MatMul") ? OpType::MatMul : OpType::Pointwise;
        prob.ops[i].base_cost = base_costs[i];
        if (i < inputs.size())
            for (auto t : inputs[i]) prob.ops[i].inputs.push_back((size_t)t);
        if (i < outputs.size())
            for (auto t : outputs[i]) prob.ops[i].outputs.push_back((size_t)t);
    }

    std::set<size_t> produced, consumed;
    std::map<size_t, int> consumer_count;
    for (auto& op : prob.ops) {
        for (auto t : op.outputs) produced.insert(t);
        for (auto t : op.inputs) {
            consumed.insert(t);
            consumer_count[t]++;
        }
    }

    for (auto t : produced) {
        if (!consumed.count(t)) continue;
        if (prob.tensors[t].size() > prob.fast_memory_capacity) continue;
        // Single-consumer produced tensors: still retainable (useful when
        // producer and consumer are in different steps, e.g. T5, T11 in custom)
        prob.retainable_tensors.insert(t);
    }

    // Graph inputs with >1 consumer: retainable (saves re-reading from slow
    // memory across steps). Single-consumer graph inputs: not retainable
    // (only read once, nothing to save).
    for (size_t t = 0; t < prob.num_tensors(); t++) {
        if (produced.count(t)) continue;           // already handled above
        if (!consumed.count(t)) continue;           // not used at all
        if (consumer_count[t] <= 1) continue;       // single consumer → useless
        if (prob.tensors[t].size() > prob.fast_memory_capacity) continue;
        prob.retainable_tensors.insert(t);
    }

    return prob;
}

void write_solution(const std::string& filename, const Solution& sol) {
    std::ofstream f(filename);
    f << "{\n  \"steps\": [\n";
    auto& steps = sol.steps();
    for (size_t i = 0; i < steps.size(); i++) {
        f << "    {\"ops\": [";
        auto ops = steps[i].subgraph.ops();
        for (size_t j = 0; j < ops.size(); j++) {
            f << ops[j]; if (j + 1 < ops.size()) f << ", ";
        }
        f << "], \"retain\": [";
        size_t k = 0;
        for (auto t : steps[i].retain_these) { f << t; if (++k < steps[i].retain_these.size()) f << ", "; }
        f << "]}";
        if (i + 1 < steps.size()) f << ",";
        f << "\n";
    }
    f << "  ],\n  \"total_latency\": " << sol.total_latency() << "\n}\n";
}