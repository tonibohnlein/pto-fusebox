// cross_validate.cpp — analytical vs tile-by-tile simulation
// Verifies compute_cost() against a step-by-step simulation that
// mirrors the reference evaluator (main.cpp) exactly.
// Build: make cross_validate

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "core/cost.h"
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <vector>

// Tile-by-tile simulation matching the reference evaluator's logic.
static CostResult simulate(const Subgraph& sg, TileConfig cfg,
    const FlatSet<size_t>& rfp, const FlatSet<size_t>& rt)
{
    CostResult result;
    result.config = cfg;
    const auto& prob = sg.problem();
    result.working_set = sg.working_set(cfg, rfp, rt);
    if (result.working_set > prob.fast_memory_capacity) return result;
    result.feasible = true;

    double B = (double)prob.slow_memory_bandwidth;
    const auto& bi = sg.boundary_inputs();
    const auto& bo = sg.boundary_outputs();

    // Build internal topology
    FlatSet<size_t> ops_set(sg.ops().begin(), sg.ops().end());
    std::map<size_t, FlatSet<size_t>> op_succs;
    for (auto i : sg.ops()) {
        op_succs[i] = {};
        for (auto t : prob.ops[i].outputs)
            for (auto j : sg.ops())
                if (j != i)
                    for (auto it : prob.ops[j].inputs)
                        if (it == t) op_succs[i].insert(j);
    }

    // Find sinks
    std::vector<size_t> sinks;
    for (auto i : sg.ops())
        if (op_succs[i].empty()) sinks.push_back(i);

    size_t sink_out = prob.ops[sinks[0]].outputs[0];
    int ntw = std::max((int)(prob.tensors[sink_out].width / cfg.w), 1);
    int nth = std::max((int)(prob.tensors[sink_out].height / cfg.h), 1);

    bool has_pw_sink = false;
    for (auto s : sinks)
        if (prob.ops[s].type == OpType::Pointwise) has_pw_sink = true;

    int64_t output_K = 1;
    if (!has_pw_sink)
        for (auto s : sinks)
            if (prob.ops[s].type == OpType::MatMul) {
                output_K = prob.tensors[prob.ops[s].inputs[0]].width;
                break;
            }

    int d_tiles = has_pw_sink ? 1 : std::max((int)(output_K / cfg.k), 1);
    result.num_spatial_tiles = ntw * nth;
    result.num_k_passes = d_tiles;

    // Build reverse topo (sinks first)
    std::map<size_t, int> out_deg;
    for (auto i : sg.ops()) out_deg[i] = (int)op_succs[i].size();
    std::vector<size_t> rev_topo;
    {
        std::vector<size_t> q;
        for (auto i : sg.ops()) if (out_deg[i] == 0) q.push_back(i);
        while (!q.empty()) {
            size_t u = q.back(); q.pop_back();
            rev_topo.push_back(u);
            for (auto j : sg.ops())
                if (op_succs[j].count(u)) {
                    out_deg[j]--;
                    if (out_deg[j] == 0) q.push_back(j);
                }
        }
    }

    // Reference-style tile-by-tile simulation
    std::map<size_t, int> h_tiles, v_tiles;
    std::map<size_t, int> h_pos, v_pos, h_last, v_last;

    // Seed sink output
    h_tiles[sink_out] = ntw;
    v_tiles[sink_out] = nth;

    bool first_tile = true;
    double total = 0;
    auto traversal = make_traversal(ntw, nth, cfg.snake);

    for (int tp = 0; tp < (int)traversal.size(); tp++) {
        h_pos[sink_out] = traversal[tp] / ntw;
        v_pos[sink_out] = traversal[tp] % ntw;

        double regular_cost = 0;
        for (int d = 0; d < d_tiles; d++) {
            if (d >= 2 && d < d_tiles - 1) continue;  // optimization

            double io = 0, compute = 0;
            for (auto op_idx : rev_topo) {
                const auto& op = prob.ops[op_idx];
                size_t out = op.outputs[0];
                bool is_sink = op_succs[op_idx].empty();

                // Propagate h/v tiles and positions
                if (op.type == OpType::Pointwise) {
                    for (auto t : op.inputs) {
                        if (first_tile) { h_tiles[t] = h_tiles[out]; v_tiles[t] = v_tiles[out]; }
                        h_pos[t] = h_pos[out]; v_pos[t] = v_pos[out];
                    }
                } else if (!is_sink || d_tiles == 1) {
                    size_t lhs = op.inputs[0], rhs = op.inputs[1];
                    if (first_tile) {
                        h_tiles[lhs] = 1; v_tiles[lhs] = v_tiles[out];
                        h_tiles[rhs] = h_tiles[out]; v_tiles[rhs] = 1;
                    }
                    h_pos[lhs] = 0; v_pos[lhs] = v_pos[out];
                    h_pos[rhs] = h_pos[out]; v_pos[rhs] = 0;
                } else {
                    size_t lhs = op.inputs[0], rhs = op.inputs[1];
                    if (first_tile) {
                        h_tiles[lhs] = d_tiles; v_tiles[lhs] = v_tiles[out];
                        h_tiles[rhs] = h_tiles[out]; v_tiles[rhs] = d_tiles;
                    }
                    h_pos[lhs] = d; v_pos[lhs] = v_pos[out];
                    h_pos[rhs] = h_pos[out]; v_pos[rhs] = d;
                }

                // IO: input loads
                for (auto t : op.inputs) {
                    if (!bi.count(t) || rfp.count(t)) continue;
                    bool needs_load = first_tile;
                    if (!first_tile && h_last.count(t))
                        needs_load = (h_last[t] != h_pos[t] || v_last[t] != v_pos[t]);
                    if (needs_load) {
                        int ht = std::max(h_tiles.count(t) ? h_tiles[t] : 1, 1);
                        int vt = std::max(v_tiles.count(t) ? v_tiles[t] : 1, 1);
                        io += (double)(prob.tensors[t].width / ht) *
                              (double)(prob.tensors[t].height / vt) / B;
                    }
                }
                // IO: output evictions
                for (auto t : op.outputs) {
                    if (!bo.count(t) || rt.count(t)) continue;
                    if (d_tiles > 1 && d < d_tiles - 1) continue;
                    int ht = std::max(h_tiles.count(t) ? h_tiles[t] : 1, 1);
                    int vt = std::max(v_tiles.count(t) ? v_tiles[t] : 1, 1);
                    io += (double)(prob.tensors[t].width / ht) *
                          (double)(prob.tensors[t].height / vt) / B;
                }

                // Compute: only SINK ops divide by d_tiles (temporal split-K).
                // Non-sink ops execute once per tile regardless of nk.
                size_t co = op.outputs[0];
                int ht = std::max(h_tiles.count(co) ? h_tiles[co] : 1, 1);
                int vt = std::max(v_tiles.count(co) ? v_tiles[co] : 1, 1);
                double sw = (double)prob.tensors[co].width / ht;
                double sh = (double)prob.tensors[co].height / vt;
                double scale = std::max(sw / cfg.w, 1.0) * std::max(sh / cfg.h, 1.0);
                double nk_adj = is_sink ? (double)d_tiles : 1.0;
                compute += (double)op.base_cost / nk_adj * scale;

                for (auto t : op.inputs) { h_last[t] = h_pos[t]; v_last[t] = v_pos[t]; }
            }

            double step = std::max(compute, io);
            total += step;
            if (d == 1) regular_cost = step;
            first_tile = false;
        }
        if (d_tiles > 3) total += (double)(d_tiles - 3) * regular_cost;
    }

    result.latency = total;
    return result;
}

struct TC { std::string nm; Problem p; std::vector<size_t> ops; TileConfig cfg; FlatSet<size_t> rfp, rt; };

int main() {
    int pass = 0, fail = 0;
    auto run = [&](const TC& tc) {
        DAG dag = DAG::build(tc.p);
        auto sg = Subgraph::create(tc.p, dag, tc.ops);
        if (!sg) { std::cout << "SKIP " << tc.nm << "\n"; return; }
        auto a = sg->compute_cost(tc.cfg, tc.rfp, tc.rt);
        auto s = simulate(*sg, tc.cfg, tc.rfp, tc.rt);
        if (!a.feasible && !s.feasible) { pass++; return; }
        if (a.feasible != s.feasible) { fail++; std::cout << "FAIL " << tc.nm
            << ": feasibility mismatch a=" << a.feasible << " s=" << s.feasible << "\n"; return; }
        if (std::abs(a.latency - s.latency) < 0.01) pass++;
        else { fail++; std::cout << "FAIL " << tc.nm << ": a=" << a.latency << " s=" << s.latency << "\n"; }
    };

    auto N = SnakeDir::None; auto R = SnakeDir::RowMajor; auto C = SnakeDir::ColMajor;

    // PROBLEM.md examples
    { Problem p; p.tensors={{128,128},{128,128},{128,128}};
      p.ops={{OpType::Pointwise,{0},{1},1000},{OpType::Pointwise,{1},{2},100}};
      p.fast_memory_capacity=35000;p.slow_memory_bandwidth=10;p.native_w=128;p.native_h=128;
      run({"1A0",p,{0},{128,128,1,N},{},{}});run({"1A1",p,{1},{128,128,1,N},{},{}});
      run({"1B",p,{0,1},{128,128,1,N},{},{}});run({"1C",p,{0,1},{64,64,1,N},{},{}}); }
    { Problem p; p.tensors={{256,256},{256,256},{256,256}};
      p.ops={{OpType::Pointwise,{0},{1},1000},{OpType::Pointwise,{1},{2},100}};
      p.fast_memory_capacity=25000;p.slow_memory_bandwidth=10;p.native_w=128;p.native_h=128;
      run({"2A",p,{0},{128,128,1,N},{},{}});run({"2B",p,{0,1},{128,128,1,N},{},{}}); }
    { Problem p; p.tensors={{128,128},{128,128},{128,128},{128,128}};
      p.ops={{OpType::Pointwise,{0},{1},1500},{OpType::Pointwise,{1},{2},1500},{OpType::Pointwise,{1,2},{3},1500}};
      p.fast_memory_capacity=50000;p.slow_memory_bandwidth=10;p.native_w=128;p.native_h=128;
      run({"3B0",p,{0,1},{128,128,1,N},{},{2}});run({"3B1",p,{0,2},{128,128,1,N},{2},{}});
      run({"3C0",p,{0},{128,128,1,N},{},{1}});run({"3C1",p,{1,2},{128,128,1,N},{1},{}}); }
    { Problem p; p.tensors={{128,128},{128,128},{128,128}};
      p.ops={{OpType::MatMul,{0,1},{2},1500}};
      p.fast_memory_capacity=25000;p.slow_memory_bandwidth=10;p.native_w=128;p.native_h=128;
      run({"4A",p,{0},{64,64,128,N},{},{}});run({"4Br",p,{0},{64,64,128,R},{},{}});
      run({"4Bc",p,{0},{64,64,128,C},{},{}}); }
    { Problem p; p.tensors={{128,128},{128,128},{128,128},{128,128},{128,128}};
      p.ops={{OpType::MatMul,{0,1},{3},2000},{OpType::MatMul,{3,2},{4},2000}};
      p.fast_memory_capacity=45000;p.slow_memory_bandwidth=10;p.native_w=128;p.native_h=128;
      run({"5B",p,{0,1},{128,128,32,N},{},{}}); }
    // Larger grids
    auto mm=[](int64_t sz,int64_t c,int64_t b){Problem p;p.tensors={{sz,sz},{sz,sz},{sz,sz}};
      p.ops={{OpType::MatMul,{0,1},{2},2000}};p.fast_memory_capacity=c;p.slow_memory_bandwidth=b;
      p.native_w=128;p.native_h=128;return p;};
    { auto p=mm(256,30000,10);
      run({"m256n",p,{0},{64,64,128,N},{},{}});run({"m256r",p,{0},{64,64,128,R},{},{}});
      run({"m256c",p,{0},{64,64,128,C},{},{}});run({"m256k64",p,{0},{64,64,64,R},{},{}});
      run({"m256k32",p,{0},{64,64,32,R},{},{}}); }
    { auto p=mm(512,200000,20);
      run({"m512n",p,{0},{128,128,128,N},{},{}});run({"m512r",p,{0},{128,128,128,R},{},{}});
      run({"m512c",p,{0},{128,128,128,C},{},{}});run({"m512k64",p,{0},{128,128,64,R},{},{}});
      run({"m512k32",p,{0},{128,128,32,R},{},{}}); }
    { auto p=mm(512,50000,20);
      run({"ms512n",p,{0},{64,64,128,N},{},{}});run({"ms512r",p,{0},{64,64,128,R},{},{}});
      run({"ms512c",p,{0},{64,64,128,C},{},{}});run({"ms512k64",p,{0},{64,64,64,R},{},{}});
      run({"ms512k32r",p,{0},{64,64,32,R},{},{}});run({"ms512k32c",p,{0},{64,64,32,C},{},{}}); }
    { auto p=mm(1024,200000,25);
      run({"m1k128",p,{0},{128,128,128,R},{},{}});run({"m1k64r",p,{0},{128,128,64,R},{},{}});
      run({"m1k64c",p,{0},{128,128,64,C},{},{}});run({"m1k32",p,{0},{128,128,32,R},{},{}}); }
    // Non-square
    { Problem p;p.tensors={{512,512},{256,512},{256,512}};
      p.ops={{OpType::MatMul,{0,1},{2},3000}};p.fast_memory_capacity=200000;p.slow_memory_bandwidth=20;
      p.native_w=128;p.native_h=128;
      run({"rectn",p,{0},{128,128,128,N},{},{}});run({"rectr",p,{0},{128,128,128,R},{},{}});
      run({"rectc",p,{0},{128,128,128,C},{},{}});run({"rect64r",p,{0},{64,128,128,R},{},{}});
      run({"rect64c",p,{0},{64,128,128,C},{},{}}); }
    // Fused MM+PW (PW sink, nk=1)
    { Problem p;p.tensors={{256,256},{256,256},{256,256},{256,256}};
      p.ops={{OpType::MatMul,{0,1},{2},2000},{OpType::Pointwise,{2},{3},500}};
      p.fast_memory_capacity=100000;p.slow_memory_bandwidth=15;p.native_w=128;p.native_h=128;
      run({"fusen",p,{0,1},{128,128,1,N},{},{}});run({"fuser",p,{0,1},{128,128,1,R},{},{}});
      run({"fusec",p,{0,1},{128,128,1,C},{},{}});run({"fuse64",p,{0,1},{64,64,1,R},{},{}});
      run({"fusek128",p,{0,1},{128,128,128,R},{},{}}); }
    // Retained
    { Problem p;p.tensors={{256,256},{256,256},{256,256}};
      p.ops={{OpType::MatMul,{0,1},{2},2000}};p.fast_memory_capacity=200000;p.slow_memory_bandwidth=10;
      p.native_w=128;p.native_h=128;
      run({"retln",p,{0},{128,128,128,N},{0},{}});run({"retlr",p,{0},{128,128,128,R},{0},{}});
      run({"retrn",p,{0},{128,128,128,N},{1},{}});run({"retrr",p,{0},{128,128,128,R},{1},{}});
      run({"reton",p,{0},{128,128,128,N},{},{2}});run({"retor",p,{0},{128,128,128,R},{},{2}}); }
    // PW
    { Problem p;p.tensors={{512,512},{512,512}};
      p.ops={{OpType::Pointwise,{0},{1},1000}};p.fast_memory_capacity=100000;p.slow_memory_bandwidth=10;
      p.native_w=128;p.native_h=128;
      run({"pwn",p,{0},{128,128,1,N},{},{}});run({"pwr",p,{0},{128,128,1,R},{},{}});
      run({"pwc",p,{0},{128,128,1,C},{},{}}); }
    // Fan-in and chain of 3
    { Problem p;p.tensors={{128,128},{128,128},{128,128},{128,128},{128,128},{128,128},{128,128}};
      p.ops={{OpType::MatMul,{0,1},{2},2000},{OpType::MatMul,{3,4},{5},2000},{OpType::MatMul,{2,5},{6},2000}};
      p.fast_memory_capacity=70000;p.slow_memory_bandwidth=10;p.native_w=128;p.native_h=128;
      run({"fanin32",p,{0,1,2},{128,128,32,N},{},{}});
      run({"fanin64",p,{0,1,2},{128,128,64,N},{},{}}); }
    { Problem p;p.tensors={{128,128},{128,128},{128,128},{128,128},{128,128},{128,128},{128,128}};
      p.ops={{OpType::MatMul,{0,1},{2},2000},{OpType::MatMul,{2,3},{4},2000},{OpType::MatMul,{4,5},{6},2000}};
      p.fast_memory_capacity=60000;p.slow_memory_bandwidth=10;p.native_w=128;p.native_h=128;
      run({"chain3k32",p,{0,1,2},{128,128,32,N},{},{}}); }

    std::cout << "\n" << pass << " passed, " << fail << " failed out of " << (pass+fail) << " tests\n";
    return fail > 0 ? 1 : 0;
}