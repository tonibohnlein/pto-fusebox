// cross_validate.cpp — analytical vs tile-by-tile simulation
// Build: make cross_validate

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "core/cost.h"
#include <cmath>
#include <iostream>
#include <set>

static CostResult simulate(const Subgraph& sg, TileConfig cfg,
    const std::set<size_t>& rfp, const std::set<size_t>& rt)
{
    CostResult result;
    result.config = cfg;
    const auto& prob = sg.problem();
    result.working_set = sg.working_set(cfg, rfp, rt);
    if (result.working_set > prob.fast_memory_capacity) return result;
    result.feasible = true;

    double B = (double)prob.slow_memory_bandwidth;
    int ntw = (int)(sg.output_width() / cfg.w);
    int nth = (int)(sg.output_height() / cfg.h);
    int nk = sg.has_matmul() ? (int)(sg.max_K() / cfg.k) : 1;
    result.num_spatial_tiles = ntw * nth;
    result.num_k_passes = nk;

    auto ceil_div = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
    int64_t scale = ceil_div(cfg.w, prob.native_w)
                  * ceil_div(cfg.h, prob.native_h);

    // Separate MM compute (per k-step) from PW compute (once per tile)
    double mm_comp = 0, pw_comp = 0;
    for (auto i : sg.ops()) {
        double c = (double)prob.ops[i].base_cost;
        if (prob.ops[i].type == OpType::MatMul)
            mm_comp += c * ((double)cfg.k / sg.op_K(i));
        else
            pw_comp += c;
    }
    mm_comp *= (double)scale;
    pw_comp *= (double)scale;

    const auto& bi = sg.boundary_inputs();
    double total = 0; int pr = -1, pc = -1;

    for (int tp = 0; tp < (int)tiles.size(); tp++) {
        int tr = tiles[tp] / ntw, tc = tiles[tp] % ntw;
        for (int ks = 0; ks < nk; ks++) {
            double mi = 0, mo = 0;
            std::set<size_t> xfer_done;  // dedup shared boundary inputs
            for (auto i : sg.ops()) {
                const auto& op = prob.ops[i];
                if (op.type == OpType::MatMul) {
                    size_t lhs = op.inputs[0], rhs = op.inputs[1];
                    if (bi.count(lhs) && !rfp.count(lhs) && !xfer_done.count(lhs)) {
                        if (ks == 0 && (tp == 0 || tr != pr)) { mi += (double)(cfg.h * sg.op_K(i)) / B; xfer_done.insert(lhs); }
                    }
                    if (bi.count(rhs) && !rfp.count(rhs) && !xfer_done.count(rhs)) {
                        if (ks > 0 || (ks == 0 && (tp == 0 || tc != pc || nk > 1))) { mi += (double)(cfg.k * cfg.w) / B; xfer_done.insert(rhs); }
                    }
                } else {
                    if (ks == 0) for (auto t : op.inputs)
                        if (bi.count(t) && !rfp.count(t) && !xfer_done.count(t)) {
                            mi += (double)(cfg.h * cfg.w) / B;
                            xfer_done.insert(t);
                        }
                }
            }
            if (ks == nk - 1) {
                for (auto t : sg.boundary_outputs())
                    if (!rt.count(t))
                        mo += (double)(cfg.h * cfg.w) / B;
            }
            double step_comp = (ks == nk - 1) ? mm_comp + pw_comp : mm_comp;
            total += std::max(step_comp, mi + mo);
        }
        pr = tr; pc = tc;
    }
    result.latency = total;
    return result;
}

struct TC { std::string nm; Problem p; std::vector<size_t> ops; TileConfig cfg; std::set<size_t> rfp, rt; };

int main() {
    int pass = 0, fail = 0;
    auto run = [&](const TC& tc) {
        DAG dag = DAG::build(tc.p);
        auto sg = Subgraph::create(tc.p, dag, tc.ops);
        if (!sg) { std::cout << "SKIP " << tc.nm << "\n"; return; }
        auto a = sg->compute_cost(tc.cfg, tc.rfp, tc.rt);
        auto s = simulate(*sg, tc.cfg, tc.rfp, tc.rt);
        if (!a.feasible && !s.feasible) pass++;
        else if (a.feasible && s.feasible && std::abs(a.latency - s.latency) < 0.01) pass++;
        else { fail++; std::cout << "FAIL " << tc.nm << ": a=" << a.latency << " s=" << s.latency << " (fea=" << a.feasible << "," << s.feasible << ")\n"; }
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
    // Fused MM+PW (PW sink rule forces k=1)
    { Problem p;p.tensors={{256,256},{256,256},{256,256},{256,256}};
      p.ops={{OpType::MatMul,{0,1},{2},2000},{OpType::Pointwise,{2},{3},500}};
      p.fast_memory_capacity=100000;p.slow_memory_bandwidth=15;p.native_w=128;p.native_h=128;
      run({"fusen",p,{0,1},{128,128,1,N},{},{}});run({"fuser",p,{0,1},{128,128,1,R},{},{}});
      run({"fusec",p,{0,1},{128,128,1,C},{},{}});run({"fuse64",p,{0,1},{64,64,1,R},{},{}}); }
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

    std::cout << "\n" << pass << " passed, " << fail << " failed out of " << (pass+fail) << " tests\n";
    return fail > 0 ? 1 : 0;
}