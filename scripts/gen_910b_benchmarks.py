#!/usr/bin/env python3
"""Generate 910B benchmark JSON instances for the cost-model regression cases.

These mirror the C++ instances in test/ascend_910b_test.cpp so the fusion-DAG
visualizations (fusion_dag/*.png) cover the cases we added: realistic model
stages (FFN, attention, transformer block), the chained-2mm k-split variants,
large-W streaming softmax, and a deep matmul chain. Run, then render with
scripts/render_fusion_dag.sh.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "benchmarks")

# 910B machine config (matches set_910b in the C++ tests).
CFG = dict(
    fast_memory_capacity=1 << 30, slow_memory_bandwidth=10, native_granularity=[128, 128],
    num_cube_cores=24, num_vector_cores=48, cube_capacity=131072, l1_capacity=524288,
    vec_capacity=196608, vector_compute_cost=1, vector_lanes=256)

MM, PW, RED = "MatMul", "Pointwise", "Reduction"


def emit(name, tensors, ops, cube_compute_cost=4096):
    """tensors: [(w,h)]; ops: [(type, [inputs], [outputs])]."""
    widths = [w for w, _ in tensors]
    heights = [h for _, h in tensors]
    base_costs = []
    for t, ins, outs in ops:
        if t == MM:
            base_costs.append(16384 * tensors[ins[0]][0])  # 16384 * K
        else:
            ow, oh = tensors[outs[0]]
            base_costs.append(ow * oh)
    j = dict(widths=widths, heights=heights, op_types=[t for t, _, _ in ops],
             inputs=[ins for _, ins, _ in ops], outputs=[outs for _, _, outs in ops],
             base_costs=base_costs, cube_compute_cost=cube_compute_cost, **CFG)
    path = os.path.join(OUT, f"910b-{name}.json")
    with open(path, "w") as f:
        json.dump(j, f, indent=1)
    print(f"wrote {path}")


# (1) Transformer block: scores -> softmax(4) -> PV -> @Wo -> @W1 -> relu -> @W2
emit("transformer-block",
     [(512, 128), (256, 512), (256, 128), (1, 128), (256, 128), (1, 128), (256, 128),
      (512, 256), (512, 128), (512, 512), (512, 128), (2048, 512), (2048, 128),
      (2048, 128), (512, 2048), (512, 128)],
     [(MM, [0, 1], [2]), (RED, [2], [3]), (PW, [2, 3], [4]), (RED, [4], [5]),
      (PW, [4, 5], [6]), (MM, [6, 7], [8]), (MM, [8, 9], [10]), (MM, [10, 11], [12]),
      (PW, [12], [13]), (MM, [13, 14], [15])])

# (2) FFN at batch 128: X@W1 -> relu -> @W2  (d=512, d_ff=2048)
emit("ffn-b128",
     [(512, 128), (2048, 512), (2048, 128), (2048, 128), (512, 2048), (512, 128)],
     [(MM, [0, 1], [2]), (PW, [2], [3]), (MM, [3, 4], [5])])

# (3) Attention at batch 128: Q@K^T -> softmax(over keys) -> @V  (d=64, S=2048)
emit("attention-b128",
     [(64, 128), (2048, 64), (2048, 128), (1, 128), (2048, 128), (1, 128), (2048, 128),
      (64, 2048), (64, 128)],
     [(MM, [0, 1], [2]), (RED, [2], [3]), (PW, [2, 3], [4]), (RED, [4], [5]),
      (PW, [4, 5], [6]), (MM, [6, 7], [8])])

# (4) Large-W streaming softmax (W=32768): must stream the reduced axis.
emit("softmax-stream-32k",
     [(32768, 128), (1, 128), (32768, 128), (1, 128), (32768, 128)],
     [(RED, [0], [1]), (PW, [0, 1], [2]), (RED, [2], [3]), (PW, [2, 3], [4])])


# (5) Chained-2mm k-split variants (C=(A@B)@D). cube_compute_cost per the test.
def chain(M, K1, N1, N2):
    # A[K1,M] B[N1,K1] -> T[N1,M]; T,D[N2,N1] -> C[N2,M]
    return ([(K1, M), (N1, K1), (N1, M), (N2, N1), (N2, M)],
            [(MM, [0, 1], [2]), (MM, [2, 3], [4])])

t, o = chain(256, 256, 64, 512);  emit("chain-no-ksplit", t, o, cube_compute_cost=1000)
t, o = chain(1024, 8192, 64, 1024); emit("chain-internal-k", t, o, cube_compute_cost=4096)
t, o = chain(64, 128, 512, 64);   emit("chain-parallel-k", t, o, cube_compute_cost=1000)

# (6) Deep matmul chain (8 cube ops), D=M=256.
tns = [(256, 256)]
ops = []
prev = 0
for _ in range(8):
    tns.append((256, 256)); w = len(tns) - 1
    tns.append((256, 256)); out = len(tns) - 1
    ops.append((MM, [prev, w], [out])); prev = out
emit("deep-chain-8mm", tns, ops)
