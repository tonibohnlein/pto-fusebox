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

# 910B machine config (matches set_910b in the C++ tests). These are the GROUNDED
# coefficients of the current model (pto-isa cycles @ 1.85 GHz): the cube clock, the
# six per-port GiB/s bandwidths, the L0 fractal tile, and the vector startup/drain +
# per-element slopes. Key order mirrors the emitted JSON so regeneration is churn-free.
CFG = dict(
    fast_memory_capacity=1 << 30,
    num_cube_cores=24, num_vector_cores=48,
    cube_capacity=131072, l1_capacity=524288, vec_capacity=196608,
    kernel_fill_cost=10000,
    cube_freq_hz=1.85e9,                                              # 910B cube clock
    bw_gm_l1=135.0, bw_l0c_gm=70.0, bw_l1_l0a=441.0, bw_l1_l0b=220.5,  # cube-path GiB/s
    bw_gm_ub=100.9, bw_ub_gm=188.46,                                 # vector-path GiB/s
    l0_tile_m=128, l0_tile_n=256,                                    # L0 fractal tile
    vec_reg_bytes=256, vec_op_head=14.0, vec_op_tail=18.0,           # vector startup/drain
    vec_slope_pw=2.0, vec_slope_reduce=14.0)                         # per-element slopes (pw, reduce)

MM, PW, RED = "MatMul", "Pointwise", "Reduction"


def emit(name, tensors, ops, cube_compute_cost=1, **overrides):
    """tensors: [(w,h)]; ops: [(type, [inputs], [outputs])].
    cube_compute_cost is a grounded unit multiplier (=1); compute-bound-ness now comes
    from geometry (MAC count vs DDR bytes), not this knob.
    overrides: per-instance CFG overrides (e.g. vec_capacity=16384 for a tight-UB
    architecture) applied on top of the shared CFG."""
    widths = [w for w, _ in tensors]
    heights = [h for _, h in tensors]
    j = dict(widths=widths, heights=heights, op_types=[t for t, _, _ in ops],
             inputs=[ins for _, ins, _ in ops], outputs=[outs for _, _, outs in ops],
             cube_compute_cost=cube_compute_cost, **CFG)
    j.update(overrides)
    path = os.path.join(OUT, f"910b-{name}.json")
    with open(path, "w") as f:
        json.dump(j, f, indent=1)
        f.write("\n")
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

t, o = chain(256, 256, 64, 512);  emit("chain-no-ksplit", t, o)
t, o = chain(1024, 8192, 64, 1024); emit("chain-internal-k", t, o)
t, o = chain(64, 128, 512, 64);   emit("chain-parallel-k", t, o)

# (6) Deep matmul chain (8 cube ops), D=M=256.
tns = [(256, 256)]
ops = []
prev = 0
for _ in range(8):
    tns.append((256, 256)); w = len(tns) - 1
    tns.append((256, 256)); out = len(tns) - 1
    ops.append((MM, [prev, w], [out])); prev = out
emit("deep-chain-8mm", tns, ops)

# (7) Mixed cube+vector kernels (rendered with mlsys_mixed = Ascend910BMixed).
#     Memory-bound matmul (cube_compute_cost=64) — the realistic MM->PW regime.
# MM -> PW:  C = A@B; D = relu(C).  The canonical fused mixed kernel. Memory-bound
# (small square K=256 matmul) — the realistic MM->PW regime.
emit("mixed-mm-pw",
     [(256, 256), (256, 256), (256, 256), (256, 256)],  # A[K,M] B[N,K] C[N,M] D
     [(MM, [0, 1], [2]), (PW, [2], [3])])
# MM -> MM -> PW:  T1=A@B; C=T1@D; E=relu(C).  T1 is a held cube L1 operand band.
emit("mixed-mm-mm-pw",
     [(256, 256), (1024, 256), (1024, 256), (256, 1024), (256, 256), (256, 256)],
     [(MM, [0, 1], [2]), (MM, [2, 3], [4]), (PW, [4], [5])])
# MM -> PW -> PW:  C=A@B; T=relu(C); D=gelu(T).  T is a held vector UB band.
emit("mixed-mm-pw-pw",
     [(256, 256), (256, 256), (256, 256), (256, 256), (256, 256)],
     [(MM, [0, 1], [2]), (PW, [2], [3]), (PW, [3], [4])])
# Compute-bound MM->PW: a deep contraction (K=4096) at a modest [256,256] output —
# MACs (~2.7e8) dwarf the DDR round-trip, so the cube dominates and the vector stage
# is fully hidden under the overlap. Grounded compute-bound-ness comes from geometry,
# not a cost knob (the memory-bound mixed-mm-pw above shares the [256,256] output).
emit("mixed-mm-pw-compute",
     [(4096, 256), (256, 4096), (256, 256), (256, 256)],  # A[K=4096,M] B[N,K=4096] C[N,M] D
     [(MM, [0, 1], [2]), (PW, [2], [3])])
# Tight-UB architecture (UB 16KB): the shared tile is forced small, the matmul
# reload blows up, so the mixed solver picks SEPARATE over fusion — the
# fuse-vs-separate decision driven by the on-chip memory size.
emit("mixed-mm-pw-tight-ub",
     [(256, 256), (256, 256), (256, 256), (256, 256)],
     [(MM, [0, 1], [2]), (PW, [2], [3])], vec_capacity=16384)
# 2-layer MLP: X@W1 -> relu -> @W2 -> relu -> @W3.  Each matmul fuses its relu
# epilogue (mixed kernel); the final matmul has no epilogue.
emit("mixed-mlp",
     [(512, 128), (2048, 512), (2048, 128), (2048, 128),   # X W1 H1 A1=relu(H1)
      (512, 2048), (512, 128), (512, 128),                 # W2 H2 A2=relu(H2)
      (2048, 512), (2048, 128)],                           # W3 Y
     [(MM, [0, 1], [2]), (PW, [2], [3]),
      (MM, [3, 4], [5]), (PW, [5], [6]),
      (MM, [6, 7], [8])])
