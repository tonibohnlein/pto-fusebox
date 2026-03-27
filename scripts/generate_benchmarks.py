#!/usr/bin/env python3
"""
Expanded benchmark generator for MLSys 2026 Track A.

Coverage matrix — each benchmark targets specific solver stress dimensions:

  GAP COVERED                      | BENCHMARKS
  ---------------------------------+---------------------------------------------
  Scale stress (200+ ops)          | stacked_transformer, large_moe, mega_gnn
  Non-power-of-2 dims              | llama_mlp, bert_block, gpt_neo_attention
  Extreme aspect ratio             | tall_skinny_chain, embedding_lookup
  Pure PW graph                    | pointwise_only_dag
  Multi-output graph               | multi_head_output, hydra_net
  Recompute-dominant               | cheap_producer_diamond, recompute_ladder
  Very large tensors               | large_tensor_matmul, stacked_transformer
  Extreme tight memory             | ultra_tight_splitk
  Asymmetric native granularity    | asymmetric_native_*
  Stacked/nested blocks            | stacked_transformer
  Wide + deep                      | mega_gnn, large_moe
  Sparse/irregular connectivity    | mega_gnn
  Mixed dimension graph            | mixed_scale_unet
  Bandwidth extremes               | ultra_low_bw, ultra_high_bw
  Many shared inputs               | multi_shared_weights
"""
import json
import os


class BenchmarkBuilder:
    def __init__(self, cap, bw, native_w=128, native_h=128):
        self.widths = []
        self.heights = []
        self.op_types = []
        self.inputs = []
        self.outputs = []
        self.base_costs = []
        self.cap = cap
        self.bw = bw
        self.native = [native_w, native_h]

    def tensor(self, w, h):
        idx = len(self.widths)
        self.widths.append(w)
        self.heights.append(h)
        return idx

    def matmul(self, lhs, rhs, out, base_cost):
        idx = len(self.op_types)
        self.op_types.append("MatMul")
        self.inputs.append([lhs, rhs])
        self.outputs.append([out])
        self.base_costs.append(base_cost)
        return idx

    def pointwise(self, ins, out, base_cost):
        idx = len(self.op_types)
        self.op_types.append("Pointwise")
        self.inputs.append(ins if isinstance(ins, list) else [ins])
        self.outputs.append([out])
        self.base_costs.append(base_cost)
        return idx

    def linear(self, x, weight, out, base_cost):
        return self.matmul(x, weight, out, base_cost)

    def to_json(self):
        return {
            "widths": self.widths,
            "heights": self.heights,
            "op_types": self.op_types,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "base_costs": self.base_costs,
            "fast_memory_capacity": self.cap,
            "slow_memory_bandwidth": self.bw,
            "native_granularity": self.native,
        }

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_json(), f, indent=2)
        n_ops = len(self.op_types)
        n_tensors = len(self.widths)
        mm = sum(1 for t in self.op_types if t == "MatMul")
        pw = n_ops - mm
        sizes = sorted(set(w * h for w, h in zip(self.widths, self.heights)))
        print(f"  {os.path.basename(path):45s} {n_ops:4d} ops ({mm:3d} MM + {pw:3d} PW), "
              f"{n_tensors:4d} tensors, cap={self.cap:>9d}, bw={self.bw:>4d}, "
              f"native={self.native}")


# ============================================================================
# GAP 1: Scale stress (200+ ops)
# ============================================================================

def stacked_transformer(n_blocks=4, seq=256, d_model=384, d_ff=1536,
                        n_heads=6, cap=200000, bw=30, name="stacked-xformer"):
    """
    Multiple stacked transformer blocks. Exercises:
    - 200+ ops (realistic LLM-scale DAG)
    - Non-power-of-2 dims (384, 1536)
    - Large tensors (384×256 = 98K per activation)
    - Deep DAG with cross-block retention opportunities
    """
    d_head = d_model // n_heads
    b = BenchmarkBuilder(cap, bw)
    X = b.tensor(d_model, seq)
    current = X

    for blk in range(n_blocks):
        # Multi-head self-attention
        head_outputs = []
        for hd in range(n_heads):
            W_Q = b.tensor(d_head, d_model)
            W_K = b.tensor(d_head, d_model)
            W_V = b.tensor(d_head, d_model)
            Q = b.tensor(d_head, seq)
            K = b.tensor(d_head, seq)
            V = b.tensor(d_head, seq)
            bc = max(500, seq * d_model * d_head // 10000)
            b.linear(current, W_Q, Q, bc)
            b.linear(current, W_K, K, bc)
            b.linear(current, W_V, V, bc)

            QK = b.tensor(d_head, seq)
            b.pointwise([Q, K], QK, max(100, seq * d_head // 5000))
            W_A = b.tensor(seq, d_head)
            A = b.tensor(seq, seq)
            b.linear(QK, W_A, A, max(500, seq * seq * d_head // 10000))
            S = b.tensor(seq, seq)
            b.pointwise([A], S, max(200, seq * seq // 5000))
            O = b.tensor(d_head, seq)
            b.linear(S, V, O, max(500, seq * seq * d_head // 10000))
            head_outputs.append(O)

        # Combine heads
        combined = head_outputs[0]
        for i in range(1, len(head_outputs)):
            nxt = b.tensor(d_head, seq)
            b.pointwise([combined, head_outputs[i]], nxt, 200)
            combined = nxt

        W_O = b.tensor(d_model, d_head)
        Y = b.tensor(d_model, seq)
        b.linear(combined, W_O, Y, max(500, seq * d_model * d_model // 10000))
        R1 = b.tensor(d_model, seq)
        b.pointwise([current, Y], R1, max(200, seq * d_model // 5000))

        # FFN
        W1 = b.tensor(d_ff, d_model)
        H = b.tensor(d_ff, seq)
        b.linear(R1, W1, H, max(500, seq * d_model * d_ff // 10000))
        G = b.tensor(d_ff, seq)
        b.pointwise([H], G, max(200, seq * d_ff // 5000))
        W2 = b.tensor(d_model, d_ff)
        F = b.tensor(d_model, seq)
        b.linear(G, W2, F, max(500, seq * d_ff * d_model // 10000))
        R2 = b.tensor(d_model, seq)
        b.pointwise([R1, F], R2, max(200, seq * d_model // 5000))
        current = R2

    b.save(f"benchmarks/{name}.json")


def large_moe(seq=128, d_model=256, d_expert=512, n_experts=8,
              cap=80000, bw=20, name="large-moe"):
    """
    MoE with 8 experts — tests massive parallelism + fan-in.
    ~100 ops, wide and moderately deep.
    """
    b = BenchmarkBuilder(cap, bw)
    X = b.tensor(d_model, seq)

    # Router
    W_R = b.tensor(n_experts, d_model)
    R_raw = b.tensor(n_experts, seq)
    b.linear(X, W_R, R_raw, max(200, seq * d_model * n_experts // 10000))
    R = b.tensor(n_experts, seq)
    b.pointwise([R_raw], R, max(100, seq * n_experts // 1000))

    expert_outputs = []
    for i in range(n_experts):
        W1 = b.tensor(d_expert, d_model)
        H = b.tensor(d_expert, seq)
        b.linear(X, W1, H, max(500, seq * d_model * d_expert // 10000))
        G = b.tensor(d_expert, seq)
        b.pointwise([H], G, max(200, seq * d_expert // 5000))
        W2 = b.tensor(d_model, d_expert)
        E = b.tensor(d_model, seq)
        b.linear(G, W2, E, max(500, seq * d_expert * d_model // 10000))
        expert_outputs.append(E)

    # Project router scores to model dimension
    W_R_proj = b.tensor(d_model, n_experts)
    R_proj = b.tensor(d_model, seq)
    b.matmul(R, W_R_proj, R_proj, max(200, seq * n_experts * d_model // 10000))

    # Combine expert outputs
    combined = expert_outputs[0]
    for i in range(1, len(expert_outputs)):
        tmp = b.tensor(d_model, seq)
        b.pointwise([combined, expert_outputs[i]], tmp,
                     max(200, seq * d_model // 5000))
        combined = tmp

    # Mix with router projection
    Y = b.tensor(d_model, seq)
    b.pointwise([combined, R_proj], Y, max(200, seq * d_model // 2000))
    Z = b.tensor(d_model, seq)
    b.pointwise([X, Y], Z, max(200, seq * d_model // 5000))
    b.save(f"benchmarks/{name}.json")


# ============================================================================
# GAP 2: Non-power-of-2 dimensions (realistic ML dims)
# ============================================================================

def llama_mlp(seq=256, d_model=768, d_ff=2048, cap=120000, bw=25,
              name="llama-mlp"):
    """
    LLaMA-style MLP: gate projection + up projection + SiLU + down projection.
    Key: d_model=768 is NOT a power of 2 (divisors: 1,2,3,4,...,768).
    Forces tiling to find compatible divisors across 768 and 2048.
    """
    b = BenchmarkBuilder(cap, bw)
    X = b.tensor(d_model, seq)  # 768×256

    W_gate = b.tensor(d_ff, d_model)  # 2048×768
    W_up = b.tensor(d_ff, d_model)    # 2048×768
    W_down = b.tensor(d_model, d_ff)  # 768×2048

    gate = b.tensor(d_ff, seq)  # 2048×256
    up = b.tensor(d_ff, seq)

    b.linear(X, W_gate, gate, max(500, seq * d_model * d_ff // 10000))
    b.linear(X, W_up, up, max(500, seq * d_model * d_ff // 10000))

    # SiLU(gate) * up
    gate_act = b.tensor(d_ff, seq)
    b.pointwise([gate], gate_act, max(200, seq * d_ff // 5000))
    combined = b.tensor(d_ff, seq)
    b.pointwise([gate_act, up], combined, max(200, seq * d_ff // 5000))

    # Down projection
    out = b.tensor(d_model, seq)
    b.linear(combined, W_down, out, max(500, seq * d_ff * d_model // 10000))

    # Residual
    result = b.tensor(d_model, seq)
    b.pointwise([X, out], result, max(200, seq * d_model // 5000))

    b.save(f"benchmarks/{name}.json")


def bert_block(seq=384, d_model=768, d_ff=3072, n_heads=12,
               cap=150000, bw=30, name="bert-block"):
    """
    BERT-style transformer block with non-pow2 dimensions.
    seq=384, d_model=768, d_head=64, d_ff=3072.
    384 is divisible by 128 (3 tiles) but NOT 256 → interesting tiling constraints.
    """
    d_head = d_model // n_heads  # 64
    b = BenchmarkBuilder(cap, bw)
    X = b.tensor(d_model, seq)

    # Single-head attention for simplicity (12 heads would be 200+ ops)
    # Use 4 representative heads
    head_outs = []
    for _ in range(4):
        W_Q = b.tensor(d_head, d_model)
        W_K = b.tensor(d_head, d_model)
        W_V = b.tensor(d_head, d_model)
        Q = b.tensor(d_head, seq)
        K = b.tensor(d_head, seq)
        V = b.tensor(d_head, seq)
        bc = max(500, seq * d_model * d_head // 10000)
        b.linear(X, W_Q, Q, bc)
        b.linear(X, W_K, K, bc)
        b.linear(X, W_V, V, bc)
        QK = b.tensor(d_head, seq)
        b.pointwise([Q, K], QK, 200)
        W_A = b.tensor(seq, d_head)
        A = b.tensor(seq, seq)
        b.linear(QK, W_A, A, max(500, seq * seq * d_head // 10000))
        S = b.tensor(seq, seq)
        b.pointwise([A], S, max(200, seq * seq // 5000))
        O = b.tensor(d_head, seq)
        b.linear(S, V, O, max(500, seq * seq * d_head // 10000))
        head_outs.append(O)

    combined = head_outs[0]
    for i in range(1, len(head_outs)):
        nxt = b.tensor(d_head, seq)
        b.pointwise([combined, head_outs[i]], nxt, 200)
        combined = nxt

    W_O = b.tensor(d_model, d_head)
    Y = b.tensor(d_model, seq)
    b.linear(combined, W_O, Y, max(500, seq * d_model ** 2 // 10000))
    R1 = b.tensor(d_model, seq)
    b.pointwise([X, Y], R1, 200)

    # FFN
    W1 = b.tensor(d_ff, d_model)
    H = b.tensor(d_ff, seq)
    b.linear(R1, W1, H, max(500, seq * d_model * d_ff // 10000))
    G = b.tensor(d_ff, seq)
    b.pointwise([H], G, max(200, seq * d_ff // 5000))
    W2 = b.tensor(d_model, d_ff)
    F = b.tensor(d_model, seq)
    b.linear(G, W2, F, max(500, seq * d_ff * d_model // 10000))
    R2 = b.tensor(d_model, seq)
    b.pointwise([R1, F], R2, max(200, seq * d_model // 5000))

    b.save(f"benchmarks/{name}.json")


# ============================================================================
# GAP 3: Extreme aspect ratios
# ============================================================================

def tall_skinny_chain(n_ops=8, seq=4096, d=64, cap=50000, bw=15,
                      name="tall-skinny"):
    """
    Tall-skinny tensors (4096×64 = 262K). Common in long-context attention
    where sequence length >> head dimension.
    Forces many spatial tiles in one axis, very few in the other.
    """
    b = BenchmarkBuilder(cap, bw)
    prev = b.tensor(d, seq)  # 64×4096

    for i in range(n_ops):
        if i % 3 == 0:
            # MM: project to wider dim then back
            W = b.tensor(d * 4, d)  # 256×64
            H = b.tensor(d * 4, seq)  # 256×4096
            b.matmul(prev, W, H, max(500, seq * d * d * 4 // 10000))
            prev = H
        elif i % 3 == 1:
            out = b.tensor(d * 4, seq)
            b.pointwise([prev], out, max(200, seq * d * 4 // 5000))
            prev = out
        else:
            W = b.tensor(d, d * 4)  # 64×256
            out = b.tensor(d, seq)  # 64×4096
            b.matmul(prev, W, out, max(500, seq * d * 4 * d // 10000))
            prev = out

    b.save(f"benchmarks/{name}.json")


def embedding_lookup(vocab=8192, d_model=256, seq=128, n_downstream=4,
                     cap=80000, bw=20, name="embedding"):
    """
    Embedding-style: very wide weight matrix (8192×256).
    Input tensor is 8192×128 (huge), output is 256×128 (small).
    Extreme ratio between input and output sizes.
    """
    b = BenchmarkBuilder(cap, bw)
    X = b.tensor(vocab, seq)  # 8192×128

    # Multiple parallel projections from the large input
    proj_outs = []
    for i in range(n_downstream):
        W = b.tensor(d_model, vocab)
        O = b.tensor(d_model, seq)
        b.linear(X, W, O, max(500, seq * vocab * d_model // 10000))
        A = b.tensor(d_model, seq)
        b.pointwise([O], A, max(200, seq * d_model // 5000))
        proj_outs.append(A)

    combined = proj_outs[0]
    for i in range(1, len(proj_outs)):
        out = b.tensor(d_model, seq)
        b.pointwise([combined, proj_outs[i]], out, 200)
        combined = out

    b.save(f"benchmarks/{name}.json")


# ============================================================================
# GAP 4: Pure pointwise graph
# ============================================================================

def pointwise_only_dag(n_layers=5, width=512, seq=256, branches=3,
                       cap=60000, bw=15, name="pure-pw"):
    """
    Graph with ONLY Pointwise ops — no MatMul at all.
    Tests: k=1 always, grouping purely based on memory and data flow,
    no split-K decisions, snake traversal irrelevant.
    Diamond patterns where all intermediates are ephemeral.
    """
    b = BenchmarkBuilder(cap, bw)
    X = b.tensor(width, seq)
    current = X

    for layer in range(n_layers):
        branch_outs = []
        for br in range(branches):
            H = b.tensor(width, seq)
            b.pointwise([current], H, max(100, width * seq // 5000))
            branch_outs.append(H)

        # Pairwise combine
        combined = branch_outs[0]
        for i in range(1, len(branch_outs)):
            out = b.tensor(width, seq)
            b.pointwise([combined, branch_outs[i]], out,
                         max(100, width * seq // 5000))
            combined = out

        # Residual
        if layer > 0:
            res = b.tensor(width, seq)
            b.pointwise([current, combined], res, max(100, width * seq // 5000))
            current = res
        else:
            current = combined

    b.save(f"benchmarks/{name}.json")


# ============================================================================
# GAP 5: Multi-output graph
# ============================================================================

def multi_head_output(n_heads=8, d_model=256, seq=128,
                      cap=50000, bw=15, name="multi-output"):
    """
    Graph with 8 independent graph outputs (no consumer for each head output).
    Shared encoder → parallel output heads.
    Tests: solution must cover all 8 output paths, scheduling flexibility.
    """
    b = BenchmarkBuilder(cap, bw)
    X = b.tensor(d_model, seq)

    # Shared encoder
    W_enc = b.tensor(d_model, d_model)
    H = b.tensor(d_model, seq)
    b.linear(X, W_enc, H, max(500, seq * d_model ** 2 // 10000))
    A = b.tensor(d_model, seq)
    b.pointwise([H], A, max(200, seq * d_model // 5000))

    # N independent output heads — each produces a graph output
    for i in range(n_heads):
        d_out = 64 + 32 * (i % 4)  # varying output dims
        W_head = b.tensor(d_out, d_model)
        head_out = b.tensor(d_out, seq)  # this has no consumer → graph output
        b.linear(A, W_head, head_out,
                 max(300, seq * d_model * d_out // 10000))

    b.save(f"benchmarks/{name}.json")


def hydra_net(n_branches=6, depth=3, d=192, seq=128,
              cap=45000, bw=12, name="hydra-net"):
    """
    Hydra-net: shared trunk + multiple branching output paths.
    Wider than multi_head_output with deeper branches.
    """
    b = BenchmarkBuilder(cap, bw)
    X = b.tensor(d, seq)

    # Shared trunk (2 layers)
    W1 = b.tensor(d, d)
    H = b.tensor(d, seq)
    b.linear(X, W1, H, max(500, seq * d * d // 10000))
    trunk = b.tensor(d, seq)
    b.pointwise([H], trunk, max(200, seq * d // 5000))

    # Independent branches, each producing a graph output
    for br in range(n_branches):
        prev = trunk
        d_br = d + 32 * (br % 3)
        for step in range(depth):
            d_in = d if step == 0 else d_br
            d_out = d_br
            W = b.tensor(d_out, d_in)
            out = b.tensor(d_out, seq)
            b.matmul(prev, W, out, max(400, seq * d_in * d_out // 10000))
            act = b.tensor(d_out, seq)
            b.pointwise([out], act, max(100, seq * d_out // 5000))
            prev = act
        # No further consumer → graph output

    b.save(f"benchmarks/{name}.json")


# ============================================================================
# GAP 6: Recompute-dominant
# ============================================================================

def cheap_producer_diamond(n_diamonds=6, dim=256, cap=40000, bw=5,
                           name="recompute-diamond"):
    """
    Diamond graphs where the producer op is very cheap but the tensor is large.
    With bw=5 (very slow memory), recomputing the cheap op is much cheaper
    than spilling to slow memory.
    Tests: solver should prefer recomputation over materialization.
    """
    b = BenchmarkBuilder(cap, bw)
    X = b.tensor(dim, dim)
    current = X

    for i in range(n_diamonds):
        # Cheap producer
        produced = b.tensor(dim, dim)
        b.pointwise([current], produced, 50)  # very cheap!

        # Two expensive consumers (diamond)
        W1 = b.tensor(dim, dim)
        out1 = b.tensor(dim, dim)
        b.matmul(produced, W1, out1, 2000)

        W2 = b.tensor(dim, dim)
        out2 = b.tensor(dim, dim)
        b.matmul(produced, W2, out2, 2000)

        # Merge
        merged = b.tensor(dim, dim)
        b.pointwise([out1, out2], merged, 200)
        current = merged

    b.save(f"benchmarks/{name}.json")


def recompute_ladder(n_rungs=8, dim=128, cap=20000, bw=3,
                     name="recompute-ladder"):
    """
    Ladder graph: two parallel chains connected at every rung.
    With extremely low bandwidth, solver must recompute shared intermediates.
    """
    b = BenchmarkBuilder(cap, bw)
    left = b.tensor(dim, dim)
    right = b.tensor(dim, dim)

    for i in range(n_rungs):
        # Left chain step
        Wl = b.tensor(dim, dim)
        Hl = b.tensor(dim, dim)
        b.matmul(left, Wl, Hl, 500)
        Al = b.tensor(dim, dim)
        b.pointwise([Hl], Al, 100)

        # Right chain step
        Wr = b.tensor(dim, dim)
        Hr = b.tensor(dim, dim)
        b.matmul(right, Wr, Hr, 500)
        Ar = b.tensor(dim, dim)
        b.pointwise([Hr], Ar, 100)

        # Cross-connection (rung)
        new_left = b.tensor(dim, dim)
        b.pointwise([Al, Ar], new_left, 80)
        new_right = b.tensor(dim, dim)
        b.pointwise([Al, Ar], new_right, 80)

        left = new_left
        right = new_right

    # Final merge
    result = b.tensor(dim, dim)
    b.pointwise([left, right], result, 100)

    b.save(f"benchmarks/{name}.json")


# ============================================================================
# GAP 7: Very large tensors
# ============================================================================

def large_tensor_matmul(dim=8192, cap=2000000, bw=100,
                        name="large-tensor"):
    """
    Single large MatMul chain with 8192×8192 tensors (67M elements each!).
    Forces many spatial tiles and aggressive split-K.
    Tests: tiling enumeration with large divisor sets, cost model at scale.
    """
    b = BenchmarkBuilder(cap, bw)
    A = b.tensor(dim, dim)
    W1 = b.tensor(dim, dim)
    T1 = b.tensor(dim, dim)
    b.matmul(A, W1, T1, 10000)

    # PW activation
    T1a = b.tensor(dim, dim)
    b.pointwise([T1], T1a, 2000)

    # Second MM
    W2 = b.tensor(dim, dim)
    T2 = b.tensor(dim, dim)
    b.matmul(T1a, W2, T2, 10000)

    # Residual
    out = b.tensor(dim, dim)
    b.pointwise([A, T2], out, 2000)

    b.save(f"benchmarks/{name}.json")


# ============================================================================
# GAP 8: Extreme tight memory
# ============================================================================

def ultra_tight_splitk(n_ops=6, dim=512, cap=18000, bw=10,
                       name="ultra-tight"):
    """
    Memory so tight that even a single 512×512 MM at native granularity
    (128×128 slices) BARELY fits. Forces aggressive split-K everywhere.
    3 slices of 128×128 = 49152 > 18000. Must use smaller tiles.
    """
    b = BenchmarkBuilder(cap, bw)
    prev = b.tensor(dim, dim)

    for i in range(n_ops):
        W = b.tensor(dim, dim)
        out = b.tensor(dim, dim)
        b.matmul(prev, W, out, 1000 + i * 300)
        prev = out

    b.save(f"benchmarks/{name}.json")


# ============================================================================
# GAP 9: Asymmetric native granularity
# ============================================================================

def asymmetric_native_attention(seq=256, d=128, cap=40000, bw=15,
                                name="asym-native-attn"):
    """
    Attention-like pattern with asymmetric native granularity 256×64.
    Tests: tiling must align with non-square native, different padding
    penalties along each axis.
    """
    b = BenchmarkBuilder(cap, bw, native_w=256, native_h=64)
    X = b.tensor(d, seq)

    W_Q = b.tensor(d, d)
    W_K = b.tensor(d, d)
    W_V = b.tensor(d, d)
    Q = b.tensor(d, seq)
    K = b.tensor(d, seq)
    V = b.tensor(d, seq)
    b.linear(X, W_Q, Q, max(500, seq * d * d // 10000))
    b.linear(X, W_K, K, max(500, seq * d * d // 10000))
    b.linear(X, W_V, V, max(500, seq * d * d // 10000))

    QK = b.tensor(d, seq)
    b.pointwise([Q, K], QK, 200)
    W_A = b.tensor(seq, d)
    A = b.tensor(seq, seq)
    b.linear(QK, W_A, A, max(500, seq * seq * d // 10000))
    S = b.tensor(seq, seq)
    b.pointwise([A], S, 200)
    O = b.tensor(d, seq)
    b.linear(S, V, O, max(500, seq * seq * d // 10000))

    W_O = b.tensor(d, d)
    Y = b.tensor(d, seq)
    b.linear(O, W_O, Y, max(500, seq * d * d // 10000))
    R = b.tensor(d, seq)
    b.pointwise([X, Y], R, 200)

    b.save(f"benchmarks/{name}.json")


def asymmetric_native_chain(n_ops=10, dim=256, cap=30000, bw=10,
                            name="asym-native-chain"):
    """
    MM/PW chain with native 64×256 — inverted asymmetry.
    """
    b = BenchmarkBuilder(cap, bw, native_w=64, native_h=256)
    prev = b.tensor(dim, dim)

    for i in range(n_ops):
        if i % 2 == 0:
            W = b.tensor(dim, dim)
            out = b.tensor(dim, dim)
            b.matmul(prev, W, out, max(500, dim ** 3 // 10000))
        else:
            out = b.tensor(dim, dim)
            b.pointwise([prev], out, max(200, dim ** 2 // 5000))
        prev = out

    b.save(f"benchmarks/{name}.json")


# ============================================================================
# GAP 10: Wide + deep simultaneously
# ============================================================================

def mega_gnn(n_layers=6, n_neighbors=5, d=128, n_nodes=4,
             cap=50000, bw=15, name="mega-gnn"):
    """
    GNN-like: each "node" has a feature vector, updated by aggregating
    from neighbors. Irregular connectivity: not all nodes connect to all others.
    Both wide (n_nodes * n_neighbors paths per layer) and deep (n_layers).
    ~200+ ops with sparse, irregular structure.
    """
    b = BenchmarkBuilder(cap, bw)

    # Initialize node features
    node_feats = []
    for n in range(n_nodes):
        feat = b.tensor(d, d)
        node_feats.append(feat)

    # Define irregular adjacency (not fully connected)
    import random
    random.seed(42)
    adjacency = {}
    for n in range(n_nodes):
        # Each node connects to 2-4 others (wrapping)
        n_nbrs = 2 + (n % 3)
        nbrs = []
        for k in range(n_nbrs):
            target = (n + k + 1) % n_nodes
            if target != n:
                nbrs.append(target)
        adjacency[n] = list(set(nbrs))

    for layer in range(n_layers):
        new_feats = []
        W_self = b.tensor(d, d)  # shared self-weight per layer
        W_nbr = b.tensor(d, d)   # shared neighbor weight per layer

        for n in range(n_nodes):
            # Self transform
            self_out = b.tensor(d, d)
            b.matmul(node_feats[n], W_self, self_out,
                     max(300, d ** 3 // 50000))

            # Aggregate from neighbors
            nbr_outs = []
            for nbr in adjacency[n]:
                nbr_out = b.tensor(d, d)
                b.matmul(node_feats[nbr], W_nbr, nbr_out,
                         max(300, d ** 3 // 50000))
                nbr_outs.append(nbr_out)

            # Combine: self + mean(neighbors)
            combined = self_out
            for no in nbr_outs:
                tmp = b.tensor(d, d)
                b.pointwise([combined, no], tmp, max(100, d * d // 5000))
                combined = tmp

            # Activation
            activated = b.tensor(d, d)
            b.pointwise([combined], activated, max(100, d * d // 5000))
            new_feats.append(activated)

        node_feats = new_feats

    # Final readout: combine all node features
    combined = node_feats[0]
    for i in range(1, n_nodes):
        out = b.tensor(d, d)
        b.pointwise([combined, node_feats[i]], out, max(100, d * d // 5000))
        combined = out

    b.save(f"benchmarks/{name}.json")


# ============================================================================
# GAP 11: Mixed dimension graph (U-Net style)
# ============================================================================

def mixed_scale_unet(cap=100000, bw=20, name="mixed-unet"):
    """
    U-Net-like: encoder (128→256→512→1024) then decoder (1024→512→256→128)
    with skip connections. Adjacent subgraph candidates have very different
    tensor sizes → interesting tiling incompatibilities.
    """
    b = BenchmarkBuilder(cap, bw)
    seq = 256

    # Encoder path: increasing channel dim
    dims = [128, 256, 512, 1024]
    enc_features = []
    X = b.tensor(dims[0], seq)
    current = X

    for i in range(len(dims) - 1):
        d_in, d_out = dims[i], dims[i + 1]
        W = b.tensor(d_out, d_in)
        H = b.tensor(d_out, seq)
        b.linear(current, W, H, max(500, seq * d_in * d_out // 10000))
        A = b.tensor(d_out, seq)
        b.pointwise([H], A, max(200, seq * d_out // 5000))
        enc_features.append(current)  # save for skip connection
        current = A

    # Bottleneck
    d_bottle = dims[-1]
    W_b = b.tensor(d_bottle, d_bottle)
    H_b = b.tensor(d_bottle, seq)
    b.linear(current, W_b, H_b, max(500, seq * d_bottle ** 2 // 10000))
    current = H_b

    # Decoder path: decreasing channel dim with skip connections
    for i in range(len(dims) - 2, -1, -1):
        d_in = dims[i + 1] if i < len(dims) - 1 else d_bottle
        d_out = dims[i]
        # Project down
        W = b.tensor(d_out, d_in)
        H = b.tensor(d_out, seq)
        b.linear(current, W, H, max(500, seq * d_in * d_out // 10000))
        # Skip connection (add encoder feature)
        skip = enc_features[i]
        combined = b.tensor(d_out, seq)
        b.pointwise([H, skip], combined, max(200, seq * d_out // 5000))
        A = b.tensor(d_out, seq)
        b.pointwise([combined], A, max(200, seq * d_out // 5000))
        current = A

    b.save(f"benchmarks/{name}.json")


# ============================================================================
# GAP 12: Bandwidth extremes
# ============================================================================

def ultra_low_bw(n_ops=8, dim=256, cap=80000, bw=1, name="ultra-low-bw"):
    """
    Extremely low bandwidth (bw=1). Almost everything is memory-bound.
    Grouping to eliminate intermediate IO is critical.
    Tests: solver must aggressively group to minimize boundary crossings.
    """
    b = BenchmarkBuilder(cap, bw)
    prev = b.tensor(dim, dim)

    for i in range(n_ops):
        if i % 2 == 0:
            W = b.tensor(dim, dim)
            out = b.tensor(dim, dim)
            b.matmul(prev, W, out, max(500, dim ** 3 // 10000))
        else:
            out = b.tensor(dim, dim)
            b.pointwise([prev], out, max(200, dim ** 2 // 5000))
        prev = out

    b.save(f"benchmarks/{name}.json")


def ultra_high_bw(n_ops=12, dim=512, cap=60000, bw=500, name="ultra-high-bw"):
    """
    Very high bandwidth (bw=500). Memory transfers are nearly free.
    Everything is compute-bound → grouping has minimal IO benefit,
    but compute costs from sub-native tiling become the bottleneck.
    Tests: solver should prefer native-sized tiles, minimize spatial tiling waste.
    """
    b = BenchmarkBuilder(cap, bw)
    prev = b.tensor(dim, dim)

    for i in range(n_ops):
        if i % 3 == 0:
            W = b.tensor(dim, dim)
            out = b.tensor(dim, dim)
            b.matmul(prev, W, out, 3000)
        elif i % 3 == 1:
            out = b.tensor(dim, dim)
            b.pointwise([prev], out, 500)
        else:
            # Fan-out + merge: diamond pattern
            W1 = b.tensor(dim, dim)
            o1 = b.tensor(dim, dim)
            b.matmul(prev, W1, o1, 3000)
            o1a = b.tensor(dim, dim)
            b.pointwise([o1], o1a, 500)
            prev = o1a
            continue
        prev = out

    b.save(f"benchmarks/{name}.json")


# ============================================================================
# GAP 13: Many shared inputs (retention stress)
# ============================================================================

def multi_shared_weights(n_stages=6, d=256, seq=128,
                         cap=50000, bw=10, name="multi-shared"):
    """
    Three weight matrices are shared across all stages.
    Retaining all three would save massive IO but may not fit simultaneously.
    Tests: solver must choose which shared weights to retain, tradeoff analysis.
    """
    b = BenchmarkBuilder(cap, bw)

    # Three shared weights
    W_a = b.tensor(d, d)  # size = 256×256 = 65536
    W_b = b.tensor(d, d)
    W_c = b.tensor(d // 2, d)

    X = b.tensor(d, seq)
    prev = X

    for s in range(n_stages):
        # Path A: prev @ W_a
        Ha = b.tensor(d, seq)
        b.matmul(prev, W_a, Ha, max(500, seq * d * d // 10000))
        Aa = b.tensor(d, seq)
        b.pointwise([Ha], Aa, max(200, seq * d // 5000))

        # Path B: prev @ W_b
        Hb = b.tensor(d, seq)
        b.matmul(prev, W_b, Hb, max(500, seq * d * d // 10000))
        Ab = b.tensor(d, seq)
        b.pointwise([Hb], Ab, max(200, seq * d // 5000))

        # Combine A+B
        combined = b.tensor(d, seq)
        b.pointwise([Aa, Ab], combined, max(200, seq * d // 5000))

        # Project down with shared W_c
        proj = b.tensor(d // 2, seq)
        b.matmul(combined, W_c, proj, max(300, seq * d * (d // 2) // 10000))

        # Project back up
        W_up = b.tensor(d, d // 2)
        up = b.tensor(d, seq)
        b.matmul(proj, W_up, up, max(300, seq * (d // 2) * d // 10000))

        # Residual
        res = b.tensor(d, seq)
        b.pointwise([prev, up], res, max(200, seq * d // 5000))
        prev = res

    b.save(f"benchmarks/{name}.json")


# ============================================================================
# GAP 14: Deep chain with many partition points (combinatorial)
# ============================================================================

def deep_alternating_chain(n_ops=40, dim=128, cap=25000, bw=8,
                           name="deep-chain-40"):
    """
    Very long alternating MM/PW chain (40 ops).
    Each cut point is a valid partition boundary → exponentially many partitions.
    Tests: search must efficiently explore the huge partition space.
    """
    b = BenchmarkBuilder(cap, bw)
    prev = b.tensor(dim, dim)

    for i in range(n_ops):
        if i % 2 == 0:
            W = b.tensor(dim, dim)
            out = b.tensor(dim, dim)
            b.matmul(prev, W, out, max(300, dim ** 3 // 100000))
        else:
            out = b.tensor(dim, dim)
            b.pointwise([prev], out, max(100, dim ** 2 // 1000))
        prev = out

    b.save(f"benchmarks/{name}.json")


# ============================================================================
# GAP 15: Stress test — biggest benchmark
# ============================================================================

def mega_transformer(n_blocks=6, seq=512, d_model=768, d_ff=3072,
                     n_heads=4, cap=500000, bw=50, name="mega-xformer"):
    """
    The big one: 6 transformer blocks with 4 heads each.
    ~300+ ops, non-pow2 dims, deep, wide, many retention opportunities.
    This is the stress test for overall solver performance.
    """
    d_head = d_model // n_heads  # 192
    b = BenchmarkBuilder(cap, bw)
    X = b.tensor(d_model, seq)
    current = X

    for blk in range(n_blocks):
        head_outputs = []
        for hd in range(n_heads):
            W_Q = b.tensor(d_head, d_model)
            W_K = b.tensor(d_head, d_model)
            W_V = b.tensor(d_head, d_model)
            Q = b.tensor(d_head, seq)
            K = b.tensor(d_head, seq)
            V = b.tensor(d_head, seq)
            bc = max(500, seq * d_model * d_head // 10000)
            b.linear(current, W_Q, Q, bc)
            b.linear(current, W_K, K, bc)
            b.linear(current, W_V, V, bc)

            QK = b.tensor(d_head, seq)
            b.pointwise([Q, K], QK, 200)
            W_A = b.tensor(seq, d_head)
            A = b.tensor(seq, seq)
            b.linear(QK, W_A, A, max(500, seq * seq * d_head // 10000))
            S = b.tensor(seq, seq)
            b.pointwise([A], S, max(200, seq * seq // 5000))
            O = b.tensor(d_head, seq)
            b.linear(S, V, O, max(500, seq * seq * d_head // 10000))
            head_outputs.append(O)

        combined = head_outputs[0]
        for i in range(1, len(head_outputs)):
            nxt = b.tensor(d_head, seq)
            b.pointwise([combined, head_outputs[i]], nxt, 200)
            combined = nxt

        W_O = b.tensor(d_model, d_head)
        Y = b.tensor(d_model, seq)
        b.linear(combined, W_O, Y, max(500, seq * d_model ** 2 // 10000))
        R1 = b.tensor(d_model, seq)
        b.pointwise([current, Y], R1, 200)

        W1 = b.tensor(d_ff, d_model)
        H = b.tensor(d_ff, seq)
        b.linear(R1, W1, H, max(500, seq * d_model * d_ff // 10000))
        G = b.tensor(d_ff, seq)
        b.pointwise([H], G, max(200, seq * d_ff // 5000))
        W2 = b.tensor(d_model, d_ff)
        F = b.tensor(d_model, seq)
        b.linear(G, W2, F, max(500, seq * d_ff * d_model // 10000))
        R2 = b.tensor(d_model, seq)
        b.pointwise([R1, F], R2, 200)
        current = R2

    b.save(f"benchmarks/{name}.json")


# ============================================================================
# Custom benchmarks (architecture-focused)
# ============================================================================

def custom_transformer_block(seq=256, d_model=512, d_ff=2048, n_heads=8,
                             cap=80000, bw=20, name="transformer"):
    """Single transformer block: MHA + FFN + residuals."""
    d_head = d_model // n_heads
    b = BenchmarkBuilder(cap, bw)
    X = b.tensor(d_model, seq)

    head_outputs = []
    for i in range(n_heads):
        W_Q = b.tensor(d_head, d_model)
        W_K = b.tensor(d_head, d_model)
        W_V = b.tensor(d_head, d_model)
        Q = b.tensor(d_head, seq)
        K = b.tensor(d_head, seq)
        V = b.tensor(d_head, seq)

        bc_proj = max(500, seq * d_model * d_head // 10000)
        b.linear(X, W_Q, Q, bc_proj)
        b.linear(X, W_K, K, bc_proj)
        b.linear(X, W_V, V, bc_proj)

        # QK interaction + score projection
        QK = b.tensor(d_head, seq)
        b.pointwise([Q, K], QK, max(100, seq * d_head // 5000))
        W_A = b.tensor(seq, d_head)
        A = b.tensor(seq, seq)
        bc_attn = max(500, seq * seq * d_head // 10000)
        b.linear(QK, W_A, A, bc_attn)

        S = b.tensor(seq, seq)
        b.pointwise([A], S, max(200, seq * seq // 5000))

        O = b.tensor(d_head, seq)
        b.linear(S, V, O, bc_attn)
        head_outputs.append(O)

    # Combine heads
    combined = head_outputs[0]
    for i in range(1, len(head_outputs)):
        nxt = b.tensor(d_head, seq)
        b.pointwise([combined, head_outputs[i]], nxt, 200)
        combined = nxt

    W_O = b.tensor(d_model, d_head)
    Y = b.tensor(d_model, seq)
    b.linear(combined, W_O, Y, max(500, seq * d_model * d_head // 10000))

    R1 = b.tensor(d_model, seq)
    b.pointwise([X, Y], R1, max(200, seq * d_model // 5000))

    # FFN
    W1 = b.tensor(d_ff, d_model)
    H = b.tensor(d_ff, seq)
    b.linear(R1, W1, H, max(500, seq * d_model * d_ff // 10000))
    G = b.tensor(d_ff, seq)
    b.pointwise([H], G, max(200, seq * d_ff // 5000))
    W2 = b.tensor(d_model, d_ff)
    F = b.tensor(d_model, seq)
    b.linear(G, W2, F, max(500, seq * d_ff * d_model // 10000))

    R2 = b.tensor(d_model, seq)
    b.pointwise([R1, F], R2, max(200, seq * d_model // 5000))

    b.save(f"benchmarks/custom-{name}.json")


def custom_moe_block(seq=128, d_model=512, d_expert=1024, n_experts=4,
                     cap=60000, bw=15, name="moe"):
    """Mixture of Experts: router + parallel expert FFNs + combine."""
    b = BenchmarkBuilder(cap, bw)
    X = b.tensor(d_model, seq)

    # Router
    W_R = b.tensor(n_experts, d_model)
    R_raw = b.tensor(n_experts, seq)
    b.linear(X, W_R, R_raw, max(200, seq * d_model * n_experts // 10000))
    R = b.tensor(n_experts, seq)
    b.pointwise([R_raw], R, max(100, seq * n_experts // 1000))

    # Expert FFNs
    expert_outputs = []
    for i in range(n_experts):
        W1 = b.tensor(d_expert, d_model)
        H = b.tensor(d_expert, seq)
        b.linear(X, W1, H, max(500, seq * d_model * d_expert // 10000))
        G = b.tensor(d_expert, seq)
        b.pointwise([H], G, max(200, seq * d_expert // 5000))
        W2 = b.tensor(d_model, d_expert)
        E = b.tensor(d_model, seq)
        b.linear(G, W2, E, max(500, seq * d_expert * d_model // 10000))
        expert_outputs.append(E)

    # Project router to model dim, then combine with experts
    W_R_proj = b.tensor(d_model, n_experts)
    R_proj = b.tensor(d_model, seq)
    b.matmul(R, W_R_proj, R_proj, max(200, seq * n_experts * d_model // 10000))

    combined = expert_outputs[0]
    for i in range(1, len(expert_outputs)):
        tmp = b.tensor(d_model, seq)
        b.pointwise([combined, expert_outputs[i]], tmp,
                     max(200, seq * d_model // 5000))
        combined = tmp

    Y = b.tensor(d_model, seq)
    b.pointwise([combined, R_proj], Y, max(200, seq * d_model // 2000))
    Z = b.tensor(d_model, seq)
    b.pointwise([X, Y], Z, max(200, seq * d_model // 5000))

    b.save(f"benchmarks/custom-{name}.json")


def custom_deep_residual_chain(n_blocks=6, dim=256, cap=50000, bw=10,
                               name="reschain"):
    """Deep residual chain with skip connections."""
    b = BenchmarkBuilder(cap, bw)
    X = b.tensor(dim, dim)
    current = X

    for i in range(n_blocks):
        W1 = b.tensor(dim, dim)
        H1 = b.tensor(dim, dim)
        bc = max(500, dim * dim * dim // 10000)
        b.linear(current, W1, H1, bc)
        A1 = b.tensor(dim, dim)
        b.pointwise([H1], A1, max(200, dim * dim // 2000))
        W2 = b.tensor(dim, dim)
        H2 = b.tensor(dim, dim)
        b.linear(A1, W2, H2, bc)
        R = b.tensor(dim, dim)
        b.pointwise([current, H2], R, max(200, dim * dim // 2000))
        current = R

    b.save(f"benchmarks/custom-{name}.json")


def custom_asymmetric_bottleneck(seq=512, d_in=1024, d_bottle=64, d_out=1024,
                                 n_branches=3, cap=40000, bw=10,
                                 name="bottleneck"):
    """Bottleneck with extreme dimension changes and self-attention."""
    b = BenchmarkBuilder(cap, bw)
    X = b.tensor(d_in, seq)

    branch_outputs = []
    for i in range(n_branches):
        d_b = d_bottle * (i + 1)

        W_down = b.tensor(d_b, d_in)
        H_down = b.tensor(d_b, seq)
        b.linear(X, W_down, H_down, max(500, seq * d_in * d_b // 10000))

        A = b.tensor(d_b, seq)
        b.pointwise([H_down], A, max(200, seq * d_b // 2000))

        # Self-interaction: A -> scaled -> project to (seq, seq) via weight
        A_scaled = b.tensor(d_b, seq)
        b.pointwise([A], A_scaled, max(100, seq * d_b // 5000))
        W_self = b.tensor(seq, d_b)
        S = b.tensor(seq, seq)
        b.linear(A_scaled, W_self, S, max(300, seq * seq * d_b // 10000))

        P = b.tensor(d_b, seq)
        b.linear(S, A, P, max(300, seq * d_b * seq // 10000))

        W_up = b.tensor(d_out, d_b)
        B_out = b.tensor(d_out, seq)
        b.linear(P, W_up, B_out, max(500, seq * d_b * d_out // 10000))
        branch_outputs.append(B_out)

    combined = branch_outputs[0]
    for i in range(1, len(branch_outputs)):
        out = b.tensor(d_out, seq)
        b.pointwise([combined, branch_outputs[i]], out,
                     max(200, seq * d_out // 2000))
        combined = out

    result = b.tensor(d_out, seq)
    b.pointwise([X, combined], result, max(200, seq * d_out // 2000))

    b.save(f"benchmarks/custom-{name}.json")


def custom_conv_like_block(h=64, w=64, c_in=256, c_out=256, n_layers=4,
                           cap=60000, bw=15, name="convlike"):
    """Conv-like block modeled as MatMul (im2col style)."""
    spatial = h * w
    b = BenchmarkBuilder(cap, bw)
    X = b.tensor(c_in, spatial)
    current = X

    for i in range(n_layers):
        W = b.tensor(c_out, c_in)
        H = b.tensor(c_out, spatial)
        bc = max(500, spatial * c_in * c_out // 10000)
        b.linear(current, W, H, bc)
        A = b.tensor(c_out, spatial)
        b.pointwise([H], A, max(200, spatial * c_out // 5000))

        if i % 2 == 1 and i > 0:
            R = b.tensor(c_out, spatial)
            if c_in == c_out:
                b.pointwise([current, A], R, max(100, spatial * c_out // 5000))
            else:
                W_skip = b.tensor(c_out, c_in)
                proj = b.tensor(c_out, spatial)
                b.linear(current, W_skip, proj,
                         max(200, spatial * c_in * c_out // 20000))
                b.pointwise([proj, A], R, max(100, spatial * c_out // 5000))
            current = R
        else:
            current = A
        c_in = c_out

    b.save(f"benchmarks/custom-{name}.json")


def custom_wide_fan_diamond(n_parallel=8, dim=256, cap=50000, bw=10,
                            name="fandia"):
    """Wide fan-out followed by pairwise reduction tree."""
    b = BenchmarkBuilder(cap, bw)
    X = b.tensor(dim, dim)

    branch_outputs = []
    for i in range(n_parallel):
        W = b.tensor(dim, dim)
        H = b.tensor(dim, dim)
        bc = max(500, dim * dim * dim // 10000)
        b.linear(X, W, H, bc)
        A = b.tensor(dim, dim)
        b.pointwise([H], A, max(200, dim * dim // 2000))
        branch_outputs.append(A)

    level = branch_outputs
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                out = b.tensor(dim, dim)
                b.pointwise([level[i], level[i + 1]], out,
                            max(200, dim * dim // 2000))
                next_level.append(out)
            else:
                next_level.append(level[i])
        level = next_level

    result = b.tensor(dim, dim)
    b.pointwise([X, level[0]], result, max(200, dim * dim // 2000))

    b.save(f"benchmarks/custom-{name}.json")


def custom_tight_memory_chain(n_ops=12, dim=512, cap=35000, bw=5,
                              name="tight"):
    """Chain of MatMuls with very tight memory."""
    b = BenchmarkBuilder(cap, bw)
    current = b.tensor(dim, dim)

    for i in range(n_ops):
        if i % 2 == 0:
            W = b.tensor(dim, dim)
            out = b.tensor(dim, dim)
            bc = 1000 + i * 200
            b.linear(current, W, out, bc)
        else:
            out = b.tensor(dim, dim)
            b.pointwise([current], out, 300 + i * 50)
        current = out

    b.save(f"benchmarks/custom-{name}.json")


def custom_multi_scale_attention(seq=128, dims=[64, 128, 256, 512],
                                 cap=45000, bw=10, name="multiscale"):
    """Multi-scale attention at different resolutions."""
    b = BenchmarkBuilder(cap, bw)
    d_in = max(dims)
    X = b.tensor(d_in, seq)

    scale_outputs = []
    for d in dims:
        W_Q = b.tensor(d, d_in)
        W_K = b.tensor(d, d_in)
        W_V = b.tensor(d, d_in)
        Q = b.tensor(d, seq)
        K = b.tensor(d, seq)
        V = b.tensor(d, seq)

        bc_proj = max(300, seq * d_in * d // 10000)
        b.linear(X, W_Q, Q, bc_proj)
        b.linear(X, W_K, K, bc_proj)
        b.linear(X, W_V, V, bc_proj)

        # QK interaction + score projection
        QK = b.tensor(d, seq)
        b.pointwise([Q, K], QK, max(100, seq * d // 5000))
        W_A = b.tensor(seq, d)
        A = b.tensor(seq, seq)
        bc_attn = max(300, seq * seq * d // 10000)
        b.linear(QK, W_A, A, bc_attn)

        S = b.tensor(seq, seq)
        b.pointwise([A], S, max(100, seq * seq // 5000))
        O = b.tensor(d, seq)
        b.linear(S, V, O, bc_attn)

        W_out = b.tensor(d_in, d)
        O_proj = b.tensor(d_in, seq)
        b.linear(O, W_out, O_proj, bc_proj)
        scale_outputs.append(O_proj)

    combined = scale_outputs[0]
    for i in range(1, len(scale_outputs)):
        out = b.tensor(d_in, seq)
        b.pointwise([combined, scale_outputs[i]], out,
                     max(200, seq * d_in // 2000))
        combined = out

    result = b.tensor(d_in, seq)
    b.pointwise([X, combined], result, max(200, seq * d_in // 2000))

    b.save(f"benchmarks/custom-{name}.json")


def custom_irregular_dag(n_layers=6, branches_per_layer=[3, 2, 4, 2, 3, 2],
                         cap=60000, bw=15, name="irregular-dag"):
    """Irregular DAG with varying fan-out and skip connections."""
    b = BenchmarkBuilder(cap, bw)
    d_model = 256
    X = b.tensor(d_model, 128)
    prev_outputs = [X]

    for layer in range(n_layers):
        n_branches = branches_per_layer[layer % len(branches_per_layer)]
        layer_outputs = []

        for br in range(n_branches):
            inp1 = prev_outputs[br % len(prev_outputs)]
            d_hidden = 128 + 64 * (br % 3)
            W = b.tensor(d_hidden, d_model)
            H = b.tensor(d_hidden, 128)
            bc = max(500, d_model * d_hidden * 128 // 100000)
            b.matmul(inp1, W, H, bc)
            A = b.tensor(d_hidden, 128)
            b.pointwise([H], A, max(100, d_hidden * 128 // 5000))
            W2 = b.tensor(d_model, d_hidden)
            O = b.tensor(d_model, 128)
            b.matmul(A, W2, O, bc)
            layer_outputs.append(O)

        combined = layer_outputs[0]
        for i in range(1, len(layer_outputs)):
            out = b.tensor(d_model, 128)
            b.pointwise([combined, layer_outputs[i]], out,
                         max(150, d_model * 128 // 3000))
            combined = out

        if layer % 2 == 1 and layer < n_layers - 1:
            skip = b.tensor(d_model, 128)
            b.pointwise([combined, X], skip, max(150, d_model * 128 // 3000))
            combined = skip

        prev_outputs = layer_outputs + [combined]

    b.save(f"benchmarks/custom-{name}.json")


def custom_shared_weight_network(n_stages=4, d=256, seq=128,
                                 cap=50000, bw=10, name="shared-weights"):
    """Network with shared weight matrices across stages."""
    b = BenchmarkBuilder(cap, bw)
    W_shared = b.tensor(d, d)
    W_proj = b.tensor(d // 2, d)
    X = b.tensor(d, seq)
    prev = X

    for s in range(n_stages):
        H = b.tensor(d, seq)
        bc_main = max(800, d * d * seq // 50000)
        b.matmul(prev, W_shared, H, bc_main)
        A = b.tensor(d, seq)
        b.pointwise([H], A, max(100, d * seq // 5000))
        O = b.tensor(d // 2, seq)
        bc_proj = max(400, d * (d // 2) * seq // 50000)
        b.matmul(A, W_proj, O, bc_proj)
        W_up = b.tensor(d, d // 2)
        U = b.tensor(d, seq)
        b.matmul(O, W_up, U, bc_proj)
        R = b.tensor(d, seq)
        b.pointwise([prev, U], R, max(200, d * seq // 3000))
        prev = R

    b.save(f"benchmarks/custom-{name}.json")


def custom_deep_narrow_chain(n_ops=25, dim=128, cap=25000, bw=8,
                             name="deep-narrow"):
    """Long alternating MM/PW chain with tight memory."""
    b = BenchmarkBuilder(cap, bw)
    prev = b.tensor(dim, dim)

    for i in range(n_ops):
        if i % 2 == 0:
            W = b.tensor(dim, dim)
            out = b.tensor(dim, dim)
            bc = max(300, dim * dim * dim // 100000)
            b.matmul(prev, W, out, bc)
        else:
            out = b.tensor(dim, dim)
            b.pointwise([prev], out, max(100, dim * dim // 1000))
        prev = out

    b.save(f"benchmarks/custom-{name}.json")


def custom_multi_path_reduction(n_paths=6, depth=3, dim=192, cap=45000,
                                bw=12, name="multipath-reduce"):
    """N independent paths with different depths that merge at the end."""
    b = BenchmarkBuilder(cap, bw)
    X = b.tensor(dim, dim)
    path_outputs = []

    for p in range(n_paths):
        prev = X
        path_depth = depth + (p % 3) - 1
        d_hidden = dim + 32 * (p % 4)
        bc = max(400, dim * d_hidden * dim // 80000)

        for d_idx in range(path_depth):
            in_dim = dim if d_idx == 0 else d_hidden
            out_dim = (d_hidden if d_idx == 0
                       else dim if d_idx == path_depth - 1
                       else d_hidden)
            W = b.tensor(out_dim, in_dim)
            H = b.tensor(out_dim, dim)
            b.matmul(prev, W, H, bc)
            A = b.tensor(out_dim, dim)
            b.pointwise([H], A, max(80, out_dim * dim // 5000))
            prev = A

        path_outputs.append(prev)

    combined = path_outputs[0]
    for i in range(1, len(path_outputs)):
        out = b.tensor(dim, dim)
        b.pointwise([combined, path_outputs[i]], out,
                     max(150, dim * dim // 3000))
        combined = out

    W_final = b.tensor(dim, dim)
    result = b.tensor(dim, dim)
    b.matmul(combined, W_final, result, max(500, dim * dim * dim // 60000))

    b.save(f"benchmarks/custom-{name}.json")


def custom_encoder_decoder(enc_layers=3, dec_layers=3, d=256, seq_enc=128,
                           seq_dec=64, cap=55000, bw=12, name="enc-dec"):
    """Encoder-decoder with cross-attention."""
    b = BenchmarkBuilder(cap, bw)
    d_head = d // 4
    bc_mm = max(500, d * d * max(seq_enc, seq_dec) // 80000)
    bc_pw = max(100, d * max(seq_enc, seq_dec) // 4000)

    enc_input = b.tensor(d, seq_enc)
    prev_enc = enc_input

    for layer in range(enc_layers):
        W_Q = b.tensor(d_head, d)
        W_K = b.tensor(d_head, d)
        W_V = b.tensor(d_head, d)
        Q = b.tensor(d_head, seq_enc)
        K = b.tensor(d_head, seq_enc)
        V = b.tensor(d_head, seq_enc)
        b.matmul(prev_enc, W_Q, Q, bc_mm)
        b.matmul(prev_enc, W_K, K, bc_mm)
        b.matmul(prev_enc, W_V, V, bc_mm)
        # QK interaction + score projection
        QK = b.tensor(d_head, seq_enc)
        b.pointwise([Q, K], QK, bc_pw)
        W_A = b.tensor(seq_enc, d_head)
        A = b.tensor(seq_enc, seq_enc)
        b.matmul(QK, W_A, A, bc_mm)
        S = b.tensor(seq_enc, seq_enc)
        b.pointwise([A], S, bc_pw)
        O = b.tensor(d_head, seq_enc)
        b.matmul(S, V, O, bc_mm)
        W_O = b.tensor(d, d_head)
        P = b.tensor(d, seq_enc)
        b.matmul(O, W_O, P, bc_mm)
        R = b.tensor(d, seq_enc)
        b.pointwise([prev_enc, P], R, bc_pw)
        prev_enc = R

    enc_output = prev_enc

    dec_input = b.tensor(d, seq_dec)
    prev_dec = dec_input

    for layer in range(dec_layers):
        # Decoder query and value
        W_Q = b.tensor(d_head, d)
        W_V = b.tensor(d_head, d)
        Q = b.tensor(d_head, seq_dec)
        V_dec = b.tensor(d_head, seq_dec)
        b.matmul(prev_dec, W_Q, Q, bc_mm)
        b.matmul(prev_dec, W_V, V_dec, bc_mm)
        # Cross-attention keys/values from encoder
        W_K_c = b.tensor(d_head, d)
        W_V_c = b.tensor(d_head, d)
        K_c = b.tensor(d_head, seq_enc)
        V_c = b.tensor(d_head, seq_enc)
        b.matmul(enc_output, W_K_c, K_c, bc_mm)
        b.matmul(enc_output, W_V_c, V_c, bc_mm)
        # Cross-attention: Q(d_head, seq_dec) and K_c(d_head, seq_enc)
        # Different heights -> can't pointwise directly.
        # Scale K_c and combine with V_c (both seq_enc height)
        K_scaled = b.tensor(d_head, seq_enc)
        b.pointwise([K_c], K_scaled, bc_pw)
        V_mod = b.tensor(d_head, seq_enc)
        b.pointwise([K_scaled, V_c], V_mod, bc_pw)
        # Score from Q via weight projection
        W_score = b.tensor(seq_enc, d_head)
        A = b.tensor(seq_enc, seq_dec)
        b.matmul(Q, W_score, A, bc_mm)
        S = b.tensor(seq_enc, seq_dec)
        b.pointwise([A], S, bc_pw)
        # S @ V_mod -> O (seq_dec × d_head)
        O = b.tensor(d_head, seq_dec)
        b.matmul(S, V_mod, O, bc_mm)
        W_O = b.tensor(d, d_head)
        P = b.tensor(d, seq_dec)
        b.matmul(O, W_O, P, bc_mm)
        R = b.tensor(d, seq_dec)
        b.pointwise([prev_dec, P], R, bc_pw)
        prev_dec = R

    b.save(f"benchmarks/custom-{name}.json")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    os.makedirs("benchmarks", exist_ok=True)

    print("=" * 80)
    print("GENERATING BENCHMARK SUITE")
    print("=" * 80)

    print("\n--- Scale stress ---")
    stacked_transformer()                                    # ~200 ops
    large_moe()                                              # ~50 ops, 8-wide parallel
    mega_transformer()                                       # ~300+ ops, stress test

    print("\n--- Non-power-of-2 dimensions ---")
    llama_mlp()                                              # 768, 2048
    bert_block()                                             # 768, 384, 3072

    print("\n--- Extreme aspect ratios ---")
    tall_skinny_chain()                                      # 4096×64 tensors
    embedding_lookup()                                       # 8192×128 input

    print("\n--- Pure pointwise ---")
    pointwise_only_dag()                                     # 0 MatMuls

    print("\n--- Multi-output ---")
    multi_head_output()                                      # 8 graph outputs
    hydra_net()                                              # 6 deep output branches

    print("\n--- Recompute-dominant ---")
    cheap_producer_diamond()                                 # cheap PW, expensive spill
    recompute_ladder()                                       # ladder graph, bw=3

    print("\n--- Large tensors ---")
    large_tensor_matmul()                                    # 8192×8192

    print("\n--- Extreme tight memory ---")
    ultra_tight_splitk()                                     # cap=18000, dim=512

    print("\n--- Asymmetric native ---")
    asymmetric_native_attention()                            # native 256×64
    asymmetric_native_chain()                                # native 64×256

    print("\n--- Wide + deep ---")
    mega_gnn()                                               # GNN-like irregular

    print("\n--- Mixed dimension (U-Net) ---")
    mixed_scale_unet()                                       # 128→1024→128

    print("\n--- Bandwidth extremes ---")
    ultra_low_bw()                                           # bw=1
    ultra_high_bw()                                          # bw=500

    print("\n--- Retention stress ---")
    multi_shared_weights()                                   # 3 shared weights

    print("\n--- Combinatorial stress ---")
    deep_alternating_chain()                                 # 40-op chain

    # Custom architecture-focused benchmarks
    print("\n--- Custom: transformer blocks ---")
    custom_transformer_block(seq=128, d_model=256, d_ff=512, n_heads=4,
                             cap=50000, bw=10, name="transformer-small")
    custom_transformer_block(seq=256, d_model=512, d_ff=2048, n_heads=8,
                             cap=80000, bw=20, name="transformer-large")
    custom_transformer_block(seq=512, d_model=1024, d_ff=4096, n_heads=16,
                             cap=120000, bw=50, name="transformer-stress")

    print("\n--- Custom: MoE ---")
    custom_moe_block(seq=128, d_model=256, d_expert=512, n_experts=4,
                     cap=40000, bw=10, name="moe-4expert")

    print("\n--- Custom: residual / chain ---")
    custom_deep_residual_chain(n_blocks=8, dim=256, cap=50000, bw=10,
                               name="reschain-8")
    custom_deep_narrow_chain(n_ops=20, dim=128, cap=25000, bw=8,
                             name="deep-narrow-20")
    custom_tight_memory_chain(n_ops=10, dim=512, cap=35000, bw=5,
                              name="tight-10ops")

    print("\n--- Custom: bottleneck / conv ---")
    custom_asymmetric_bottleneck(seq=256, d_in=512, d_bottle=64, d_out=512,
                                 n_branches=3, cap=40000, bw=10,
                                 name="bottleneck-3br")
    custom_conv_like_block(h=32, w=32, c_in=128, c_out=256, n_layers=6,
                           cap=50000, bw=15, name="convlike-6layer")

    print("\n--- Custom: fan-out / multi-path ---")
    custom_wide_fan_diamond(n_parallel=8, dim=256, cap=50000, bw=10,
                            name="fandia-8wide")
    custom_multi_scale_attention(seq=128, dims=[64, 128, 256],
                                 cap=40000, bw=10, name="multiscale-3")
    custom_multi_path_reduction(n_paths=6, depth=3, dim=192, cap=45000, bw=12,
                                name="multipath-6")

    print("\n--- Custom: harder benchmarks ---")
    custom_irregular_dag(n_layers=5, branches_per_layer=[3, 2, 4, 2, 3],
                         cap=60000, bw=15, name="irregular-5layer")
    custom_shared_weight_network(n_stages=5, d=256, seq=128,
                                 cap=50000, bw=10, name="shared-weights-5")
    custom_encoder_decoder(enc_layers=2, dec_layers=3, d=256, seq_enc=128,
                           seq_dec=64, cap=55000, bw=12, name="enc-dec-2x3")

    n_main = 22
    n_custom = 15
    print(f"\nGenerated {n_main + n_custom} benchmarks in benchmarks/")