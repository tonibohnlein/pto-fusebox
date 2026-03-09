#!/usr/bin/env python3
"""
Generate benchmark instances based on realistic ML model architectures.
Each instance exercises different aspects of the cost model and search.
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
        """Add a tensor, return its index."""
        idx = len(self.widths)
        self.widths.append(w)
        self.heights.append(h)
        return idx
    
    def matmul(self, lhs, rhs, out, base_cost):
        """Add MatMul: lhs @ rhs -> out. K = width of lhs = height of rhs."""
        idx = len(self.op_types)
        self.op_types.append("MatMul")
        self.inputs.append([lhs, rhs])
        self.outputs.append([out])
        self.base_costs.append(base_cost)
        return idx
    
    def pointwise(self, ins, out, base_cost):
        """Add Pointwise: ins -> out."""
        idx = len(self.op_types)
        self.op_types.append("Pointwise")
        self.inputs.append(ins if isinstance(ins, list) else [ins])
        self.outputs.append([out])
        self.base_costs.append(base_cost)
        return idx
    
    def linear(self, x, weight, out, base_cost):
        """x @ weight -> out (MatMul wrapper)."""
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
        print(f"  {path}: {n_ops} ops ({mm} MM + {pw} PW), {n_tensors} tensors, "
              f"cap={self.cap}, bw={self.bw}")


def transformer_block(seq=256, d_model=512, d_ff=2048, n_heads=8, 
                      cap=80000, bw=20, name="transformer"):
    """
    Single transformer block: Multi-Head Attention + FFN + residuals.
    
    MHA: For each head i:
      Q_i = X @ W_Qi  (seq×d_model @ d_model×d_head -> seq×d_head)
      K_i = X @ W_Ki
      V_i = X @ W_Vi
      A_i = Q_i @ K_i^T  (seq×d_head @ d_head×seq -> seq×seq)  [attention scores]
      S_i = PW(A_i)      [softmax-like activation]
      O_i = S_i @ V_i    (seq×seq @ seq×d_head -> seq×d_head)  [attention output]
    
    Concat + project: 
      concat is implicit (PW chain combining heads)
      Y = concat @ W_O   (seq×d_model @ d_model×d_model -> seq×d_model)
    
    Residual: R1 = PW(X, Y)  [add + layernorm]
    
    FFN:
      H = R1 @ W1     (seq×d_model @ d_model×d_ff -> seq×d_ff)
      G = PW(H)        [GELU activation]
      F = G @ W2       (seq×d_ff @ d_ff×d_model -> seq×d_model)
    
    Residual: R2 = PW(R1, F)
    """
    d_head = d_model // n_heads
    b = BenchmarkBuilder(cap, bw)
    
    # Input tensor: X (d_model × seq) - note: width=cols, height=rows
    # Convention: tensor(width, height) where width=cols, height=rows
    # For MatMul A(h×K) @ B(K×w) -> C(h×w): LHS has width=K, height=h
    X = b.tensor(d_model, seq)  # X: seq rows × d_model cols -> (w=d_model, h=seq)
    
    head_outputs = []
    for i in range(n_heads):
        # Weight matrices for this head
        W_Q = b.tensor(d_head, d_model)   # d_model×d_head
        W_K = b.tensor(d_head, d_model)
        W_V = b.tensor(d_head, d_model)
        
        # Q, K, V projections: X @ W -> (seq × d_head)
        Q = b.tensor(d_head, seq)
        K = b.tensor(d_head, seq)
        V = b.tensor(d_head, seq)
        
        bc_proj = max(500, seq * d_model * d_head // 10000)
        b.linear(X, W_Q, Q, bc_proj)
        b.linear(X, W_K, K, bc_proj)
        b.linear(X, W_V, V, bc_proj)
        
        # Attention: Q @ K^T -> A (seq × seq)
        # K^T has width=seq, height=d_head. So A = Q(d_head×seq) @ K^T(seq×d_head)
        # Wait, MatMul: LHS(h×K) @ RHS(K×w) -> out(h×w). K = LHS.width.
        # Q is (w=d_head, h=seq). K^T is (w=seq, h=d_head).
        # Q @ K^T: LHS=Q(w=d_head,h=seq), RHS=K^T(w=seq,h=d_head)
        # K = d_head. Output: (w=seq, h=seq).
        K_T = b.tensor(seq, d_head)  # K transposed
        A = b.tensor(seq, seq)
        
        # Transpose K -> K_T (modeled as PW since it's a reshape)
        bc_transpose = max(100, seq * d_head // 5000)
        b.pointwise([K], K_T, bc_transpose)
        
        bc_attn = max(500, seq * seq * d_head // 10000)
        b.linear(Q, K_T, A, bc_attn)
        
        # Softmax (PW)
        S = b.tensor(seq, seq)
        b.pointwise([A], S, max(200, seq * seq // 5000))
        
        # S @ V -> O (seq × d_head)
        O = b.tensor(d_head, seq)
        b.linear(S, V, O, bc_attn)
        
        head_outputs.append(O)
    
    # Combine heads via PW chain (simulating concat + interactions)
    if len(head_outputs) >= 2:
        combined = head_outputs[0]
        for i in range(1, len(head_outputs)):
            new_combined = b.tensor(d_head, seq) if i < len(head_outputs)-1 else b.tensor(d_model, seq)
            b.pointwise([combined, head_outputs[i]], new_combined, 200)
            combined = new_combined
    else:
        combined = head_outputs[0]
    
    # Output projection: combined @ W_O -> Y
    W_O = b.tensor(d_model, d_model)
    Y = b.tensor(d_model, seq)
    b.linear(combined, W_O, Y, max(500, seq * d_model * d_model // 10000))
    
    # Residual + LayerNorm
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
    
    # Residual
    R2 = b.tensor(d_model, seq)
    b.pointwise([R1, F], R2, max(200, seq * d_model // 5000))
    
    b.save(f"benchmarks/custom-{name}.json")
    return b


def moe_block(seq=128, d_model=512, d_expert=1024, n_experts=4,
              cap=60000, bw=15, name="moe"):
    """
    Mixture of Experts: router + parallel expert FFNs + combine.
    
    Router: R = X @ W_R -> softmax -> gating weights
    Expert i: H_i = X @ W1_i -> GELU -> @ W2_i -> E_i
    Output: Y = PW(R, E_0, E_1, ..., E_n)  [weighted combination]
    Residual: Z = PW(X, Y)
    
    Tests: fan-out from X to all experts, parallel independent paths,
    fan-in at combination, shared input X.
    """
    b = BenchmarkBuilder(cap, bw)
    
    X = b.tensor(d_model, seq)
    
    # Router
    W_R = b.tensor(n_experts, d_model)
    R_raw = b.tensor(n_experts, seq)
    b.linear(X, W_R, R_raw, max(200, seq * d_model * n_experts // 10000))
    
    R = b.tensor(n_experts, seq)
    b.pointwise([R_raw], R, max(100, seq * n_experts // 1000))
    
    # Expert FFNs (parallel)
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
    
    # Weighted combination: PW(R, E_0, ..., E_{n-1}) -> Y
    Y = b.tensor(d_model, seq)
    b.pointwise([R] + expert_outputs, Y, max(200, seq * d_model // 2000))
    
    # Residual
    Z = b.tensor(d_model, seq)
    b.pointwise([X, Y], Z, max(200, seq * d_model // 5000))
    
    b.save(f"benchmarks/custom-{name}.json")
    return b


def deep_residual_chain(n_blocks=6, dim=256, cap=50000, bw=10, name="reschain"):
    """
    Deep residual chain with skip connections every 2 layers.
    
    Block i: H = X @ W_i -> PW(activation) -> @ W_i' -> PW(X + result)
    
    Tests: long chains with skip connections creating diamonds,
    recompute vs spill decisions, retain opportunities.
    """
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
        
        # Residual: add current + H2
        R = b.tensor(dim, dim)
        b.pointwise([current, H2], R, max(200, dim * dim // 2000))
        
        current = R
    
    b.save(f"benchmarks/custom-{name}.json")
    return b


def asymmetric_bottleneck(seq=512, d_in=1024, d_bottle=64, d_out=1024,
                          n_branches=3, cap=40000, bw=10, name="bottleneck"):
    """
    Bottleneck architecture with extreme dimension changes.
    
    X(seq × d_in) -> project_down(seq × d_bottle) -> process -> 
    project_up(seq × d_out) -> combine
    
    Multiple branches with different bottleneck sizes force different
    tiling strategies due to vastly different K dimensions.
    Tests: asymmetric tensor sizes, varying K, tight memory.
    """
    b = BenchmarkBuilder(cap, bw)
    
    X = b.tensor(d_in, seq)
    
    branch_outputs = []
    for i in range(n_branches):
        d_b = d_bottle * (i + 1)  # 64, 128, 192 — different bottleneck sizes
        
        # Down-project: X @ W_down -> H_down
        W_down = b.tensor(d_b, d_in)
        H_down = b.tensor(d_b, seq)
        b.linear(X, W_down, H_down, max(500, seq * d_in * d_b // 10000))
        
        # Process in bottleneck space
        A = b.tensor(d_b, seq)
        b.pointwise([H_down], A, max(200, seq * d_b // 2000))
        
        # Self-interaction: A @ A^T -> S (seq × seq) — small because d_b is small
        A_T = b.tensor(seq, d_b)
        b.pointwise([A], A_T, max(100, seq * d_b // 5000))  # transpose
        
        S = b.tensor(seq, seq)
        b.linear(A, A_T, S, max(300, seq * seq * d_b // 10000))
        
        # Apply to values: S @ A -> processed
        P = b.tensor(d_b, seq)
        b.linear(S, A, P, max(300, seq * d_b * seq // 10000))
        
        # Up-project: P @ W_up -> branch_out
        W_up = b.tensor(d_out, d_b)
        B_out = b.tensor(d_out, seq)
        b.linear(P, W_up, B_out, max(500, seq * d_b * d_out // 10000))
        
        branch_outputs.append(B_out)
    
    # Combine branches via PW chain
    combined = branch_outputs[0]
    for i in range(1, len(branch_outputs)):
        out = b.tensor(d_out, seq)
        b.pointwise([combined, branch_outputs[i]], out, max(200, seq * d_out // 2000))
        combined = out
    
    # Final residual
    result = b.tensor(d_out, seq)
    b.pointwise([X, combined], result, max(200, seq * d_out // 2000))
    
    b.save(f"benchmarks/custom-{name}.json")
    return b


def conv_like_block(h=64, w=64, c_in=256, c_out=256, n_layers=4,
                    cap=60000, bw=15, name="convlike"):
    """
    Conv-like block modeled as MatMul (im2col style).
    
    Each "conv" layer: input (c_in × spatial) @ filter (c_out × c_in) -> output
    With batch-norm (PW) and skip connections.
    
    Tests: square-ish tensors, moderate K, many layers.
    """
    spatial = h * w  # flattened spatial dim
    b = BenchmarkBuilder(cap, bw)
    
    X = b.tensor(c_in, spatial)
    current = X
    
    for i in range(n_layers):
        # "Convolution" as MatMul: input @ filter
        W = b.tensor(c_out, c_in)
        H = b.tensor(c_out, spatial)
        bc = max(500, spatial * c_in * c_out // 10000)
        b.linear(current, W, H, bc)
        
        # BatchNorm + ReLU (PW)
        A = b.tensor(c_out, spatial)
        b.pointwise([H], A, max(200, spatial * c_out // 5000))
        
        # Skip connection every 2 layers
        if i % 2 == 1 and i > 0:
            R = b.tensor(c_out, spatial)
            # Need a 1x1 conv if dimensions changed, otherwise just add
            if c_in == c_out:
                b.pointwise([current, A], R, max(100, spatial * c_out // 5000))
            else:
                # 1x1 projection for dimension change
                W_skip = b.tensor(c_out, c_in)
                proj = b.tensor(c_out, spatial)
                b.linear(current, W_skip, proj, max(200, spatial * c_in * c_out // 20000))
                b.pointwise([proj, A], R, max(100, spatial * c_out // 5000))
            current = R
        else:
            current = A
        
        c_in = c_out  # for next layer
    
    b.save(f"benchmarks/custom-{name}.json")
    return b


def wide_fan_diamond(n_parallel=8, dim=256, cap=50000, bw=10, name="fandia"):
    """
    Wide fan-out followed by diamond fan-in.
    
    X -> {branch_i: X @ W_i -> PW -> H_i} -> pairwise combine -> final
    
    Tests: wide fan-out from shared input, pairwise reduction tree,
    recompute of X in multiple branches, many retain opportunities.
    """
    b = BenchmarkBuilder(cap, bw)
    
    X = b.tensor(dim, dim)
    
    # Fan-out: X through n_parallel branches
    branch_outputs = []
    for i in range(n_parallel):
        W = b.tensor(dim, dim)
        H = b.tensor(dim, dim)
        bc = max(500, dim * dim * dim // 10000)
        b.linear(X, W, H, bc)
        
        A = b.tensor(dim, dim)
        b.pointwise([H], A, max(200, dim * dim // 2000))
        
        branch_outputs.append(A)
    
    # Pairwise reduction tree
    level = branch_outputs
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                out = b.tensor(dim, dim)
                b.pointwise([level[i], level[i+1]], out, max(200, dim * dim // 2000))
                next_level.append(out)
            else:
                next_level.append(level[i])
        level = next_level
    
    # Final residual with X
    result = b.tensor(dim, dim)
    b.pointwise([X, level[0]], result, max(200, dim * dim // 2000))
    
    b.save(f"benchmarks/custom-{name}.json")
    return b


def tight_memory_chain(n_ops=12, dim=512, cap=35000, bw=5, name="tight"):
    """
    Chain of MatMuls with very tight memory, forcing aggressive split-K.
    
    All tensors are dim×dim. cap is set so even a single MatMul at native
    granularity barely fits, forcing split-K and small tiles.
    Alternating MM and PW with varying base costs.
    
    Tests: split-K optimization, below-native tiling, memory-bound regime.
    """
    b = BenchmarkBuilder(cap, bw)
    
    current = b.tensor(dim, dim)
    
    for i in range(n_ops):
        if i % 2 == 0:
            # MatMul
            W = b.tensor(dim, dim)
            out = b.tensor(dim, dim)
            bc = 1000 + i * 200  # increasing compute
            b.linear(current, W, out, bc)
        else:
            # PW
            out = b.tensor(dim, dim)
            b.pointwise([current], out, 300 + i * 50)
        current = out
    
    b.save(f"benchmarks/custom-{name}.json")
    return b


def multi_scale_attention(seq=128, dims=[64, 128, 256, 512],
                          cap=45000, bw=10, name="multiscale"):
    """
    Multi-scale attention: process at different resolutions then combine.
    
    For each scale d:
      Q = X @ W_Q(d), K = X @ W_K(d), V = X @ W_V(d)
      A = Q @ K^T (seq×seq attention matrix)
      O = A @ V
    
    Then combine all O_i via PW chain + residual.
    
    Tests: varying K dimensions in same graph, shared input X,
    fan-out/fan-in pattern, interesting tiling because dims vary.
    """
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
        
        # K^T
        K_T = b.tensor(seq, d)
        b.pointwise([K], K_T, max(100, seq * d // 5000))
        
        # Attention: Q @ K_T -> A
        A = b.tensor(seq, seq)
        bc_attn = max(300, seq * seq * d // 10000)
        b.linear(Q, K_T, A, bc_attn)
        
        # Softmax
        S = b.tensor(seq, seq)
        b.pointwise([A], S, max(100, seq * seq // 5000))
        
        # S @ V -> O
        O = b.tensor(d, seq)
        b.linear(S, V, O, bc_attn)
        
        # Project back to d_in
        W_out = b.tensor(d_in, d)
        O_proj = b.tensor(d_in, seq)
        b.linear(O, W_out, O_proj, bc_proj)
        
        scale_outputs.append(O_proj)
    
    # Combine via PW chain
    combined = scale_outputs[0]
    for i in range(1, len(scale_outputs)):
        out = b.tensor(d_in, seq)
        b.pointwise([combined, scale_outputs[i]], out, max(200, seq * d_in // 2000))
        combined = out
    
    # Residual
    result = b.tensor(d_in, seq)
    b.pointwise([X, combined], result, max(200, seq * d_in // 2000))
    
    b.save(f"benchmarks/custom-{name}.json")
    return b


if __name__ == "__main__":
    os.makedirs("benchmarks", exist_ok=True)
    
    print("Generating custom benchmarks:\n")
    
    # 1. Small transformer block (manageable for exhaustive analysis)
    transformer_block(seq=128, d_model=256, d_ff=512, n_heads=4,
                      cap=50000, bw=10, name="transformer-small")
    
    # 2. Larger transformer with tight memory  
    transformer_block(seq=256, d_model=512, d_ff=2048, n_heads=8,
                      cap=80000, bw=20, name="transformer-large")
    
    # 3. Mixture of Experts
    moe_block(seq=128, d_model=256, d_expert=512, n_experts=4,
              cap=40000, bw=10, name="moe-4expert")
    
    # 4. Deep residual chain
    deep_residual_chain(n_blocks=8, dim=256, cap=50000, bw=10, name="reschain-8")
    
    # 5. Asymmetric bottleneck with varying K
    asymmetric_bottleneck(seq=256, d_in=512, d_bottle=64, d_out=512,
                          n_branches=3, cap=40000, bw=10, name="bottleneck-3br")
    
    # 6. Conv-like block
    conv_like_block(h=32, w=32, c_in=128, c_out=256, n_layers=6,
                    cap=50000, bw=15, name="convlike-6layer")
    
    # 7. Wide fan-out diamond (tests recompute)
    wide_fan_diamond(n_parallel=8, dim=256, cap=50000, bw=10, name="fandia-8wide")
    
    # 8. Tight memory chain (tests split-K)
    tight_memory_chain(n_ops=10, dim=512, cap=35000, bw=5, name="tight-10ops")
    
    # 9. Multi-scale attention (varying K)
    multi_scale_attention(seq=128, dims=[64, 128, 256], 
                          cap=40000, bw=10, name="multiscale-3")
    
    # 10. Stress test: large transformer 
    transformer_block(seq=512, d_model=1024, d_ff=4096, n_heads=16,
                      cap=120000, bw=50, name="transformer-stress")
    
    print(f"\nGenerated {10} custom benchmarks in benchmarks/")