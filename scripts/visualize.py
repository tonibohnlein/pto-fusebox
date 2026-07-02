import json
import sys

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def hw_legend(d):
    """Instance-legend label: the Ascend 910B per-die grounded pto-isa specs."""
    kb = lambda x: f"{d.get(x, 0) // 1024}KB"
    bw_line = (f"grounded (pto-isa cycles, {(d.get('cube_freq_hz', 0) or 0)/1e9:.2f}GHz): "
               f"GM->L1 {d.get('bw_gm_l1')}  L0c->GM {d.get('bw_l0c_gm')}  "
               f"GM<->UB {d.get('bw_gm_ub')}/{d.get('bw_ub_gm')} GiB/s")
    lines = [
        "Ascend 910B (1 die)",
        f"Cube/AIC: {d.get('num_cube_cores')} cores   Vector/AIV: {d.get('num_vector_cores')} cores",
        f"L1/Mat: {kb('l1_capacity')}   L0c/Acc: {kb('cube_capacity')}   UB: {kb('vec_capacity')}",
        bw_line,
        "Tile: cube 16x16 fractal  |  vector sub-16 rows + 32B DMA-block width",
        "Grid-only (P x Q x split-K); double-buffered (implicit, full pool)",
    ]
    return "Instance Info\\n" + "\\n".join(lines)

def generate_instance_dot(input_data, out_filepath):
    """Generates a DOT file for the raw problem instance."""
    lines = [
        "digraph Instance {",
        "    rankdir=TB;",
        "    node [fontname=\"Helvetica,Arial,sans-serif\", fontsize=10];",
        "    edge [fontname=\"Helvetica,Arial,sans-serif\", fontsize=10];",
        ""
    ]

    lines.append(f'    InstanceInfo [label="{hw_legend(input_data)}", shape=note, style=filled, fillcolor=lightyellow, margin=0.2];')
    lines.append('')

    # 1. Define Tensors (Data)
    lines.append("    // --- Tensors ---")
    for i, (w, h) in enumerate(zip(input_data['widths'], input_data['heights'])):
        size_kb = (w * h) / 1000.0
        label = f"Tensor {i}\\n{w}x{h}\\n({size_kb:.1f} K)"
        lines.append(f"    T{i} [label=\"{label}\", shape=box, style=filled, fillcolor=lightblue];")

    # 2. Define Ops (Compute)
    lines.append("\n    // --- Operations ---")
    for i in range(len(input_data['op_types'])):
        op_type = input_data['op_types'][i]
        lines.append(f"    Op{i} [label=\"Op {i}\\n{op_type}\", shape=ellipse, style=filled, fillcolor=lightgreen];")

    # 3. Define Edges
    lines.append("\n    // --- Edges ---")
    for i in range(len(input_data['op_types'])):
        # Inputs to Op
        for in_t in input_data['inputs'][i]:
            lines.append(f"    T{in_t} -> Op{i};")
        # Op to Outputs
        for out_t in input_data['outputs'][i]:
            lines.append(f"    Op{i} -> T{out_t};")

    lines.append("}")
    
    with open(out_filepath, 'w') as f:
        f.write("\n".join(lines))
    print(f"Instance DOT written to {out_filepath}")

def generate_solution_dot(input_data, output_data, out_filepath):
    """Generates a DOT file showing the scheduled/fused solution using identical layout."""
    
    # A distinct color palette to visually group Ops that belong to the same step
    colors = ["#ffb3ba", "#ffdfba", "#ffffba", "#baffc9", "#bae1ff", "#e8baff", "#ffbaff", "#c2c2f0", "#ffb3e6"]
    
    lines = [
        "digraph Solution {",
        "    rankdir=TB;",
        "    node [fontname=\"Helvetica,Arial,sans-serif\", fontsize=10];",
        "    edge [fontname=\"Helvetica,Arial,sans-serif\", fontsize=10];",
        ""
    ]

    lines.append(f'    InstanceInfo [label="{hw_legend(input_data)}", shape=note, style=filled, fillcolor=lightyellow, margin=0.2];')
    lines.append('')

    # Pre-calculate solution metadata
    retained_tensors = set()
    step_retains = {}
    for step_idx, retains in enumerate(output_data.get('tensors_to_retain', [])):
        for t in retains:
            retained_tensors.add(t)
            if t not in step_retains:
                step_retains[t] = []
            step_retains[t].append(step_idx)

    op_steps = {}
    for step_idx, sub_ops in enumerate(output_data['subgraphs']):
        for op in sub_ops:
            if op not in op_steps:
                op_steps[op] = []
            op_steps[op].append(step_idx)

    # --- Ephemeral Tensor Deduction ---
    tensor_producers = {}
    tensor_consumers = {i: [] for i in range(len(input_data['widths']))}

    for op_idx in range(len(input_data['op_types'])):
        for t_in in input_data['inputs'][op_idx]:
            tensor_consumers[t_in].append(op_idx)
        for t_out in input_data['outputs'][op_idx]:
            tensor_producers[t_out] = op_idx

    prod_steps_for_t = {}
    for t in range(len(input_data['widths'])):
        prod_op = tensor_producers.get(t, -1)
        prod_steps_for_t[t] = set(op_steps.get(prod_op, []))

    cons_steps_for_t = {t: set() for t in range(len(input_data['widths']))}
    for t in range(len(input_data['widths'])):
        for cons_op in tensor_consumers[t]:
            for s in op_steps.get(cons_op, []):
                cons_steps_for_t[t].add(s)

    # 1. Define Tensors
    lines.append("    // --- Tensors ---")
    
    cap_kb_raw = input_data.get('fast_memory_capacity', 0) / 1000.0
    
    for i, (w, h) in enumerate(zip(input_data['widths'], input_data['heights'])):
        size_kb = (w * h) / 1000.0
        
        # Base style parameters
        style_list = ["filled"]
        border_color = "black"
        penwidth = 1.0

        # Mark with red border if tensor doesn't fit in fast memory
        if cap_kb_raw > 0 and size_kb > cap_kb_raw:
            border_color = "red"
            penwidth = 3.0
            
        is_retained = i in retained_tensors
        is_ephemeral = False
        is_nonephemeral = False
        
        prod_op = tensor_producers.get(i, -1)
        
        if prod_op == -1 or len(tensor_consumers[i]) == 0:
            is_nonephemeral = True # Network input/output
        else:
            # We examine each specific producer-consumer pair individually to check for ephemerality
            # A tensor can be ephemeral to Step A, if Step A produces it and Step A consumes it, AND Step A contains ALL global consumers of this tensor.
            # Wait, if Step A contains all global consumers, then it doesn't leave Step A. It's purely ephemeral.
            # If Step B ALSO consumes it, then Step A does NOT contain all global consumers, so it leaves Step A.
            # Thus, the tensor must be written to Mem. It is Mem.
            # When would a tensor be BOTH Mem and Eph?
            # E.g., produced by Step A. Consumed by Step A. Also consumed by Step B.
            # In this case, inside Step A it might be forwarded directly in fast memory, but it STILL must be written to slow memory for Step B.
            # So is it Eph and Mem? Yes. It's used ephemerally in A, and via Mem in B.
            
            # Re-evaluating ephemerality:
            # Ephemeral: Produced in Step X, and at least one consumer is ALSO in Step X.
            # Non-ephemeral (Mem): At least one consumer is in Step Y (where Y != X), OR no consumers exist.
            
            for s in prod_steps_for_t[i]:
                # Does step 's' also consume it?
                if s in cons_steps_for_t[i]:
                    is_ephemeral = True
                # Does any other step consume it?
                for cons_s in cons_steps_for_t[i]:
                    if cons_s != s:
                        is_nonephemeral = True
            
            # If produced but never consumed (network output), it's Mem
            if not cons_steps_for_t[i]:
                is_nonephemeral = True
            
            # If it's pure ephemeral, is_nonephemeral will remain False.
            # If it's both, both become True.
            # If it's purely cross-subgraph, is_ephemeral remains False.
            
        # Now decide colors based on mixture of roles
        roles = []
        colors_for_roles = []
        
        if is_retained:
            steps_str = ",".join(map(str, step_retains[i]))
            roles.append(f"Retained: {steps_str}")
            colors_for_roles.append("gold")
            
        if is_nonephemeral and not is_retained:
            roles.append("Mem")
            colors_for_roles.append("lightblue")
            
        if is_ephemeral:
            roles.append("Eph")
            colors_for_roles.append("silver")
            
        # Fallback if no roles caught (should be impossible)
        if not roles:
            roles.append("Unknown")
            colors_for_roles.append("lightblue")
            
        # Compile styling string
        if len(colors_for_roles) > 1:
            style_list.append("wedged")
            
        fillcolor_str = ":".join(colors_for_roles)
        style_str = ",".join(style_list)
        role_label = "|".join(roles)
            
        label = f"Tensor {i}\\n{w}x{h}\\n({size_kb:.1f} K)\\n[{role_label}]"
        
        # Add to graph
        lines.append(f"    T{i} [label=\"{label}\", shape=box, style=\"{style_str}\", fillcolor=\"{fillcolor_str}\", color=\"{border_color}\", penwidth={penwidth}];")

    # 2. Ops, grouped into a rounded CLUSTER BOX per fused kernel (subgraph/step). The box carries
    #    the KERNEL-level decision — kind (cube / vector / MIXED), max region, P x Q grid, split-K,
    #    cores, latency — so a fused MIXED kernel (cube + vector ops in one box) is visually obvious.
    #    Each op node inside shows only its per-op role: the single-core seq-k / UB stream. A MIXED
    #    kernel runs on the 1 AIC : 2 AIV mix-cluster, so its cores span BOTH pools.
    lines.append("\n    // --- Fused kernels (clusters) + ops ---")
    op_types = input_data['op_types']

    def first_step(i):
        st = op_steps.get(i, [])
        return st[0] if st else None

    def op_node(i, indent="    "):
        steps = op_steps.get(i, [])
        if not steps:
            return f'{indent}Op{i} [label="Op {i}\\n{op_types[i]}\\n(unscheduled)", shape=ellipse, style=filled, fillcolor="lightgrey"];'
        # single-core k-stream (matmul seq-k / vector UB chunk), by op_order slot
        extra = ""
        s0 = steps[0]
        seq_k_list, order_list = output_data.get('seq_k', []), output_data.get('op_order', [])
        if (len(seq_k_list) > s0 and seq_k_list[s0] is not None
                and len(order_list) > s0 and i in order_list[s0]):
            kk = seq_k_list[s0][order_list[s0].index(i)]
            if kk and kk > 0:
                extra += f"\\n{'seq-k' if op_types[i] == 'MatMul' else 'stream'} {kk} (1-core)"
        if len(steps) > 1:  # recomputed across steps — wedged colors + note
            extra += f"\\n(recomputed: steps {','.join(map(str, steps))})"
        style = "filled,wedged" if len(steps) > 1 else "filled"
        color_list = ":".join(colors[s % len(colors)] for s in steps)
        return f'{indent}Op{i} [label="Op {i}\\n{op_types[i]}{extra}", shape=ellipse, style="{style}", fillcolor="{color_list}"];'

    for s, sub_ops in enumerate(output_data['subgraphs']):
        members = [o for o in sub_ops if first_step(o) == s]
        if not members:
            continue
        has_mm = any(op_types[o] == "MatMul" for o in sub_ops)
        has_vec = any(op_types[o] in ("Pointwise", "Reduction") for o in sub_ops)
        kind = "MIXED (cube+vector)" if (has_mm and has_vec) else ("cube (AIC)" if has_mm else "vector (AIV)")

        def at(lst, default):
            v = output_data.get(lst, [])
            return v[s] if len(v) > s else default
        gran = at('granularities', ['?', '?', '?'])
        parts = at('parts', [0, 0])
        if parts and parts[0] > 0:  # non-uniform P x Q grid; the tile is the MAX region
            grid, tile_word = f"{parts[0]}x{parts[1]} grid = {parts[0] * parts[1]} regions", "max region"
        else:
            grid, tile_word = "1 tile", "tile"
        split = at('splits', 1)
        split_str = f"  split-K x{split} (parallel)" if split and split > 1 else ""
        cores = at('cores', None)
        if cores:
            cores_str = (f"{cores} cores (AIC + 2xAIV mix-cluster)" if (has_mm and has_vec)
                         else f"{cores} {'cube (AIC)' if has_mm else 'vector (AIV)'} cores")
        else:
            cores_str = ""
        lat = at('subgraph_latencies', 0.0)
        klabel = (f"Kernel {s} — {kind}\\n{tile_word} {gran[0]}x{gran[1]}  |  {grid}{split_str}"
                  f"\\n{cores_str}\\nLat {lat:.1f}")
        bg = "#e6f0ff" if (has_mm and has_vec) else ("#fff0f0" if has_mm else "#eefbe6")
        lines.append(f'    subgraph cluster_k{s} {{')
        lines.append(f'      label="{klabel}"; labelloc="t"; fontsize=11; style="rounded,filled"; '
                     f'fillcolor="{bg}"; color="gray40"; penwidth=1.5;')
        for o in members:
            lines.append(op_node(o, indent="      "))
        lines.append("    }")
    for i in range(len(op_types)):  # unscheduled ops sit outside every cluster
        if first_step(i) is None:
            lines.append(op_node(i))

    # 3. Define Edges
    lines.append("\n    // --- Data Edges ---")
    for i in range(len(input_data['op_types'])):
        for in_t in input_data['inputs'][i]:
            lines.append(f"    T{in_t} -> Op{i};")
        for out_t in input_data['outputs'][i]:
            lines.append(f"    Op{i} -> T{out_t};")

    lines.append("}")
    
    with open(out_filepath, 'w') as f:
        f.write("\n".join(lines))
    print(f"Solution DOT written to {out_filepath}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python visualize.py instance <input.json> <out.dot>")
        print("  python visualize.py solution <input.json> <output.json> <out.dot>")
        sys.exit(1)

    mode = sys.argv[1]
    
    if mode == "instance":
        in_json = sys.argv[2]
        out_dot = sys.argv[3]
        data = load_json(in_json)
        generate_instance_dot(data, out_dot)
        
    elif mode == "solution":
        in_json = sys.argv[2]
        out_json = sys.argv[3]
        out_dot = sys.argv[4]
        in_data = load_json(in_json)
        out_data = load_json(out_json)
        generate_solution_dot(in_data, out_data, out_dot)