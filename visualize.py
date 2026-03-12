import json
import sys
import os

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def generate_instance_dot(input_data, out_filepath):
    """Generates a DOT file for the raw problem instance."""
    lines = [
        "digraph Instance {",
        "    rankdir=TB;",
        "    node [fontname=\"Helvetica,Arial,sans-serif\", fontsize=10];",
        "    edge [fontname=\"Helvetica,Arial,sans-serif\", fontsize=10];",
        ""
    ]

    cap_kb = input_data.get('fast_memory_capacity', 0) / 1024.0
    if cap_kb > 0:
        lines.append(f'    label="Fast Memory Capacity: {cap_kb:.1f} KB\\n";')
        lines.append('    labelloc="t";')
        lines.append('')

    # 1. Define Tensors (Data)
    lines.append("    // --- Tensors ---")
    for i, (w, h) in enumerate(zip(input_data['widths'], input_data['heights'])):
        size_kb = (w * h) / 1024.0
        label = f"Tensor {i}\\n{w}x{h}\\n({size_kb:.1f} KB)"
        lines.append(f"    T{i} [label=\"{label}\", shape=box, style=filled, fillcolor=lightblue];")

    # 2. Define Ops (Compute)
    lines.append("\n    // --- Operations ---")
    for i in range(len(input_data['op_types'])):
        op_type = input_data['op_types'][i]
        cost = input_data['base_costs'][i]
        label = f"Op {i}\\n{op_type}\\nCost: {cost}"
        lines.append(f"    Op{i} [label=\"{label}\", shape=ellipse, style=filled, fillcolor=lightgreen];")

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

    cap_kb = input_data.get('fast_memory_capacity', 0) / 1024.0
    if cap_kb > 0:
        lines.append(f'    label="Fast Memory Capacity: {cap_kb:.1f} KB\\n";')
        lines.append('    labelloc="t";')
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
    
    cap_kb_raw = input_data.get('fast_memory_capacity', 0) / 1024.0
    
    for i, (w, h) in enumerate(zip(input_data['widths'], input_data['heights'])):
        size_kb = (w * h) / 1024.0
        
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
            
        label = f"Tensor {i}\\n{w}x{h}\\n({size_kb:.1f} KB)\\n[{role_label}]"
        
        # Add to graph
        lines.append(f"    T{i} [label=\"{label}\", shape=box, style=\"{style_str}\", fillcolor=\"{fillcolor_str}\", color=\"{border_color}\", penwidth={penwidth}];")

    # 2. Define Ops
    lines.append("\n    // --- Operations ---")
    for i in range(len(input_data['op_types'])):
        op_type = input_data['op_types'][i]
        cost = input_data['base_costs'][i]
        
        steps = op_steps.get(i, [])
        if steps:
            # Use wedged multi-colors to vividly indicate sharing/multiple instances of recomputations
            style = "filled,wedged" if len(steps) > 1 else "filled"
            color_list = ":".join(colors[s % len(colors)] for s in steps)
            
            first_step = steps[0]
            gran_list = output_data.get('granularities', [])
            gran = gran_list[first_step] if len(gran_list) > first_step else ['?','?','?']
            
            lat_list = output_data.get('subgraph_latencies', [])
            lat = lat_list[first_step] if len(lat_list) > first_step else 0.0
            
            trav_order = output_data.get('traversal_orders', [])
            snake_str = "None"
            if len(trav_order) > first_step and trav_order[first_step] is not None and len(trav_order[first_step]) > 1:
                t_ord = trav_order[first_step]
                if abs(t_ord[1] - t_ord[0]) == 1:
                    snake_str = "RowMajor"
                else:
                    snake_str = "ColMajor"
            
            step_str = ",".join(map(str, steps))
            label = f"Op {i}\\n{op_type}\\nCost: {cost}\\n---\\nSteps {{ {step_str} }}\\nTile: {gran[0]}x{gran[1]}x{gran[2]}\\nLat: {lat:.1f}\\n{snake_str}"
            lines.append(f'    Op{i} [label="{label}", shape=ellipse, style="{style}", fillcolor="{color_list}"];')
        else:
            label = f"Op {i}\\n{op_type}\\nCost: {cost}\\n(Unscheduled)"
            color = "lightgrey"
            lines.append(f"    Op{i} [label=\"{label}\", shape=ellipse, style=filled, fillcolor=\"{color}\"];")

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