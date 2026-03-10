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

    # Pre-calculate solution metadata
    retained_tensors = set()
    step_retains = {}
    for step_idx, retains in enumerate(output_data['tensors_to_retain']):
        for t in retains:
            retained_tensors.add(t)
            if t not in step_retains:
                step_retains[t] = []
            step_retains[t].append(step_idx)

    op_to_step = {}
    for step_idx, sub_ops in enumerate(output_data['subgraphs']):
        for op in sub_ops:
            op_to_step[op] = step_idx

    # 1. Define Tensors (Identical structural definition to instance)
    lines.append("    // --- Tensors ---")
    for i, (w, h) in enumerate(zip(input_data['widths'], input_data['heights'])):
        size_kb = (w * h) / 1024.0
        
        if i in retained_tensors:
            steps_str = ",".join(map(str, step_retains[i]))
            label = f"Tensor {i}\\n{w}x{h}\\n({size_kb:.1f} KB)\\n[Retained: Steps {steps_str}]"
            lines.append(f"    T{i} [label=\"{label}\", shape=box, style=filled, fillcolor=gold];")
        else:
            label = f"Tensor {i}\\n{w}x{h}\\n({size_kb:.1f} KB)"
            lines.append(f"    T{i} [label=\"{label}\", shape=box, style=filled, fillcolor=lightblue];")

    # 2. Define Ops (Identical structural definition, but with Step colors and labels)
    lines.append("\n    // --- Operations ---")
    for i in range(len(input_data['op_types'])):
        op_type = input_data['op_types'][i]
        cost = input_data['base_costs'][i]
        
        step_idx = op_to_step.get(i, -1)
        if step_idx != -1:
            gran = output_data['granularities'][step_idx]
            lat = output_data['subgraph_latencies'][step_idx]
            
            # Safe snake traversal check
            trav_order = output_data.get('traversal_orders', [])[step_idx]
            snake_str = "None"
            if trav_order is not None and len(trav_order) > 1:
                if abs(trav_order[1] - trav_order[0]) == 1:
                    snake_str = "RowMajor"
                else:
                    snake_str = "ColMajor"
            
            # Combine basic info with solution info
            label = f"Op {i}\\n{op_type}\\nCost: {cost}\\n---\\nStep {step_idx}\\nTile: {gran[0]}x{gran[1]}x{gran[2]}\\nLat: {lat:.1f}\\n{snake_str}"
            color = colors[step_idx % len(colors)]
        else:
            label = f"Op {i}\\n{op_type}\\nCost: {cost}\\n(Unscheduled)"
            color = "lightgrey"
            
        lines.append(f"    Op{i} [label=\"{label}\", shape=ellipse, style=filled, fillcolor=\"{color}\"];")

    # 3. Define Edges (Exactly matching the instance topology)
    lines.append("\n    // --- Data Edges ---")
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