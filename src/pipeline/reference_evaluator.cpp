/*
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef MLSYS_H_
#define MLSYS_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <deque>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

//#include "third_party/absl/status/statusor.h"

////////////////////////////////////////////////////////////////////////////////
/////////  Basic definitions for problem & solution data structures.   /////////
/////////  Contest participants do not need to modify this code.       /////////
////////////////////////////////////////////////////////////////////////////////

namespace mlsys {

using BaseCost = int64_t;
using Depth = int64_t;
using FastMemoryCapacity = int64_t;
using Height = int64_t;
using Inputs = std::vector<size_t>;
using OpType = std::string;
using Outputs = std::vector<size_t>;
using SlowMemoryBandwidth = int64_t;
using SubgraphLatency = double;
using TotalLatency = double;
using TraversalOrder = std::vector<int64_t>;
using Width = int64_t;

struct Tensor {
  Width width;
  Height height;
  Inputs inputs;
  Outputs outputs;
  bool operator==(const Tensor& other) const = default;
};

struct Op {
  OpType op_type;
  Inputs inputs;
  Outputs outputs;
  BaseCost base_cost;
  bool operator==(const Op& other) const = default;
};

struct Granularity {
  Width width;
  Height height;
  Depth depth;
  bool operator==(const Granularity& other) const = default;
};

struct Problem {
  std::vector<Tensor> tensors;
  std::vector<Op> ops;
  FastMemoryCapacity fast_memory_capacity;
  SlowMemoryBandwidth slow_memory_bandwidth;
  Granularity native_granularity;
  std::vector<size_t> ops_reverse_top_order;
  bool operator==(const Problem& other) const = default;
  void initReverseTopOrder();
};

/*absl::StatusOr<Problem>*/ bool ReadProblem(const std::string& filename);

struct Subgraph {
  std::vector<size_t> ops;
  std::unordered_set<size_t> tensors_to_retain;
  Granularity granularity;
  std::optional<TraversalOrder> traversal_order;
  std::string traversal_code = "none";
  SubgraphLatency subgraph_latency;
  std::vector<size_t> ops_reverse_top_order;
  bool operator==(const Subgraph& other) const = default;
};

struct Solution {
  std::vector<Subgraph> subgraphs;
  std::vector<std::set<size_t>> ops_to_subgraphs;
  void initOpsToSubgraphs(const Problem& instance);
  void initTopOrders(const Problem& instance);
  bool operator==(const Solution& other) const = default;
};

enum VALIDITY {
  VALID,
  OUT_OF_MEMORY,
  SHAPES_MISALIGNED,
  POINTWISE_WITH_DEPTH
};

bool ReadSolution(const std::string& filename);

TotalLatency Evaluate(const Problem& problem,
                                      const Solution& solution);

std::pair<Inputs, Outputs> getInputsAndOutputs (const Problem& problem, const Solution& solution, size_t subgraph_index)
{
    std::set<size_t> op_set, inputs, outputs;
    for(size_t op : solution.subgraphs[subgraph_index].ops)
        op_set.insert(op);

    for(size_t op : solution.subgraphs[subgraph_index].ops) {
        for (size_t tensor : problem.ops[op].inputs) {
            if(problem.tensors[tensor].inputs.empty())
                inputs.insert(tensor);
            for (size_t parent_op : problem.tensors[tensor].inputs) {
                if (solution.ops_to_subgraphs[parent_op].find(subgraph_index) == solution.ops_to_subgraphs[parent_op].end())
                    inputs.insert(tensor);
            }
        }
        for (size_t tensor : problem.ops[op].outputs) {
            if(problem.tensors[tensor].outputs.empty())
                outputs.insert(tensor);
            else {
                bool ephemeral = false;
                for (size_t child_op : problem.tensors[tensor].outputs) {
                    if (solution.ops_to_subgraphs[child_op].find(subgraph_index) != solution.ops_to_subgraphs[child_op].end()){
                        ephemeral = true;
                        break;
                    }
                }
                if(ephemeral)
                    continue;

                for (size_t child_op : problem.tensors[tensor].outputs) {
                    for(size_t subgraph : solution.ops_to_subgraphs[child_op]) {
                        if (solution.ops_to_subgraphs[op].find(subgraph) == solution.ops_to_subgraphs[op].end())
                            outputs.insert(tensor);
                    }
                }
            }

        }
    }
    if(subgraph_index > 0) {
        for (size_t tensor : solution.subgraphs[subgraph_index-1].tensors_to_retain)
            inputs.erase(tensor);
    }
    for (size_t tensor : solution.subgraphs[subgraph_index].tensors_to_retain)
        outputs.erase(tensor);

    std::pair<Inputs, Outputs> to_return;
    std::copy(inputs.begin(), inputs.end(), std::back_inserter(to_return.first));
    std::copy(outputs.begin(), outputs.end(), std::back_inserter(to_return.second));
    return to_return;
}

bool checkAcyclicity (const Problem& instance, const Solution& solution) {

    std::vector<std::set<size_t>> deps(solution.subgraphs.size());
    std::vector<std::vector<std::pair<size_t, size_t>>> frees_dependencies(solution.subgraphs.size());

    for(size_t op = 0; op <instance.ops.size(); ++op)
        for(size_t tens : instance.ops[op].inputs)
            for(size_t in_op : instance.tensors[tens].inputs) {
                for(size_t target_sub : solution.ops_to_subgraphs[op]) {
                    if(solution.ops_to_subgraphs[in_op].find(target_sub) == solution.ops_to_subgraphs[in_op].end()){
                        deps[target_sub].insert(tens);
                        for(size_t source_sub : solution.ops_to_subgraphs[in_op])
                            frees_dependencies[source_sub].emplace_back(target_sub, tens);
                    }
                }
            }

    std::deque<size_t> Q;
    size_t visited = 0;
    for(size_t subgraph_index = 0; subgraph_index < solution.subgraphs.size(); ++subgraph_index) {
        if(deps[subgraph_index].empty())
            Q.push_back(subgraph_index);
    }
    while(!Q.empty()) {
        size_t subgraph_index = Q.front();
        Q.pop_front();
        ++visited;
        for(auto idx_and_itr : frees_dependencies[subgraph_index]) {
            auto itr = deps[idx_and_itr.first].find(idx_and_itr.second);
            if(itr != deps[idx_and_itr.first].end())
            {
                deps[idx_and_itr.first].erase(itr);
                if(deps[idx_and_itr.first].empty()) {
                    Q.push_back(idx_and_itr.first);
                }
            }
        }
    }

    return (visited == solution.subgraphs.size());
}

bool checkConnectivity (const Problem& instance, const Solution& solution) {

    for(size_t subgraph_index = 0; subgraph_index < solution.subgraphs.size(); ++subgraph_index)
    {
        if(solution.subgraphs[subgraph_index].ops.empty())
            continue;

        std::set<size_t> seen;
        size_t last = solution.subgraphs[subgraph_index].ops_reverse_top_order[0];
        seen.insert(last);
        std::deque<size_t> Q;
        Q.push_back(last);
        while(!Q.empty()) {
            size_t op = Q.front();
            Q.pop_front();
            for(size_t tens : instance.ops[op].inputs) {
                for(size_t other_op : instance.tensors[tens].inputs) {
                    if(seen.find(other_op) == seen.end() &&
                       solution.ops_to_subgraphs[other_op].find(subgraph_index) != solution.ops_to_subgraphs[other_op].end()) {
                        seen.insert(other_op);
                        Q.push_back(other_op);
                       }
                }
                for(size_t other_op : instance.tensors[tens].outputs) {
                    if(seen.find(other_op) == seen.end() &&
                       solution.ops_to_subgraphs[other_op].find(subgraph_index) != solution.ops_to_subgraphs[other_op].end()) {
                        seen.insert(other_op);
                        Q.push_back(other_op);
                       }
                }
            }
            for(size_t tens : instance.ops[op].outputs) {
                for(size_t other_op : instance.tensors[tens].inputs) {
                    if(seen.find(other_op) == seen.end() &&
                       solution.ops_to_subgraphs[other_op].find(subgraph_index) != solution.ops_to_subgraphs[other_op].end()) {
                        seen.insert(other_op);
                        Q.push_back(other_op);
                       }
                }
                for(size_t other_op : instance.tensors[tens].outputs) {
                    if(seen.find(other_op) == seen.end() &&
                       solution.ops_to_subgraphs[other_op].find(subgraph_index) != solution.ops_to_subgraphs[other_op].end()) {
                        seen.insert(other_op);
                        Q.push_back(other_op);
                       }
                }
            }
        }
        if(seen.size() != solution.subgraphs[subgraph_index].ops.size()) {
            std::cout<<"ERROR (subgraph "<<subgraph_index<<"): the subgraph seems to be disconnected."<<std::endl;
            return false;
        }

    }
    return true;
}

bool checkEphemeralization (const Problem& instance, const Solution& solution) {

    std::vector<bool> can_be_input(instance.tensors.size(), false);
    for(size_t tens = 0; tens <instance.tensors.size(); ++tens)
        if(instance.tensors[tens].inputs.empty())
            can_be_input[tens] = true;

    for(size_t op = 0; op <instance.ops.size(); ++op)
        for(size_t subgraph_index : solution.ops_to_subgraphs[op])
            for(size_t tens : instance.ops[op].outputs) {
                bool is_output = true;
                for(size_t child_op : instance.tensors[tens].outputs) {
                    if(solution.ops_to_subgraphs[child_op].find(subgraph_index) != solution.ops_to_subgraphs[child_op].end()) {
                        is_output = false;
                        break;
                    }
                }
                if(is_output)
                    can_be_input[tens] = true;
            }

    for(size_t op = 0; op <instance.ops.size(); ++op)
        for(size_t subgraph_index : solution.ops_to_subgraphs[op])
            for(size_t tens : instance.ops[op].inputs) {
                bool is_internal = false;
                for(size_t parent_op : instance.tensors[tens].inputs)
                    if(solution.ops_to_subgraphs[parent_op].find(subgraph_index) != solution.ops_to_subgraphs[parent_op].end())
                        is_internal = true;


                if(!is_internal && !can_be_input[tens]) {
                    std::cout<<"ERROR (subgraph "<<subgraph_index<<"): Tensor "<<tens<<" produced by operation "<< instance.tensors[tens].inputs[0]<<
                            " cannot be read in the subgraph; it is only produced ephemerally in previous subgraphs, not as an output."<<std::endl;
                    return false;
                }

            }

    return true;
}

bool checkRetainment (const Problem& instance, const Solution& solution) {

    for(size_t subgraph_index = 0; subgraph_index < solution.subgraphs.size(); ++subgraph_index)
    {
        for(size_t tens : solution.subgraphs[subgraph_index].tensors_to_retain) {
            size_t op_idx = instance.tensors[tens].inputs[0];
            if(solution.ops_to_subgraphs[op_idx].find(subgraph_index) == solution.ops_to_subgraphs[op_idx].end()) {
                std::cout<<"ERROR (subgraph "<<subgraph_index<<"): Tensor "<<tens<<" cannot be retained because it's not produced in subgraph "<<
                        subgraph_index<<" (its parent op "<<op_idx<<" is not there)."<<std::endl;
                return false;
            }
            bool has_output = false;
            if(subgraph_index < solution.subgraphs.size() - 1 &&
                solution.ops_to_subgraphs[op_idx].find(subgraph_index + 1) == solution.ops_to_subgraphs[op_idx].end()) {
                    for(size_t child_op : instance.tensors[tens].outputs) {
                        if(solution.ops_to_subgraphs[child_op].find(subgraph_index + 1) != solution.ops_to_subgraphs[child_op].end())
                            has_output = true;
                }
            }
            if(!has_output) {
                std::cout<<"WARNING: (subgraph "<<subgraph_index<<"): No reason to retain tensor "<<tens<<
                    ", since it is not used in the next subgraph."<<std::endl;
            }
        }
    }
    return true;
}

std::vector<int64_t> Snakify(std::vector<int64_t> v, size_t cycle)
{
    std::vector<int64_t> snake = v;
    for(size_t start = cycle; start < v.size(); start+= 2*cycle){
        for(size_t i=0; i<cycle; ++i)
            snake[start+i] = v[start+cycle-1-i];
    }
    return snake;
}

TraversalOrder createOrder(size_t h_dim, size_t v_dim, const std::string& traversal_code)
{
    TraversalOrder order;
    order.resize(h_dim * v_dim);
    if(traversal_code == "hsnake") {
        for(size_t i=0; i < order.size(); ++i)
            order[i] = i;
        order = Snakify(order, h_dim);

    } else if(traversal_code == "vsnake") {
        for(size_t i=0; i < order.size(); ++i)
            order[i] = (i%v_dim) * h_dim + i/v_dim;
        order = Snakify(order, v_dim);

    } else {
        for(size_t i=0; i < order.size(); ++i)
            order[i] = i;
    }
    return order;
}

TotalLatency computeSubgraphCost(const Problem& problem, const Solution& solution, size_t subgraph_index, VALIDITY& valid) {

    if (solution.subgraphs[subgraph_index].ops.empty())
        return 0;

    const Subgraph& subgraph = solution.subgraphs[subgraph_index];
    const Granularity& op_gran = solution.subgraphs[subgraph_index].granularity;

    bool time_optimized = true;

    TotalLatency time = 0;
    std::pair<Inputs, Outputs> in_and_out = getInputsAndOutputs (problem, solution, subgraph_index);

    std::unordered_map<size_t, bool> is_sink;
    for(size_t op_idx : subgraph.ops_reverse_top_order) {
        is_sink[op_idx] = true;
        for(size_t out_mtx : problem.ops[op_idx].outputs)
            for(size_t out_op : problem.tensors[out_mtx].outputs)
                if(solution.ops_to_subgraphs[out_op].find(subgraph_index) != solution.ops_to_subgraphs[out_op].end())
                    is_sink[op_idx] = false;

    }
    bool has_matmul_sink = false;
    Width matmul_sink_d_width;
    for(size_t op_idx : subgraph.ops_reverse_top_order)
        if(is_sink[op_idx] && problem.ops[op_idx].op_type == "MatMul") {
            has_matmul_sink = true;
            matmul_sink_d_width = problem.tensors[problem.ops[op_idx].inputs[0]].width;
        }
        
    if(!has_matmul_sink && op_gran.depth > 1) {
        std::cout<<"WARNING (subgraph "<<subgraph_index<<"): No MatMul sink operator, but depth granularity is set to "
            <<op_gran.depth<<" > 1; is this intended?"<<std::endl;
    }

    std::unordered_map<size_t, size_t> h_tiles, v_tiles, h_pos, v_pos, tile_dim; // per tensor
    std::unordered_map<size_t, size_t> h_last_pos, v_last_pos; // per op
    bool first_tile = true;
    valid = VALIDITY::VALID;

    size_t last_op_idx = subgraph.ops_reverse_top_order.front();
    const Op& last_op = problem.ops[last_op_idx];
    h_tiles[last_op.outputs[0]] = std::max(problem.tensors[last_op.outputs[0]].width / op_gran.width, 1L);
    v_tiles[last_op.outputs[0]] = std::max(problem.tensors[last_op.outputs[0]].height / op_gran.height, 1L);
    size_t d_tiles = has_matmul_sink ? std::max(matmul_sink_d_width/op_gran.depth, 1L) : 1;

    tile_dim[last_op.outputs[0]] = (d_tiles > 1) ? 3 : ((h_tiles[last_op.outputs[0]] > 1 && h_tiles[last_op.outputs[0]] > 1) ? 2 : 1);

    if(d_tiles > 1) {
        for(size_t op : solution.subgraphs[subgraph_index].ops)
            if(is_sink[op] && problem.ops[op].op_type == "Pointwise") {
                std::cout<<"ERROR (subgraph "<<subgraph_index<<"): There is a Pointwise sink operator "
                    <<op<<" but we have "<<d_tiles<<">1 tiles in the depth dimension."<<std::endl;
                valid = VALIDITY::POINTWISE_WITH_DEPTH;
                return 0;
            }
    }

    for(size_t op_idx : solution.subgraphs[subgraph_index].ops) {
        if(!is_sink[op_idx] || op_idx == last_op_idx)
            continue;
        const Op& op = problem.ops[op_idx];
        if(problem.tensors[last_op.outputs[0]].width != problem.tensors[op.outputs[0]].width ||
            problem.tensors[last_op.outputs[0]].height != problem.tensors[op.outputs[0]].height ||
            (d_tiles > 1 && problem.tensors[last_op.inputs[0]].width != problem.tensors[op.inputs[0]].width)) {
            valid = VALIDITY::SHAPES_MISALIGNED; //output size mismatch
            std::cout<<"ERROR (subgraph "<<subgraph_index<<"): Tile shape misalignment between sink operators "
                    <<last_op_idx<<" and "<<op_idx<<"."<<std::endl;
            return 0;
        }
        h_tiles[op.outputs[0]] = h_tiles[last_op.outputs[0]];
        v_tiles[op.outputs[0]] = v_tiles[last_op.outputs[0]];
        tile_dim[op.outputs[0]] = tile_dim[last_op.outputs[0]];
    }

    TotalLatency costSecondDim = 0;
    for(size_t idx = 0; idx < 5; ++idx) {
        const std::string& code = solution.subgraphs[subgraph_index].traversal_code;

        if((idx == 1 || idx == 4) && ((code != "vsnake" && h_tiles[last_op.outputs[0]] < 2)
                        || (code == "vsnake" && v_tiles[last_op.outputs[0]] < 2)))
            continue;
        if(idx == 2 && ((code != "vsnake" && h_tiles[last_op.outputs[0]] < 3)
                        || (code == "vsnake" && v_tiles[last_op.outputs[0]] < 3)))
            continue;
        if(idx > 2 && ((code != "vsnake" && v_tiles[last_op.outputs[0]] < 2)
                        || (code == "vsnake" && h_tiles[last_op.outputs[0]] < 2)))
            continue;


        if(code == "hsnake" || code == "none") {
            if(idx < 3)
                v_pos[last_op.outputs[0]] = 0;
            else
                v_pos[last_op.outputs[0]] = 1;
            if(idx < 2)
                h_pos[last_op.outputs[0]] = idx;
            else if(idx == 2)
                h_pos[last_op.outputs[0]] = h_tiles[last_op.outputs[0]]-1;
            else {
                if(code == "hsnake")
                    h_pos[last_op.outputs[0]] = h_tiles[last_op.outputs[0]]+2-idx;
                if(code == "none")
                    h_pos[last_op.outputs[0]] = idx - 3;
            }
        } else {
            if(idx < 3)
                h_pos[last_op.outputs[0]] = 0;
            else
                h_pos[last_op.outputs[0]] = 1;
            if(idx < 2)
                v_pos[last_op.outputs[0]] = idx;
            else if(idx == 2)
                v_pos[last_op.outputs[0]] = v_tiles[last_op.outputs[0]]-1;
            else
                v_pos[last_op.outputs[0]] = v_tiles[last_op.outputs[0]]+2-idx;
        }

        TotalLatency costThirdDim = 0.0, costDTile = 0.0;
        for(size_t d=0; d < d_tiles; ++d)
        {
            if(time_optimized && d == 2)
                d = d_tiles - 1;

            TotalLatency io_time = 0.0, compute_time = 0.0;
            double mem_used = 0.0;

            std::unordered_set<size_t> inputs, outputs;

            // go through operators
            for(size_t op_idx : subgraph.ops_reverse_top_order)
            {
                const Op& op = problem.ops[op_idx];
                size_t out_tensor = op.outputs[0];

                if(is_sink[op_idx]) {
                    h_pos[out_tensor] = h_pos[last_op.outputs[0]];
                    v_pos[out_tensor] = v_pos[last_op.outputs[0]];
                }

                if(op.op_type == "Pointwise") {
                    for (size_t in_tens : op.inputs) {
                        if(first_tile) {
                            if(h_tiles.find(in_tens) == h_tiles.end()) {
                                h_tiles[in_tens] = h_tiles[out_tensor];
                                v_tiles[in_tens] = v_tiles[out_tensor];
                                tile_dim[in_tens] = tile_dim[out_tensor];
                            } else if (h_tiles[in_tens] != h_tiles[out_tensor] || v_tiles[in_tens] != v_tiles[out_tensor]
                                       || tile_dim[in_tens] != tile_dim[out_tensor]) {
                                valid = VALIDITY::SHAPES_MISALIGNED;
                                std::cout<<"ERROR (subgraph "<<subgraph_index<<"): Tile shape misalignment for Pointwise operator "<<op_idx<<":";
                                if(h_tiles[in_tens] != h_tiles[out_tensor])
                                    std::cout<<" input tensor "<<in_tens<<" has "<<h_tiles[in_tens]<<" horizontal tiles, while output tensor "
                                        <<out_tensor<<" has "<<h_tiles[out_tensor]<<" horizontal tiles.";
                                if(v_tiles[in_tens] != v_tiles[out_tensor])
                                    std::cout<<" input tensor "<<in_tens<<" has "<<v_tiles[in_tens]<<" vertical tiles, while output tensor "
                                        <<out_tensor<<" has "<<v_tiles[out_tensor]<<" vertical tiles.";
                                if(tile_dim[in_tens] != tile_dim[out_tensor])
                                    std::cout<<" input tensor "<<in_tens<<" has abstract tiling dimension of both "<<tile_dim[in_tens]<<"D and "
                                        <<tile_dim[out_tensor]<<"D.";
                                std::cout<<std::endl;
                                return 0;
                            }
                        }
                        h_pos[in_tens] = h_pos[out_tensor];
                        v_pos[in_tens] = v_pos[out_tensor];
                    }
                } else if(!is_sink[op_idx] || d_tiles == 1) {
                    if(first_tile) {
                        if(h_tiles.find(op.inputs[0]) == h_tiles.end()) {
                            h_tiles[op.inputs[0]] = 1;
                            v_tiles[op.inputs[0]] = v_tiles[out_tensor];
                            tile_dim[op.inputs[0]] = std::max(tile_dim[out_tensor]-1, (size_t)1);
                        } else if(h_tiles[op.inputs[0]] != 1 || v_tiles[op.inputs[0]] != v_tiles[out_tensor]
                                    || tile_dim[op.inputs[0]] != std::max(tile_dim[out_tensor]-1, (size_t)1) ) {
                            valid = VALIDITY::SHAPES_MISALIGNED;
                            std::cout<<"ERROR (subgraph "<<subgraph_index<<"): Tile shape misalignment for Matmul operator "<<op_idx<<":";
                            if(h_tiles[op.inputs[0]] != 1)
                                std::cout<<" first input tensor "<<op.inputs[0]<<" has "<<h_tiles[op.inputs[0]]<<" (i.e. more than 1) horizontal tiles.";
                            if(v_tiles[op.inputs[0]] != v_tiles[out_tensor])
                                std::cout<<" first input tensor "<<op.inputs[0]<<" has "<<v_tiles[op.inputs[0]]<<" vertical tiles, while output tensor "
                                    <<out_tensor<<" has "<<v_tiles[out_tensor]<<" vertical tiles.";
                                    if(tile_dim[op.inputs[0]] != std::max(tile_dim[out_tensor]-1, (size_t)1))
                                std::cout<<" input tensor "<<op.inputs[0]<<" has abstract tiling dimension of both "<<tile_dim[op.inputs[0]]<<"D and "
                                    <<std::max(tile_dim[out_tensor]-1, (size_t)1)<<"D.";
                            std::cout<<std::endl;
                            return 0;
                        }
                        if(h_tiles.find(op.inputs[1]) == h_tiles.end()) {
                            h_tiles[op.inputs[1]] = h_tiles[out_tensor];
                            v_tiles[op.inputs[1]] = 1;
                            tile_dim[op.inputs[1]] = std::max(tile_dim[out_tensor]-1, (size_t)1);
                        } else if(h_tiles[op.inputs[1]] != h_tiles[out_tensor] || v_tiles[op.inputs[1]] != 1
                                  || tile_dim[op.inputs[1]] != std::max(tile_dim[out_tensor]-1, (size_t)1) ) {
                            valid = VALIDITY::SHAPES_MISALIGNED;
                            std::cout<<"ERROR (subgraph "<<subgraph_index<<"): Tile shape misalignment for Matmul operator "<<op_idx<<":";
                            if(h_tiles[op.inputs[1]] != h_tiles[out_tensor])
                                std::cout<<" second input tensor "<<op.inputs[1]<<" has "<<h_tiles[op.inputs[1]]<<" horizontal tiles, while output tensor "
                                    <<out_tensor<<" has "<<h_tiles[out_tensor]<<" horizontal tiles.";
                            if(v_tiles[op.inputs[1]] != 1)
                                std::cout<<" second input tensor "<<op.inputs[1]<<" has "<<v_tiles[op.inputs[1]]<<" (i.e. more than 1) vertical tiles.";
                            if(tile_dim[op.inputs[1]] != std::max(tile_dim[out_tensor]-1, (size_t)1))
                                std::cout<<" input tensor "<<op.inputs[1]<<" has abstract tiling dimension of both "<<tile_dim[op.inputs[1]]<<"D and "
                                    <<std::max(tile_dim[out_tensor]-1, (size_t)1)<<"D.";
                            std::cout<<std::endl;
                            return 0;
                        }
                    }
                    h_pos[op.inputs[0]] = 0;
                    v_pos[op.inputs[0]] = v_pos[out_tensor];
                    h_pos[op.inputs[1]] = h_pos[out_tensor];
                    v_pos[op.inputs[1]] = 0;
                } else { // depth magic
                    if(first_tile) {
                        if(h_tiles.find(op.inputs[0]) == h_tiles.end()) {
                            h_tiles[op.inputs[0]] = d_tiles;
                            v_tiles[op.inputs[0]] = v_tiles[out_tensor];
                            tile_dim[op.inputs[0]] = std::max(tile_dim[out_tensor]-1, (size_t)1);
                        } else if(h_tiles[op.inputs[0]] != d_tiles || v_tiles[op.inputs[0]] != v_tiles[out_tensor]
                                  || tile_dim[op.inputs[0]] != std::max(tile_dim[out_tensor]-1, (size_t)1)) {
                            valid = VALIDITY::SHAPES_MISALIGNED;
                            std::cout<<"ERROR (subgraph "<<subgraph_index<<"): Tile shape misalignment for Matmul operator "<<op_idx<<" with "<<
                                        d_tiles<<" tiles in depth dimension:";
                            if(h_tiles[op.inputs[0]] != d_tiles)
                                std::cout<<" first input tensor "<<op.inputs[0]<<" has "<<h_tiles[op.inputs[0]]<<
                                            " horizontal tiles, whereas depth-tiling would require "<<d_tiles<<".";
                            if(v_tiles[op.inputs[0]] != v_tiles[out_tensor])
                                std::cout<<" first input tensor "<<op.inputs[0]<<" has "<<v_tiles[op.inputs[0]]<<" vertical tiles, while output tensor "
                                    <<out_tensor<<" has "<<v_tiles[out_tensor]<<" vertical tiles.";
                                    if(tile_dim[op.inputs[0]] != std::max(tile_dim[out_tensor]-1, (size_t)1))
                                std::cout<<" input tensor "<<op.inputs[0]<<" has abstract tiling dimension of both "<<tile_dim[op.inputs[0]]<<"D and "
                                    <<std::max(tile_dim[out_tensor]-1, (size_t)1)<<"D.";
                            std::cout<<std::endl;
                            return 0;
                        }

                        if(h_tiles.find(op.inputs[1]) == h_tiles.end()) {
                            h_tiles[op.inputs[1]] = h_tiles[out_tensor];
                            v_tiles[op.inputs[1]] = d_tiles;
                            tile_dim[op.inputs[1]] = std::max(tile_dim[out_tensor]-1, (size_t)1);
                        } else if(h_tiles[op.inputs[1]] != h_tiles[out_tensor] || v_tiles[op.inputs[1]] != d_tiles
                                  || tile_dim[op.inputs[1]] != std::max(tile_dim[out_tensor]-1, (size_t)1)) {
                            valid = VALIDITY::SHAPES_MISALIGNED;
                            std::cout<<"ERROR (subgraph "<<subgraph_index<<"): Tile shape misalignment for Matmul operator "<<op_idx<<" with "<<
                                        d_tiles<<" tiles in depth dimension:";
                            if(h_tiles[op.inputs[1]] != h_tiles[out_tensor])
                                std::cout<<" second input tensor "<<op.inputs[1]<<" has "<<h_tiles[op.inputs[1]]<<" horizontal tiles, while output tensor "
                                    <<out_tensor<<" has "<<h_tiles[out_tensor]<<" horizontal tiles.";
                            if(v_tiles[op.inputs[1]] != d_tiles)
                                std::cout<<" second input tensor "<<op.inputs[1]<<" has "<<v_tiles[op.inputs[1]]<<
                                            " vertical tiles, whereas depth-tiling would require "<<d_tiles<<".";
                                            if(tile_dim[op.inputs[1]] != std::max(tile_dim[out_tensor]-1, (size_t)1))
                                std::cout<<" input tensor "<<op.inputs[1]<<" has abstract tiling dimension of both "<<tile_dim[op.inputs[1]]<<"D and "
                                    <<std::max(tile_dim[out_tensor]-1, (size_t)1)<<"D.";
                            std::cout<<std::endl;
                            return 0;
                        }
                    }
                    h_pos[op.inputs[0]] = d;
                    v_pos[op.inputs[0]] = v_pos[out_tensor];
                    h_pos[op.inputs[1]] = h_pos[out_tensor];
                    v_pos[op.inputs[1]] = d;
                }

                for(size_t input : op.inputs) {
                    if(find(in_and_out.first.begin(), in_and_out.first.end(), input) != in_and_out.first.end())
                        inputs.insert(input);
                }
                for(size_t output : op.outputs) {
                    if(find(in_and_out.second.begin(), in_and_out.second.end(), output) != in_and_out.second.end())
                        outputs.insert(output);
                }

                double v_size = static_cast<double>(problem.tensors[op.outputs[0]].height / v_tiles[op.outputs[0]]);
                double h_size = static_cast<double>(problem.tensors[op.outputs[0]].width / h_tiles[op.outputs[0]]);
                compute_time += static_cast<double>(op.base_cost)  / (is_sink[op_idx] ? static_cast<double>(d_tiles) : 1) *
                                std::max(static_cast<double>(h_size)
                                         / static_cast<double>(problem.native_granularity.width), 1.0) *
                                std::max(static_cast<double>(v_size)
                                         / static_cast<double>(problem.native_granularity.height), 1.0);
            }
            for(size_t input : inputs) {
                if(first_tile) {
                    mem_used += static_cast<double>(problem.tensors[input].height / v_tiles[input]) *
                                static_cast<double>(problem.tensors[input].width / h_tiles[input]);
                }
                if(first_tile || v_last_pos[input] != v_pos[input] || h_last_pos[input] != h_pos[input]) {
                    double h_size = static_cast<double>(problem.tensors[input].height / v_tiles[input]);
                    double v_size = static_cast<double>(problem.tensors[input].width / h_tiles[input]);
                    io_time += h_size * v_size / static_cast<double>(problem.slow_memory_bandwidth);
                }
            }
            for(size_t output : outputs) {
                if(first_tile) {
                    mem_used += static_cast<double>(problem.tensors[output].height / v_tiles[output]) *
                                static_cast<double>(problem.tensors[output].width / h_tiles[output]);
                }
                if(d_tiles <= 1 || d == d_tiles - 1) {
                    double h_size = static_cast<double>(problem.tensors[output].height / v_tiles[output]);
                    double v_size = static_cast<double>(problem.tensors[output].width / h_tiles[output]);
                    io_time += h_size * v_size / static_cast<double>(problem.slow_memory_bandwidth);
                }
            }

            if(first_tile) {
                if (subgraph_index > 0) {
                    for(size_t tensor_idx : solution.subgraphs[subgraph_index-1].tensors_to_retain) {
                        mem_used += static_cast<double>(problem.tensors[tensor_idx].height * problem.tensors[tensor_idx].width);
                    }
                }
                for(size_t tensor_idx : solution.subgraphs[subgraph_index].tensors_to_retain) {
                    mem_used += static_cast<double>(problem.tensors[tensor_idx].height * problem.tensors[tensor_idx].width);
                }

                if(mem_used > static_cast<double>(problem.fast_memory_capacity) + 0.0001) {
                    valid = VALIDITY::OUT_OF_MEMORY;
                    std::cout<<"ERROR (subgraph "<<subgraph_index<<"): Memory capacity violated. Requires "<<mem_used<<" memory, but only "<<
                        problem.fast_memory_capacity<<" is available."<<std::endl;
                    return 0;
                }
            }

            costDTile += std::max(io_time, compute_time);
            if(time_optimized && d==1)
                costThirdDim = std::max(io_time, compute_time);

            first_tile = false;
            for(size_t op_idx : subgraph.ops_reverse_top_order) {
                for(size_t input : problem.ops[op_idx].inputs) {
                    h_last_pos[input] = h_pos[input];
                    v_last_pos[input] = v_pos[input];
                }
            }
        }
        if(time_optimized && d_tiles > 3) {
            costDTile += static_cast<double>(d_tiles-3) * costThirdDim;
        }

        if(idx == 0)
            time += costDTile;

        size_t dim = (code != "vsnake") ? h_tiles[last_op.outputs[0]] : v_tiles[last_op.outputs[0]];
        if(idx == 1)
            time += static_cast<double>(dim-1) * costDTile;

        if(idx == 3)
            costSecondDim += costDTile;

        if(idx == 4)
            costSecondDim += static_cast<double>(dim-1) * costDTile;

        if(idx == 4 || (dim == 1 && idx == 3)) {
            size_t other_dim = (code != "vsnake") ? v_tiles[last_op.outputs[0]] : h_tiles[last_op.outputs[0]];
            time += static_cast<double>(other_dim-1) * costSecondDim;
        }

    }

    return time;
}

TotalLatency evaluate (const Problem& problem, Solution& solution, std::vector<size_t>& cost_mismatch_subgraphs) {

    TotalLatency latency = 0;
    bool isValid = true;

    if(!checkAcyclicity(problem, solution)) {
        std::cout<<"ERROR: seems like this subgraph configuration is not acyclic."<<std::endl;
        return -1;
    }

    if(!checkConnectivity(problem, solution)) {
        return -1;
    }

    if(!checkEphemeralization(problem, solution)) {
        return -1;
    }

    if(!checkRetainment(problem, solution)) {
        return -1;
    }

    for(size_t subgraph_index = 0; subgraph_index < solution.subgraphs.size(); ++subgraph_index) {
        VALIDITY subgraph_valid = VALIDITY::VALID;
        TotalLatency refCost = solution.subgraphs[subgraph_index].subgraph_latency;
        solution.subgraphs[subgraph_index].subgraph_latency = computeSubgraphCost (problem, solution, subgraph_index, subgraph_valid);
        if(subgraph_valid == VALIDITY::VALID)
            latency += solution.subgraphs[subgraph_index].subgraph_latency;

        if(abs(refCost - solution.subgraphs[subgraph_index].subgraph_latency) > 0.0001)
            cost_mismatch_subgraphs.push_back(subgraph_index);

        isValid = isValid && (subgraph_valid == VALIDITY::VALID);
    }
    return isValid ? latency : -1;
}


void Problem::initReverseTopOrder() {

    std::vector<std::vector<size_t> > parent_ops(ops.size()); // re-check the assumptions here
    std::deque<size_t> out_degree(ops.size());
    for(size_t op=0; op < ops.size(); ++op) {
        for(auto in_tensor : ops[op].inputs) {
            for(auto in_op : tensors[in_tensor].inputs) {
                parent_ops[op].push_back(in_op);
            }
        }
        for(auto out_tensor : ops[op].outputs) {
            out_degree[op] += tensors[out_tensor].outputs.size();
        }
    }

    ops_reverse_top_order.clear();
    std::deque<size_t> Q;
    for(size_t op=0; op < ops.size(); ++op)
        if(out_degree[op] == 0)
            Q.push_back(op);

    while(!Q.empty()) {
        size_t op = Q.front();
        Q.pop_front();
        ops_reverse_top_order.push_back(op);
        for(auto op_in : parent_ops[op]) {
            --out_degree[op_in];
            if(out_degree[op_in] == 0)
                Q.push_back(op_in);
        }
    }
}

void Solution::initTopOrders(const Problem& instance) {
    for(size_t subgraph_index = 0; subgraph_index < subgraphs.size(); ++subgraph_index)
        subgraphs[subgraph_index].ops_reverse_top_order.clear();
    for(size_t op : instance.ops_reverse_top_order) {
        for(size_t subgraph_index : ops_to_subgraphs[op])
            subgraphs[subgraph_index].ops_reverse_top_order.push_back(op);
    }
}

void Solution::initOpsToSubgraphs(const Problem& instance) {
    ops_to_subgraphs.clear();
    ops_to_subgraphs.resize(instance.ops.size());
    for(size_t subgraph_index = 0; subgraph_index < subgraphs.size(); ++subgraph_index)
        for(size_t op : subgraphs[subgraph_index].ops)
            ops_to_subgraphs[op].insert(subgraph_index);
}

std::vector<std::string> Split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

bool ReadProblem(const std::string& filename, Problem& instance) {

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Failed to open problem input file.\n";
        return false;
    }

    std::vector<Width> widths;
    std::vector<Height> heights;
    std::vector<Inputs> inputs;
    std::vector<Outputs> outputs;
    std::vector<BaseCost> baseCosts;
    std::vector<OpType> opTypes;

    std::string line;
    while (std::getline(infile, line)) {

        std::size_t colonPos = line.find(':');
        if (line.empty() || colonPos == std::string::npos) {
            continue;
        }

        std::string dataType = line.substr(0, colonPos-1);
        std::string data = line.substr(colonPos + 1);

        // further formatting
        std::size_t tempPos = data.find('[');
        if(tempPos != std::string::npos)
            data = data.substr(tempPos + 1);

        if(dataType.find("fast_memory_capacity") == std::string::npos &&
            dataType.find("slow_memory_bandwidth") == std::string::npos) {
            int diff = static_cast<int>(std::count(data.begin(), data.end(), ']')) - static_cast<int>(std::count(data.begin(), data.end(), '['));
            while(diff < 1) {
                if (!std::getline(infile, line))
                    break;
                data = data + line;
                diff = static_cast<int>(std::count(data.begin(), data.end(), ']')) - static_cast<int>(std::count(data.begin(), data.end(), '['));
            }

            tempPos = data.rfind(']');
            if(tempPos != std::string::npos)
                data = data.substr(0, tempPos);
        }
        data.erase(remove_if(data.begin(), data.end(), isspace), data.end());
        data.erase(remove(data.begin(), data.end(), '\n'), data.end());
        data.erase(remove(data.begin(), data.end(), '\"'), data.end());
        std::vector<std::string> dataItems;
        if(dataType.find("inputs") != std::string::npos || dataType.find("outputs") != std::string::npos) {
            dataItems = Split(data, ']');
        } else {
            dataItems = Split(data, ',');
        }
        std::replace(data.begin(), data.end(), ',', ' ');
        for(auto &entry : dataItems) {
            std::replace(entry.begin(), entry.end(), ',', ' ');
        }

        std::istringstream dataStream(data);

        if(dataType.find("fast_memory_capacity") != std::string::npos) {
            if(!(dataStream >> instance.fast_memory_capacity)) {
                std::cerr << "Error: Malformed problem input file.\n";
                return false;
            }
        } else if(dataType.find("slow_memory_bandwidth") != std::string::npos) {
            if(!(dataStream >> instance.slow_memory_bandwidth)) {
                std::cerr << "Error: Malformed problem input file.\n";
                return false;
            }
        } else if(dataType.find("native_granularity") != std::string::npos) {
            Width w;
            Height h;
            if(!(dataStream >> w >> h)) {
                std::cerr << "Error: Malformed problem input file.\n";
                return false;
            }
            instance.native_granularity.width = w;
            instance.native_granularity.height = h;

        } else if(dataType.find("widths") != std::string::npos) {
            Width w;
            while((dataStream >> w)) {
                widths.push_back(w);
            }
        } else if(dataType.find("heights") != std::string::npos) {
            Height h;
            while((dataStream >> h)) {
                heights.push_back(h);
            }
        } else if(dataType.find("base_costs") != std::string::npos) {
            BaseCost bc;
            while((dataStream >> bc)) {
                baseCosts.push_back(bc);
            }
        } else if(dataType.find("op_types") != std::string::npos) {
            for(auto entry : dataItems) {
                opTypes.push_back(entry);
            }
        } else if(dataType.find("inputs") != std::string::npos) {
            for(auto entry : dataItems) {
                inputs.emplace_back();
                tempPos = entry.find('[');
                if(tempPos != std::string::npos)
                    entry = entry.substr(tempPos + 1);
                tempPos = entry.rfind(']');
                if(tempPos != std::string::npos)
                    entry = entry.substr(0, tempPos);
                size_t i;
                std::istringstream localDataStream(entry);
                while((localDataStream >> i)) {
                    inputs.back().push_back(i);
                }
            }
        } else if(dataType.find("outputs") != std::string::npos) {
            for(auto entry : dataItems) {
                outputs.emplace_back();
                tempPos = entry.find('[');
                if(tempPos != std::string::npos)
                    entry = entry.substr(tempPos + 1);
                tempPos = entry.rfind(']');
                if(tempPos != std::string::npos)
                    entry = entry.substr(0, tempPos);
                size_t o;
                std::istringstream localDataStream(entry);
                while((localDataStream >> o)) {
                    outputs.back().push_back(o);
                }
            }
        } else {
            std::cerr << "Error: Unknown data type in problem input file.\n";
            return false;
        }
    }

    instance.tensors.resize(widths.size());
    for(size_t i=0; i < widths.size(); ++i) {
        instance.tensors[i].width = widths[i];
        instance.tensors[i].height = heights[i];
    }
    instance.ops.resize(opTypes.size());
    for(size_t i=0; i < opTypes.size(); ++i) {
        instance.ops[i].op_type = opTypes[i];
        instance.ops[i].inputs = inputs[i];
        instance.ops[i].outputs = outputs[i];
        for(auto in : inputs[i])
            instance.tensors[in].outputs.push_back(i);
        for(auto out : outputs[i])
            instance.tensors[out].inputs.push_back(i);
        instance.ops[i].base_cost = baseCosts[i];
    }
    instance.initReverseTopOrder();

    return true;
}

bool ReadSolution(const std::string& filename, const Problem& instance, Solution& solution) {

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Failed to open solution input file.\n";
        return false;
    }

    std::vector<Granularity> granularities;
    std::vector<std::vector<size_t> > ops;
    std::vector<std::unordered_set<size_t> > tensors_to_retain;
    std::vector<TraversalOrder> traversal_orders;
    std::vector<SubgraphLatency> latencies;

    std::string line;
    while (std::getline(infile, line)) {

        std::size_t colonPos = line.find(':');
        if (line.empty() || colonPos == std::string::npos) {
            continue;
        }

        std::string dataType = line.substr(0, colonPos-1);
        std::string data = line.substr(colonPos + 1);

        // further formatting
        std::size_t tempPos = data.find('[');
        if(tempPos != std::string::npos)
            data = data.substr(tempPos + 1);

        int diff = static_cast<int>(std::count(data.begin(), data.end(), ']')) - static_cast<int>(std::count(data.begin(), data.end(), '['));
        while(diff < 1) {
            if (!std::getline(infile, line))
                break;
            data = data + line;
            diff = static_cast<int>(std::count(data.begin(), data.end(), ']')) - static_cast<int>(std::count(data.begin(), data.end(), '['));
        }

        tempPos = data.rfind(']');
        if(tempPos != std::string::npos)
            data = data.substr(0, tempPos);
        data.erase(remove_if(data.begin(), data.end(), isspace), data.end());
        data.erase(remove(data.begin(), data.end(), '\"'), data.end());
        data.erase(remove(data.begin(), data.end(), '\n'), data.end());
        std::vector<std::string> dataItems;
        if(dataType.find("subgraph_latencies") != std::string::npos) {
            dataItems = Split(data, ','); // TODO traversal orders is not good here
        } else if (dataType.find("traversal_orders") != std::string::npos) {
            std::vector<std::string> tempDataItems = Split(data, ']');
            for(auto &entry : tempDataItems)
            {
                std::vector<std::string> innerDataItems = Split(entry, 'l');
                for(auto str: innerDataItems)
                    if(!str.empty())
                        dataItems.push_back(str);
            }
        } else {
            dataItems = Split(data, ']');
        }
        std::replace(data.begin(), data.end(), ',', ' ');
        for(auto &entry : dataItems) {
            std::replace(entry.begin(), entry.end(), ',', ' ');
        }

        std::istringstream dataStream(data);

        if(dataType.find("subgraphs") != std::string::npos) {
            for(auto entry : dataItems) {
                ops.emplace_back();
                tempPos = entry.find('[');
                if(tempPos != std::string::npos)
                    entry = entry.substr(tempPos + 1);
                tempPos = entry.rfind(']');
                if(tempPos != std::string::npos)
                    entry = entry.substr(0, tempPos);
                size_t op;
                std::istringstream localDataStream(entry);
                while((localDataStream >> op)) {
                    ops.back().push_back(op);
                }
            }
        } else if(dataType.find("granularities") != std::string::npos) {
            for(auto entry : dataItems) {
                granularities.emplace_back();
                tempPos = entry.find('[');
                if(tempPos != std::string::npos)
                    entry = entry.substr(tempPos + 1);
                tempPos = entry.rfind(']');
                if(tempPos != std::string::npos)
                    entry = entry.substr(0, tempPos);
                Width w; Height h; Depth d;
                std::istringstream localDataStream(entry);
                if(!(localDataStream >> w >> h >> d)) {
                    std::cerr << "Error: Malformed solution input file.\n";
                    return false;
                }
                granularities.back().width = w;
                granularities.back().height = h;
                granularities.back().depth = d;
            }
        } else if(dataType.find("tensors_to_retain") != std::string::npos) {
            for(auto entry : dataItems) {
                tensors_to_retain.emplace_back();
                tempPos = entry.find('[');
                if(tempPos != std::string::npos)
                    entry = entry.substr(tempPos + 1);
                tempPos = entry.rfind(']');
                if(tempPos != std::string::npos)
                    entry = entry.substr(0, tempPos);
                size_t t;
                std::istringstream localDataStream(entry);
                while((localDataStream >> t)) {
                    tensors_to_retain.back().insert(t);
                }
            }
        } else if(dataType.find("traversal_orders") != std::string::npos) {
            for(auto entry : dataItems) {
                traversal_orders.emplace_back();
                if(entry.find("nu") != std::string::npos)
                    continue;

                tempPos = entry.find('[');
                if(tempPos != std::string::npos)
                    entry = entry.substr(tempPos + 1);
                tempPos = entry.rfind(']');
                if(tempPos != std::string::npos)
                    entry = entry.substr(0, tempPos);
                size_t t;
                std::istringstream localDataStream(entry);
                while((localDataStream >> t)) {
                    traversal_orders.back().push_back(t);
                }
            }
        } else if(dataType.find("subgraph_latencies") != std::string::npos) {
            SubgraphLatency latency;
            while((dataStream >> latency)) {
                latencies.push_back(latency);
            }
        } else {
            std::cerr << "Error: Unknown data type in solution input file.\n";
            return false;
        }
    }

    solution.subgraphs.clear();
    solution.subgraphs.resize(ops.size());

    size_t nr_ops = 0;
    for(size_t i=0; i < ops.size(); ++i)
        for(size_t op : ops[i])
            nr_ops = std::max(nr_ops, op+1);
    solution.ops_to_subgraphs.resize(nr_ops);
    for(size_t i=0; i < ops.size(); ++i) {
        solution.subgraphs[i].ops = ops[i];
        for(size_t op : ops[i])
            solution.ops_to_subgraphs[op].insert(i);
        solution.subgraphs[i].granularity = granularities[i];
        solution.subgraphs[i].tensors_to_retain = tensors_to_retain[i];
        solution.subgraphs[i].subgraph_latency = latencies[i];
    }
    solution.initTopOrders(instance);

    for(size_t i=0; i < solution.subgraphs.size(); ++i) {
        if(!traversal_orders[i].empty()) {
            size_t last_op_idx = solution.subgraphs[i].ops_reverse_top_order.front();
            const Op& last_op = instance.ops[last_op_idx];
            size_t h_dim = std::max(instance.tensors[last_op.outputs[0]].width /
                                    solution.subgraphs[i].granularity.width, 1L);
            size_t v_dim = std::max(instance.tensors[last_op.outputs[0]].height /
                                    solution.subgraphs[i].granularity.height, 1L);
            if(traversal_orders[i] == createOrder(h_dim, v_dim, "hsnake"))
                solution.subgraphs[i].traversal_code = "hsnake";
            else if(traversal_orders[i] == createOrder(h_dim, v_dim, "vsnake"))
                solution.subgraphs[i].traversal_code = "vsnake";
        }

    }

    return true;
}

}  // namespace mlsys

using namespace mlsys;

int main(int argc, char* argv[]) {

    if(argc != 3) {
        std::cout<<"The evaluator requires 2 arguments: first the problem file name, then the solution file name."<<std::endl;
        std::cout<<"For example: ./evaluator mlsys-2026-1.json mlsys-2026-1_sol.json"<<std::endl;
        return 1;
    }

    std::string problem_filename = argv[1],  solution_filename = argv[2];

    Problem instance;
    if(!ReadProblem(problem_filename, instance))
        return 1;

    Solution solution;
    if(!ReadSolution(solution_filename, instance, solution))
        return 1;

    TotalLatency refCost;
    for(size_t idx = 0; idx < solution.subgraphs.size(); ++idx)
        refCost += solution.subgraphs[idx].subgraph_latency;

    std::vector<size_t> cost_mismatch_subgraphs;
    TotalLatency computedCost = evaluate(instance, solution, cost_mismatch_subgraphs);

    if(computedCost > -0.5) {
        std::cout<<"Reference cost: "<<refCost<<std::endl;
        std::cout<<"Computed cost:  "<<computedCost<<std::endl;
        if(abs(refCost-computedCost) > 0.001) {
            std::cout<<std::endl<<"ERROR: The costs do not seem to match in subgraphs ";
            for(size_t i = 0; i < cost_mismatch_subgraphs.size(); ++i) {
                if(i > 0)
                    std::cout<<", ";
                std::cout<<cost_mismatch_subgraphs[i];
            }
            std::cout<<"."<<std::endl;
        }
        else
            std::cout<<std::endl<<"Cost calculation seems correct."<<std::endl;
    }

    return 0;
}

#endif  // MLSYS_H_
