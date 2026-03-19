#pragma once

#include "core/types.h"
#include "core/dag.h"
#include "symmetry/merkle_hash.h"
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <iostream>
#include <sstream>

// ============================================================================
// SeriesPattern: a repeating motif along the topological order.
//
// Unlike SymmetricPattern (parallel, disconnected components), a series
// pattern represents consecutive identical blocks in a chain:
//
//   Block_0 → Block_1 → Block_2 → ... → Block_{N-1}
//
// where each block has K ops with identical internal structure and
// identical inter-block wiring.
//
// The solver can solve the partition for one block and replicate it
// across all N blocks (shifting op indices by K per block).
// ============================================================================

struct SeriesPattern {
    size_t block_size;                     // K ops per block
    size_t num_blocks;                     // N consecutive identical blocks
    size_t first_block;                    // index of first matching block
                                           // (block i covers topo positions
                                           //  [first_block*K .. (first_block+1)*K))
    std::vector<size_t> representative;    // ops in the first matching block
    std::vector<std::vector<size_t>> blocks; // all N blocks' op lists

    size_t total_ops() const { return block_size * num_blocks; }
    double coverage(size_t total) const {
        return 100.0 * (double)total_ops() / (double)total;
    }

    std::string to_string() const {
        std::ostringstream os;
        os << "series: " << num_blocks << " blocks x " << block_size
           << " ops (total=" << total_ops() << ")";
        for (size_t i = 0; i < blocks.size() && i < 4; i++) {
            os << "  [";
            for (size_t j = 0; j < blocks[i].size(); j++) {
                if (j) os << ",";
                os << blocks[i][j];
            }
            os << "]";
        }
        if (blocks.size() > 4) os << " ...+" << (blocks.size() - 4);
        return os.str();
    }
};

// ============================================================================
// SeriesDetector
//
// Finds repeating motifs in the topological order by:
//   1. Assigning each op an init hash (type + cost + shapes)
//   2. Splitting the topo order into candidate blocks of size K
//   3. Finding the longest run of blocks with:
//      - identical init-hash sequences
//      - identical internal edge structure (same local edge positions)
//      - identical cross-block edge structure (same stride pattern)
//   4. Reporting all maximal series patterns
// ============================================================================

class SeriesDetector {
public:
    static std::vector<SeriesPattern> discover(
        const Problem& prob, const DAG& dag, const MerkleHashes& merkle,
        bool verbose = false)
    {
        const size_t N = prob.num_ops();
        if (N < 4) return {};

        const auto& topo = dag.topological_order();

        // Inverse: op → position in topo order
        std::vector<size_t> pos(N);
        for (size_t i = 0; i < N; i++)
            pos[topo[i]] = i;

        // ================================================================
        // 1. Build init-hash sequence along topo order
        // ================================================================
        std::vector<size_t> seq(N);
        for (size_t i = 0; i < N; i++)
            seq[i] = merkle.init[topo[i]];

        // ================================================================
        // 2. For each candidate block size K, find longest matching run
        // ================================================================

        // Precompute: for each op in topo order, its internal edges
        // (encoded as local position pairs within a hypothetical block).
        // We'll reuse this for each K.

        // For a block starting at topo position `base` with size K:
        //   internal_edges: set of (a, b) where topo[base+a] → topo[base+b]
        //                   and 0 <= a, b < K
        //   cross_edges:    set of (a, b) where topo[base+a] → topo[base+K+b]
        //                   and 0 <= a < K and 0 <= b < K

        auto get_internal_edges = [&](size_t base, size_t K)
            -> std::vector<std::pair<int,int>>
        {
            std::vector<std::pair<int,int>> edges;
            for (size_t a = 0; a < K && base + a < N; a++) {
                size_t u = topo[base + a];
                for (auto v_set : dag.op_succs[u]) {
                    size_t vpos = pos[v_set];
                    if (vpos >= base && vpos < base + K) {
                        edges.push_back({(int)a, (int)(vpos - base)});
                    }
                }
            }
            std::sort(edges.begin(), edges.end());
            return edges;
        };

        auto get_cross_edges = [&](size_t base, size_t K)
            -> std::vector<std::pair<int,int>>
        {
            std::vector<std::pair<int,int>> edges;
            size_t next_base = base + K;
            if (next_base + K > N) return edges;  // no full next block
            for (size_t a = 0; a < K; a++) {
                size_t u = topo[base + a];
                for (auto v_set : dag.op_succs[u]) {
                    size_t vpos = pos[v_set];
                    if (vpos >= next_base && vpos < next_base + K) {
                        edges.push_back({(int)a, (int)(vpos - next_base)});
                    }
                }
            }
            std::sort(edges.begin(), edges.end());
            return edges;
        };

        // Also count "long" edges (skip more than one block) and
        // "backward/external" edges (go before the block)
        auto get_back_edges = [&](size_t base, size_t K)
            -> std::vector<std::pair<int, int>>
        {
            // Edges from outside (before base) into this block
            std::vector<std::pair<int,int>> edges;
            for (size_t b = 0; b < K && base + b < N; b++) {
                size_t v = topo[base + b];
                for (auto u : dag.op_preds[v]) {
                    size_t upos = pos[u];
                    if (upos < base) {
                        // Encode as (distance_back, local_target)
                        // For matching: we need the relative structure, not absolute
                        // Use (upos offset from previous block start, b)
                        size_t prev_base = (base >= K) ? base - K : 0;
                        if (upos >= prev_base && upos < base) {
                            edges.push_back({(int)(upos - prev_base), (int)b});
                        }
                        // Edges from further back (e.g., residual from 2 blocks ago)
                        // are encoded differently — they break the pattern
                    }
                }
            }
            std::sort(edges.begin(), edges.end());
            return edges;
        };

        struct RunInfo {
            size_t K;           // block size
            size_t start;       // first matching block (index, not topo pos)
            size_t count;       // number of consecutive matching blocks
        };

        std::vector<RunInfo> best_runs;

        // Try block sizes from 2 up to N/2
        for (size_t K = 2; K <= N / 2; K++) {
            size_t num_full_blocks = N / K;
            if (num_full_blocks < 2) continue;

            // Check each pair of consecutive blocks for matching
            // hash sequence + internal edges + cross edges
            std::vector<bool> matches(num_full_blocks, false);

            // Reference: first block's structures
            // We compare block i with block i-1 (adjacent matching)
            for (size_t bi = 1; bi < num_full_blocks; bi++) {
                size_t base_prev = (bi - 1) * K;
                size_t base_curr = bi * K;

                // 1. Hash sequence match
                bool hash_match = true;
                for (size_t j = 0; j < K; j++) {
                    if (seq[base_prev + j] != seq[base_curr + j]) {
                        hash_match = false;
                        break;
                    }
                }
                if (!hash_match) continue;

                // 2. Internal edge structure match
                auto ie_prev = get_internal_edges(base_prev, K);
                auto ie_curr = get_internal_edges(base_curr, K);
                if (ie_prev != ie_curr) continue;

                // 3. Back-edge structure match (incoming from previous block)
                // This is the key inter-block check: each block receives
                // the same inputs from its predecessor.
                auto be_prev = get_back_edges(base_prev, K);
                auto be_curr = get_back_edges(base_curr, K);
                if (be_prev != be_curr) continue;

                matches[bi] = true;
            }

            // Find longest runs of consecutive matches
            size_t run_start = 0;
            size_t run_len = 1;
            for (size_t bi = 1; bi < num_full_blocks; bi++) {
                if (matches[bi]) {
                    run_len++;
                } else {
                    if (run_len >= 2) {
                        best_runs.push_back({K, run_start, run_len});
                    }
                    run_start = bi;
                    run_len = 1;
                }
            }
            if (run_len >= 2) {
                best_runs.push_back({K, run_start, run_len});
            }
        }

        // ================================================================
        // 3. Filter: keep only Pareto-optimal runs
        //    (no other run covers strictly more ops with >= block count)
        // ================================================================

        // Sort by total_ops desc
        std::sort(best_runs.begin(), best_runs.end(),
                  [](const RunInfo& a, const RunInfo& b) {
            size_t ta = a.K * a.count, tb = b.K * b.count;
            if (ta != tb) return ta > tb;
            return a.count > b.count;
        });

        // Remove runs whose ops are fully covered by a larger run
        std::vector<bool> run_dead(best_runs.size(), false);
        for (size_t i = 0; i < best_runs.size(); i++) {
            if (run_dead[i]) continue;
            size_t i_start = best_runs[i].start * best_runs[i].K;
            size_t i_end = i_start + best_runs[i].K * best_runs[i].count;

            for (size_t j = i + 1; j < best_runs.size(); j++) {
                if (run_dead[j]) continue;
                size_t j_start = best_runs[j].start * best_runs[j].K;
                size_t j_end = j_start + best_runs[j].K * best_runs[j].count;

                // j is covered by i if [j_start, j_end) ⊆ [i_start, i_end)
                if (j_start >= i_start && j_end <= i_end) {
                    run_dead[j] = true;
                }
            }
        }

        // ================================================================
        // 4. Build output patterns
        // ================================================================

        std::vector<SeriesPattern> patterns;
        for (size_t i = 0; i < best_runs.size(); i++) {
            if (run_dead[i]) continue;
            auto& run = best_runs[i];

            SeriesPattern pat;
            pat.block_size = run.K;
            pat.num_blocks = run.count;
            pat.first_block = run.start;

            for (size_t bi = 0; bi < run.count; bi++) {
                size_t base = (run.start + bi) * run.K;
                std::vector<size_t> block_ops;
                for (size_t j = 0; j < run.K; j++)
                    block_ops.push_back(topo[base + j]);
                if (bi == 0) pat.representative = block_ops;
                pat.blocks.push_back(std::move(block_ops));
            }

            patterns.push_back(std::move(pat));
        }

        // Sort: total ops desc, then block count desc
        std::sort(patterns.begin(), patterns.end(),
                  [](const SeriesPattern& a, const SeriesPattern& b) {
            if (a.total_ops() != b.total_ops())
                return a.total_ops() > b.total_ops();
            return a.num_blocks > b.num_blocks;
        });

        if (verbose) {
            std::cerr << "[series] discovered " << patterns.size()
                      << " patterns:\n";
            for (auto& p : patterns)
                std::cerr << "  " << p.to_string() << "\n";
        }

        return patterns;
    }
};