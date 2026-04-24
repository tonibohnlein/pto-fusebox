# mlsys26

Track A submission for the MLSys 2026 Scheduling Contest. C++ solver
that takes a JSON description of a computation graph plus hardware
constraints and emits a JSON schedule (subgraph grouping, tile
granularities, retention sets, traversal orders) minimising total
roofline latency.

## Requirements

- Linux x86_64 (Ubuntu 22.04 LTS or compatible)
- C++20 compiler — `g++ 12` or newer recommended (gcc 11.4 has a
  miscompile that produces infeasible solutions on this code base)
- CMake ≥ 3.16
- `nlohmann-json3-dev` (Debian/Ubuntu) or `nlohmann-json-devel`
  (Fedora/RHEL)

On Ubuntu 22.04:

```bash
sudo apt install build-essential cmake g++-12 nlohmann-json3-dev
```

## Build

Standard release build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

To produce a portable binary with libstdc++ and libgcc statically
linked (only glibc remains dynamic — required for the contest
submission), add `-DSTATIC_STDLIB=ON`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTATIC_STDLIB=ON
cmake --build build -j2 --target mlsys
ldd build/mlsys      # should list only linux-vdso, libm, libc, ld-linux
```

If your default `g++` is older than 12, set `CXX` explicitly:

```bash
CXX=g++-12 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTATIC_STDLIB=ON
```

Use `-j2` rather than higher parallelism — the larger source files in
`src/search/` are memory-hungry to compile.

## Run

```bash
./build/mlsys <input.json> <output.json>
```

The time budget is parsed from the benchmark filename (e.g.
`mlsys-2026-9.json` → 15s); for any other filename it falls back to a
size-based heuristic. The solver writes the solution to `<output.json>`
before its deadline expires, so it works correctly under an external
timeout-and-kill harness.

A reference evaluator binary is also produced; it replays a solution
and cross-checks per-step feasibility and latency against the
problem's cost model:

```bash
./build/evaluate <input.json> <output.json>
```

## Tests

47 unit and integration tests cover the cost model, every move type,
FM/greedy correctness, ordering, coupling, and end-to-end regression.

```bash
cd build && ctest -j2 --output-on-failure
```

## Layout

```
src/
  core/        DAG, subgraph, tiling, cost cache
  partition/   Partition state, eval_set, group bookkeeping
  init/        Initialisation strategies (chain+edge, seed+grow, ...)
  search/      Greedy, FM (inner pass + outer loop), evolution,
               coupling-aware search, parallel orchestration
  solution/    Schedule construction (DFS / beam ordering), latency
  symmetry/    Merkle hashing, parallel + series pattern detection
  io/          JSON I/O
  pipeline/    main, evaluate, solver entry points
benchmarks/    Released contest benchmarks + custom test graphs
test/          Unit and integration tests
doc/           Cost model, FM, ephemeral-gap, acyclicity notes
scripts/       Regression harness, dimension verifier, build helpers
```

## License

MIT (see `LICENSE`).
