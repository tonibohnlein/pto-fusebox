CXX      := g++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra
SRCDIR   := src
BUILDDIR := build
TESTDIR  := test
TARGET   := mlsys

# Find all source files recursively
SRCS := $(shell find $(SRCDIR) -name '*.cpp')
OBJS := $(SRCS:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)

.PHONY: all clean verify run test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Pattern rule: src/subdir/file.cpp -> build/subdir/file.o
$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -c -o $@ $<

clean:
	rm -rf $(BUILDDIR) $(TARGET) cross_validate unit_tests problem_examples_test \
		chained_matmul_test fanin_matmul_test local_search_test \
		init_strategies_test fm_test

verify: $(TARGET)
	./$(TARGET) --verify

# --- Test infrastructure ---

# All object files except main.o (tests provide their own main)
TEST_OBJS := $(filter-out $(BUILDDIR)/pipeline/main.o, $(OBJS))

# Analytical vs simulation cross-validation
cross_validate: $(TESTDIR)/cross_validate.cpp $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $^
	./$@

# Unit tests for Subgraph, DAG, cost, traversal, Solution, retain
unit_tests: $(TESTDIR)/unit_tests.cpp $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $^
	./$@

# All PROBLEM.md examples with per-step latency checks
problem_examples_test: $(TESTDIR)/problem_examples_test.cpp $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $^
	./$@

# Chained MatMul topology tests
chained_matmul_test: $(TESTDIR)/chained_matmul_test.cpp $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $^
	./$@

# Fan-in MatMul topology tests
fanin_matmul_test: $(TESTDIR)/fanin_matmul_test.cpp $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $^
	./$@

# Local search component tests
local_search_test: $(TESTDIR)/local_search_test.cpp $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $^
	./$@

# Initialization strategy tests
init_strategies_test: $(TESTDIR)/init_strategies_test.cpp $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $^
	./$@

# FM search building block + pass + outer loop tests
fm_test: $(TESTDIR)/fm_test.cpp $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $^
	./$@

test: verify cross_validate unit_tests problem_examples_test chained_matmul_test \
      fanin_matmul_test local_search_test init_strategies_test fm_test

# --- Benchmark runs ---

run: $(TARGET)
	@for f in ../MLSys/benchmarks/mlsys-2026-*.json; do \
		name=$$(basename $$f .json); \
		echo "=== $$name ==="; \
		timeout 120 ./$(TARGET) $$f /tmp/$${name}_sol.json; \
		echo ""; \
	done
