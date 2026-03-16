cmake_minimum_required(VERSION 3.16)
project(mlsys-solver CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address")

# Threading support
find_package(Threads REQUIRED)

# All source files can include from src/
include_directories(${CMAKE_SOURCE_DIR}/src)

# --- Library: all solver source (no main) ---
file(GLOB_RECURSE LIB_SOURCES
    src/core/*.cpp
    src/solution/*.cpp
    src/partition/*.cpp
    src/init/*.cpp
    src/search/*.cpp
    src/postopt/*.cpp
    src/io/*.cpp
    src/pipeline/solver.cpp
)

add_library(solver_lib STATIC ${LIB_SOURCES})
target_link_libraries(solver_lib Threads::Threads)

# --- Main executable ---
add_executable(mlsys src/pipeline/main.cpp)
target_link_libraries(mlsys solver_lib Threads::Threads)

add_executable(evaluate src/pipeline/evaluate.cpp)
target_link_libraries(evaluate solver_lib Threads::Threads)

# --- Tests ---
enable_testing()

set(TEST_SOURCES
    test/cross_validate.cpp
    test/unit_tests.cpp
    test/problem_examples_test.cpp
    test/chained_matmul_test.cpp
    test/fanin_matmul_test.cpp
    test/local_search_test.cpp
    test/init_strategies_test.cpp
    test/fm_test.cpp
    test/move_gain_test.cpp
    test/retain_snake_test.cpp
    test/constructed_optima_test.cpp
    test/cost_cache_test.cpp
    test/evolution_test.cpp
    test/solution_search_test.cpp
    test/sanity_test.cpp
    test/partition_fm_test.cpp
    test/more_unit_tests.cpp
    test/io_test.cpp
    test/dag_test.cpp
    test/subgraph_test.cpp
    test/cost_model_test.cpp
    test/fm_search_test.cpp
    test/partition_test.cpp
    test/parallel_search_test.cpp
    test/ordering_test.cpp
    test/solver_test.cpp
)

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} solver_lib)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# --- Verify target (uses the main binary) ---
add_custom_target(verify
    COMMAND $<TARGET_FILE:mlsys> --verify
    DEPENDS mlsys
    COMMENT "Verifying against PROBLEM.md examples"
)

# --- Benchmark run targets ---
# Expects benchmarks/ directory in project root
set(BENCH_DIR ${CMAKE_SOURCE_DIR}/benchmarks)

add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E echo "=== Running all benchmarks ==="
    DEPENDS mlsys
)

file(GLOB BENCH_FILES "${BENCH_DIR}/mlsys-2026-*.json")
foreach(bench ${BENCH_FILES})
    get_filename_component(bench_name ${bench} NAME_WE)
    add_custom_target(run_${bench_name}
        COMMAND $<TARGET_FILE:mlsys> ${bench} ${CMAKE_BINARY_DIR}/${bench_name}_sol.json
        DEPENDS mlsys
        COMMENT "Running ${bench_name}"
    )
endforeach()

add_custom_target(run_all
    DEPENDS mlsys
    COMMENT "Running all benchmarks"
)
add_custom_command(TARGET run_all POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Solutions written to build directory"
)