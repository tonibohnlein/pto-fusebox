cmake_minimum_required(VERSION 3.16)
project(pto-fusebox CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address")

# Option: keep O2 but enable gain-mismatch verification (no ASAN overhead)
# Usage: cmake -DGAIN_CHECK=ON -DCMAKE_BUILD_TYPE=Release ..
option(GAIN_CHECK "Enable gain-mismatch verification at O2" OFF)
if(GAIN_CHECK)
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g")
    # CMake adds -DNDEBUG for Release via COMPILE_DEFINITIONS; override it
    foreach(flag_var CMAKE_CXX_FLAGS_RELEASE CMAKE_C_FLAGS_RELEASE)
        string(REPLACE "-DNDEBUG" "" ${flag_var} "${${flag_var}}")
    endforeach()
    # Also strip from the per-config property CMake uses internally
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -UNDEBUG")
endif()

# Experimental dependency-constrained Gorder pebbling policy. The default DFS
# ordering remains the production policy until model and device comparison.
option(PTO_FUSEBOX_GORDER "Use dependency-constrained Gorder scheduling" OFF)

# Threading support
find_package(Threads REQUIRED)

# Option: statically link libstdc++ and libgcc so the binary is portable
# across Linux distros (glibc still dynamic — required for pthread, DNS).
# Usage: cmake -DSTATIC_STDLIB=ON -DCMAKE_BUILD_TYPE=Release ..
option(STATIC_STDLIB "Statically link libstdc++ and libgcc" OFF)
if(STATIC_STDLIB)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
endif()

# All source files can include from src/
include_directories(${CMAKE_SOURCE_DIR}/src)

# --- Library: all solver source (no main) ---
file(GLOB_RECURSE LIB_SOURCES
    src/core/*.cpp
    src/solution/*.cpp
    src/partition/*.cpp
    src/init/*.cpp
    src/search/*.cpp
    src/io/*.cpp
    src/pipeline/solver.cpp
)

add_library(solver_lib STATIC ${LIB_SOURCES})
if(PTO_FUSEBOX_GORDER)
    target_compile_definitions(solver_lib PUBLIC PTO_FUSEBOX_GORDER)
endif()
target_link_libraries(solver_lib Threads::Threads)

# --- Main executable ---
add_executable(mlsys src/pipeline/main.cpp)
target_link_libraries(mlsys solver_lib Threads::Threads)

# --- Mixed-model variant (cube+vector fusion) ---
# Same solver, compiled with the compile-time arch switch flipped to
# Ascend910BMixed (see src/core/subgraph.h). Used to solve/visualise fused mixed
# kernels; the default `mlsys` stays the unit-homogeneous model.
add_executable(mlsys_mixed ${LIB_SOURCES} src/pipeline/main.cpp)
target_compile_definitions(mlsys_mixed PRIVATE PYPTO_FUSE_CUBE_VECTOR)
if(PTO_FUSEBOX_GORDER)
    target_compile_definitions(mlsys_mixed PRIVATE PTO_FUSEBOX_GORDER)
endif()
target_link_libraries(mlsys_mixed Threads::Threads)

add_executable(symmetry_app src/pipeline/symmetry_app.cpp)
target_link_libraries(symmetry_app solver_lib Threads::Threads)


# --- Tests ---
enable_testing()

set(TEST_SOURCES
    # 910B cost-model suite (the only cost model after the competition removal).
    test/ascend_910b_test.cpp
    # Cost-agnostic structural infra still used by the 910B solver.
    test/dag_test.cpp
    test/pebbling_order_test.cpp
    test/pairing_heap_test.cpp
)

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} solver_lib)
    set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
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

# Per-case 910B fusion-DAG visualizations are rendered via
# scripts/render_fusion_dag.sh; the cost-model regression lives in the C++ test
# suite (test/ascend_910b_test.cpp).
