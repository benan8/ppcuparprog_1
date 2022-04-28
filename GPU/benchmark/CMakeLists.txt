include(FetchContent)

SET(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)

FetchContent_Declare(
    Bench
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.6.1
)
FetchContent_MakeAvailable(Bench)

add_executable(${BENCH_BINARY} ${CMAKE_CURRENT_LIST_DIR}/main.cpp)
target_include_directories(${BENCH_BINARY} PRIVATE benchmark)
# links to the SRC_LIBRARY which contains all the "real" code and the benchmark library
target_link_libraries(${BENCH_BINARY} ${SRC_LIBRARY} benchmark::benchmark)