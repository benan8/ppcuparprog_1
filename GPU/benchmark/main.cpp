#include "bSimulator.hpp"
#include "benchmark/benchmark.h"

auto from = 32;
auto to = 1024;

BENCHMARK(BM_initU)->RangeMultiplier(2)->Range(from, to);
BENCHMARK(BM_initV)->RangeMultiplier(2)->Range(from, to);
BENCHMARK(BM_initP)->RangeMultiplier(2)->Range(from, to);
BENCHMARK(BM_solveUMomentum)->RangeMultiplier(2)->Range(from, to);
BENCHMARK(BM_solveVMomentum)->RangeMultiplier(2)->Range(from, to);
BENCHMARK(BM_solveContinuityEquationP)->RangeMultiplier(2)->Range(from, to);
BENCHMARK(BM_applyBoundaryU)->RangeMultiplier(2)->Range(from, to);
BENCHMARK(BM_applyBoundaryV)->RangeMultiplier(2)->Range(from, to);
BENCHMARK(BM_applyBoundaryP)->RangeMultiplier(2)->Range(from, to);
BENCHMARK(BM_calculateError)->RangeMultiplier(2)->Range(from, to);
BENCHMARK(BM_iterateU)->RangeMultiplier(2)->Range(from, to);
BENCHMARK(BM_iterateV)->RangeMultiplier(2)->Range(from, to);
BENCHMARK(BM_iterateP)->RangeMultiplier(2)->Range(from, to);
BENCHMARK(BM_constructor)->RangeMultiplier(2)->Range(from, to);
BENCHMARK(BM_run)->RangeMultiplier(2)->Range(from, to);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}