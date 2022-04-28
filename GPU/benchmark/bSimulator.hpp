#include "simulator.hpp"
#include "benchmark/benchmark.h"

void BM_initU(benchmark::State& state) {
    auto grid = static_cast<Simulator::SizeType>(state.range(0));
    Simulator s{grid};
    for(auto _ : state) {
        s.initU();
    }
}

void BM_initV(benchmark::State& state) {
    auto grid = static_cast<Simulator::SizeType>(state.range(0));
    Simulator s{grid};
    for(auto _ : state) {
        s.initV();
    }
}

void BM_initP(benchmark::State& state) {
    auto grid = static_cast<Simulator::SizeType>(state.range(0));
    Simulator s{grid};
    for(auto _ : state) {
        s.initP();
    }
}

void BM_solveUMomentum(benchmark::State& state) {
    auto grid = static_cast<Simulator::SizeType>(state.range(0));
    Simulator s{grid};
    for(auto _ : state) {
        s.solveUMomentum(100.0);
    }
}

void BM_solveVMomentum(benchmark::State& state) {
    auto grid = static_cast<Simulator::SizeType>(state.range(0));
    Simulator s{grid};
    for(auto _ : state) {
        s.solveVMomentum(100.0);
    }
}

void BM_solveContinuityEquationP(benchmark::State& state) {
    auto grid = static_cast<Simulator::SizeType>(state.range(0));
    Simulator s{grid};
    for(auto _ : state) {
        s.solveContinuityEquationP(4.5);
    }
}

void BM_applyBoundaryU(benchmark::State& state) {
    auto grid = static_cast<Simulator::SizeType>(state.range(0));
    Simulator s{grid};
    for(auto _ : state) {
        s.applyBoundaryU();
    }
}

void BM_applyBoundaryV(benchmark::State& state) {
    auto grid = static_cast<Simulator::SizeType>(state.range(0));
    Simulator s{grid};
    for(auto _ : state) {
        s.applyBoundaryV();
    }
}

void BM_applyBoundaryP(benchmark::State& state) {
    auto grid = static_cast<Simulator::SizeType>(state.range(0));
    Simulator s{grid};
    for(auto _ : state) {
        s.applyBoundaryP();
    }
}

void BM_calculateError(benchmark::State& state) {
    auto grid = static_cast<Simulator::SizeType>(state.range(0));
    Simulator s{grid};
    for(auto _ : state) {
        s.calculateError();
    }
}

void BM_iterateU(benchmark::State& state) {
    auto grid = static_cast<Simulator::SizeType>(state.range(0));
    Simulator s{grid};
    for(auto _ : state) {
        s.iterateU();
    }
}

void BM_iterateV(benchmark::State& state) {
    auto grid = static_cast<Simulator::SizeType>(state.range(0));
    Simulator s{grid};
    for(auto _ : state) {
        s.iterateV();
    }
}

void BM_iterateP(benchmark::State& state) {
    auto grid = static_cast<Simulator::SizeType>(state.range(0));
    Simulator s{grid};
    for(auto _ : state) {
        s.iterateP();
    }
}

void BM_constructor(benchmark::State& state) {
    auto grid = static_cast<Simulator::SizeType>(state.range(0));
    for(auto _ : state) {
        Simulator s{grid};
    }
}

void BM_run(benchmark::State& state) {
    auto grid = static_cast<Simulator::SizeType>(state.range(0));
    Simulator s{grid};
    for(auto _ : state) {
        s.run(4.5, 100.0, 100);
    }
}