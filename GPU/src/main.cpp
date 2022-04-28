#include "fmt/core.h"
#include "simulator.hpp"
#include <iostream>
#include <optional>
#include <stdlib.h>
#include "omp.h"

int main(int argc, char** argv) {
    // Size from command line if specified
    Simulator::SizeType grid = 256;
    if (argc > 1) {
        auto tmp = atoi(argv[1]);
        if (tmp <= 0) {
            std::cerr << "Grid should be larger than 0\n";
            return EXIT_FAILURE;
        }
        grid = static_cast<Simulator::SizeType>(tmp);
    }
    std::optional<unsigned> maxSteps;
    if (argc > 2) {
        auto tmp = atoi(argv[2]);
        if (tmp <= 0) {
            std::cerr << "MaxSteps should be larger than 0\n";
            return EXIT_FAILURE;
        }
        maxSteps = static_cast<decltype(maxSteps)::value_type>(tmp);
    }
    fmt::print("The size of the grid is {}\n", grid);
    Simulator s{ grid };
    Simulator::setPrinting(true);
    fmt::print("Number of used OpenMP threads is {}\n", omp_get_max_threads());
    if (maxSteps) {
        fmt::print("The maximum number of steps is {}\n", maxSteps.value());
        s.run(4.5, 100.0, maxSteps.value());
    } else {
        fmt::print("There is no maximum steps, it will run until it goes into a steady state");
        s.run(4.5, 100.0);
    }
}
