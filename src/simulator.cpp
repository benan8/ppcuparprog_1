#include "simulator.hpp"
#include "fmt/core.h"
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>

#include <ctime>
#include <vector>
#include <chrono>

struct data {
    std::string name;
    long unsigned int opC = 0;
    long int time;

    data(std::string _name, long int _time, long unsigned int o){
        name = _name;
        time = _time;
        opC = o;
    }
    
    void print(){

        float timeS = float(time)/1000;
        float speed = (float(opC)/1048576)/timeS;
        std::cout << name << "\t\t" << opC << "\t\t" << time << " ms" << "\t\t\t" << speed << std::endl;
    }
};

std::vector<data> adatok;

void Simulator::setPrinting(bool toPrint) { printing = toPrint; }

void Simulator::initU() {
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (SizeType i = 0; i <= (grid - 1); i++) {
        u[(i) * (grid + 1) + grid] = 1.0;
        u[(i) * (grid + 1) + grid - 1] = 1.0;
        #pragma omp parallel for
        for (SizeType j = 0; j < (grid - 1); j++) {
            u[(i) * (grid + 1) + j] = 0.0;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    data d("initU", std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count(),(grid - 1)*2*sizeof(float) + (grid - 1)*sizeof(float));
    adatok.push_back(d);
}

void Simulator::initV() {
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (SizeType i = 0; i <= (grid); i++) {
        #pragma omp parallel for
        for (SizeType j = 0; j <= (grid - 1); j++) {
            v[(i)*grid + j] = 0.0;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    data d("initV", std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count(),grid*(grid-1)*sizeof(float));
    adatok.push_back(d);
}

void Simulator::initP() {
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (SizeType i = 0; i <= (grid); i++) {
        #pragma omp parallel for
        for (SizeType j = 0; j <= (grid); j++) {
            p[(i) * (grid + 1) + j] = 1.0;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    data d("initP", std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count(),(grid)*(grid)*sizeof(float));
    adatok.push_back(d);
}

void Simulator::solveUMomentum(const FloatType Re) {
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (SizeType i = 1; i <= (grid - 2); i++) {
        #pragma omp parallel for
        for (SizeType j = 1; j <= (grid - 1); j++) {
            un[(i) * (grid + 1) + j] = u[(i) * (grid + 1) + j]
                - dt
                    * ((u[(i + 1) * (grid + 1) + j] * u[(i + 1) * (grid + 1) + j] - u[(i - 1) * (grid + 1) + j] * u[(i - 1) * (grid + 1) + j]) / 2.0 / dx
                    + 0.25 * ((u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j + 1]) * (v[(i)*grid + j] + v[(i + 1) * grid + j])
                            - (u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j - 1]) * (v[(i + 1) * grid + j - 1] + v[(i)*grid + j - 1])) / dy)
                    - dt / dx * (p[(i + 1) * (grid + 1) + j] - p[(i) * (grid + 1) + j]) + dt * 1.0 / Re
                    * ((u[(i + 1) * (grid + 1) + j] - 2.0 * u[(i) * (grid + 1) + j] + u[(i - 1) * (grid + 1) + j]) / dx / dx
                     + (u[(i) * (grid + 1) + j + 1] - 2.0 * u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j - 1]) / dy / dy);
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    data d("solveUMomentum", std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count(),(grid - 2)*(grid - 1)*(17*sizeof(float)));
    adatok.push_back(d);
}

void Simulator::applyBoundaryU() {
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (SizeType j = 1; j <= (grid - 1); j++) {
        un[(0) * (grid + 1) + j] = 0.0;
        un[(grid - 1) * (grid + 1) + j] = 0.0;
    }

    #pragma omp parallel for
    for (SizeType i = 0; i <= (grid - 1); i++) {
        un[(i) * (grid + 1) + 0] = -un[(i) * (grid + 1) + 1];
        un[(i) * (grid + 1) + grid] = 2 - un[(i) * (grid + 1) + grid - 1];
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    data d("applyBoundaryU", std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count(),2*(grid - 1)*sizeof(float)+(grid-1)*4*sizeof(float));
    adatok.push_back(d);
}

void Simulator::solveVMomentum(const FloatType Re) {
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (SizeType i = 1; i <= (grid - 1); i++) {
        #pragma omp parallel for
        for (SizeType j = 1; j <= (grid - 2); j++) {
            vn[(i)*grid + j] = v[(i)*grid + j]
                - dt * (0.25 * ((u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j + 1]) * (v[(i)*grid + j] + v[(i + 1) * grid + j])
                              - (u[(i - 1) * (grid + 1) + j] + u[(i - 1) * (grid + 1) + j + 1]) * (v[(i)*grid + j] + v[(i - 1) * grid + j])) / dx
                              + (v[(i)*grid + j + 1] * v[(i)*grid + j + 1] - v[(i)*grid + j - 1] * v[(i)*grid + j - 1]) / 2.0 / dy)
                              - dt / dy * (p[(i) * (grid + 1) + j + 1] - p[(i) * (grid + 1) + j]) + dt * 1.0 / Re
                              * ((v[(i + 1) * grid + j] - 2.0 * v[(i)*grid + j] + v[(i - 1) * grid + j]) / dx / dx
                              + (v[(i)*grid + j + 1] - 2.0 * v[(i)*grid + j] + v[(i)*grid + j - 1]) / dy / dy);
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    data d("solveVMomentum", std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count(),(grid - 2)*(grid - 1)*(17*sizeof(float)));
    adatok.push_back(d);
}

void Simulator::applyBoundaryV() {
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (SizeType j = 1; j <= (grid - 2); j++) {
        vn[(0) * grid + j] = -vn[(1) * grid + j];
        vn[(grid)*grid + j] = -vn[(grid - 1) * grid + j];
    }

    #pragma omp parallel for
    for (SizeType i = 0; i <= (grid); i++) {
        vn[(i)*grid + 0] = 0.0;
        vn[(i)*grid + grid - 1] = 0.0;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    data d("applyBoundaryV", std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count(),2*(grid - 2)*sizeof(float)+(grid)*4*sizeof(float));
    adatok.push_back(d);
}


void Simulator::solveContinuityEquationP(const FloatType delta) {
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (SizeType i = 1; i <= (grid - 1); i++) {
        #pragma omp parallel for
        for (SizeType j = 1; j <= (grid - 1); j++) {
            pn[(i) * (grid + 1) + j] = p[(i) * (grid + 1) + j]
                - dt * delta * ((un[(i) * (grid + 1) + j] - un[(i - 1) * (grid + 1) + j]) / dx + (vn[(i)*grid + j] - vn[(i)*grid + j - 1]) / dy);
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    data d("solveContinuityEquationP", std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count(),(grid - 1)*(grid - 1)*8*sizeof(float));
    adatok.push_back(d);
}

void Simulator::applyBoundaryP() {
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (SizeType i = 1; i <= (grid - 1); i++) {
        pn[(i) * (grid + 1) + 0] = pn[(i) * (grid + 1) + 1];
        pn[(i) * (grid + 1) + grid] = pn[(i) * (grid + 1) + grid - 1];
    }

   #pragma omp parallel for
    for (SizeType j = 0; j <= (grid); j++) {
        pn[(0) * (grid + 1) + j] = pn[(1) * (grid + 1) + j];
        pn[(grid) * (grid + 1) + j] = pn[(grid - 1) * (grid + 1) + j];
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    data d("applyBoundaryP", std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count(),(grid - 1)*4*sizeof(float)+grid*4*sizeof(float));
    adatok.push_back(d);
}

Simulator::FloatType Simulator::calculateError() {
    auto t1 = std::chrono::high_resolution_clock::now();
    FloatType error = 0.0;
    #pragma omp parallel for reduction(+:error)
    for (SizeType i = 1; i <= (grid - 1); i++) {
        #pragma omp parallel for reduction(+:error)
        for (SizeType j = 1; j <= (grid - 1); j++) {
            m[(i) * (grid + 1) + j] =
                ((un[(i) * (grid + 1) + j] - un[(i - 1) * (grid + 1) + j]) / dx + (vn[(i)*grid + j] - vn[(i)*grid + j - 1]) / dy);
            error += fabs(m[(i) * (grid + 1) + j]);
        }
    }
     auto t2 = std::chrono::high_resolution_clock::now();
    data d("calculateError", std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count(),(grid - 1)*(grid - 1)*9*sizeof(float));
    adatok.push_back(d);

    return error;
}

void Simulator::iterateU() {
    auto t1 = std::chrono::high_resolution_clock::now();
    //std::swap(u, un);
    #pragma omp parallel for
     for (SizeType i = 0; i <= (grid - 1); i++) {
         #pragma omp parallel for
         for (SizeType j = 0; j <= (grid); j++) {
             u[(i) * (grid + 1) + j] = un[(i) * (grid + 1) + j];
         }
     }

      auto t2 = std::chrono::high_resolution_clock::now();
    data d("iterateU", std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count(),(grid - 1)*(grid)*2*sizeof(float));
    adatok.push_back(d);
}

void Simulator::iterateV() {
    auto t1 = std::chrono::high_resolution_clock::now();
    //std::swap(v, vn);
    #pragma omp parallel for
     for (SizeType i = 0; i <= (grid); i++) {
         #pragma omp parallel for
         for (SizeType j = 0; j <= (grid - 1); j++) {
             v[(i)*grid + j] = vn[(i)*grid + j];
         }
     }

     auto t2 = std::chrono::high_resolution_clock::now();
    data d("initU", std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count(),(grid - 1)*(grid)*2*sizeof(float));
    adatok.push_back(d);
}

void Simulator::iterateP() {
    auto t1 = std::chrono::high_resolution_clock::now();
    //std::swap(p, pn);
    #pragma omp parallel for
     for (SizeType i = 0; i <= (grid); i++) {
         #pragma omp parallel for
         for (SizeType j = 0; j <= (grid); j++) {
             p[(i) * (grid + 1) + j] = pn[(i) * (grid + 1) + j];
         }
     }
     auto t2 = std::chrono::high_resolution_clock::now();
    data d("initU", std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count(),(grid)*(grid)*2*sizeof(float));
    adatok.push_back(d);
}

void Simulator::deallocate() {
    // it doesn't do anything until we use vectors
    // because that deallocates automatically
    // but if we have to use a more raw data structure later it is needed
    // and when the the Tests overwrites some member those might won't deallocate
}

Simulator::Simulator(SizeType gridP)
    : grid([](auto g) {
          if (g <= 1) {
              throw std::runtime_error("Grid is smaller or equal to 1.0, give larger number");
          }
          return g;
      }(gridP)),
      dx(1.0 / static_cast<FloatType>(grid - 1)),
      dy(1.0 / static_cast<FloatType>(grid - 1)),
      dt(0.001 / std::pow(grid / 128.0 * 2.0, 2.0)),
      u(grid * (grid + 1)),
      un(grid * (grid + 1)),
      v((grid + 1) * grid),
      vn((grid + 1) * grid),
      p((grid + 1) * (grid + 1)),
      pn((grid + 1) * (grid + 1)),
      m((grid + 1) * (grid + 1)) {
    initU();
    initV();
    initP();
}

void Simulator::run(const FloatType delta, const FloatType Re, unsigned maxSteps) {
    if (printing) {
        fmt::print("Running simulation with delta: {}, Re: {}\n", delta, Re);
    }
    auto error = std::numeric_limits<FloatType>::max();
    unsigned step = 1;
    while (error > 0.00000001 && step <= maxSteps) {
        solveUMomentum(Re);
        applyBoundaryU();

        solveVMomentum(Re);
        applyBoundaryV();

        solveContinuityEquationP(delta);
        applyBoundaryP();

        error = calculateError();

        if (printing && (step % 1000 == 1)) {
            fmt::print("Error is {} for the step {}\n", error, step);
        }

        iterateU();
        iterateV();
        iterateP();
        ++step;
    }
    std::cout << "Name" << "\t\t" << "count" << "\t\t" << "Time" << "\t\t\t" << "GB/s" << std::endl;
    for (unsigned i = 0; i < adatok.size(); i++){
        adatok[i].print();
    }
}

Simulator::~Simulator() { deallocate(); }