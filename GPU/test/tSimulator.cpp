#include "simulator.hpp"
#include "gtest/gtest.h"
#include <exception>
#include <random>
#include <vector>

class SimulatorGrids : public ::testing::Test {
protected:
    std::vector<Simulator::SizeType> tc{ 4, 64, 327, 512 };
    std::default_random_engine engine{ 0 };

    std::vector<Simulator::FloatType> generateRandomArray(Simulator::SizeType length, Simulator::FloatType min = -5.0, Simulator::FloatType max = 5.0) {
        std::uniform_real_distribution<Simulator::FloatType> uniform_dist(min, max);
        std::vector<Simulator::FloatType> arr;
        arr.reserve(length);
        for (Simulator::SizeType i = 0; i < length; ++i) {
            arr.push_back(uniform_dist(engine));
        }
        return arr;
    }

    Simulator getNoisyState(Simulator::SizeType grid) {
        Simulator s{ grid };
        s.deallocate();
        s.u = generateRandomArray((grid + 1) * (grid + 1));
        s.un = generateRandomArray((grid + 1) * (grid + 1));
        s.v = generateRandomArray((grid + 1) * (grid + 1));
        s.vn = generateRandomArray((grid + 1) * (grid + 1));
        s.p = generateRandomArray((grid + 1) * (grid + 1), 0.0);
        s.pn = generateRandomArray((grid + 1) * (grid + 1), 0.0);
        s.m = generateRandomArray((grid + 1) * (grid + 1), 0.0);
        return s;
    }
};

TEST_F(SimulatorGrids, initU) {
    for (auto grid : tc) {
        auto s = getNoisyState(grid);
        s.initU();
        for (Simulator::SizeType i = 0; i <= (grid - 1); i++) {
            ASSERT_DOUBLE_EQ(s.u[(i) * (grid + 1) + grid], 1.0);
            ASSERT_DOUBLE_EQ(s.u[(i) * (grid + 1) + grid - 1], 1.0);
            for (Simulator::SizeType j = 0; j < (grid - 1); j++) {
                ASSERT_DOUBLE_EQ(s.u[(i) * (grid + 1) + j], 0.0);
            }
        }
    }
}

TEST_F(SimulatorGrids, initV) {
    for (auto grid : tc) {
        auto s = getNoisyState(grid);
        s.initV();
        for (const auto e : s.v) {
            ASSERT_DOUBLE_EQ(e, 0.0);
        }
    }
}

TEST_F(SimulatorGrids, initP) {
    for (auto grid : tc) {
        auto s = getNoisyState(grid);
        s.initP();
        for (const auto e : s.p) {
            ASSERT_DOUBLE_EQ(e, 1.0);
        }
    }
}

TEST(SimulatorInit, ConstructorSafe) {
    EXPECT_NO_THROW(Simulator s{ 1024 });
    EXPECT_THROW(Simulator s{ 0 }, std::runtime_error);
    EXPECT_THROW(Simulator s{ 1 }, std::runtime_error);
}

TEST_F(SimulatorGrids, ConstructorValues) {
    for (const auto grid : tc) {
        Simulator s{ grid };
        EXPECT_EQ(s.grid, grid);
        EXPECT_DOUBLE_EQ(s.dx, 1.0 / (grid - 1));
        EXPECT_DOUBLE_EQ(s.dy, 1.0 / (grid - 1));
        EXPECT_DOUBLE_EQ(s.dt, 0.001 / ((grid / 128.0 * 2) * (grid / 128.0 * 2)));
        EXPECT_EQ(s.u.size(), (grid + 1) * (grid + 1));
        EXPECT_EQ(s.un.size(), (grid + 1) * (grid + 1));
        EXPECT_EQ(s.v.size(), (grid + 1) * (grid + 1));
        EXPECT_EQ(s.vn.size(), (grid + 1) * (grid + 1));
        EXPECT_EQ(s.p.size(), (grid + 1) * (grid + 1));
        EXPECT_EQ(s.pn.size(), (grid + 1) * (grid + 1));
        EXPECT_EQ(s.m.size(), (grid + 1) * (grid + 1));
        for (const auto e : s.v) {
            ASSERT_DOUBLE_EQ(e, 0.0);
        }
        for (const auto e : s.p) {
            ASSERT_DOUBLE_EQ(e, 1.0);
        }
        for (Simulator::SizeType i = 0; i <= (grid - 1); i++) {
            ASSERT_DOUBLE_EQ(s.u[(i) * (grid + 1) + grid], 1.0);
            ASSERT_DOUBLE_EQ(s.u[(i) * (grid + 1) + grid - 1], 1.0);
            for (Simulator::SizeType j = 0; j < (grid - 1); j++) {
                ASSERT_DOUBLE_EQ(s.u[(i) * (grid + 1) + j], 0.0);
            }
        }
    }
}

TEST_F(SimulatorGrids, solveU) {
    const double Re = 121.0;
    for (auto grid : tc) {
        auto s = getNoisyState(grid);
        s.solveUMomentum(Re);
        for (Simulator::SizeType i = 1; i <= (grid - 2); i++) {
            for (Simulator::SizeType j = 1; j <= (grid - 1); j++) {
                ASSERT_DOUBLE_EQ(s.un[(i) * (grid + 1) + j],
                    s.u[(i) * (grid + 1) + j]
                        - s.dt
                            * ((s.u[(i + 1) * (grid + 1) + j] * s.u[(i + 1) * (grid + 1) + j]
                                   - s.u[(i - 1) * (grid + 1) + j] * s.u[(i - 1) * (grid + 1) + j])
                                    / 2.0 / s.dx
                                + 0.25
                                    * ((s.u[(i) * (grid + 1) + j] + s.u[(i) * (grid + 1) + j + 1]) * (s.v[(i)*(grid + 1) + j] + s.v[(i + 1) *(grid + 1) + j])
                                        - (s.u[(i) * (grid + 1) + j] + s.u[(i) * (grid + 1) + j - 1])
                                            * (s.v[(i + 1) *(grid + 1) + j - 1] + s.v[(i)*(grid + 1) + j - 1]))
                                    / s.dy)
                        - s.dt / s.dx * (s.p[(i + 1) * (grid + 1) + j] - s.p[(i) * (grid + 1) + j])
                        + s.dt * 1.0 / Re
                            * ((s.u[(i + 1) * (grid + 1) + j] - 2.0 * s.u[(i) * (grid + 1) + j] + s.u[(i - 1) * (grid + 1) + j]) / s.dx / s.dx
                                + (s.u[(i) * (grid + 1) + j + 1] - 2.0 * s.u[(i) * (grid + 1) + j] + s.u[(i) * (grid + 1) + j - 1]) / s.dy / s.dy));
            }
        }
    }
}

TEST_F(SimulatorGrids, boundU) {
    for (auto grid : tc) {
        auto s = getNoisyState(grid);
        s.applyBoundaryU();
        for (Simulator::SizeType j = 1; j <= (grid - 1); j++) {
            ASSERT_DOUBLE_EQ(s.un[(0) * (grid + 1) + j], 0.0);
            ASSERT_DOUBLE_EQ(s.un[(grid - 1) * (grid + 1) + j], 0.0);
        }

        for (Simulator::SizeType i = 0; i <= (grid - 1); i++) {
            ASSERT_DOUBLE_EQ(s.un[(i) * (grid + 1) + 0], -s.un[(i) * (grid + 1) + 1]);
            ASSERT_DOUBLE_EQ(s.un[(i) * (grid + 1) + grid], 2 - s.un[(i) * (grid + 1) + grid - 1]);
        }
    }
}

TEST_F(SimulatorGrids, solveV) {
    const double Re = 61.9;
    for (auto grid : tc) {
        auto s = getNoisyState(grid);
        s.solveVMomentum(Re);
        for (Simulator::SizeType i = 1; i <= (grid - 1); i++) {
            for (Simulator::SizeType j = 1; j <= (grid - 2); j++) {
                ASSERT_DOUBLE_EQ(s.vn[(i)*(grid + 1) + j],
                    s.v[(i)*(grid + 1) + j]
                        - s.dt
                            * (0.25
                                    * ((s.u[(i) * (grid + 1) + j] + s.u[(i) * (grid + 1) + j + 1]) * (s.v[(i)*(grid + 1) + j] + s.v[(i + 1) *(grid + 1) + j])
                                        - (s.u[(i - 1) * (grid + 1) + j] + s.u[(i - 1) * (grid + 1) + j + 1])
                                            * (s.v[(i)*(grid + 1) + j] + s.v[(i - 1) *(grid + 1) + j]))
                                    / s.dx
                                + (s.v[(i)*(grid + 1) + j + 1] * s.v[(i)*(grid + 1) + j + 1] - s.v[(i)*(grid + 1) + j - 1] * s.v[(i)*(grid + 1) + j - 1]) / 2.0 / s.dy)
                        - s.dt / s.dy * (s.p[(i) * (grid + 1) + j + 1] - s.p[(i) * (grid + 1) + j])
                        + s.dt * 1.0 / Re
                            * ((s.v[(i + 1) *(grid + 1) + j] - 2.0 * s.v[(i)*(grid + 1) + j] + s.v[(i - 1) *(grid + 1) + j]) / s.dx / s.dx
                                + (s.v[(i)*(grid + 1) + j + 1] - 2.0 * s.v[(i)*(grid + 1) + j] + s.v[(i)*(grid + 1) + j - 1]) / s.dy / s.dy));
            }
        }
    }
}

TEST_F(SimulatorGrids, boundV) {
    for (auto grid : tc) {
        auto s = getNoisyState(grid);
        s.applyBoundaryV();
        for (Simulator::SizeType j = 1; j <= (grid - 2); j++) {
            ASSERT_DOUBLE_EQ(s.vn[(0) *(grid + 1) + j], -s.vn[(1) *(grid + 1) + j]);
            ASSERT_DOUBLE_EQ(s.vn[(grid)*(grid + 1) + j], -s.vn[(grid - 1) *(grid + 1) + j]);
        }

        for (Simulator::SizeType i = 0; i <= (grid); i++) {
            ASSERT_DOUBLE_EQ(s.vn[(i)*(grid + 1) + 0], 0.0);
            ASSERT_DOUBLE_EQ(s.vn[(i)*(grid + 1) + grid - 1], 0.0);
        }
    }
}

TEST_F(SimulatorGrids, solveP) {
    const double delta = 5.2;
    for (auto grid : tc) {
        auto s = getNoisyState(grid);
        s.solveContinuityEquationP(delta);
        for (Simulator::SizeType i = 1; i <= (grid - 1); i++) {
            for (Simulator::SizeType j = 1; j <= (grid - 1); j++) {
                ASSERT_DOUBLE_EQ(s.pn[(i) * (grid + 1) + j],
                    s.p[(i) * (grid + 1) + j]
                        - s.dt * delta
                            * ((s.un[(i) * (grid + 1) + j] - s.un[(i - 1) * (grid + 1) + j]) / s.dx
                                + (s.vn[(i)*(grid + 1) + j] - s.vn[(i)*(grid + 1) + j - 1]) / s.dy));
            }
        }
    }
}

TEST_F(SimulatorGrids, boundP) {
    for (auto grid : tc) {
        auto s = getNoisyState(grid);
        s.applyBoundaryP();
        for (Simulator::SizeType i = 1; i <= (grid - 1); i++) {
            ASSERT_DOUBLE_EQ(s.pn[(i) * (grid + 1) + 0], s.pn[(i) * (grid + 1) + 1]);
            ASSERT_DOUBLE_EQ(s.pn[(i) * (grid + 1) + grid], s.pn[(i) * (grid + 1) + grid - 1]);
        }

        for (Simulator::SizeType j = 0; j <= (grid); j++) {
            ASSERT_DOUBLE_EQ(s.pn[(0) * (grid + 1) + j], s.pn[(1) * (grid + 1) + j]);
            ASSERT_DOUBLE_EQ(s.pn[(grid) * (grid + 1) + j], s.pn[(grid - 1) * (grid + 1) + j]);
        }
    }
}

TEST_F(SimulatorGrids, calculateError) {
    for (auto grid : tc) {
        auto s = getNoisyState(grid);
        auto err = s.calculateError();

        double error = 0.0;
        for (Simulator::SizeType i = 1; i <= (grid - 1); i++) {
            for (Simulator::SizeType j = 1; j <= (grid - 1); j++) {
                error += fabs(s.m[(i) * (grid + 1) + j]);
            }
        }

        ASSERT_NEAR(err, error, 0.0001);
    }
}

TEST_F(SimulatorGrids, iterU) {
    for (auto grid : tc) {
        auto s = getNoisyState(grid);
        auto length = grid * (grid + 1);
        std::vector<double> old{ std::begin(s.un), std::begin(s.un) + length };
        s.iterateU();
        for (Simulator::SizeType i = 0; i < length; ++i) {
            ASSERT_DOUBLE_EQ(s.u[i], old[i]);
        }
    }
}

TEST_F(SimulatorGrids, iterV) {
    for (auto grid : tc) {
        auto s = getNoisyState(grid);
        auto length = grid * (grid + 1);
        std::vector<double> old{ std::begin(s.vn), std::begin(s.vn) + length };
        s.iterateV();
        for (Simulator::SizeType i = 0; i < length; ++i) {
            ASSERT_DOUBLE_EQ(s.v[i], old[i]);
        }
    }
}

TEST_F(SimulatorGrids, iterP) {
    for (auto grid : tc) {
        auto s = getNoisyState(grid);
        auto length = (grid + 1) * (grid + 1);
        std::vector<double> old{ std::begin(s.pn), std::begin(s.pn) + length };
        s.iterateP();
        for (Simulator::SizeType i = 0; i < length; ++i) {
            ASSERT_DOUBLE_EQ(s.p[i], old[i]);
        }
    }
}
