#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>

int pi(int argc, char* argv[]) {
    int numtasks, rank, len, rc;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        std::cout << "Error starting MPI program. Terminating.\n";
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(hostname, &len);

    int N = 100000;
    int num = N / numtasks;

    int count = 0;
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    for (int i = 0; i <= num; ++i) {
        double x = dist(mt);
        double y = dist(mt);
        if ((pow(x, 2) + pow(y, 2)) <= 0.25) {
            ++count;
        }
    }

    int all_count;
    // MPI_Allreduce(&count, &all_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);// or
    MPI_Reduce(&count, &all_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);


    double pi;
    pi = 4.0 * float(all_count) / N;
    if (rank == 0) {
        std::cout << "approximated pi=" << pi << std::endl;
    }


    MPI_Finalize();
    return EXIT_SUCCESS;
}