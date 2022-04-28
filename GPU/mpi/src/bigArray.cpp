#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

int bigArray(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int numtasks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int my_size;
    std::vector<float> myarray;
    std::vector<float> bigarray;
    if (rank == 0) {
        srand(1);
        int bigsize = numtasks * ((double)rand() / (double)RAND_MAX * 10000.0 + 1000);
        bigarray.resize(bigsize);
        for (int i = 0; i < bigsize; i++) {
            bigarray[i] = (float)rand() / (float)RAND_MAX;
        }
        my_size = bigsize / numtasks;
    }
    MPI_Bcast(&my_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    myarray.resize(my_size);
    double t1 = MPI_Wtime();
    MPI_Scatter(bigarray.data(), my_size, MPI_FLOAT, &myarray[0], my_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    double t2 = MPI_Wtime();
    if (rank == 0)
        printf("Time on rank 0 %g\n", t2 - t1);

    float l2sum = 0.0f;
    for (int i = 0; i < my_size; i++) {
        float s = sinf(myarray[i]);
        l2sum += s * s;
    }
    t1 = MPI_Wtime();
    printf("Compute time on rank %d: %g\n", rank, t1 - t2);
    float allsum;
    MPI_Reduce(&l2sum, &allsum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("L2 norm %g\n", sqrt(allsum));

    MPI_Finalize();
    return EXIT_SUCCESS;
}