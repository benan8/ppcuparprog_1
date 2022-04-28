#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int pingPong(int argc, char* argv[]) {
    int numtasks, rank, len, rc;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (!(numtasks == 2)) {
        printf("Can only run with 2 tasks\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    MPI_Status status;
    int* array = (int*)malloc(100000000 * sizeof(int));
    int* array2 = (int*)malloc(100000000 * sizeof(int));
    for (unsigned int i = 1; i < 100000000; i *= 2) {
        double t1 = MPI_Wtime();
        if (rank == 0) {
            MPI_Send(array, i, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(array2, i, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
        } else {
            MPI_Recv(array, i, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Send(array2, i, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
        double t2 = MPI_Wtime();
        if (rank == 0)
            printf("Sent %d bytes, latency: %g, bandwidth %g MB/s\n", i, t2 - t1, 2.0 * (double)(i * 4) / (t2 - t1) / 1024.0 / 1024.0);
    }
    free(array);
    free(array2);
    MPI_Finalize();
}