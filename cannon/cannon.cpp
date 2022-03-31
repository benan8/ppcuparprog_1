#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

int allocMatrix(int ***mat, int rows, int cols){

	int *p = (int *)malloc(sizeof(int *) * rows * cols);
	if (!p){
		return -1;
	}
	*mat = (int **)malloc(rows * sizeof(int *));
	if (!mat){
		free(p);
		return -1;
	}

	for (int i = 0; i < rows; i++){
		(*mat)[i] = &(p[i * cols]);
	}
	return 0;
}

int freeMatrix(int ***mat){
	free(&((*mat)[0][0]));
	free(*mat);
	return 0;
}

void matrixMultiply(int **a, int **b, int rows, int cols, int ***c){
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			int val = 0;
			for (int k = 0; k < rows; k++){
				val += a[i][k] * b[k][j];
			}
			(*c)[i][j] = val;
		}
	}
}

void printMatrix(int **mat, int size){
	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			std::cout << mat[i][j] << ' ';
		}
		std::cout << std::endl;
	}
}

void printMatrixFile(int **mat, int size, FILE *fp){
	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			fprintf(fp, "%d ", mat[i][j]);
		}
		fprintf(fp, "\n");
	}
}

int main(int argc, char *argv[]){
	MPI_Comm cannon_comm;
	int dim[2], period[2], reorder;
	int coord[2], id;
	FILE *fp;
	int **A = NULL, **B = NULL, **C = NULL;
	int **localA = NULL, **localB = NULL, **localC = NULL;
	int **localARec = NULL, **localBRec = NULL;
	int rows = 0;
	int cols;
	int count = 0;
	int w_size;
	int procDim;
	int blockDim;
	int left, right, up, down;
	int bCastData[4];

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &w_size);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0){
		int n;
		char ch;

		fp = fopen("A.txt", "r");
		if (fp == NULL){
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		while (fscanf(fp, "%d", &n) != EOF){
			ch = fgetc(fp);
			if (ch == '\n'){
				rows = rows + 1;
			}
			count++;
		}
		cols = count / rows;

		if (cols != rows){
			printf("Matrix must be square!\n");
			MPI_Abort(MPI_COMM_WORLD, 2);
		}
		double sqroot = sqrt(w_size);
		if ((sqroot - floor(sqroot)) != 0){
			printf("Number of processes must be a perfect square!\n");
			MPI_Abort(MPI_COMM_WORLD, 2);
		}
		int intRoot = (int)sqroot;
		if (cols % intRoot != 0 || rows % intRoot != 0){
			printf("Number of rows/cols not divisible by %d!\n", intRoot);
			MPI_Abort(MPI_COMM_WORLD, 3);
		}
		procDim = intRoot;
		blockDim = cols / intRoot;

		fseek(fp, 0, SEEK_SET);

		if (allocMatrix(&A, rows, cols) != 0){
			printf("Matrix alloc for A failed!\n");
			MPI_Abort(MPI_COMM_WORLD, 4);
		}
		if (allocMatrix(&B, rows, cols) != 0){
			printf("Matrix alloc for B failed!\n");
			MPI_Abort(MPI_COMM_WORLD, 5);
		}

		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				fscanf(fp, "%d", &n);
				A[i][j] = n;
			}
		}
		printf("A matrix:\n");
		printMatrix(A, rows);
		fclose(fp);

		fp = fopen("B.txt", "r");
		if (fp == NULL){
			return 1;
		}
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				fscanf(fp, "%d", &n);
				B[i][j] = n;
			}
		}
		printf("B matrix:\n");
		printMatrix(B, rows);
		fclose(fp);

		if (allocMatrix(&C, rows, cols) != 0){
			printf("Matrix C alloc failed!\n");
			MPI_Abort(MPI_COMM_WORLD, 6);
		}

		bCastData[0] = procDim;
		bCastData[1] = blockDim;
		bCastData[2] = rows;
		bCastData[3] = cols;
	}

	MPI_Bcast(&bCastData, 4, MPI_INT, 0, MPI_COMM_WORLD);
	procDim = bCastData[0];
	blockDim = bCastData[1];
	rows = bCastData[2];
	cols = bCastData[3];

	dim[0] = procDim;
	dim[1] = procDim;
	period[0] = 1;
	period[1] = 1;
	reorder = 1;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &cannon_comm);

	allocMatrix(&localA, blockDim, blockDim);
	allocMatrix(&localB, blockDim, blockDim);

	int globalSize[2] = {rows, cols};
	int localSize[2] = {blockDim, blockDim};
	int starts[2] = {0, 0};
	MPI_Datatype type, subarrtype;
	MPI_Type_create_subarray(2, globalSize, localSize, starts, MPI_ORDER_C, MPI_INT, &type);
	MPI_Type_create_resized(type, 0, blockDim * sizeof(int), &subarrtype);
	MPI_Type_commit(&subarrtype);

	int *globalptrA = NULL;
	int *globalptrB = NULL;
	int *globalptrC = NULL;
	if (rank == 0){
		globalptrA = &(A[0][0]);
		globalptrB = &(B[0][0]);
		globalptrC = &(C[0][0]);
	}

	int *sendCounts = (int *)malloc(sizeof(int) * w_size);
	int *displacements = (int *)malloc(sizeof(int) * w_size);

	if (rank == 0){
		for (int i = 0; i < w_size; i++){
			sendCounts[i] = 1;
		}
		int disp = 0;
		for (int i = 0; i < procDim; i++){
			for (int j = 0; j < procDim; j++){
				displacements[i * procDim + j] = disp;
				disp += 1;
			}
			disp += (blockDim - 1) * procDim;
		}
	}

	MPI_Scatterv(globalptrA, sendCounts, displacements, subarrtype, &(localA[0][0]),
				 rows * cols / (w_size), MPI_INT,
				 0, MPI_COMM_WORLD);
	MPI_Scatterv(globalptrB, sendCounts, displacements, subarrtype, &(localB[0][0]),
				 rows * cols / (w_size), MPI_INT,
				 0, MPI_COMM_WORLD);

	if (allocMatrix(&localC, blockDim, blockDim) != 0){
		printf("Matrix alloc in rank %d failed!\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 7);
	}

	MPI_Cart_coords(cannon_comm, rank, 2, coord);
	MPI_Cart_shift(cannon_comm, 1, coord[0], &left, &right);
	MPI_Sendrecv_replace(&(localA[0][0]), blockDim * blockDim, MPI_INT, left, 1, right, 1, cannon_comm, MPI_STATUS_IGNORE);
	MPI_Cart_shift(cannon_comm, 0, coord[1], &up, &down);
	MPI_Sendrecv_replace(&(localB[0][0]), blockDim * blockDim, MPI_INT, up, 1, down, 1, cannon_comm, MPI_STATUS_IGNORE);


	for (int i = 0; i < blockDim; i++){
		for (int j = 0; j < blockDim; j++){
			localC[i][j] = 0;
		}
	}

	int **multiplyRes = NULL;
	if (allocMatrix(&multiplyRes, blockDim, blockDim) != 0){
		printf("Matrix alloc in rank %d failed!\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 8);
	}
	for (int k = 0; k < procDim; k++){
		matrixMultiply(localA, localB, blockDim, blockDim, &multiplyRes);

		for (int i = 0; i < blockDim; i++){
			for (int j = 0; j < blockDim; j++){
				localC[i][j] += multiplyRes[i][j];
			}
		}

		MPI_Cart_shift(cannon_comm, 1, 1, &left, &right);
		MPI_Cart_shift(cannon_comm, 0, 1, &up, &down);
		MPI_Sendrecv_replace(&(localA[0][0]), blockDim * blockDim, MPI_INT, left, 1, right, 1, cannon_comm, MPI_STATUS_IGNORE);
		MPI_Sendrecv_replace(&(localB[0][0]), blockDim * blockDim, MPI_INT, up, 1, down, 1, cannon_comm, MPI_STATUS_IGNORE);
	}

	MPI_Gatherv(&(localC[0][0]), rows * cols / w_size, MPI_INT,
				globalptrC, sendCounts, displacements, subarrtype,
				0, MPI_COMM_WORLD);

	freeMatrix(&localC);
	freeMatrix(&multiplyRes);

	if (rank == 0){
		printf("C is:\n");
		printMatrix(C, rows);
	}

	MPI_Finalize();

	return 0;
}
