#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <malloc.h>
#include <math.h>
#include <stdbool.h>

int *sended_size, *started_positions, *part_of_rows, *number_of_full_rows;
int cntOfProcesses, rank;
const double epsilon = 0.000001;

double *matrix_multiplication(double *part_of_matrix, const double *x, int N) {
    int i, j, ringIteration;
    double *result = (double*)calloc(sended_size[rank], sizeof(double));
    double *receiveVectBuf = (double*)malloc(sended_size[0] * sizeof(double));
    for (i = 0; i < sended_size[rank]; ++i) {
        receiveVectBuf[i] = x[i];
    }

    int ourSenderRank = (rank + 1) % cntOfProcesses;
    int ourRecipientRank = (rank + cntOfProcesses - 1) % cntOfProcesses;
    int curSenderRank;
    for (ringIteration = 0, curSenderRank = rank; ringIteration < cntOfProcesses; ++ringIteration,
            curSenderRank = (curSenderRank + 1) % cntOfProcesses) {

        for (i = 0; i < sended_size[rank]; ++i) {
            for (j = 0; j < sended_size[curSenderRank]; ++j) {
                result[i] += part_of_matrix[i * N + started_positions[curSenderRank] + j] * receiveVectBuf[j];
            }
        }
        MPI_Sendrecv_replace(receiveVectBuf, sended_size[0], MPI_DOUBLE, ourRecipientRank, 5,
                             ourSenderRank, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    return result;
}

void next_y(double *part_of_a, double *part_of_x, const double *part_of_b, double *part_of_y, int N) {
    int i;
    double *part_of_new_y = matrix_multiplication(part_of_a, part_of_x, N);
    for (i = 0; i < sended_size[rank]; ++i) {
        part_of_y[i] = part_of_new_y[i] - part_of_b[i];
    }
    free(part_of_new_y);
}

double scalar_product(const double *v1, const double *v2) {
    int i;
    double curNodeRes = 0;
    double scalarMulRes = 0;
    for (i = 0; i < sended_size[rank]; ++i) {
        curNodeRes += v1[i] * v2[i];
    }
    MPI_Allreduce(&curNodeRes, &scalarMulRes, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return scalarMulRes;
                                                                                   }

double tau_func(double *part_of_a, double *part_of_y, int N) {
    double *part_of_A_yn = matrix_multiplication(part_of_a, part_of_y, N);

    double numerator = scalar_product(part_of_y, part_of_A_yn);
    double denominator = scalar_product(part_of_A_yn, part_of_A_yn);
    free(part_of_A_yn);
    return numerator / denominator;
}

void get_next_x(double *last_x, double tau, double *y) {
    int i;
    for (i = 0; i < sended_size[rank]; ++i) {
        last_x[i] -= tau * y[i];
    }
}

double norm(double *v) {
    return sqrt(scalar_product(v, v));
}

bool stop_criteria(double *aPart, double *xnPart, const double *bPart, double b_norm, int N) {
    double *numerator = matrix_multiplication(aPart, xnPart, N);
    int i;
    for (i = 0; i < sended_size[rank]; ++i) {
        numerator[i] -= bPart[i];
    }
    bool flag = (norm(numerator) / b_norm) < epsilon;
    free(numerator);
    return flag;
}

void get_X(double *aPart, double *bPart, double *xnPart, double b_norm, int N) {
    double *ynPart = (double*) malloc(sended_size[rank] * sizeof(double));
    double tau;
    while (!stop_criteria(aPart, xnPart, bPart, b_norm, N)) {
        next_y(aPart, xnPart, bPart, ynPart, N);
        tau = tau_func(aPart, ynPart, N);
        get_next_x(xnPart, tau, ynPart);
    }
    free(ynPart);
}

void initialize(int N) {
    sended_size = (int*) malloc(cntOfProcesses * sizeof(int));
    started_positions = (int*) malloc(cntOfProcesses * sizeof(int));
    part_of_rows = (int*) malloc(cntOfProcesses * sizeof(int));
    number_of_full_rows = (int*) malloc(cntOfProcesses * sizeof(int));
     int offsetIdx = 0;
    int procRank;
    for (procRank = 0; procRank < cntOfProcesses; ++procRank) {
        if (procRank < N % cntOfProcesses) {
            part_of_rows[procRank] = (N / cntOfProcesses + 1) * N;
        } else {
            part_of_rows[procRank] = (N / cntOfProcesses) * N;
        }
        number_of_full_rows[procRank] = offsetIdx;
        started_positions[procRank] = offsetIdx / N;
        sended_size[procRank] = part_of_rows[procRank] / N;
        offsetIdx += part_of_rows[procRank];
    }
}

void allocMem(double **aPart, double **bPart, double **xPart) {
    *aPart = (double*) malloc(part_of_rows[rank] * sizeof(double));
    *bPart = (double*) malloc(part_of_rows[rank] * sizeof(double));
    *xPart = (double*) malloc(part_of_rows[rank] * sizeof(double));
}

void fill_matrix(double *A, double *B, double *X, int N) {
    int i, j;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            if (i == j)
                A[i * N+j] = 2;
            else
                A[i * N + j] = 0.004;
        }
    }
    for (i = 0; i < N; ++i) {
        B[i] = i;
    }
    for (i = 0; i < N; ++i) {
        X[i] = 0;
    }

}

void deallocate(double *aPart, double *bPart, double *xPart, double *A, double *B, double *X)
{
    free(aPart);
    free(bPart);
    free(xPart);
    free(sended_size);
    free(part_of_rows);
    free(started_positions);
    free(number_of_full_rows);
 free(A);
    free(B);
    free(X);
}

int main(int argc, char **argv) {
    int N = 15000;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &cntOfProcesses);

    double *aPart, *bPart, *xPart;
    double *A = (double*) malloc(N * N * sizeof(double));
    double *B = (double*) calloc(N, sizeof(double));
    double *X = (double*) malloc(N * sizeof(double));


    initialize(N);
    allocMem(&aPart, &bPart, &xPart);
    double start;
    if (rank == 0) {
        start = MPI_Wtime();
        fill_matrix(A, B, X, N);
    }

    MPI_Scatterv(A, part_of_rows, number_of_full_rows, MPI_DOUBLE,
                 aPart, part_of_rows[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, sended_size, started_positions, MPI_DOUBLE,
                 bPart, sended_size[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(X, sended_size, started_positions, MPI_DOUBLE,
                 xPart, sended_size[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double b_norm = norm(bPart);
    get_X(aPart, bPart, xPart, b_norm, N);
    MPI_Gatherv(xPart, sended_size[rank], MPI_DOUBLE,
                X, sended_size, started_positions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double end = MPI_Wtime();
        printf("N = %d, cntOfProcesses = %d, time = %f\n", N, cntOfProcesses, end - start);
    }
    MPI_Finalize();
    deallocate(aPart, bPart, xPart, A, B, X);
}

