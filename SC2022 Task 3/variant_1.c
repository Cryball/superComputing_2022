#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <malloc.h>
#include <math.h>
#include <stdbool.h>


int *sended_size, *started_positions, *part_of_rows, *number_of_full_rows;
int cntOfProcesses, rank;
const double epsilon = 0.000001;

double *matrix_multiplication(double *part_of_matrix, double *x, int N) {
    int i, j;
    double *res = (double *) calloc(part_of_rows[rank], sizeof(double));
    for (i = 0; i < part_of_rows[rank]; ++i) {
        for (j = 0; j < N; ++j) {
            res[i] += part_of_matrix[i * N + j] * x[j];
        }
    }
    return res;
}

void next_y(double *part_of_A, double *x, double *b, double *part_of_y, double *final_y, int N) {
    int i;
    double *part_of_b = matrix_multiplication(part_of_A, x, N);
    for (i = 0; i < part_of_rows[rank]; ++i) {
        part_of_y[i] = part_of_b[i] - b[number_of_full_rows[rank] + i];
    }
    MPI_Allgatherv(part_of_y, part_of_rows[rank], MPI_DOUBLE,
                   final_y, part_of_rows, number_of_full_rows, MPI_DOUBLE, MPI_COMM_WORLD);
    free(part_of_b);
}

double scalar_product(const double *v1, const double *v2, int N) {
    int i;
    double current_process_sum = 0, result = 0;
    for (i = 0; i < part_of_rows[rank]; ++i) {
        current_process_sum += v1[i] * v2[i];
    }
    MPI_Allreduce(&current_process_sum, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return result;
}

double tau_func(double *part_of_a, double *part_of_y_last, double *last_x, int N) {
    double *part_of_y = matrix_multiplication(part_of_a, last_x, N);
    double numerator = scalar_product(part_of_y_last, part_of_y, N);
    double denominator = scalar_product(part_of_y, part_of_y, N);
    free(part_of_y);
    return numerator / denominator;
}

void calcNextX(double *x_last, double tau, const double *y, int N) {
   int i;
    for (i = 0; i < N; ++i) {
        x_last[i] -= tau * y[i];
    }
}

double norm(double *v, int N) {
    return sqrt(scalar_product(v, v, N));
}

bool stop_criteria(double *aPart, double *xn, const double *B, int N, double b_norm) {

    int i;
    double *numerator = matrix_multiplication(aPart, xn, N);
    for (i = 0; i < part_of_rows[rank]; ++i) {
        numerator[i] -= B[number_of_full_rows[rank] + i];
    }

    double given_norm = (norm(numerator, N) / b_norm);
    bool flag = given_norm < epsilon;

    free(numerator);
    return flag;
}

void solution(double *aPart, double *B, double *xn, int N, double b_norm) {
    double *ynPart = (double *)malloc(part_of_rows[rank] * sizeof(double));
    double *fullYN = (double *)malloc(N * sizeof(double ));
    double tau;
    while (!stop_criteria(aPart, xn, B, N, b_norm)) {
        next_y(aPart, xn, B, ynPart, fullYN, N);
        tau = tau_func(aPart, ynPart, fullYN, N);
        calcNextX(xn, tau, fullYN, N);
    }
    free(ynPart);
    free(fullYN);
}

void set_params(int N) {
    sended_size = (int*)malloc(cntOfProcesses * sizeof(int));
    started_positions = (int*)malloc(cntOfProcesses * sizeof(int));
    part_of_rows = (int*)malloc(cntOfProcesses * sizeof(int));
    number_of_full_rows = (int*)malloc(cntOfProcesses * sizeof(int));
    int offsetIdx = 0;
    int procRank = 0;
    for (procRank = 0; procRank < cntOfProcesses; ++procRank) {
        if (procRank < N % cntOfProcesses) {
            sended_size[procRank] = (N / cntOfProcesses + 1) * N;
} else {
            sended_size[procRank] = (N / cntOfProcesses) * N;
        }
        started_positions[procRank] = offsetIdx;
        number_of_full_rows[procRank] = offsetIdx / N;
        offsetIdx += sended_size[procRank];
        part_of_rows[procRank] = sended_size[procRank] / N;
    }
}

void allocate(double **aPart, double **B, double **X, int N) {
    *aPart = (double*)malloc(sended_size[rank] * sizeof(double));;
    *B = (double*)calloc(N, sizeof(double));
    *X = (double*)malloc(N * sizeof(double));
}

void deallocate(double* partial_a, double* A, double* B,
                double* X)
{
    free(partial_a);
    free(A);
    free(B);
    free(sended_size);
    free(started_positions);
    free(part_of_rows);
    free(number_of_full_rows);
    free(X);
}

void loadData(double *A, double *B, double *X, int N) {
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

int main(int argc, char **argv) {
    int N = 15000;
    srand(1);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &cntOfProcesses);

    double *aPart, *B, *X;
    double *A = (double*) malloc(N * N * sizeof(double));
    set_params(N);
    allocate(&aPart, &B, &X, N);
    double start = 0;
    if (rank == 0) {
        loadData(A, B, X, N);
        start = MPI_Wtime();
    }

    MPI_Scatterv(A, sended_size, started_positions, MPI_DOUBLE,
                 aPart, sended_size[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(X, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double b_norm = norm(B + number_of_full_rows[rank], N);
    solution(aPart, B, X, N, b_norm);

    if (rank == 0) {
        double end = MPI_Wtime();
        printf("%d\n", N);
        printf( "N = %d, cntOfProcesses = %d, time = %f\n", N, cntOfProcesses, end - start);
    }
    deallocate(aPart, A, B, X);
    MPI_Finalize();
}

