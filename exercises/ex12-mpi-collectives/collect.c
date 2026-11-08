#include "collect.h"

#include <math.h>
#include <mpi.h>
#include <stdlib.h>

static void build_counts_displs(int n, int size, int *counts, int *displs)
{
    int base = n / size;
    int rem = n % size;
    int off = 0;

    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = off;
        off += counts[r];
    }
}

double dist_sum(const double *data, int n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *counts = malloc((size_t)size * sizeof(*counts));
    int *displs = malloc((size_t)size * sizeof(*displs));

    build_counts_displs(n, size, counts, displs);

    int local_n = counts[rank];
    double *local = malloc((size_t)(local_n > 0 ? local_n : 1) * sizeof(*local));

    MPI_Scatterv(data, counts, displs, MPI_DOUBLE,
                 local, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double local_sum = 0.0;
    for (int i = 0; i < local_n; ++i)
        local_sum += local[i];

    double total = 0.0;
    MPI_Allreduce(&local_sum, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    free(local);
    free(counts);
    free(displs);

    return total;
}

double dist_max(const double *data, int n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *counts = malloc((size_t)size * sizeof(*counts));
    int *displs = malloc((size_t)size * sizeof(*displs));

    build_counts_displs(n, size, counts, displs);

    int local_n = counts[rank];
    double *local = malloc((size_t)(local_n > 0 ? local_n : 1) * sizeof(*local));

    MPI_Scatterv(data, counts, displs, MPI_DOUBLE,
                 local, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double local_max = -INFINITY;
    for (int i = 0; i < local_n; ++i) {
        if (local[i] > local_max)
            local_max = local[i];
    }

    double global_max = -INFINITY;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);

    free(local);
    free(counts);
    free(displs);

    return global_max;
}

void dist_scale(const double *data, int n, double scale, double *out)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *counts = malloc((size_t)size * sizeof(*counts));
    int *displs = malloc((size_t)size * sizeof(*displs));

    build_counts_displs(n, size, counts, displs);

    int local_n = counts[rank];
    double *local = malloc((size_t)(local_n > 0 ? local_n : 1) * sizeof(*local));

    MPI_Scatterv(data, counts, displs, MPI_DOUBLE,
                 local, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_n; ++i)
        local[i] *= scale;

    MPI_Gatherv(local, local_n, MPI_DOUBLE,
                out, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(local);
    free(counts);
    free(displs);
}
