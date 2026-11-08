/*
 * Assignment 3 — distributed 2-D heat diffusion (five-point Jacobi).
 *
 * Build with mpicc -O2 -std=c11 -Wall -Wextra -Werror.
 * Run as: mpirun -np P ./assign3 N STEPS REPETITIONS [VERIFY]
 *
 * The PARTNER token in this working filename must be replaced with the
 * second roll number before producing the pair-named submission artifact.
 */

#include <mpi.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TAG_NORTH = 10,
    TAG_SOUTH = 11,
    TAG_WEST = 12,
    TAG_EAST = 13
};

typedef struct {
    double wall_min;
    double wall_max;
    double comm_max_mean;
    double compute_max_mean;
    double comm_fraction_max_mean;
} Timing;

static void mpi_check(int code, const char *call, const char *file, int line)
{
    if (code != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(code, message, &length);
        fprintf(stderr, "%s:%d: %s failed: %.*s\n",
                file, line, call, length, message);
        MPI_Abort(MPI_COMM_WORLD, code);
        exit(EXIT_FAILURE);
    }
}

#define MPI_CHECK(call) mpi_check((call), #call, __FILE__, __LINE__)

static size_t index_of(int row, int col, int stride)
{
    return (size_t)row * (size_t)stride + (size_t)col;
}

static int parse_positive(const char *text, int allow_zero, int *value)
{
    char *end = NULL;
    long parsed;

    if (text == NULL || *text == '\0')
        return 0;
    parsed = strtol(text, &end, 10);
    if (*end != '\0' || parsed < (allow_zero ? 0 : 1) || parsed > INT32_MAX)
        return 0;
    *value = (int)parsed;
    return 1;
}

static int block_size(int extent, int parts, int coord)
{
    int base = extent / parts;
    int remainder = extent % parts;
    return base + (coord < remainder ? 1 : 0);
}

static int block_start(int extent, int parts, int coord)
{
    int base = extent / parts;
    int remainder = extent % parts;
    return coord * base + (coord < remainder ? coord : remainder);
}

static double initial_value(int global_row, int global_col)
{
    unsigned value = (unsigned)global_row * 17u + (unsigned)global_col * 13u;
    return (double)(value % 101u) / 101.0;
}

static void initialize_local(double *grid, int local_rows, int local_cols,
                             int start_row, int start_col)
{
    int stride = local_cols + 2;
    size_t cells = (size_t)(local_rows + 2) * (size_t)stride;
    memset(grid, 0, cells * sizeof(*grid));

    for (int row = 1; row <= local_rows; ++row) {
        for (int col = 1; col <= local_cols; ++col) {
            grid[index_of(row, col, stride)] =
                initial_value(start_row + row - 1, start_col + col - 1);
        }
    }
}

static void update_cell(const double *src, double *dst, int row, int col,
                        int stride, int global_row, int global_col, int n)
{
    if (global_row == 0 || global_row == n - 1 ||
        global_col == 0 || global_col == n - 1)
        return;

    size_t at = index_of(row, col, stride);
    dst[at] = 0.25 * (src[at - (size_t)stride] +
                      src[at + (size_t)stride] +
                      src[at - 1] + src[at + 1]);
}

static void compute_strict_interior(const double *src, double *dst,
                                    int local_rows, int local_cols,
                                    int start_row, int start_col, int n)
{
    int stride = local_cols + 2;
    for (int row = 2; row < local_rows; ++row) {
        for (int col = 2; col < local_cols; ++col) {
            update_cell(src, dst, row, col, stride,
                        start_row + row - 1, start_col + col - 1, n);
        }
    }
}

static void compute_local_boundary(const double *src, double *dst,
                                   int local_rows, int local_cols,
                                   int start_row, int start_col, int n)
{
    int stride = local_cols + 2;
    for (int row = 1; row <= local_rows; ++row) {
        for (int col = 1; col <= local_cols; ++col) {
            if (row == 1 || row == local_rows ||
                col == 1 || col == local_cols) {
                update_cell(src, dst, row, col, stride,
                            start_row + row - 1, start_col + col - 1, n);
            }
        }
    }
}

static void exchange_blocking(double *grid, int local_rows, int local_cols,
                              int north, int south, int west, int east,
                              MPI_Datatype column_type, MPI_Comm cart,
                              double *comm_time)
{
    int stride = local_cols + 2;
    double t;

    t = MPI_Wtime();
    MPI_CHECK(MPI_Sendrecv(&grid[index_of(1, 1, stride)], local_cols,
                          MPI_DOUBLE, north, TAG_NORTH,
                          &grid[index_of(local_rows + 1, 1, stride)],
                          local_cols, MPI_DOUBLE, south, TAG_NORTH,
                          cart, MPI_STATUS_IGNORE));
    *comm_time += MPI_Wtime() - t;

    t = MPI_Wtime();
    MPI_CHECK(MPI_Sendrecv(&grid[index_of(local_rows, 1, stride)], local_cols,
                          MPI_DOUBLE, south, TAG_SOUTH,
                          &grid[index_of(0, 1, stride)], local_cols,
                          MPI_DOUBLE, north, TAG_SOUTH,
                          cart, MPI_STATUS_IGNORE));
    *comm_time += MPI_Wtime() - t;

    t = MPI_Wtime();
    MPI_CHECK(MPI_Sendrecv(&grid[index_of(1, 1, stride)], 1, column_type,
                          west, TAG_WEST,
                          &grid[index_of(1, local_cols + 1, stride)], 1,
                          column_type, east, TAG_WEST,
                          cart, MPI_STATUS_IGNORE));
    *comm_time += MPI_Wtime() - t;

    t = MPI_Wtime();
    MPI_CHECK(MPI_Sendrecv(&grid[index_of(1, local_cols, stride)], 1,
                          column_type, east, TAG_EAST,
                          &grid[index_of(1, 0, stride)], 1, column_type,
                          west, TAG_EAST, cart, MPI_STATUS_IGNORE));
    *comm_time += MPI_Wtime() - t;
}

static void serial_reference(double *grid, int n, int steps)
{
    size_t cells = (size_t)n * (size_t)n;
    double *a = malloc(cells * sizeof(*a));
    double *b = malloc(cells * sizeof(*b));
    if (a == NULL || b == NULL) {
        fprintf(stderr, "could not allocate serial reference\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col)
            a[(size_t)row * (size_t)n + (size_t)col] =
                initial_value(row, col);

    for (int step = 0; step < steps; ++step) {
        memcpy(b, a, cells * sizeof(*a));
        for (int row = 1; row < n - 1; ++row) {
            for (int col = 1; col < n - 1; ++col) {
                size_t at = (size_t)row * (size_t)n + (size_t)col;
                b[at] = 0.25 * (a[at - (size_t)n] +
                                a[at + (size_t)n] +
                                a[at - 1] + a[at + 1]);
            }
        }
        double *swap = a;
        a = b;
        b = swap;
    }

    memcpy(grid, a, cells * sizeof(*grid));
    free(a);
    free(b);
}

static Timing run_solver(int n, int steps, int repetitions, int use_overlap,
                         int local_rows, int local_cols,
                         int start_row, int start_col,
                         int north, int south, int west, int east,
                         MPI_Datatype column_type, MPI_Comm cart,
                         int rank, double *final_grid)
{
    int stride = local_cols + 2;
    size_t cells = (size_t)(local_rows + 2) * (size_t)stride;
    double *a = malloc(cells * sizeof(*a));
    double *b = malloc(cells * sizeof(*b));
    if (a == NULL || b == NULL) {
        fprintf(stderr, "rank %d: could not allocate local grid\n", rank);
        MPI_Abort(cart, EXIT_FAILURE);
    }

    Timing result = {0.0, 0.0, 0.0, 0.0, 0.0};
    double wall_min = HUGE_VAL;
    double wall_max = 0.0;
    double comm_max_sum = 0.0;
    double compute_max_sum = 0.0;
    double comm_fraction_max_sum = 0.0;

    /* One warm-up run is discarded; `repetitions` timed runs follow. */
    for (int repetition = 0; repetition <= repetitions; ++repetition) {
        initialize_local(a, local_rows, local_cols, start_row, start_col);
        initialize_local(b, local_rows, local_cols, start_row, start_col);
        double local_comm = 0.0;
        double local_compute = 0.0;

        MPI_CHECK(MPI_Barrier(cart));
        double wall_start = MPI_Wtime();
        for (int step = 0; step < steps; ++step) {
            if (use_overlap) {
                MPI_Request requests[8];
                int request_count = 0;
                /* Post halo receives and sends, timing their MPI-call cost. */
                double t = MPI_Wtime();
                MPI_CHECK(MPI_Irecv(&a[index_of(0, 1, stride)], local_cols,
                                    MPI_DOUBLE, north, TAG_SOUTH, cart,
                                    &requests[request_count++]));
                local_comm += MPI_Wtime() - t;
                t = MPI_Wtime();
                MPI_CHECK(MPI_Irecv(&a[index_of(local_rows + 1, 1, stride)],
                                    local_cols, MPI_DOUBLE, south, TAG_NORTH,
                                    cart, &requests[request_count++]));
                local_comm += MPI_Wtime() - t;
                t = MPI_Wtime();
                MPI_CHECK(MPI_Irecv(&a[index_of(1, 0, stride)], 1, column_type,
                                    west, TAG_EAST, cart,
                                    &requests[request_count++]));
                local_comm += MPI_Wtime() - t;
                t = MPI_Wtime();
                MPI_CHECK(MPI_Irecv(&a[index_of(1, local_cols + 1, stride)], 1,
                                    column_type, east, TAG_WEST, cart,
                                    &requests[request_count++]));
                local_comm += MPI_Wtime() - t;
                t = MPI_Wtime();
                MPI_CHECK(MPI_Isend(&a[index_of(1, 1, stride)], local_cols,
                                    MPI_DOUBLE, north, TAG_NORTH, cart,
                                    &requests[request_count++]));
                local_comm += MPI_Wtime() - t;
                t = MPI_Wtime();
                MPI_CHECK(MPI_Isend(&a[index_of(local_rows, 1, stride)],
                                    local_cols, MPI_DOUBLE, south, TAG_SOUTH,
                                    cart, &requests[request_count++]));
                local_comm += MPI_Wtime() - t;
                t = MPI_Wtime();
                MPI_CHECK(MPI_Isend(&a[index_of(1, 1, stride)], 1, column_type,
                                    west, TAG_WEST, cart,
                                    &requests[request_count++]));
                local_comm += MPI_Wtime() - t;
                t = MPI_Wtime();
                MPI_CHECK(MPI_Isend(&a[index_of(1, local_cols, stride)], 1,
                                    column_type, east, TAG_EAST, cart,
                                    &requests[request_count++]));
                local_comm += MPI_Wtime() - t;

                t = MPI_Wtime();
                compute_strict_interior(a, b, local_rows, local_cols,
                                        start_row, start_col, n);
                local_compute += MPI_Wtime() - t;

                t = MPI_Wtime();
                MPI_CHECK(MPI_Waitall(request_count, requests,
                                      MPI_STATUSES_IGNORE));
                local_comm += MPI_Wtime() - t;

                t = MPI_Wtime();
                compute_local_boundary(a, b, local_rows, local_cols,
                                       start_row, start_col, n);
                local_compute += MPI_Wtime() - t;
            } else {
                exchange_blocking(a, local_rows, local_cols,
                                  north, south, west, east,
                                  column_type, cart, &local_comm);
                double t = MPI_Wtime();
                compute_strict_interior(a, b, local_rows, local_cols,
                                        start_row, start_col, n);
                compute_local_boundary(a, b, local_rows, local_cols,
                                       start_row, start_col, n);
                local_compute += MPI_Wtime() - t;
            }

            double *swap = a;
            a = b;
            b = swap;
        }
        double local_wall = MPI_Wtime() - wall_start;
        if (repetition == 0)
            continue;

        double global_wall = 0.0;
        double global_comm_max = 0.0;
        double global_compute_max = 0.0;
        double local_fraction = 0.0;
        double global_fraction = 0.0;
        double instrumented_local = local_comm + local_compute;
        if (instrumented_local > 0.0)
            local_fraction = local_comm / instrumented_local;

        MPI_CHECK(MPI_Reduce(&local_wall, &global_wall, 1, MPI_DOUBLE, MPI_MAX,
                             0, cart));
        MPI_CHECK(MPI_Reduce(&local_comm, &global_comm_max, 1, MPI_DOUBLE,
                             MPI_MAX, 0, cart));
        MPI_CHECK(MPI_Reduce(&local_compute, &global_compute_max, 1, MPI_DOUBLE,
                             MPI_MAX, 0, cart));
        MPI_CHECK(MPI_Reduce(&local_fraction, &global_fraction, 1, MPI_DOUBLE,
                             MPI_MAX, 0, cart));
        if (rank == 0) {
            if (global_wall < wall_min)
                wall_min = global_wall;
            if (global_wall > wall_max)
                wall_max = global_wall;
            comm_max_sum += global_comm_max;
            compute_max_sum += global_compute_max;
            comm_fraction_max_sum += global_fraction;
        }
    }

    if (rank == 0) {
        result.wall_min = wall_min;
        result.wall_max = wall_max;
        result.comm_max_mean = comm_max_sum / (double)repetitions;
        result.compute_max_mean = compute_max_sum / (double)repetitions;
        result.comm_fraction_max_mean =
            comm_fraction_max_sum / (double)repetitions;
    }

    /* Copy the local final interior into a compact result buffer. */
    for (int row = 0; row < local_rows; ++row)
        for (int col = 0; col < local_cols; ++col)
            final_grid[(size_t)row * (size_t)local_cols + (size_t)col] =
                a[index_of(row + 1, col + 1, stride)];

    free(a);
    free(b);
    return result;
}

static double verify_against_serial(const double *local_grid,
                                   const double *reference, int n,
                                   int local_rows, int local_cols,
                                   int start_row, int start_col, MPI_Comm cart)
{
    double local_error = 0.0;
    for (int row = 0; row < local_rows; ++row) {
        for (int col = 0; col < local_cols; ++col) {
            size_t local_at = (size_t)row * (size_t)local_cols + (size_t)col;
            size_t global_at = (size_t)(start_row + row) * (size_t)n +
                               (size_t)(start_col + col);
            double error = fabs(local_grid[local_at] - reference[global_at]);
            if (error > local_error)
                local_error = error;
        }
    }

    double global_error = 0.0;
    MPI_CHECK(MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE,
                            MPI_MAX, cart));
    return global_error;
}

static int run_mode(const char *name, int use_overlap,
                    int n, int steps, int repetitions, int ranks,
                    int local_rows, int local_cols,
                    int start_row, int start_col,
                    int north, int south, int west, int east,
                    MPI_Datatype column_type, MPI_Comm cart,
                    int rank, const double *reference)
{
    size_t local_cells = (size_t)local_rows * (size_t)local_cols;
    double *final_grid = malloc(local_cells * sizeof(*final_grid));
    if (final_grid == NULL) {
        fprintf(stderr, "rank %d: could not allocate final grid\n", rank);
        MPI_Abort(cart, EXIT_FAILURE);
    }

    Timing timing = run_solver(n, steps, repetitions, use_overlap,
                               local_rows, local_cols, start_row, start_col,
                               north, south, west, east, column_type,
                               cart, rank, final_grid);
    double max_error = reference == NULL ? 0.0 :
        verify_against_serial(final_grid, reference, n, local_rows, local_cols,
                              start_row, start_col, cart);
    if (rank == 0) {
        const char *verdict = reference == NULL ? "NOT_CHECKED" :
                              max_error <= 1e-12 ? "OK" : "MISMATCH";
        if (reference == NULL) {
            printf("RESULT %s ranks=%d grid=%d steps=%d repetitions=%d\n",
                   verdict, ranks, n, steps, repetitions);
        } else {
            printf("RESULT %s ranks=%d grid=%d steps=%d repetitions=%d max_error=%.3e\n",
                   verdict, ranks, n, steps, repetitions, max_error);
        }
        printf("TIMING mode=%s wall_min=%.9f wall_max=%.9f "
               "mpi_call_max_mean=%.9f compute_max_mean=%.9f "
               "mpi_call_fraction=%.6f\n",
               name, timing.wall_min, timing.wall_max,
               timing.comm_max_mean, timing.compute_max_mean,
               timing.comm_fraction_max_mean);
    }
    free(final_grid);
    return max_error <= 1e-12;
}

int main(int argc, char **argv)
{
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

    int rank = 0;
    int size = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    int n = 37;
    int steps = 10;
    int repetitions = 1;
    int verify = 1;
    int valid = argc <= 5;
    if (valid && argc > 1)
        valid = parse_positive(argv[1], 0, &n);
    if (valid && argc > 2)
        valid = parse_positive(argv[2], 1, &steps);
    if (valid && argc > 3)
        valid = parse_positive(argv[3], 0, &repetitions);
    if (valid && argc > 4)
        valid = parse_positive(argv[4], 1, &verify) && verify <= 1;
    if (!valid || n < 3) {
        if (rank == 0)
            fprintf(stderr, "usage: %s [N>=3 [STEPS>=0 [REPETITIONS>=1 [VERIFY=0|1]]]]\n",
                    argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int dims[2] = {0, 0};
    int periods[2] = {0, 0};
    MPI_CHECK(MPI_Dims_create(size, 2, dims));
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_CHECK(MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart));
    MPI_CHECK(MPI_Comm_set_errhandler(cart, MPI_ERRORS_RETURN));

    int coords[2] = {0, 0};
    int north, south, west, east;
    MPI_CHECK(MPI_Cart_coords(cart, rank, 2, coords));
    /* With periods={0,0}, absent Cartesian neighbors are MPI_PROC_NULL. */
    MPI_CHECK(MPI_Cart_shift(cart, 0, 1, &north, &south));
    MPI_CHECK(MPI_Cart_shift(cart, 1, 1, &west, &east));

    int local_rows = block_size(n, dims[0], coords[0]);
    int local_cols = block_size(n, dims[1], coords[1]);
    int start_row = block_start(n, dims[0], coords[0]);
    int start_col = block_start(n, dims[1], coords[1]);
    if (n < dims[0] || n < dims[1]) {
        if (rank == 0)
            fprintf(stderr, "grid N=%d too small for Cartesian process grid %dx%d\n",
                    n, dims[0], dims[1]);
        MPI_Comm_free(&cart);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Datatype column_type;
    MPI_CHECK(MPI_Type_vector(local_rows, 1, local_cols + 2, MPI_DOUBLE,
                              &column_type));
    MPI_CHECK(MPI_Type_commit(&column_type));

    double *reference = NULL;
    if (verify) {
        size_t global_cells = (size_t)n * (size_t)n;
        if (global_cells > SIZE_MAX / sizeof(double)) {
            if (rank == 0)
                fprintf(stderr, "grid is too large\n");
            MPI_Type_free(&column_type);
            MPI_Comm_free(&cart);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        reference = malloc(global_cells * sizeof(*reference));
        if (reference == NULL) {
            fprintf(stderr, "rank %d: could not allocate reference\n", rank);
            MPI_Abort(cart, EXIT_FAILURE);
        }
        serial_reference(reference, n, steps);
    }

    if (rank == 0) {
        printf("GRID N=%d ranks=%d cart=%dx%d steps=%d\n",
               n, size, dims[0], dims[1], steps);
    }

    int blocking_ok = run_mode("blocking", 0, n, steps, repetitions, size,
                               local_rows, local_cols, start_row, start_col,
                               north, south, west, east, column_type, cart,
                               rank, reference);
    int overlap_ok = run_mode("overlap", 1, n, steps, repetitions, size,
                              local_rows, local_cols, start_row, start_col,
                              north, south, west, east, column_type, cart,
                              rank, reference);

    free(reference);
    MPI_CHECK(MPI_Type_free(&column_type));
    MPI_CHECK(MPI_Comm_free(&cart));
    MPI_CHECK(MPI_Finalize());
    return blocking_ok && overlap_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
