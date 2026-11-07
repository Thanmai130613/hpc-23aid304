/* ===========================================================================
   Exercise 06 — Parallel reduction with OpenMP

   Implement the four functions declared in include/reduction.h so that each
   one uses OpenMP to compute its result in parallel.

   Rules
     * Do not add a main() function. The test harness provides one.
     * Do not change include/reduction.h.
     * Every function must give the same answer regardless of thread count.

   Build and test locally:
       ./selfcheck.sh
   =========================================================================== */

#include "reduction.h"

#include <math.h>
#include <omp.h>

#define REDUCTION_LANES 128

double par_sum(const double *a, size_t n)
{
    double partials[REDUCTION_LANES] = {0.0};
    const size_t base = n / REDUCTION_LANES;
    const size_t remainder = n % REDUCTION_LANES;

    /* Fixed input partitions keep each partial sum independent of thread count. */
    #pragma omp parallel for reduction(+:partials[:REDUCTION_LANES])
    for (size_t lane = 0; lane < REDUCTION_LANES; lane++) {
        const size_t start = lane * base + (lane < remainder ? lane : remainder);
        const size_t length = base + (lane < remainder ? 1 : 0);
        double subtotal = 0.0;
        for (size_t i = start; i < start + length; i++) {
            subtotal += a[i];
        }
        partials[lane] = subtotal;
    }

    double sum = 0.0;
    for (size_t lane = 0; lane < REDUCTION_LANES; lane++) {
        sum += partials[lane];
    }
    return sum;
}

double par_dot(const double *a, const double *b, size_t n)
{
    double partials[REDUCTION_LANES] = {0.0};
    const size_t base = n / REDUCTION_LANES;
    const size_t remainder = n % REDUCTION_LANES;

    #pragma omp parallel for reduction(+:partials[:REDUCTION_LANES])
    for (size_t lane = 0; lane < REDUCTION_LANES; lane++) {
        const size_t start = lane * base + (lane < remainder ? lane : remainder);
        const size_t length = base + (lane < remainder ? 1 : 0);
        double subtotal = 0.0;
        for (size_t i = start; i < start + length; i++) {
            subtotal += a[i] * b[i];
        }
        partials[lane] = subtotal;
    }

    double dot = 0.0;
    for (size_t lane = 0; lane < REDUCTION_LANES; lane++) {
        dot += partials[lane];
    }
    return dot;
}

double par_max(const double *a, size_t n)
{
    if (n == 0) {
        return -INFINITY;
    }

    double best = -INFINITY;

    #pragma omp parallel for reduction(max:best)
    for (size_t i = 0; i < n; i++) {
        if (a[i] > best) {
            best = a[i];
        }
    }

    return best;
}

long par_count_above(const double *a, size_t n, double threshold)
{
    long count = 0;

    #pragma omp parallel for reduction(+:count)
    for (size_t i = 0; i < n; i++) {
        if (a[i] > threshold) {
            count++;
        }
    }

    return count;
}
