#include "histogram.h"

#include <omp.h>

static int bin_of(double v, double lo, double hi)
{
    if (v < lo) return 0;
    if (v >= hi) return NBINS - 1;

    int b = (int)((v - lo) / (hi - lo) * NBINS);
    if (b < 0) b = 0;
    if (b >= NBINS) b = NBINS - 1;
    return b;
}

void histogram(const double *a, size_t n, double lo, double hi, long *bins)
{
    for (int b = 0; b < NBINS; ++b) bins[b] = 0;

    #pragma omp parallel
    {
        long local[NBINS] = {0};

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            ++local[bin_of(a[i], lo, hi)];
        }

        #pragma omp critical
        {
            for (int b = 0; b < NBINS; ++b) bins[b] += local[b];
        }
    }
}
