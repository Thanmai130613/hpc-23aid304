# Assignment 3 — Distribute It

**Working problem:** two-dimensional heat diffusion, implemented as a 5-point Jacobi stencil with fixed outer boundaries.

**Pair:** `CB.AI.U4AIM24021` and `[PARTNER ROLL NUMBER REQUIRED]`.

**Contribution statement:** `[Complete with an accurate description of each partner's work before producing the final PDF.]`

**Status:** design and local correctness implementation in progress. This draft intentionally contains no ASA or multi-node measurements.

## Part A — Design before code (7 marks)

### A1. PCAM

- **Partition:** divide the global `N × N` grid into contiguous rectangular subgrids, one per MPI rank.
- **Communication:** at each time step exchange the north, south, east, and west face values with adjacent subgrids. The 5-point stencil does not require corner data.
- **Agglomeration and granularity:** retain a rectangular block per rank. Each block performs its local interior updates for every exchanged halo value. The surface-to-volume ratio below motivates using a two-dimensional decomposition instead of narrow 1-D slabs as the rank count grows.
- **Mapping:** use a non-periodic 2-D Cartesian communicator. `MPI_Cart_shift` supplies neighbors; physical edges are `MPI_PROC_NULL`.

### A2. Communication-to-computation prediction

For a square `N × N` grid and `P` ranks, count halo grid points per interior update, ignoring physical outer boundaries and message latency:

- 1-D slabs: `2N / (N²/P) = 2P/N`.
- Approximately square 2-D blocks: `4N/√P / (N²/P) = 4√P/N`.
- For process-grid dimensions `Pr × Pc`, the corresponding approximation is `2(Pr/N + Pc/N)`.

For this square problem I use a 2-D process grid because its surface-to-volume ratio grows as `√P`, rather than `P`, for fixed `N`.

### A3. Latency/bandwidth prediction — fill before the ASA measurement

The prediction must use the measured `α` and `β` from Lab 6. For a square grid, the following table uses `N = 4096`, the nearly square process grids selected for those `P` values, and a sum-of-neighbor-messages approximation:

| P | Process grid | Local side | Non-null neighbor count used | Halo values sent | Predicted time per step |
|---:|---:|---:|---:|---:|---|
| 4 | 2 × 2 | 2048 | 2 | 4096 | `2α + 8·4096/β` |
| 16 | 4 × 4 | 1024 | 4 (interior rank) | 4096 | `4α + 8·4096/β` |
| 64 | 8 × 8 | 512 | 4 (interior rank) | 2048 | `4α + 8·2048/β` |

Insert Lab 6’s measured `α` and `β` and evaluate these expressions **before measuring this assignment**. Then compare predicted communication time with measured step time and state the rank count where communication first reaches half of the step time. Do not fill this table with unmeasured or assumed network values.

## Part B — Implement (12 marks)

The source implements uneven rectangular block sizes, a serial reference, a blocking `MPI_Sendrecv` path, and a nonblocking `MPI_Irecv`/`MPI_Isend`/`MPI_Waitall` path. In the nonblocking path, local strict-interior work occurs after communication is posted and before `MPI_Waitall`. Both paths use `MPI_PROC_NULL` at physical boundaries. Timing output separates elapsed wall time, time spent in MPI calls, and computation time; root aggregation uses `MPI_Reduce`, and reference-check status uses `MPI_Allreduce`.

| Criterion | Local evidence/status |
|---|---|
| B1 Correct MPI result at awkward rank counts 5, 6, 7 | Both modes matched the serial reference at 5, 6, and 7 local ranks (`max_error=0`). |
| B2 Genuine overlap and blocking comparison | Both code paths are present and timed. The local comparison below checks the measurement path; ASA measurements remain. |
| B3 Use collectives where appropriate | `MPI_Reduce` for timing maxima and `MPI_Allreduce` for global correctness. No send loop to a root rank. |
| B4 Separate communication/computation instrumentation | Per-rank MPI-call and compute timers are reduced and emitted for each mode. Record repeated ASA measurements below. In overlap mode, communication may progress during compute; the MPI-call fraction is not an additive fraction of wall time. |
| B5 Clean boundaries | Non-periodic Cartesian neighbors and `MPI_PROC_NULL`. |

## Part C — Measure and reconcile (11 marks)

No multi-node results are available yet. Run the strong-scaling and inter-node portions on ASA and record actual outputs only.

| P | Nodes / ranks per node | Strong-scale min time | Spread | Speedup | Efficiency | Karp–Flatt | Communication fraction | Weak-scale time | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | | | | | | | | | |
| 2 | | | | | | | | | |
| 4 | | | | | | | | | |
| 8 | | | | | | | | | |
| 16 | | | | | | | | | |
| 32 | at least two nodes overall | | | | | | | | |

Required plots to generate from these measured rows:

1. Strong-scaling speedup and ideal line, efficiency, and Karp–Flatt versus `P`.
2. Weak-scaling runtime versus `P`, with fixed work per rank.
3. Measured communication fraction versus the Part A3 prediction.
4. Intra-node versus inter-node scaling, identifying the node-boundary effect and the point where adding ranks slows the program.

Explain the trend and magnitude differences from the model, including at least two omitted effects (for example synchronization/progress overhead and contention or load imbalance). Do not substitute local single-node results for the required multi-node evidence.

### Local-only weak scaling (C2 preparation)

Command/script: `bash measure-local-weak-scaling.sh`. `N²/P` is held at approximately 65,500 grid points per rank. The timing run uses one discarded warm-up and five timed repetitions; verification is disabled for this larger timing run because correctness was checked separately at the rank counts and edge cases above.

| Ranks | N | Approx. grid points/rank | Blocking min / spread (ms) | Overlap min / spread (ms) |
|---:|---:|---:|---:|---:|
| 1 | 256 | 65,536 | 6.087 / 2.250 | 6.403 / 2.254 |
| 2 | 362 | 65,522 | 6.892 / 7.876 | 6.299 / 2.871 |
| 4 | 512 | 65,536 | 8.662 / 9.137 | 8.076 / 3.658 |
| 8 | 724 | 65,522 | 13.270 / 4.790 | 12.527 / 1.551 |

The minimum runtime rises at 4 and 8 ranks, so this local curve is not flat. MPI-call fraction also rises with rank count, while the wide blocking timing spread shows substantial local variability. This suggests communication/synchronization cost plus machine scheduling variability; separating those effects for the assignment’s network question requires ASA measurements. The plot [local-weak-scaling.svg](local-weak-scaling.svg) shows minima and min-to-max bars. This is single-node virtualized data and does not satisfy C1’s explicit two-node requirement.

## Local validation record

Compiler flags: `mpicc -O2 -std=c11 -Wall -Wextra -Werror`.

| Local case | Result |
|---|---|
| 1 rank, 3 × 3 grid, 5 steps | Both blocking and overlap: `RESULT OK`, max error `0`. |
| 1, 2, 4, 5, 6, 7 ranks; 37 × 37 grid, 9 steps; blocking and overlap checked against serial reference | Both modes: `RESULT OK` at every rank count, max error `0`. Includes uneven decompositions at 2, 5, 6, and 7 ranks. |

The 37 × 37 local checks use one timed repetition for execution/correctness plumbing only. They are too small and are not repeated enough to support a performance claim.

### Local-only overlap comparison

Build: `mpicc -O2 -std=c11 -Wall -Wextra -Werror assign3-CB.AI.U4AIM24021-PARTNER.c -o /tmp/assign3-final -lm`.

Command: `mpirun --oversubscribe -np 4 /tmp/assign3-final 512 100 5`.

Environment: Intel Core Ultra 5 125H reported through a full Microsoft Hyper-V virtual machine (9 reported cores, 2 threads/core); GCC 15.2.0 via Open MPI 5.0.10; `-O2 -std=c11 -Wall -Wextra -Werror`; 4 local MPI ranks; `N=512`, 100 steps, one discarded warm-up, 5 timed repetitions; minimum wall time reported, with max-minus-min spread.

| Mode | Minimum wall time | Maximum wall time | Spread | Instrumented MPI-call fraction | Correctness |
|---|---:|---:|---:|---:|---|
| Blocking | 17.230 ms | 51.876 ms | 34.646 ms | 0.4174 | Pass, max error 0 |
| Nonblocking with interior overlap | 16.379 ms | 23.958 ms | 7.579 ms | 0.3042 | Pass, max error 0 |

The minimum overlapped time was slightly lower, but the timing ranges overlap widely and the blocking spread is large. This noisy local result does not support a reliable speedup claim. The virtualized single-node result is only a code/measurement-path check; it is not evidence about ASA network overlap or multi-node scaling. The MPI-call fraction is instrumented MPI-call time divided by MPI-call plus compute time; for the overlapped mode, communication progressing during computation is not represented as an additive wall-time fraction.

## Required later ASA evidence

- Lab 6 measured latency `α` and bandwidth `β`, with method and units, inserted into Part A3 before assignment benchmarking.
- Correctness at 5, 6, and 7 ranks on the ASA MPI runtime.
- Blocking-versus-overlap timings and communication/computation instrumentation at the rank counts used.
- Strong scaling at `P = 1, 2, 4, 8, 16, 32`, with at least two nodes; warm-up, repetitions, statistic, spread, and node/rank placement recorded.
- Weak scaling with fixed work per rank; recorded runtimes and node placement.
- Actual comparison of measured communication fraction against Part A3, intra-node versus inter-node behavior, and the slowdown threshold.
- Pair contribution statement and final PDF filename once both roll numbers are known.
