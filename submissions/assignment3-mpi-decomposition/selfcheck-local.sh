#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
source_file="$here/assign3-CB.AI.U4AIM24021-PARTNER.c"
build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

mpicc -O2 -std=c11 -Wall -Wextra -Werror "$source_file" \
    -o "$build_dir/assign3" -lm

mpiexec_bin="${MPIEXEC:-mpirun}"
for ranks in 1 2 4 5 6 7; do
    echo "== ranks=$ranks grid=37x37 steps=9 =="
    output="$("$mpiexec_bin" --oversubscribe -np "$ranks" \
        "$build_dir/assign3" 37 9 1)"
    printf '%s\n' "$output"
    grep -q '^RESULT OK ' <<<"$output"
    test "$(grep -c '^RESULT OK ' <<<"$output")" -eq 2
done

echo '== one-rank boundary-only case =='
output="$("$mpiexec_bin" --oversubscribe -np 1 \
    "$build_dir/assign3" 3 5 1)"
printf '%s\n' "$output"
grep -q '^RESULT OK ' <<<"$output"
test "$(grep -c '^RESULT OK ' <<<"$output")" -eq 2

echo 'ALL LOCAL MPI CHECKS PASSED'
