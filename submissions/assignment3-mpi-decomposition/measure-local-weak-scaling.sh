#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
source_file="$here/assign3-CB.AI.U4AIM24021-PARTNER.c"
build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

mpicc -O2 -std=c11 -Wall -Wextra -Werror "$source_file" \
    -o "$build_dir/assign3" -lm

mpiexec_bin="${MPIEXEC:-mpirun}"
for entry in '1 256' '2 362' '4 512' '8 724'; do
    read -r ranks n <<<"$entry"
    printf '\n== local weak-scaling point: ranks=%s N=%s ==\n' "$ranks" "$n"
    "$mpiexec_bin" --oversubscribe -np "$ranks" \
        "$build_dir/assign3" "$n" 50 5 0
done

printf '\nLocal-only experiment: verification is disabled for timing; run selfcheck-local.sh first.\n'
