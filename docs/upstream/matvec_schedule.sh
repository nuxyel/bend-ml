#!/usr/bin/env bash
# Builds matvec_schedule.bend twice from the same emitted C, once as bend builds it and once with BLK_SHR flipped to 1
# (the program never forks, so the flag only changes which code is compiled around the loops), then times both on one
# pinned thread. Usage: matvec_schedule.sh [rounds]; needs clang and taskset.
set -euo pipefail
cd "$(dirname "$0")"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
bend matvec_schedule.bend -o "$T/a.c" >/dev/null
grep -q "#define BLK_SHR 0" "$T/a.c"
sed 's/#define BLK_SHR 0/#define BLK_SHR 1/' "$T/a.c" > "$T/b.c"
clang -std=c11 -O3 "$T/a.c" -lpthread -lm -o "$T/a"
clang -std=c11 -O3 "$T/b.c" -lpthread -lm -o "$T/b"
for r in $(seq 1 "${1:-5}"); do
  for op in 0 1; do
    echo "round $r op $op  as built: $(taskset -c 3 "$T/a" --threads 1 $op 50257 768 38 0)  BLK_SHR 1: $(taskset -c 3 "$T/b" --threads 1 $op 50257 768 38 0)"
  done
done
