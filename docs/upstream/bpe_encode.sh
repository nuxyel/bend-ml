#!/usr/bin/env bash
# Builds bpe_encode.bend with two Bend releases (bend -o, which builds with clang -std=c11 -O3), checks that both print
# the same three lines, then counts each binary on one pinned thread with perf stat, in user mode, in alternated rounds.
# Instructions repeat to a few thousand in 194 billion between runs, so they need no idle machine; cycles do.
#   bpe_encode.sh <bend 2.0.39> <bend 2.0.40> [rounds]
# CPU picks the core to pin to (3 by default, a P-core on my Core Ultra 7 155H); EVENTS replaces the event list, e.g.
# EVENTS=instructions:u,cycles:u,mem_load_retired.l1_miss:u for the L1 load misses on a recent Intel core.
# Needs clang, perf and taskset.
set -euo pipefail
cd "$(dirname "$0")"
A=${1:?usage: bpe_encode.sh <bend A> <bend B> [rounds]}; B=${2:?usage: bpe_encode.sh <bend A> <bend B> [rounds]}
CPU=${CPU:-3}; EVENTS=${EVENTS:-instructions:u,cycles:u}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
want=$'256\n31633\n2633430213'
for x in A B; do
  b=${!x}
  echo "$x: $("$b" version)"
  "$b" bpe_encode.bend -o "$T/$x" > /dev/null
  got=$("$T/$x")
  [ "$got" = "$want" ] || { echo "$x prints something else:"; echo "$got"; exit 1; }
done
for r in $(seq 1 "${3:-3}"); do
  for x in A B; do
    taskset -c "$CPU" perf stat -x, -e "$EVENTS" -o "$T/stat" "$T/$x" > /dev/null
    echo "round $r $x: $(grep -v -e '^#' -e '^$' "$T/stat" | grep -v 'not counted' | awk -F, '{printf "%s %s  ", $3, $1}')"
  done
done
