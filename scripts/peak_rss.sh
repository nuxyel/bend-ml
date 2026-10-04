#!/usr/bin/env bash
# Prints the peak resident memory (MB) of a command, sampled every 0.1 s from /proc.
# Usage: scripts/peak_rss.sh command [args...]
"$@" >/dev/null 2>&1 &
pid=$!
peak=0; vpeak=0
while kill -0 "$pid" 2>/dev/null; do
  rss=$(awk "/VmHWM/ {print \$2}" "/proc/$pid/status" 2>/dev/null || echo 0)
  vsz=$(awk "/VmPeak/ {print \$2}" "/proc/$pid/status" 2>/dev/null || echo 0)
  [ -n "$vsz" ] && [ "$vsz" -gt "$vpeak" ] && vpeak=$vsz
  [ -n "$rss" ] && [ "$rss" -gt "$peak" ] && peak=$rss
  sleep 0.1
done
wait "$pid" 2>/dev/null
echo "peak RSS: $((peak / 1024)) MB (peak virtual: $((vpeak / 1024)) MB)"
