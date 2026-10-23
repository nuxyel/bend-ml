"""Checks that Bend's runtime takes the thread count it is given.

Runs a parallel program twice, with `+RTS --threads 1 -RTS` and with `+RTS --threads 16 -RTS`, and compares its CPU
time with its wall time: one thread must use about one core, sixteen must use several. check_rts.py finds a runtime
option written outside the bracket; this finds the other way the thread count can be lost, a runtime that stops
reading the bracket (Bend 2.0.38 changed where the options go, and a later version could change it again).

    python3 scripts/rts_probe.py [--min N] <binary> [args...]   # --min: the cores 16 threads must use (default 3)

Local only: it skips itself on a machine with fewer than 8 cores or with SKIP_RTS_PROBE=1, as on CI, where the
runners are small and their CPU time is too noisy to judge.
"""
import os, resource, subprocess, sys, time


def cores_used(cmd):
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    wall = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    if p.returncode != 0:
        sys.exit(f"rts_probe: {' '.join(cmd)} failed ({p.returncode}): {p.stderr.strip()[:300]}")
    return ((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)) / wall, wall


def main():
    args = sys.argv[1:]
    need = 3.0
    if args[:1] == ["--min"]:
        need, args = float(args[1]), args[2:]
    if not args:
        sys.exit(__doc__)
    if os.environ.get("SKIP_RTS_PROBE") == "1" or (os.cpu_count() or 1) < 8:
        print(f"rts_probe: skipped ({os.cpu_count()} cores, SKIP_RTS_PROBE={os.environ.get('SKIP_RTS_PROBE', '')})")
        return
    exe, rest = args[0], args[1:]
    one, w1 = cores_used([exe, "+RTS", "--threads", "1", "-RTS", *rest])
    many, w16 = cores_used([exe, "+RTS", "--threads", "16", "-RTS", *rest])
    name = os.path.basename(exe)
    print(f"rts_probe: {name} uses {one:.2f} cores with +RTS --threads 1 -RTS ({w1:.2f} s) and {many:.2f} with "
          f"+RTS --threads 16 -RTS ({w16:.2f} s)")
    if one > 1.5:
        sys.exit(f"rts_probe: one thread used {one:.2f} cores: the runtime did not take --threads 1")
    if many < need:
        sys.exit(f"rts_probe: sixteen threads used {many:.2f} cores (at least {need} expected): the runtime did not take "
                 "--threads 16, or the program stopped running in parallel")


if __name__ == "__main__":
    main()
