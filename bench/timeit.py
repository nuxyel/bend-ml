"""Times a Bend binary: the median of 3 runs for each number of threads."""
import statistics, subprocess, sys, time
exe = sys.argv[1]; threads = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 4, 8, 16]
extra = sys.argv[3:]
for t in threads:
    ts = []
    for _ in range(3):
        t0 = time.time(); r = subprocess.run([exe, "--threads", str(t), *extra], capture_output=True, text=True); ts.append(time.time() - t0)
    print(f"threads={t:2d}  {statistics.median(ts):7.3f}s   output: {r.stdout.strip()[:40]}")
