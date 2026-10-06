"""Times the checker (bend X.bend --check-only) and a native build (bend X.bend -o exe, which
includes clang) for the demos and for generated networks of 8, 32 and 128 distinct typed layers
(scripts/gen_deep_mlp.py). Median of 3 runs each. Prints a Markdown table.

  python3 scripts/compile_times.py
"""
import os, statistics, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")


def timed(cmd):
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=ENV)
        ts.append(time.perf_counter() - t0)
        if r.returncode != 0:
            sys.exit(f"failed: {' '.join(cmd)}\n{r.stdout}{r.stderr}")
    return statistics.median(ts)


def row(name, path, d):
    lines = sum(1 for _ in open(path))
    check = timed([BEND, path, "--check-only"])
    build = timed([BEND, path, "-o", os.path.join(d, "exe")])
    print(f"| {name} | {lines} lines | {check:.2f} s | {build:.2f} s |", flush=True)


def main():
    print("| program | size | check | build (incl. clang) |")
    print("|---|---|---|---|")
    with tempfile.TemporaryDirectory() as d:
        for L in (8, 32, 128):
            path = os.path.join(d, f"deep{L}.bend")
            with open(path, "w") as f:
                subprocess.run([sys.executable, os.path.join(ROOT, "scripts/gen_deep_mlp.py"), str(L)], stdout=f, check=True)
            row(f"{L} dense layers, all sizes distinct", path, d)
        row("MNIST training (demos/mnist/fast.bend)", os.path.join(ROOT, "demos/mnist/fast.bend"), d)
        row("GPT-2 small inference (demos/gpt2/fast.bend)", os.path.join(ROOT, "demos/gpt2/fast.bend"), d)


if __name__ == "__main__":
    main()
