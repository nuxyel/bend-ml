"""examples/runtime_batch.bend against NumPy, with batch sizes the program only learns at run time.

Writes input files with random batch sizes (the first number of each file), runs the compiled
example and compares H = relu(X·W + b) with NumPy. Files whose row count does not match their
header must be rejected with the error message instead of producing numbers.
"""
import os, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")

# the same fixed weights as examples/runtime_batch.bend
W = np.array([[0.5, -1.0, 0.25], [1.0, 0.5, -0.5], [-0.25, 0.75, 1.0], [0.1, 0.2, 0.3]], dtype=np.float32)
B = np.array([0.1, -0.2, 0.05], dtype=np.float32)


def run(exe, path):
    return subprocess.run([exe, path], capture_output=True, text=True, env=ENV).stdout


def main():
    rng = np.random.default_rng(int(os.environ.get("FUZZ_SEED", "11")))
    failures = 0
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "runtime_batch")
        r = subprocess.run([BEND, os.path.join(ROOT, "examples/runtime_batch.bend"), "-o", exe], capture_output=True, text=True, env=ENV)
        if r.returncode != 0 or not os.path.exists(exe):
            print("ERROR could not compile the example:", r.stdout[-400:], r.stderr[-400:])
            sys.exit(1)

        def write(name, header, rows):
            p = os.path.join(d, name)
            with open(p, "w") as f:
                f.write(f"{header}\n")
                for row in rows:
                    f.write(" ".join(f"{v:.6g}" for v in row) + "\n")
            return p

        # batch sizes the program cannot know in advance, including 1 and a larger one
        for n in [1, 2, 3, 7, 16, 33, int(rng.integers(40, 120)), 200]:
            x = rng.uniform(-2, 2, (n, 4)).astype(np.float32)
            x = np.array([[float(f"{v:.6g}") for v in row] for row in x], dtype=np.float32)
            out = run(exe, write(f"b{n}.txt", n, x))
            want = np.maximum(x @ W + B, 0)
            lines = out.strip().splitlines()
            ok = len(lines) == 3 + n and lines[0] == f"batch size read from the file: {n}" and lines[2] == f"H = relu(X·W + b) : Mat<{n}, 3>"
            if ok:
                got = np.array([[float(v) for v in l.split()] for l in lines[3:]], dtype=np.float32)
                ok = got.shape == want.shape and np.allclose(got, want, rtol=1e-5, atol=1e-5)
            print(f"{'ok  ' if ok else 'ERROR'} batch size {n}, read from the file")
            failures += not ok

        # the header and the data disagree: the boundary check must refuse the file
        for name, header, rows in [("too_few", 5, rng.uniform(-1, 1, (4, 4))), ("too_many", 2, rng.uniform(-1, 1, (3, 4))), ("partial_row", 2, [[1, 2, 3, 4], [5, 6, 7]])]:
            out = run(exe, write(f"{name}.txt", header, rows))
            ok = out.startswith("error: the file says the batch has")
            print(f"{'ok  ' if ok else 'ERROR'} file with the wrong count is rejected ({name})")
            failures += not ok

    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
