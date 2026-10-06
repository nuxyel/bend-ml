"""bend-ml-tensor-array (tensor-array/cli.bend) against PyTorch/NumPy."""
import os, subprocess, sys, tempfile
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
RTOL, ATOL = 2e-4, 2e-5


def bend(*args):
    r = subprocess.run([BEND, os.path.join(ROOT, "tensor-array/cli.bend"), "--", *map(str, args)],
                       capture_output=True, text=True, env=ENV)
    out = r.stdout.strip()
    if r.returncode != 0 or not out or out[0].isalpha():
        raise RuntimeError(f"bend failed: {out[:200]} {r.stderr[:200]}")
    return np.array([float(x) for x in out.split()], dtype=np.float32)


def main():
    rng = np.random.default_rng(3)
    failures = 0
    with tempfile.TemporaryDirectory() as d:
        def w(name, a):
            p = os.path.join(d, name)
            np.savetxt(p, np.asarray(a, dtype=np.float32).reshape(1, -1), fmt="%.9g")
            return p

        def check(name, got, want):
            nonlocal failures
            want = np.asarray(want, dtype=np.float32).reshape(-1)
            ok = got.shape == want.shape and np.allclose(got, want, rtol=RTOL, atol=ATOL)
            err = float(np.max(np.abs(got - want))) if got.shape == want.shape else float("nan")
            print(f"{'ok  ' if ok else 'ERROR'} {name}  (max |error| = {err:.2e})")
            if not ok:
                failures += 1

        # matmul in the three layouts and with 0, 1 and 3 levels of parallelism
        rr = np.random.default_rng(11)
        random_shapes = [(int(rr.integers(1, 25)), int(rr.integers(1, 41)), int(rr.integers(1, 25))) for _ in range(12)]
        random_shapes += [(1, int(rr.integers(1, 60)), int(rr.integers(1, 80))) for _ in range(4)]      # matrix . vector
        for (n, k, m) in [(2, 3, 4), (5, 7, 3), (1, 6, 1), (9, 5, 11), (16, 20, 8), (1, 8, 13), (1, 40, 100), (1, 3, 5)] + random_shapes:
            A = rng.standard_normal((n, k)).astype(np.float32)
            B = rng.standard_normal((k, m)).astype(np.float32)
            for par in (0, 1, 2, 3, 4):
                check(f"matmul nn   {n}x{k}·{k}x{m} par={par}", bend("nn", par, n, k, m, w("a", A), w("b", B)), A @ B)
            check(f"matmul nt   {n}x{k}·({m}x{k})ᵀ par=2", bend("nt", 2, n, k, m, w("a", A), w("bt", B.T.copy())), A @ B)
            check(f"matmul tn   ({k}x{n})ᵀ·{k}x{m} par=2", bend("tn", 2, n, k, m, w("at", A.T.copy()), w("b", B)), A @ B)

        X = rng.standard_normal((6, 5)).astype(np.float32) * 2
        b = rng.standard_normal(5).astype(np.float32)
        check("add_row (bias)", bend("bias", 6, 5, w("x", X), w("b", b)), X + b)
        check("relu", bend("relu", 6, 5, w("x", X)), np.maximum(X, 0))
        DY = rng.standard_normal((6, 5)).astype(np.float32)
        check("relu_bwd", bend("relu_bwd", 6, 5, w("dy", DY), w("h", np.maximum(X, 0))), DY * (np.maximum(X, 0) > 0))
        W = rng.standard_normal((6, 5)).astype(np.float32)
        check("sgd", bend("sgd", 6, 5, "0.1", w("w", W), w("dw", DY)), W - 0.1 * DY)
        check("add", bend("add", 6, 5, w("a", X), w("b", DY)), X + DY)
        check("col_sums", bend("colsum", 6, 5, w("x", X)), X.sum(axis=0))

        for (n, c) in [(4, 5), (8, 10), (100, 10)]:
            L = torch.tensor(rng.standard_normal((n, c)) * 2, dtype=torch.float32, requires_grad=True)
            y = torch.tensor(rng.integers(0, c, n))
            loss = F.cross_entropy(L, y); loss.backward()
            want = np.concatenate([[loss.item()], L.grad.numpy().ravel()])
            check(f"softmax_ce n={n} c={c}", bend("ce", n, c, w("l", L.detach().numpy()), w("y", y.numpy().astype(np.float32))), want)
            hits = int((L.detach().argmax(1) == y).sum())
            check(f"count_correct n={n} c={c}", bend("hits", n, c, w("l", L.detach().numpy()), w("y", y.numpy().astype(np.float32))), [hits])

        # Bands: filled in blocks of random size, then B · x, one row and the whole matrix.
        # B · x must be bit-identical to the sequential Mat.matmul_nt (same kernel, same order).
        rb = np.random.default_rng(29)
        band_cases = [(1, 1, 0, 1), (2, 3, 1, 1), (7, 5, 3, 4), (9, 4, 5, 7), (16, 8, 2, 100)]
        # degenerate shapes: more bands than rows (empty bands), one row, one column, blocks of one
        # number, and a block larger than the whole matrix
        band_cases += [(3, 4, 5, 2), (1, 7, 3, 3), (6, 1, 2, 1), (5, 5, 2, 1), (4, 3, 1, 1000)]
        band_cases += [(int(rb.integers(1, 70)), int(rb.integers(1, 60)), int(rb.integers(0, 6)), int(rb.integers(1, 200))) for _ in range(10)]
        for (r, c, dep, blk) in band_cases:
            Wb = rng.standard_normal((r, c)).astype(np.float32)
            xb = rng.standard_normal(c).astype(np.float32)
            i = int(rb.integers(0, r))
            out = bend("bands", dep, r, c, blk, i, w("w", Wb), w("x", xb))
            y, yl, row, full = out[:r], out[r:2 * r], out[2 * r:2 * r + c], out[2 * r + c:]
            same = yl.shape == y.shape and np.array_equal(yl, y)
            print(f"{'ok  ' if same else 'ERROR'} Bands.matvec_l {r}x{c} d={dep} equals Bands.matvec bit for bit")
            failures += 0 if same else 1
            check(f"Bands.matvec {r}x{c} d={dep} blk={blk}", y, Wb @ xb)
            seq = bend("nt", 0, 1, c, r, w("x1", xb.reshape(1, c)), w("w1", Wb))
            exact = y.shape == seq.shape and np.array_equal(y, seq)
            print(f"{'ok  ' if exact else 'ERROR'} Bands.matvec {r}x{c} d={dep} equals matmul_nt bit for bit")
            failures += 0 if exact else 1
            check(f"Bands.read_row {r}x{c} d={dep} row {i}", row, Wb[i])
            check(f"Bands.to_list {r}x{c} d={dep} blk={blk}", full, Wb)

    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
