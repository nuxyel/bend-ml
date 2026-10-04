"""Compara as operações de tensor/main.bend (via tensor/cli.bend) com PyTorch."""
import os, subprocess, sys, tempfile
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
RTOL, ATOL = 2e-4, 2e-5  # F32 (não há F64 no Bend)


def bend(*args):
    r = subprocess.run([BEND, os.path.join(ROOT, "tensor/cli.bend"), *map(str, args)],
                       capture_output=True, text=True, env=ENV)
    out = r.stdout.strip()
    if r.returncode != 0 or not out or out[0].isalpha():
        raise RuntimeError(f"bend falhou: {out[:200]} {r.stderr[:200]}")
    return np.array([float(x) for x in out.split()], dtype=np.float32)


def main():
    rng = np.random.default_rng(0)
    falhas = 0
    with tempfile.TemporaryDirectory() as d:
        def w(name, a):
            p = os.path.join(d, name)
            np.savetxt(p, np.asarray(a, dtype=np.float32).reshape(1, -1), fmt="%.9g")
            return p

        def check(nome, got, want):
            nonlocal falhas
            want = np.asarray(want, dtype=np.float32).reshape(-1)
            ok = got.shape == want.shape and np.allclose(got, want, rtol=RTOL, atol=ATOL)
            err = float(np.max(np.abs(got - want))) if got.shape == want.shape else float("nan")
            print(f"{'ok  ' if ok else 'ERRO'} {nome}  (max |erro| = {err:.2e})")
            if not ok:
                falhas += 1

        # matmul: formas diferentes, incluindo vetores-linha e coluna
        for (n, k, m) in [(2, 3, 4), (5, 7, 3), (1, 6, 1), (16, 20, 8)]:
            A = rng.standard_normal((n, k)).astype(np.float32)
            B = rng.standard_normal((k, m)).astype(np.float32)
            got = bend("matmul", n, k, m, w("a", A), w("b", B))
            check(f"matmul {n}x{k} · {k}x{m}", got, A @ B)

        X = rng.standard_normal((4, 6)).astype(np.float32) * 3
        tx = torch.from_numpy(X)
        check("transpose 4x6", bend("transpose", 4, 6, w("x", X)), X.T)
        check("relu 4x6", bend("relu", 4, 6, w("x", X)), np.maximum(X, 0))
        check("softmax (por linha) 4x6", bend("softmax", 4, 6, w("x", X)), torch.softmax(tx, dim=1).numpy())
        check("gelu (tanh) 4x6", bend("gelu", 4, 6, w("x", X)), torch.nn.functional.gelu(tx, approximate="tanh").numpy())

        g = rng.standard_normal(6).astype(np.float32)
        b = rng.standard_normal(6).astype(np.float32)
        want = torch.nn.functional.layer_norm(tx, (6,), torch.from_numpy(g), torch.from_numpy(b), eps=1e-5).numpy()
        check("layernorm 4x6", bend("layernorm", 4, 6, "0.00001", w("x", X), w("g", g), w("b", b)), want)

        # softmax estável com valores grandes
        Big = np.array([[1000.0, 1001.0, 999.0], [-1000.0, 0.0, 1000.0]], dtype=np.float32)
        check("softmax estável (valores grandes)", bend("softmax", 2, 3, w("big", Big)), torch.softmax(torch.from_numpy(Big), dim=1).numpy())

    print("FALHAS:", falhas)
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
