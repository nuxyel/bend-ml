"""Gradient checking: o autograd em Bend (autograd/cli.bend) contra o autograd do PyTorch."""
import os, subprocess, sys, tempfile
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
RTOL, ATOL = 5e-4, 5e-5  # F32 (sem F64 no Bend)


def bend(*args):
    r = subprocess.run([BEND, os.path.join(ROOT, "autograd/cli.bend"), "--", *map(str, args)],
                       capture_output=True, text=True, env=ENV)
    out = r.stdout.strip()
    if r.returncode != 0 or not out or out[0].isalpha():
        raise RuntimeError(f"bend falhou: {out[:200]} {r.stderr[:200]}")
    return np.array([float(x) for x in out.split()], dtype=np.float32)


def scalar_expr(k, x0, x1, x2):
    if k == 1: return x0 * x1 + x0
    if k == 2: return torch.relu(x0 * x1 + (-1.0) * x2)
    if k == 3: return torch.tanh(x0 * x1 + x2)
    if k == 4: return torch.exp(x0) * torch.tanh(x1) + x2 * x2
    return torch.tanh(torch.relu(x0 * x1) + torch.exp(x2 * x0))


def main():
    rng = np.random.default_rng(1)
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

        # escalares: 5 expressões, 3 pontos cada (relu em ambos os lados do zero)
        for k in range(1, 6):
            for (a, b, c) in [(0.7, -1.3, 0.4), (-0.5, 0.9, 1.1), (1.2, 0.8, -0.6)]:
                xs = [torch.tensor(v, dtype=torch.float32, requires_grad=True) for v in (a, b, c)]
                y = scalar_expr(k, *xs)
                y.backward()
                want = [y.item()] + [(x.grad.item() if x.grad is not None else 0.0) for x in xs]
                check(f"escalar expr{k} em ({a},{b},{c})", bend("scalar", k, a, b, c), want)

        # camada linear: dx, dW, db
        for (n, i, o) in [(3, 4, 2), (5, 6, 3), (1, 8, 8)]:
            X = torch.tensor(rng.standard_normal((n, i)), dtype=torch.float32, requires_grad=True)
            W = torch.tensor(rng.standard_normal((i, o)), dtype=torch.float32, requires_grad=True)
            B = torch.zeros(o, requires_grad=True)
            DY = torch.tensor(rng.standard_normal((n, o)), dtype=torch.float32)
            ((X @ W + B) * DY).sum().backward()
            want = np.concatenate([X.grad.numpy().ravel(), W.grad.numpy().ravel(), B.grad.numpy().ravel()])
            check(f"linear_bwd n={n} i={i} o={o}",
                  bend("linear_bwd", n, i, o, w("x", X.detach().numpy()), w("w", W.detach().numpy()), w("dy", DY.numpy())), want)

        # relu
        X = torch.tensor(rng.standard_normal((4, 5)), dtype=torch.float32, requires_grad=True)
        DY = torch.tensor(rng.standard_normal((4, 5)), dtype=torch.float32)
        (torch.relu(X) * DY).sum().backward()
        check("relu_bwd 4x5", bend("relu_bwd", 4, 5, w("x", X.detach().numpy()), w("dy", DY.numpy())), X.grad.numpy())

        # entropia cruzada: perda e gradiente nos logits
        for (n, c) in [(4, 5), (8, 10)]:
            L = torch.tensor(rng.standard_normal((n, c)) * 2, dtype=torch.float32, requires_grad=True)
            y = torch.tensor(rng.integers(0, c, n))
            loss = F.cross_entropy(L, y)
            loss.backward()
            want = np.concatenate([[loss.item()], L.grad.numpy().ravel()])
            check(f"cross-entropy n={n} c={c}", bend("ce", n, c, w("l", L.detach().numpy()), w("y", y.numpy().astype(np.float32))), want)

    print("FALHAS:", falhas)
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
