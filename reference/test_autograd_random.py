"""Gradient checking on random expression trees: value and gradient of the Bend scalar autograd
(autograd/main.bend) against PyTorch. Each tree is compiled into its own small Bend program."""
import os, random, subprocess, sys, tempfile
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
CASES = int(os.environ.get("FUZZ_CASES", "120"))
SEED = 20261004
NVARS = 3
RTOL, ATOL = 5e-4, 5e-5

HEAD = """import Base
import {path}/autograd/main.bend as AG

def fshow(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{{}}:
      ""
    case Con{{h, t}}:
      F32.show(h) ++ " " ++ fshow(t)

def main() -> IO(Unit):
  IO.print(F32.show(AG.gval({expr}, {env})) ++ " " ++ fshow(AG.ggrad({expr}, {env})))
"""


def lit(x):
    return f"(0.0 - {abs(x)!r} : F32)" if x < 0 else f"{x!r}"


def gen(rng, depth):
    """A random expression as a nested tuple: ('var', i) | ('cst', c) | (op, child...)."""
    if depth == 0 or rng.random() < 0.2:
        if rng.random() < 0.7:
            return ("var", rng.randrange(NVARS))
        return ("cst", round(rng.uniform(-1.5, 1.5), 3))
    op = rng.choice(["add", "mul", "relu", "tanh", "exp"])
    if op in ("add", "mul"):
        return (op, gen(rng, depth - 1), gen(rng, depth - 1))
    return (op, gen(rng, depth - 1))


def render(e):
    k = e[0]
    if k == "var":
        return f"AG.GVar{{{e[1]}n}}"
    if k == "cst":
        return f"AG.GCst{{{lit(e[1])}}}"
    name = {"add": "GAdd", "mul": "GMul", "relu": "GRelu", "tanh": "GTanh", "exp": "GExp"}[k]
    return f"AG.{name}{{" + ", ".join(render(c) for c in e[1:]) + "}"


def evaluate(e, xs):
    k = e[0]
    if k == "var":
        return xs[e[1]]
    if k == "cst":
        return torch.tensor(e[1], dtype=torch.float32)
    args = [evaluate(c, xs) for c in e[1:]]
    return {"add": lambda a, b: a + b, "mul": lambda a, b: a * b, "relu": torch.relu,
            "tanh": torch.tanh, "exp": torch.exp}[k](*args)


def main():
    rng = random.Random(SEED)
    failures = checked = skipped = 0
    with tempfile.TemporaryDirectory() as d:
        for case in range(CASES):
            env = [round(rng.uniform(-1.5, 1.5), 3) for _ in range(NVARS)]
            tree = gen(rng, rng.randint(1, 4))
            xs = [torch.tensor(v, dtype=torch.float32, requires_grad=True) for v in env]
            y = evaluate(tree, xs)
            if not (y.requires_grad and torch.isfinite(y) and abs(float(y.detach())) < 1e3):
                skipped += 1                       # no variables, or out of a sane range
                continue
            y.backward()
            want = [float(y.detach())] + [float(x.grad.detach()) if x.grad is not None else 0.0 for x in xs]
            src = render(tree)
            prog = os.path.join(d, "t.bend")
            open(prog, "w").write(HEAD.format(path=ROOT, expr=src, env="[" + ", ".join(lit(v) for v in env) + "]"))
            r = subprocess.run([BEND, prog], capture_output=True, text=True, env=ENV, cwd=d)
            try:
                got = [float(t) for t in r.stdout.split()]
            except ValueError:
                got = []
            ok = len(got) == len(want) and all(abs(a - b) <= ATOL + RTOL * abs(b) for a, b in zip(got, want))
            checked += 1
            if not ok:
                failures += 1
                print(f"ERROR case {case}: {src}\n  env={env}\n  bend={got}\n  torch={want}\n  out={r.stdout[:100]} {r.stderr[:200]}")
    print(f"{'ok  ' if not failures else 'ERROR'} random autograd trees: {checked - failures}/{checked} match PyTorch ({skipped} skipped)")
    sys.exit(1 if failures else 0)


main()
