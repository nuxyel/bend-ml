"""Runs every bend-ml verification. Exit code 0 = everything is fine.

  reference/.venv/bin/python reference/check_all.py          # quick (~30 s)
  reference/.venv/bin/python reference/check_all.py --full   # also GPT-2 and 50 MNIST batches (~1.5 min extra)
"""
import os, re, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "reference/.venv/bin/python")
BEND = os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1", PATH=os.path.expanduser("~/.elan/bin") + ":" + os.environ["PATH"])
FULL = "--full" in sys.argv
results = []


def run(name, cmd, ok_if=None, timeout=3600, extra_env=None):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=dict(ENV, **(extra_env or {})), timeout=timeout)
    out = r.stdout + r.stderr
    ok = (r.returncode == 0) if ok_if is None else ok_if(out, r.returncode)
    results.append(ok)
    print(f"{'ok  ' if ok else 'ERROR'} {name} ({time.time() - t0:.0f}s)")
    if not ok:
        print("     " + out.strip().replace("\n", "\n     ")[-600:])
    return out


def check_toolchain():
    """Fail early, with a clear message, when the pinned toolchain is not the one in use."""
    bv = subprocess.run([BEND, "version"], capture_output=True, text=True, env=ENV).stdout.split()
    if not bv or bv[-1] != "2.0.35":
        sys.exit(f"ERROR: bend 2.0.35 is required (found {' '.join(bv) or 'none'}); run scripts/setup.sh")
    lv = subprocess.run(["lean", "--version"], capture_output=True, text=True, env=ENV)
    if "version 4.34.0" not in lv.stdout:
        sys.exit(f"ERROR: lean 4.34.0 is required for --verdict (found {lv.stdout.strip() or 'none'}); run scripts/setup.sh")


def main():
    check_toolchain()
    # 1. proofs: checker and audited kernel, in each package
    for pkg in ["nat-lemmas", "bpe", "tensor", "tensor-array", "autograd"]:
        f = f"{pkg}/main.bend"
        run(f"{f}: ALL PROOFS CHECK", [BEND, f], lambda o, c: "ALL PROOFS CHECK" in o and "FAIL" not in o)
        run(f"{f} --verdict (kernel em Lean)", [BEND, f, "--verdict"], lambda o, c: o.strip().endswith("ALL PROOFS CHECK"))
        src = open(os.path.join(ROOT, f)).read()
        clean = not re.search(r"@unsafe|^\s*\?\w+|sorry", re.sub(r"#.*", "", src), re.M)
        results.append(clean)
        print(f"{'ok  ' if clean else 'ERROR'} {f}: no @unsafe or ?TODO")

    # 2. shape errors that MUST fail to check
    for f, msg in [("tensor/tests/bad_matmul.bend", "expected"), ("tensor/tests/bad_reshape.bend", "expected"), ("tensor-array/tests/bad_matmul.bend", "expected"), ("tensor-array/tests/bad_grad.bend", "expected")]:
        run(f"{f} must be a type error", [BEND, f], lambda o, c, m=msg: "SOME PROOFS FAIL" in o and m in o)
    run("tensor/tests/ok.bend compiles and runs", [BEND, "tensor/tests/ok.bend"], lambda o, c: "18 18 18 18" in o)
    run("tensor-array/tests/ok.bend compiles and runs", [BEND, "tensor-array/tests/ok.bend"], lambda o, c: o.strip() == "8n")
    run("tensor-array/tests/checked.bend rejects wrong label counts", [BEND, "tensor-array/tests/checked.bend"], lambda o, c: o.strip() == "[1n, 0n, 0n, 1n, 0n, 0n]")

    # 3. tests against the Python references
    run("bpe vs Python reference (train/encode/decode)", [PY, "reference/test_bpe.py"])
    run("tensor vs PyTorch", [PY, "reference/test_tensor.py"])
    run("tensor-array vs PyTorch", [PY, "reference/test_tensor_array.py"])
    run("autograd vs PyTorch (gradient checking)", [PY, "reference/test_autograd.py"])
    cases = "80" if FULL else "30"
    run(f"autograd on {120 if FULL else 30} random expression trees vs PyTorch", [PY, "reference/test_autograd_random.py"], extra_env={"FUZZ_CASES": "120" if FULL else "30"})
    run(f"BPE fuzz ({cases} random corpora) vs the Python reference", [PY, "reference/test_fuzz_bpe.py"], extra_env={"FUZZ_CASES": cases})
    run("published packages import from BendHub and apply their laws", [PY, "reference/test_published.py"])

    # 4. GPT-2 tokenizer vs tiktoken
    data = os.path.join(ROOT, "demos/gpt2/data/merges_num.txt")
    if os.path.exists(data):
        with tempfile.TemporaryDirectory() as d:
            exe = os.path.join(d, "tok")
            run("compile the GPT-2 tokenizer", [BEND, "demos/gpt2/tok.bend", "-o", exe])
            run("GPT-2 tokenizer vs tiktoken", [PY, "reference/test_gpt2_tok.py", exe])
    else:
        print("skipped: GPT-2 tokenizer (run reference/gpt2_prep.py first)")

    # 5. full demos (--full)
    if FULL:
        with tempfile.TemporaryDirectory() as d:
            exe = os.path.join(d, "gpt2")
            run("compile GPT-2 (Array)", [BEND, "demos/gpt2/fast.bend", "-o", exe])
            run("GPT-2 (Array) in Bend vs PyTorch (11 prompts)", [PY, "reference/test_gpt2.py", exe])
            mn = os.path.join(d, "mnist")
            run("compile MNIST (Array)", [BEND, "demos/mnist/fast.bend", "-o", mn])
            out = run("MNIST (Array) in Bend, 50 batches", [mn, "1", "50", "0.1"], lambda o, c: "test_correct=7829/10000" in o)
            run("MNIST in PyTorch, 50 batches", [PY, "reference/mnist_torch.py", "--epochs", "1", "--max-batches", "50"], lambda o, c: "test_acc=0.7829" in o)

    failures = results.count(False)
    print(f"\nTOTAL: {len(results) - failures}/{len(results)} checks ok")
    sys.exit(1 if failures else 0)


main()
