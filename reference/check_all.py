"""Runs every bend-ml verification. Exit code 0 = everything is fine.

  reference/.venv/bin/python reference/check_all.py          # quick (~30 s)
  reference/.venv/bin/python reference/check_all.py --full   # also GPT-2 and 50 MNIST batches (~1.5 min extra)
"""
import os, re, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "reference/.venv/bin/python")
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
# BEND_BIN and BEND_PIN run the checks on another Bend (for example a newer release before a bump)
BEND_PIN = os.environ.get("BEND_PIN") or "2.0.37"
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
    if not bv or bv[-1] != BEND_PIN:
        sys.exit(f"ERROR: bend {BEND_PIN} is required (found {' '.join(bv) or 'none'}); run scripts/setup.sh")
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

    # 1b. binary-read: the one package with foreign code, kept to one def and out of the packages with laws
    run("binary-read/main.bend: read_f32 is its only foreign def", [BEND, "binary-read/main.bend"],
        lambda o, c: "1 def relies on unsafe or foreign code" in o and re.findall(r"^- (\S+)$", o, re.M) == ["read_f32"])
    laws = [f"{p}/main.bend" for p in ["nat-lemmas", "bpe", "tensor", "tensor-array", "autograd"]]
    clean = not any("binary-read" in open(os.path.join(ROOT, f)).read() for f in laws)
    results.append(clean)
    print(f"{'ok  ' if clean else 'ERROR'} no package with laws imports binary-read")
    want = "".join(l[3:] for l in open(os.path.join(ROOT, "binary-read/tests/read_f32.bend")) if l.startswith("#| "))
    run("binary-read/tests/read_f32.bend on the JS lane", [BEND, "binary-read/tests/read_f32.bend"], lambda o, c: o.strip() == want.strip())
    with tempfile.TemporaryDirectory() as d:
        rf = os.path.join(d, "rf")
        run("compile binary-read/tests/read_f32.bend", [BEND, "binary-read/tests/read_f32.bend", "-o", rf])
        run("binary-read/tests/read_f32.bend on the C lane", [rf], lambda o, c: o.strip() == want.strip())

    # 2. shape errors that MUST fail to check
    for f, msg in [("tensor/tests/bad_matmul.bend", "expected"), ("tensor/tests/bad_reshape.bend", "expected"), ("tensor-array/tests/bad_matmul.bend", "expected"), ("tensor-array/tests/bad_grad.bend", "expected"), ("examples/runtime_batch_bad.bend", "expected"), ("examples/symbolic_reshape_bad.bend", "expected"), ("examples/square_transpose_bad.bend", "expected"), ("tensor-array/tests/bad_bands.bend", "expected")]:
        run(f"{f} must be a type error", [BEND, f], lambda o, c, m=msg: "SOME PROOFS FAIL" in o and m in o)
    run("tensor/tests/ok.bend compiles and runs", [BEND, "tensor/tests/ok.bend"], lambda o, c: "18 18 18 18" in o)
    run("tensor-array/tests/ok.bend compiles and runs", [BEND, "tensor-array/tests/ok.bend"], lambda o, c: o.strip() == "8n")
    run("examples/symbolic_reshape.bend reshapes Mat<3, 6> into Mat<6, 3>", [BEND, "examples/symbolic_reshape.bend", "--", "3"], lambda o, c: "regroup(x) : Mat<6, 3>" in o and "15 16 17" in o)
    # the list-based baselines of the demos: nothing else builds them (train.bend went unchecked from v0.3 to v1.2.2)
    with tempfile.TemporaryDirectory() as d:
        run("demos/gpt2/gpt2.bend (lists) compiles", [BEND, "demos/gpt2/gpt2.bend", "-o", os.path.join(d, "g")])
        run("demos/mnist/train.bend (lists) compiles", [BEND, "demos/mnist/train.bend", "-o", os.path.join(d, "m")])
    with tempfile.TemporaryDirectory() as d:
        c_out = os.path.join(d, "par_bands.c")
        run("Bands.matvec compiles to a parallel join (no sequential fallback)", [BEND, "tensor-array/tests/par_bands.bend", "-o", c_out],
            lambda o, c: c == 0 and os.path.exists(c_out) and re.search(r"FID_\w*_BMV_J\d+", open(c_out).read()) is not None)
    with tempfile.TemporaryDirectory() as d:
        t8 = os.path.join(d, "t8")
        run("compile tensor-array/tests/t8.bend", [BEND, "tensor-array/tests/t8.bend", "-o", t8])
        run("the products in cells of 8 give the scalar loop's bits (tensor-array/tests/t8.bend)", [t8], lambda o, c: o.strip().strip('"') == "0 0 0 0 0 0 0 0")
        acc = os.path.join(d, "acc")
        run("compile tensor-array/tests/acc.bend", [BEND, "tensor-array/tests/acc.bend", "-o", acc])
        run("the accumulating products give the bits of Mat.add(C, A·B) (tensor-array/tests/acc.bend)", [acc], lambda o, c: o.strip().strip('"') == "0 0 0 0 0 0")
        cap = os.path.join(d, "cap")
        run("compile tensor-array/tests/cap.bend", [BEND, "tensor-array/tests/cap.bend", "-o", cap])
        run("cap_depth by doubling equals the n-step cap_pair (tensor-array/tests/cap.bend)", [cap], lambda o, c: o.strip() == "0")
        split = os.path.join(d, "split")
        run("compile tensor-array/tests/split.bend", [BEND, "tensor-array/tests/split.bend", "-o", split])
        run("split_depth picks 0 or par, and par 4 gives par 0's bits (tensor-array/tests/split.bend)", [split, "--threads", "16"],
            lambda o, c: o.strip() == '"0 3 0 0 3 0 4\\n0 0 0 0 0 0"')
        parseq = os.path.join(d, "parseq")
        run("compile tensor-array/tests/parseq.bend", [BEND, "tensor-array/tests/parseq.bend", "-o", parseq])
        run("parallel = sequential: 11,664 products at depths 1-4 give depth 0's bits (tensor-array/tests/parseq.bend)", [parseq, "--threads", "16"],
            lambda o, c: o.strip() == '"0 11664"')
        onerow = os.path.join(d, "onerow")
        run("compile tensor-array/tests/onerow.bend", [BEND, "tensor-array/tests/onerow.bend", "-o", onerow])
        run("one row at par 4 runs the scalar loop: 50,257 x 768 gives par 0's bits (tensor-array/tests/onerow.bend)", [onerow, "--threads", "16"],
            lambda o, c: o.strip() == '"0 0 0"')
    run("tensor-array/tests/checked.bend rejects wrong label counts", [BEND, "tensor-array/tests/checked.bend"], lambda o, c: o.strip() == "[1n, 0n, 0n, 1n, 0n, 0n]")

    # 3. tests against the Python references
    run("bpe vs Python reference (train/encode/decode)", [PY, "reference/test_bpe.py"])
    run("tensor vs PyTorch", [PY, "reference/test_tensor.py"])
    run("tensor-array vs PyTorch", [PY, "reference/test_tensor_array.py"])
    run("runtime batch size (examples/runtime_batch.bend) vs NumPy", [PY, "reference/test_runtime_batch.py"])
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
            ml = os.path.join(d, "mnist_lists")
            run("compile MNIST (lists)", [BEND, "demos/mnist/train.bend", "-o", ml])
            run("MNIST (lists) in Bend, 10 batches, same loss as PyTorch", [ml, "1", "10", "0.1"],
                lambda o, c: "train_loss=2.2162626" in o and "test_correct=4934/10000" in o)
            run("MNIST in PyTorch, 10 batches", [PY, "reference/mnist_torch.py", "--epochs", "1", "--max-batches", "10"],
                lambda o, c: "train_loss=2.216263" in o and "test_acc=0.4934" in o)

    failures = results.count(False)
    print(f"\nTOTAL: {len(results) - failures}/{len(results)} checks ok")
    sys.exit(1 if failures else 0)


main()
