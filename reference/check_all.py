"""Runs every bend-ml verification. Exit code 0 = everything is fine.

  reference/.venv/bin/python reference/check_all.py          # quick (~30 s)
  reference/.venv/bin/python reference/check_all.py --full   # also GPT-2 and 50 MNIST batches (~1.5 min extra)
"""
import os, re, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "reference/.venv/bin/python")
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
# BEND_BIN and BEND_PIN run the checks on another Bend (for example a newer release before a bump)
BEND_PIN = os.environ.get("BEND_PIN") or "2.0.39"
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


def outside_proofs(o):
    """The defs a file has outside the proofs, as the checker lists them ("defs that depend on @unsafe: a, b")."""
    return [d for l in re.findall(r"^defs that depend on [^:]+: (.+)$", o, re.M) for d in l.split(", ")]


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
    # 0. every runtime option given to a Bend program sits between +RTS and -RTS (Bend 2.0.38 passes one written without
    # them to the program and runs on the default thread count, silently); the self-test shows the check tells them apart
    run("scripts/check_rts.py --self-test", [PY, "scripts/check_rts.py", "--self-test"])
    run("runtime options inside +RTS ... -RTS (scripts/check_rts.py)", [PY, "scripts/check_rts.py"])
    # 1. proofs: checker and audited kernel, in each package
    for pkg in ["nat-lemmas", "bpe", "tensor", "tensor-array", "autograd"]:
        f = f"{pkg}/main.bend"
        run(f"{f}: ALL PROOFS CHECK", [BEND, f], lambda o, c: "ALL PROOFS CHECK" in o and "FAIL" not in o)
        run(f"{f} --verdict (kernel em Lean)", [BEND, f, "--verdict"], lambda o, c: o.strip().endswith("ALL PROOFS CHECK"))
        src = open(os.path.join(ROOT, f)).read()
        clean = not re.search(r"@unsafe|^\s*\?\w+|sorry", re.sub(r"#.*", "", src), re.M)
        results.append(clean)
        print(f"{'ok  ' if clean else 'ERROR'} {f}: no @unsafe or ?TODO")

    # 1b. binary-read: the one package with foreign code, kept to its six reads and out of the packages with laws
    read_defs = ["read_f32", "size64", "read_f32_at", "read_bf16_at", "read_f32_into", "read_bf16_into"]
    run("binary-read/main.bend: its foreign defs are the six reads", [BEND, "binary-read/main.bend"],
        lambda o, c: f"{len(read_defs)} defs rely on unsafe or foreign code" in o and outside_proofs(o) == read_defs)
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
        ra = os.path.join(d, "ra")
        run("compile binary-read/tests/read_at.bend", [BEND, "binary-read/tests/read_at.bend", "-o", ra])
        run("binary-read: 64-bit offsets, BF16 and _into reads on the C lane", [PY, "binary-read/tests/test_read_at.py", ra],
            lambda o, c: c == 0 and o.strip().endswith("PASS (C)"))
    run("binary-read: 64-bit offsets, BF16 and _into reads on the JS lane", [PY, "binary-read/tests/test_read_at.py", "--js"],
        lambda o, c: c == 0 and o.strip().endswith("PASS (JavaScript)"))

    # 1c. tensor-fork: the one package whose parallel blocks share arrays through Base's @unsafe Array.fork; its unsafe
    # defs are exactly the blocks, the fork tree and what reaches them, no package with laws imports it, and it gives tensor-array's bits
    fork_defs = ["blocks1", "join2", "blocks2", "join3", "blocks3", "join4", "blocks4", "par1", "par2", "par3", "par4", "pk_par", "rpar", "cpar", "t8_tree", "t8_go", "gemm_fork", "gemm_rows", "gemm_any2", "gemm_any", "matmul", "matmul_acc", "matmul_nt", "matmul_nt_acc", "matmul_tn", "matmul_tn_acc", "mp_s3", "mp_s2", "mp_step", "mp_go", "mm_precise2", "matmul_precise"]
    run("tensor-fork/main.bend: its @unsafe defs are the fork tree and what reaches it", [BEND, "tensor-fork/main.bend"],
        lambda o, c: f"{len(fork_defs)} defs rely on unsafe or foreign code" in o and outside_proofs(o) == fork_defs)
    clean = not any("tensor-fork" in open(os.path.join(ROOT, f)).read() for f in laws)
    results.append(clean)
    print(f"{'ok  ' if clean else 'ERROR'} no package with laws imports tensor-fork")
    with tempfile.TemporaryDirectory() as d:
        tf = os.path.join(d, "tf")
        run("compile tensor-fork/tests/bits.bend", [BEND, "tensor-fork/tests/bits.bend", "-o", tf])
        for t in ["1", "16"]:
            run(f"tensor-fork gives tensor-array's bits, A and B intact: 14,832 products, the six public ones, three big ones, par1..par4 ({t} threads)",
                [tf, "+RTS", "--threads", t, "-RTS"], lambda o, c: o.split() == ["0", "0", "14832"] + ["0"] * 13)
        run("tensor-fork/tests/bits.bend refuses a bare --threads 16 (a runtime option outside +RTS ... -RTS)", [tf, "--threads", "16"],
            lambda o, c: c == 2 and "+RTS" in o)
        run("the runtime takes the thread count: one core on +RTS --threads 1 -RTS, several on 16 (scripts/rts_probe.py)",
            [PY, "scripts/rts_probe.py", tf])
        tfp = os.path.join(d, "tfp")
        run("compile tensor-fork/tests/precise.bend", [BEND, "tensor-fork/tests/precise.bend", "-o", tfp])
        for t in ["1", "16"]:
            run(f"tensor-fork's matmul_precise gives tensor-array's bits: 21 products ({t} threads)", [tfp, "+RTS", "--threads", t, "-RTS"],
                lambda o, c: o.strip() == "0 21")
    if FULL:
        run("tensor-fork/tests/bits.bend on the JS lane", [BEND, "tensor-fork/tests/bits.bend"],
            lambda o, c: re.findall(r"^[0-9 ]+$", o, re.M) == ["0 0 14832", "0 0 0 0 0 0", "0 0 0 0 0 0 0"])

    # 1d. tensor-gpu: tensor-array's products as one bang each (Base's @unsafe Array.fork/join, upstream's one-block
    # pattern; Renan's decision, 2026-10-18); its unsafe defs are the tree, its leaves and what reaches them, no package
    # with laws imports it, and it gives tensor-array's bits on the CPU threads and, when CUDA is there, on the GPU
    gpu_defs = ["mm", "put", "tile", "leaf_if", "leaf", "tree", "tree_new", "bang_new", "bang", "run_e", "gemm_new", "gemm", "matmul", "matmul_acc", "matmul_nt", "matmul_nt_acc", "matmul_tn", "matmul_tn_acc", "join2", "join3", "join4", "tree1", "root1", "root1_new", "go1", "go1_new", "run1", "run1_new", "tree2", "root2", "root2_new", "go2", "go2_new", "run2", "run2_new", "tree3", "root3", "root3_new", "go3", "go3_new", "run3", "run3_new", "tree4", "root4", "root4_new", "go4", "go4_new", "run4", "run4_new", "blocks_acc", "precise_c", "mm_precise3", "mm_precise2", "matmul_precise"]
    run("tensor-gpu/main.bend: its @unsafe defs are the bang's tree and what reaches it", [BEND, "tensor-gpu/main.bend"],
        lambda o, c: f"{len(gpu_defs)} defs rely on unsafe or foreign code" in o and outside_proofs(o) == gpu_defs)
    clean = not any("tensor-gpu" in open(os.path.join(ROOT, f)).read() for f in laws)
    results.append(clean)
    print(f"{'ok  ' if clean else 'ERROR'} no package with laws imports tensor-gpu")
    gpu_want = ["0", "0", "2958"] + ["0"] * 9
    with tempfile.TemporaryDirectory() as d:
        tg = os.path.join(d, "tg")
        run("compile tensor-gpu/tests/bits.bend", [BEND, "tensor-gpu/tests/bits.bend", "-o", tg])
        for t in ["1", "16"]:
            run(f"tensor-gpu gives tensor-array's bits, A and B intact: 2,958 products, the six public ones, three big ones ({t} CPU threads)",
                [tg, "+RTS", "--threads", t, "-RTS"], lambda o, c: o.split() == gpu_want)
        tk = os.path.join(d, "tk")
        run("compile tensor-gpu/tests/kernels.bend", [BEND, "tensor-gpu/tests/kernels.bend", "-o", tk])
        for t in ["1", "16"]:
            run(f"tensor-gpu's run1..run4 and run1_new..run4_new give their leaf's bits, inputs intact: 5 sizes ({t} CPU threads)",
                [tk, "+RTS", "--threads", t, "-RTS"], lambda o, c: o.split() == ["0"] * 10)
        cuda = os.environ.get("CUDA_HOME") or os.path.expanduser("~/.local/cuda12")
        if os.path.isdir(cuda):
            tgg = os.path.join(d, "tgg")
            env = {"CUDA_HOME": cuda, "LD_LIBRARY_PATH": os.path.join(cuda, "lib")}
            run("compile tensor-gpu/tests/bits.bend for CUDA", [BEND, "tensor-gpu/tests/bits.bend", "-o", tgg], extra_env=env)
            run("tensor-gpu gives tensor-array's bits on the GPU", [tgg], lambda o, c: o.split() == gpu_want, extra_env=env)
            tkg = os.path.join(d, "tkg")
            run("compile tensor-gpu/tests/kernels.bend for CUDA", [BEND, "tensor-gpu/tests/kernels.bend", "-o", tkg], extra_env=env)
            run("tensor-gpu's run1..run4 and run1_new..run4_new give their leaf's bits on the GPU, inputs intact", [tkg],
                lambda o, c: o.split() == ["0"] * 10, extra_env=env)
            tgp = os.path.join(d, "tgp")
            run("compile tensor-gpu/tests/precise.bend for CUDA", [BEND, "tensor-gpu/tests/precise.bend", "-o", tgp], extra_env=env)
            run("tensor-gpu's matmul_precise gives tensor-array's bits on the GPU: 21 products", [tgp],
                lambda o, c: o.strip() == "0 21", extra_env=env)
            run("tensor-gpu's matmul_precise gives tensor-array's bits on the CPU threads: 21 products", [tgp, "+RTS", "--gpu", "off", "--threads", "16", "-RTS"],
                lambda o, c: o.strip() == "0 21", extra_env=env)
        else:
            print("skip  tensor-gpu on the GPU: no CUDA toolkit (CUDA_HOME)")
    if FULL:
        run("tensor-gpu/tests/bits.bend on the JS lane", [BEND, "tensor-gpu/tests/bits.bend"],
            lambda o, c: re.findall(r"^[0-9 ]+$", o, re.M) == ["0 0 2958", "0 0 0 0 0 0", "0 0 0"])

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
            lambda o, c: c == 0 and os.path.exists(c_out) and re.search(r"FID_\w*_bmv_j\d+", open(c_out).read()) is not None)
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
        run("split_depth picks 0 or par, and par 4 gives par 0's bits (tensor-array/tests/split.bend)", [split, "+RTS", "--threads", "16", "-RTS"],
            lambda o, c: o.strip() == "0 3 0 0 3 0 4\n0 0 0 0 0 0")
        parseq = os.path.join(d, "parseq")
        run("compile tensor-array/tests/parseq.bend", [BEND, "tensor-array/tests/parseq.bend", "-o", parseq])
        run("parallel = sequential: 11,664 products at depths 1-4 give depth 0's bits (tensor-array/tests/parseq.bend)", [parseq, "+RTS", "--threads", "16", "-RTS"],
            lambda o, c: o.strip() == "0 11664")
        onerow = os.path.join(d, "onerow")
        run("compile tensor-array/tests/onerow.bend", [BEND, "tensor-array/tests/onerow.bend", "-o", onerow])
        run("one row at par 4 runs the scalar loop: 50,257 x 768 gives par 0's bits (tensor-array/tests/onerow.bend)", [onerow, "+RTS", "--threads", "16", "-RTS"],
            lambda o, c: o.strip() == "0 0 0")
        blocks = os.path.join(d, "blocks")
        run("compile tensor-array/tests/blocks.bend", [BEND, "tensor-array/tests/blocks.bend", "-o", blocks])
        run("matmul_precise sums k in blocks of 1024 from dotr; up to 1024 it is matmul (tensor-array/tests/blocks.bend)", [blocks, "+RTS", "--threads", "16", "-RTS"],
            lambda o, c: o.strip() == "0 22")
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
