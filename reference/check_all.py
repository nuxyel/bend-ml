"""Roda todas as verificações do bend-ml. Saída 0 = tudo certo.

  reference/.venv/bin/python reference/check_all.py          # rápido (~3 min)
  reference/.venv/bin/python reference/check_all.py --full   # inclui GPT-2 e 50 lotes de MNIST (~5 min extra)
"""
import os, re, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "reference/.venv/bin/python")
BEND = os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1", PATH=os.path.expanduser("~/.elan/bin") + ":" + os.environ["PATH"])
FULL = "--full" in sys.argv
resultados = []


def run(nome, cmd, ok_if=None, timeout=3600):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=ENV, timeout=timeout)
    out = r.stdout + r.stderr
    ok = (r.returncode == 0) if ok_if is None else ok_if(out, r.returncode)
    resultados.append(ok)
    print(f"{'ok  ' if ok else 'ERRO'} {nome} ({time.time() - t0:.0f}s)")
    if not ok:
        print("     " + out.strip().replace("\n", "\n     ")[-600:])
    return out


def main():
    # 1. provas: checker e kernel auditado, em cada pacote
    for pkg in ["nat-lemmas", "bpe", "tensor", "tensor-array", "autograd"]:
        f = f"{pkg}/main.bend"
        run(f"{f}: ALL PROOFS CHECK", [BEND, f], lambda o, c: "ALL PROOFS CHECK" in o and "FAIL" not in o)
        run(f"{f} --verdict (kernel em Lean)", [BEND, f, "--verdict"], lambda o, c: o.strip().endswith("ALL PROOFS CHECK"))
        src = open(os.path.join(ROOT, f)).read()
        limpo = not re.search(r"@unsafe|^\s*\?\w+|sorry", re.sub(r"#.*", "", src), re.M)
        resultados.append(limpo)
        print(f"{'ok  ' if limpo else 'ERRO'} {f}: sem @unsafe nem ?TODO")

    # 2. erros de shape que DEVEM falhar ao checar
    for f, msg in [("tensor/tests/bad_matmul.bend", "expected"), ("tensor/tests/bad_reshape.bend", "expected"), ("tensor-array/tests/bad_matmul.bend", "expected"), ("tensor-array/tests/bad_grad.bend", "expected")]:
        run(f"{f} deve ser erro de tipo", [BEND, f], lambda o, c, m=msg: "SOME PROOFS FAIL" in o and m in o)
    run("tensor/tests/ok.bend compila e roda", [BEND, "tensor/tests/ok.bend"], lambda o, c: "18 18 18 18" in o)
    run("tensor-array/tests/ok.bend compila e roda", [BEND, "tensor-array/tests/ok.bend"], lambda o, c: o.strip() == "8n")

    # 3. testes contra as referências em Python
    run("bpe vs referência Python (train/encode/decode)", [PY, "reference/test_bpe.py"])
    run("tensor vs PyTorch", [PY, "reference/test_tensor.py"])
    run("tensor-array vs PyTorch", [PY, "reference/test_tensor_array.py"])
    run("autograd vs PyTorch (gradient checking)", [PY, "reference/test_autograd.py"])

    # 4. tokenizer do GPT-2 vs tiktoken
    data = os.path.join(ROOT, "demos/gpt2/data/merges_num.txt")
    if os.path.exists(data):
        with tempfile.TemporaryDirectory() as d:
            exe = os.path.join(d, "tok")
            run("compilar tokenizer do GPT-2", [BEND, "demos/gpt2/tok.bend", "-o", exe])
            run("tokenizer do GPT-2 vs tiktoken", [PY, "reference/test_gpt2_tok.py", exe])
    else:
        print("pulado: tokenizer do GPT-2 (rode reference/gpt2_prep.py antes)")

    # 5. demos completos (--full)
    if FULL:
        with tempfile.TemporaryDirectory() as d:
            exe = os.path.join(d, "gpt2")
            run("compilar GPT-2 (Array)", [BEND, "demos/gpt2/fast.bend", "-o", exe])
            run("GPT-2 (Array) em Bend vs PyTorch (3 prompts)", [PY, "reference/test_gpt2.py", exe])
            mn = os.path.join(d, "mnist")
            run("compilar MNIST (Array)", [BEND, "demos/mnist/fast.bend", "-o", mn])
            out = run("MNIST (Array) em Bend, 50 lotes", [mn, "1", "50", "0.1"], lambda o, c: "acertos_teste=7829/10000" in o)
            run("MNIST em PyTorch, 50 lotes", [PY, "reference/mnist_torch.py", "--epochs", "1", "--max-batches", "50"], lambda o, c: "acc_teste=0.7829" in o)

    falhas = resultados.count(False)
    print(f"\nTOTAL: {len(resultados) - falhas}/{len(resultados)} verificações ok")
    sys.exit(1 if falhas else 0)


main()
