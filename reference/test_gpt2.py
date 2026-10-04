"""GPT-2 em Bend (demos/gpt2/gpt2.bend) contra o mesmo modelo em PyTorch: ids e logits por passo."""
import os, re, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "reference/.venv/bin/python")
EXE = sys.argv[1]
CASES = [("The capital of France is", 8), ("Machine learning is", 8), ("1, 2, 3, 4,", 6)]
TOL = 5e-2  # os logits têm módulo ~100; F32 acumulado em 12 camadas


def main():
    falhas = 0
    for prompt, n in CASES:
        ref = subprocess.run([PY, os.path.join(ROOT, "reference/gpt2_ref.py"), prompt, str(n)], capture_output=True, text=True, cwd=ROOT).stdout
        bend = subprocess.run([EXE, prompt, str(n)], capture_output=True, text=True, cwd=ROOT).stdout
        r_ids = eval(re.search(r"gerados (\[.*\])", ref).group(1))
        r_lg = [float(x) for x in re.search(r"^logits (.*)", ref, re.M).group(1).split()]
        r_txt = re.search(r"texto (.*)", ref).group(1)
        b_ids = [int(x) for x in re.findall(r"id (\d+) +logit", bend)]
        b_lg = [float(x) for x in re.findall(r"logit (-?[\d.]+)", bend)]
        b_txt = re.search(r"texto: (.*)", bend).group(1)
        ids_ok = b_ids == r_ids
        lg_ok = len(b_lg) == len(r_lg) and max(abs(a - b) for a, b in zip(b_lg, r_lg)) < TOL
        txt_ok = repr(b_txt) == r_txt
        err = max(abs(a - b) for a, b in zip(b_lg, r_lg)) if len(b_lg) == len(r_lg) else float("nan")
        print(f"{'ok  ' if ids_ok and lg_ok and txt_ok else 'ERRO'} {prompt!r}: ids={'ok' if ids_ok else 'DIFERE'} texto={'ok' if txt_ok else 'DIFERE'} max|dlogit|={err:.2e}")
        if not (ids_ok and lg_ok and txt_ok):
            falhas += 1; print("  bend:", b_ids, "\n  torch:", r_ids)
    print("FALHAS:", falhas)
    sys.exit(1 if falhas else 0)

main()
