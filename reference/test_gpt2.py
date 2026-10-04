"""GPT-2 in Bend (demos/gpt2/gpt2.bend or fast.bend) against the same model in PyTorch: ids and logits per step."""
import os, re, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "reference/.venv/bin/python")
EXE = sys.argv[1]
CASES = [
    ("Wait — the café costs €5 😀 and", 6),
    ("The capital of France is", 8),
    ("Machine learning is", 8),
    ("1, 2, 3, 4,", 6),
    ("def fibonacci(n):", 12),
    ("Once upon a time", 32),                      # a longer generation: exercises the KV cache
    ("The quick brown fox jumps over the lazy dog. The", 10),
    ("Q: What is 2 + 2?\nA:", 8),
    ("Paris is the capital of", 6),
    ("In 1969, humans first", 10),
    ("import numpy as np\n\n", 10),
]
TOL = 5e-2  # the logits have magnitude ~100; F32 accumulated over 12 layers


def main():
    failures = 0
    for prompt, n in CASES:
        ref = subprocess.run([PY, os.path.join(ROOT, "reference/gpt2_ref.py"), prompt, str(n)], capture_output=True, text=True, cwd=ROOT).stdout
        bend = subprocess.run([EXE, prompt, str(n)], capture_output=True, text=True, cwd=ROOT).stdout
        r_ids = eval(re.search(r"^generated (\[.*\])", ref, re.M).group(1))
        r_lg = [float(x) for x in re.search(r"^logits (.*)", ref, re.M).group(1).split()]
        r_txt = re.search(r"^text (.*)", ref, re.M).group(1)
        b_ids = [int(x) for x in re.findall(r"id (\d+) +logit", bend)]
        b_lg = [float(x) for x in re.findall(r"logit (-?[\d.]+)", bend)]
        b_txt = re.search(r"^text: (.*)", bend, re.M | re.S).group(1).rstrip("\n")  # the text may span several lines
        ids_ok = b_ids == r_ids
        lg_ok = len(b_lg) == len(r_lg) and max(abs(a - b) for a, b in zip(b_lg, r_lg)) < TOL
        txt_ok = repr(b_txt) == r_txt
        err = max(abs(a - b) for a, b in zip(b_lg, r_lg)) if len(b_lg) == len(r_lg) else float("nan")
        print(f"{'ok  ' if ids_ok and lg_ok and txt_ok else 'ERROR'} {prompt!r}: ids={'ok' if ids_ok else 'DIFFERS'} text={'ok' if txt_ok else 'DIFFERS'} max|dlogit|={err:.2e}")
        if not (ids_ok and lg_ok and txt_ok):
            failures += 1; print("  bend:", b_ids, "\n  torch:", r_ids)
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)

main()
