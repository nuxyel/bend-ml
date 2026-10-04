"""Prints every `law` of every package: its name, its quantifiers and its claim, WITHOUT the proofs.

A reviewer can read the specification here and check that the statements say what the README
says. `--check-readme` also verifies that every law name appears in its package's README.

  reference/.venv/bin/python reference/list_laws.py
  reference/.venv/bin/python reference/list_laws.py --check-readme
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES = ["nat-lemmas", "bpe", "tensor", "tensor-array", "autograd"]


def laws_of(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    found = []
    i = 0
    while i < len(lines):
        m = re.match(r"^law ([\w.]+):\s*$", lines[i])
        if m:
            comment = []
            j = i - 1
            while j >= 0 and lines[j].startswith("#"):
                comment.insert(0, lines[j].lstrip("# ").rstrip())
                j -= 1
            body = []
            k = i + 1
            while k < len(lines) and lines[k].startswith("  "):
                body.append(lines[k].strip())
                k += 1
            found.append((m.group(1), [c for c in comment if c], body))
            i = k
        else:
            i += 1
    return found


def main():
    check = "--check-readme" in sys.argv
    problems = 0
    total = 0
    for pkg in PACKAGES:
        laws = laws_of(os.path.join(ROOT, pkg, "main.bend"))
        readme = open(os.path.join(ROOT, pkg, "README.md"), encoding="utf-8").read()
        print(f"\n== {pkg}: {len(laws)} laws")
        for name, comment, body in laws:
            total += 1
            quant = [b for b in body if b.startswith(("for", "exs"))]
            claim = [b for b in body if not b.startswith(("for", "exs"))]
            print(f"\nlaw {name}")
            for c in comment:
                print(f"   # {c}")
            for q in quant:
                print(f"     {q}")
            print(f"     {' '.join(claim)}")
            if check and name not in readme:
                problems += 1
                print(f"   !! '{name}' is not mentioned in {pkg}/README.md")
    print(f"\n{total} laws in {len(PACKAGES)} packages")
    if check:
        print("README check:", "ok" if not problems else f"{problems} law(s) missing from a README")
        sys.exit(1 if problems else 0)


main()
