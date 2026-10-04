"""GPT-2 tokenizer in Bend (demos/gpt2/tok.bend) against tiktoken."""
import os, subprocess, sys, tempfile, time
import tiktoken

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "demos/gpt2/data")
EXE = sys.argv[1] if len(sys.argv) > 1 else None

CASES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "The capital of France is",
    "I'm sure we're fine; they'll say it's his and I'd go, we've seen it.",
    "Numbers 123 and 4567890, price $19.99 (50% off)!",
    "  leading spaces and   multiple   spaces   inside  ",
    "line one\nline two\n\nline four\t tabbed",
    "trailing space ",
    "email@example.com http://x.org/a_b?c=d&e=f #hash @user",
    "Machine learning in Bend: types that catch shape errors at compile time.",
    "ação, coração e informação não cabem; café déjà vu naïve résumé",
    "Привет, мир! 你好 こんにちは",
    "",
    " ",
    "a",
    "\n\n\n",
]


def main():
    enc = tiktoken.get_encoding("gpt2")
    failures = 0
    with tempfile.TemporaryDirectory() as d:
        for s in CASES:
            p = os.path.join(d, "t.txt"); open(p, "wb").write(s.encode("utf-8"))
            t0 = time.time()
            r = subprocess.run([EXE, f"{D}/merges_num.txt", f"{D}/perm.txt", p], capture_output=True, text=True)
            dt = time.time() - t0
            got = [int(x) for x in r.stdout.split()] if r.stdout.strip() and r.stdout.strip()[0].isdigit() else []
            want = enc.encode(s)
            ok = got == want
            if not ok: failures += 1
            print(f"{'ok  ' if ok else 'ERROR'} {dt:5.2f}s {s[:48]!r}" + ("" if ok else f"\n   bend={got}\n   tiktoken={want}"))
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)

main()
