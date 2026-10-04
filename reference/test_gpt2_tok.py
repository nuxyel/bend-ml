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
    "def add(a, b):\n    return a + b  # sum\n\nprint(add(1, 2))",
    "# Title\n\n- item one\n- item two\n\n```python\nx = [1, 2, 3]\n```",
    "3.14159 2.71828 1e-5 0xFF 1,000,000 $5.99 100% #1 @home",
    "Don't won't can't I've she'd they're it's 'quoted' \"double\"",
    "tabs\tand\nnewlines\r\nand   triple   spaces and trailing   ",
    "UPPER lower MiXeD camelCaseWord snake_case_word kebab-case-word",
    "https://example.com/path?a=1&b=2#frag user@mail.org ~/dir/file.txt C:\\Users\\x",
    "(parentheses) [brackets] {braces} <angle> a+b-c*d/e=f!g?h;i:j,k.l",
    "The  Quick\n\nBrown   Fox\t\tJumps",
    "naïve café résumé façade jalapeño über Ärger",
    "Привет мир, как дела? Это тест токенизатора.",
    "你好，世界！这是一个测试。 こんにちは世界",
    "mixed English и русский 中文 español português in one line",
    "x" * 300,
    "ab" * 150,
    "word " * 120,
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
