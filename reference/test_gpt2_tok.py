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
    "Wait — what? That’s “quoted” text… isn’t it?",
    "Price: €25, £30, ¥500, ₹99 and 100%",
    "arrows → ← ↔ ⇒ and math ∑ ∫ ≠ ≤ ≥ ∞ ± × ÷",
    "┌──────┐\n│ box  │\n└──────┘",
    "emoji 😀🎉 and flags 🇧🇷 and 👨‍👩‍👧 family ❤️ ok",
    "non\u00a0breaking\u00a0space and ideographic\u3000space\u3000\u3000end",
    "trailing nbsp\u00a0\u00a0",
    "tab\t\u2003em space\u2003\u2003x",
    "日本語、テスト。「引用」（括弧）！？ 中文，标点：全角ＡＢＣ１２３",
    "cafe\u0301 vs café, naïve, über, Привет, мир! γειά σου, שלום, مرحبا",
    "x² + y³ = z¹⁰, ½ ¼, ①②③, ٣٤٥ ४५६",
    "«guillemets» ¿qué? ¡hola! §3 ¶ © ® ™ ° µ",
    "bullet • dot · ellipsis … dagger † ‡ per mille ‰",
    "snake_case and CamelCase, 3.14159 and 1,000,000 and 0xFF",
    "it's we've they'll I'd you're can't 'quoted'",
    "mixed 🙂text🙂 with—dashes–and_underscores",
    "\u200bzero width\u200b joiner\u200d here",
    "x" * 300,
    "ab" * 150,
    "word " * 120,
    "",
    " ",
    "a",
    "\n\n\n",
]


def random_unicode(rng, n):
    """Random text mixing ASCII, spaces and code points from every plane the table covers."""
    pools = [(32, 126), (32, 32), (0xA0, 0x24F), (0x300, 0x36F), (0x370, 0x5FF), (0x2000, 0x2BFF),
             (0x3000, 0x30FF), (0xFF00, 0xFFEF), (0x1F300, 0x1FAFF), (0x4E00, 0x4F00)]
    out = []
    for _ in range(n):
        lo, hi = rng.choice(pools)
        cp = rng.randint(lo, hi)
        if 0xD800 <= cp <= 0xDFFF:
            continue
        out.append(chr(cp))
    return "".join(out)


def main():
    import random
    rng = random.Random(int(os.environ.get("FUZZ_SEED", "7")))
    CASES.extend(random_unicode(rng, rng.randint(1, 40)) for _ in range(int(os.environ.get("FUZZ_CASES", "30"))))
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
        # a malformed merges file must be refused, not tokenized
        txt = os.path.join(d, "t.txt"); open(txt, "wb").write(b"hello world")
        for name, merges in [("odd count", "104 101 256"), ("forward reference", "104 101 300 108"), ("unknown id", "104 101 99999 108")]:
            mp = os.path.join(d, "bad.txt"); open(mp, "w").write(merges)
            r = subprocess.run([EXE, mp, f"{D}/perm.txt", txt], capture_output=True, text=True)
            ok = r.stdout.strip() == "invalid merges table"
            if not ok: failures += 1
            print(f"{'ok  ' if ok else 'ERROR'} malformed merges refused: {name}")
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)

main()
