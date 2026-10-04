"""Randomized (seeded) comparison of the Bend BPE tokenizer with the Python reference.

For many random corpora and merge counts: train, encode and decode in Bend must equal the
reference, and decode(encode(s)) must equal s (the proved roundtrip, checked on real runs).
"""
import os, random, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import bpe_ref

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
CASES = int(os.environ.get("FUZZ_CASES", "80"))
SEED = 20261004

WORDS = ["the", "quick", "brown", "fox", "ação", "coração", "naïve", "café", "日本語", "😀", "def", "return",
         "x=1", "\n", "  ", "ü", "ß", "Привет", "mañana", "e", "a", "ab", "ba", "aaa", "123"]


def bend(*args):
    r = subprocess.run([BEND, os.path.join(ROOT, "bpe/cli.bend"), *args], capture_output=True, text=True, env=ENV)
    out = r.stdout.strip()
    if r.returncode != 0 or (out and not (out[0].isdigit() or out[0] == "-")):
        raise RuntimeError(f"bend failed: {out[:200]} {r.stderr[:200]}")
    return [int(x) for x in out.split()]


def flat(table):
    return [x for pair in table for x in pair]


def gen_bytes(rng):
    kind = rng.choice(["alphabet", "words", "random", "repeat"])
    n = rng.randint(1, 160)
    if kind == "alphabet":
        letters = "".join(rng.sample("abcde", rng.randint(1, 3)))
        return "".join(rng.choice(letters) for _ in range(n)).encode()
    if kind == "words":
        return " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 30))).encode("utf-8")
    if kind == "random":
        return bytes(rng.randrange(256) for _ in range(n))
    return (rng.choice(WORDS) * rng.randint(2, 40)).encode("utf-8")


def main():
    rng = random.Random(SEED)
    failures = checks = 0
    with tempfile.TemporaryDirectory() as d:
        def w(name, data):
            p = os.path.join(d, name)
            open(p, "wb").write(data if isinstance(data, bytes) else data.encode())
            return p

        for case in range(CASES):
            corpus = gen_bytes(rng)
            n = rng.randint(0, 60)
            ref_table = bpe_ref.train(corpus, n)
            got_table = bend("train", str(n), w("c.txt", corpus)) if n > 0 else []
            ok = got_table == flat(ref_table)
            merges = w("m.txt", " ".join(map(str, flat(ref_table))) + " ")
            samples = [corpus[: rng.randint(1, len(corpus))], gen_bytes(rng)]
            for s in samples:
                ref_ids = bpe_ref.encode(ref_table, s)
                got_ids = bend("encode", merges, w("s.txt", s))
                dec = bend("decode", merges, w("i.txt", " ".join(map(str, ref_ids)) + " "))
                ok &= got_ids == ref_ids
                ok &= bytes(dec) == s == bpe_ref.decode(ref_table, ref_ids)      # the roundtrip
            checks += 1
            if not ok:
                failures += 1
                print(f"ERROR case {case}: n={n} corpus={corpus[:40]!r}")
    print(f"{'ok  ' if not failures else 'ERROR'} fuzz bpe: {checks - failures}/{checks} random cases")
    sys.exit(1 if failures else 0)


main()
