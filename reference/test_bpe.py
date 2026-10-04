"""Compares the Bend tokenizer (bpe/cli.bend) with the Python reference."""
import os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import bpe_ref

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")

CORPORA = {
    "classic": "aaabdaaabac".encode(),
    "pangram": "the quick brown fox jumps over the lazy dog. the dog barks; the fox runs away. " * 3,
    "accents": "coração, ação, pão, não, informação, canção; coração de pão. ação e reação. " * 3,
    "mixed": "naïve café — résumé: déjà vu, über, 日本語 テスト, emoji 😀 e cedilha ç. " * 2,
}


def bend(*args):
    r = subprocess.run([BEND, os.path.join(ROOT, "bpe/cli.bend"), *args], capture_output=True, text=True, env=ENV)
    if r.returncode != 0 or "rror" in r.stdout[:200]:
        raise RuntimeError(f"bend failed: {r.stdout[:300]} {r.stderr[:300]}")
    return [int(x) for x in r.stdout.split()]


def flat(table):
    return [x for pair in table for x in pair]


def main():
    failures = 0
    with tempfile.TemporaryDirectory() as d:
        def w(name, data):
            p = os.path.join(d, name)
            open(p, "wb").write(data if isinstance(data, bytes) else data.encode())
            return p

        for name, text in CORPORA.items():
            data = text if isinstance(text, bytes) else text.encode("utf-8")
            n = 30
            ref_table = bpe_ref.train(data, n)
            got_table = bend("train", str(n), w(name + ".txt", data))
            ok_train = got_table == flat(ref_table)
            merges = w(name + ".merges", " ".join(map(str, flat(ref_table))))
            for sample_name, sample in [("same", data), ("short", data[:7]), ("empty", b""),
                                          ("other", "ação não é pão 😀 the dog".encode("utf-8"))]:
                ref_ids = bpe_ref.encode(ref_table, sample)
                got_ids = bend("encode", merges, w("a.txt", sample)) if sample else []
                ok_enc = got_ids == ref_ids
                ids_file = w("ids.txt", " ".join(map(str, ref_ids)) + " ")
                got_dec = bend("decode", merges, ids_file) if ref_ids else []
                ok_dec = bytes(got_dec) == sample and bytes(got_dec) == bpe_ref.decode(ref_table, ref_ids)
                status = "ok " if (ok_enc and ok_dec) else "ERROR"
                if not (ok_enc and ok_dec):
                    failures += 1
                print(f"{status} {name}/{sample_name}: encode={'ok' if ok_enc else 'DIFFERS'} decode/roundtrip={'ok' if ok_dec else 'DIFFERS'}")
            print(f"{'ok ' if ok_train else 'ERROR'} {name}: train ({len(ref_table)} rules)")
            if not ok_train:
                failures += 1
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
