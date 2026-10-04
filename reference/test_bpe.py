"""Compara o tokenizer em Bend (bpe/cli.bend) com a referência em Python."""
import os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import bpe_ref

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")

CORPORA = {
    "classico": "aaabdaaabac".encode(),
    "pangrama": "the quick brown fox jumps over the lazy dog. the dog barks; the fox runs away. " * 3,
    "acentos": "coração, ação, pão, não, informação, canção; coração de pão. ação e reação. " * 3,
    "misto": "naïve café — résumé: déjà vu, über, 日本語 テスト, emoji 😀 e cedilha ç. " * 2,
}


def bend(*args):
    r = subprocess.run([BEND, os.path.join(ROOT, "bpe/cli.bend"), *args], capture_output=True, text=True, env=ENV)
    if r.returncode != 0 or "rror" in r.stdout[:200]:
        raise RuntimeError(f"bend falhou: {r.stdout[:300]} {r.stderr[:300]}")
    return [int(x) for x in r.stdout.split()]


def flat(table):
    return [x for pair in table for x in pair]


def main():
    falhas = 0
    with tempfile.TemporaryDirectory() as d:
        def w(name, data):
            p = os.path.join(d, name)
            open(p, "wb").write(data if isinstance(data, bytes) else data.encode())
            return p

        for nome, texto in CORPORA.items():
            data = texto if isinstance(texto, bytes) else texto.encode("utf-8")
            n = 30
            ref_table = bpe_ref.train(data, n)
            got_table = bend("train", str(n), w(nome + ".txt", data))
            ok_train = got_table == flat(ref_table)
            merges = w(nome + ".merges", " ".join(map(str, flat(ref_table))))
            for amostra_nome, amostra in [("proprio", data), ("curto", data[:7]), ("vazio", b""),
                                          ("outro", "ação não é pão 😀 the dog".encode("utf-8"))]:
                ref_ids = bpe_ref.encode(ref_table, amostra)
                got_ids = bend("encode", merges, w("a.txt", amostra)) if amostra else []
                ok_enc = got_ids == ref_ids
                ids_file = w("ids.txt", " ".join(map(str, ref_ids)) + " ")
                got_dec = bend("decode", merges, ids_file) if ref_ids else []
                ok_dec = bytes(got_dec) == amostra and bytes(got_dec) == bpe_ref.decode(ref_table, ref_ids)
                status = "ok " if (ok_enc and ok_dec) else "ERRO"
                if not (ok_enc and ok_dec):
                    falhas += 1
                print(f"{status} {nome}/{amostra_nome}: encode={'ok' if ok_enc else 'DIFERE'} decode/roundtrip={'ok' if ok_dec else 'DIFERE'}")
            print(f"{'ok ' if ok_train else 'ERRO'} {nome}: train ({len(ref_table)} regras)")
            if not ok_train:
                falhas += 1
    print("FALHAS:", falhas)
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
