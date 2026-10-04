"""Prepares the GPT-2 small data for Bend and validates the BPE equivalence.

Writes to demos/gpt2/data:
  weights.bin     little-endian float32, in the order of WEIGHT_ORDER below
  merges_num.txt  one rule per line: "a b" with a, b in the bpe/ scheme (byte -> 0..255, M{k} -> 256+k)
  perm.txt        256 numbers: the GPT-2 id of byte b
It also checks, with tiktoken, that applying the rules in order (what Bend does) gives the same ids.
"""
import json, os, sys
import numpy as np
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(__file__))
import bpe_ref

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "demos/gpt2/data")


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


def weight_order():
    names = ["wte.weight", "wpe.weight"]
    for l in range(12):
        p = f"h.{l}."
        names += [p + "ln_1.weight", p + "ln_1.bias", p + "attn.c_attn.weight", p + "attn.c_attn.bias",
                  p + "attn.c_proj.weight", p + "attn.c_proj.bias", p + "ln_2.weight", p + "ln_2.bias",
                  p + "mlp.c_fc.weight", p + "mlp.c_fc.bias", p + "mlp.c_proj.weight", p + "mlp.c_proj.bias"]
    return names + ["ln_f.weight", "ln_f.bias"]


def main():
    vocab = json.load(open(f"{D}/vocab.json"))
    b2u = bytes_to_unicode()
    perm = [vocab[b2u[b]] for b in range(256)]
    inv = {i: b for b, i in enumerate(perm)}
    merges = [l.split() for l in open(f"{D}/merges.txt", encoding="utf-8").read().split("\n")[1:] if l.strip()]
    assert len(merges) == 50000, len(merges)
    lines = []
    for k, (a, b) in enumerate(merges):
        ia, ib, im = vocab[a], vocab[b], vocab[a + b]
        assert im == 256 + k, (k, a, b, im)  # the id of merge k is 256+k
        na = inv[ia] if ia < 256 else ia
        nb = inv[ib] if ib < 256 else ib
        lines.append(f"{na} {nb}")
    open(f"{D}/merges_num.txt", "w").write("\n".join(lines) + "\n")
    open(f"{D}/perm.txt", "w").write(" ".join(map(str, perm)) + "\n")
    print("merges_num.txt and perm.txt ok (50000 rules, ids 256+k confirmed)")

    # weights
    if not os.path.exists(f"{D}/weights.bin") or "--force" in sys.argv:
        total = 0
        with safe_open(f"{D}/model.safetensors", "pt") as f, open(f"{D}/weights.bin", "wb") as out:
            keys = set(f.keys())
            pre = "transformer." if any(k.startswith("transformer.") for k in keys) else ""
            for n in weight_order():
                t = f.get_tensor(pre + n).float().numpy().astype("<f4")
                out.write(t.tobytes()); total += t.size
        print("weights.bin written:", total, "floats")

    # BPE equivalence (rules in order) with tiktoken
    try:
        import tiktoken, regex
        enc = tiktoken.get_encoding("gpt2")
        table = [tuple(map(int, l.split())) for l in lines]
        pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        samples = ["Hello, world! The quick brown fox jumps over the lazy dog.",
                   "Machine learning in Bend: types that catch shape errors at compile time.",
                   "ação, coração e informação não cabem num ASCII; café déjà vu.",
                   "  multiple   spaces\nand\nnewlines\t tabs 123 4567 it's we're they'll"]
        ok = True
        for s in samples:
            ids = []
            for w in regex.findall(pat, s):
                ids += [perm[x] if x < 256 else x for x in bpe_ref.encode(table, w.encode("utf-8"))]
            ref = enc.encode(s)
            same = ids == ref
            ok &= same
            print("tiktoken == rules in order:", same, "|", s[:40])
        print("EQUIVALENCE WITH TIKTOKEN:", "ok" if ok else "FAILED")
    except Exception as e:
        print("tiktoken not verified:", repr(e)[:200])


if __name__ == "__main__":
    main()


def split_weights():
    """One .bin file per tensor in demos/gpt2/data/w/; Conv1D weights written TRANSPOSED ([output, input])."""
    os.makedirs(f"{D}/w", exist_ok=True)
    raw = np.fromfile(f"{D}/weights.bin", dtype="<f4")
    pos = 0
    def take(*shape):
        nonlocal pos
        n = int(np.prod(shape)); a = raw[pos:pos + n].reshape(*shape); pos += n
        return a
    def put(name, a):
        np.ascontiguousarray(a, dtype="<f4").tofile(f"{D}/w/{name}.bin")
    put("wte", take(50257, 768)); put("wpe", take(1024, 768))
    for l in range(12):
        put(f"h{l}.ln1_g", take(768)); put(f"h{l}.ln1_b", take(768))
        put(f"h{l}.aw", take(768, 2304).T); put(f"h{l}.ab", take(2304))
        put(f"h{l}.pw", take(768, 768).T); put(f"h{l}.pb", take(768))
        put(f"h{l}.ln2_g", take(768)); put(f"h{l}.ln2_b", take(768))
        put(f"h{l}.fw", take(768, 3072).T); put(f"h{l}.fb", take(3072))
        put(f"h{l}.mw", take(3072, 768).T); put(f"h{l}.mb", take(768))
    put("lnf_g", take(768)); put("lnf_b", take(768))
    assert pos == raw.size
    print("w/*.bin written:", len(os.listdir(f"{D}/w")), "files")


if __name__ == "__main__" and "--split" in sys.argv:
    split_weights()
