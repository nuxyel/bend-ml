"""Forward do GPT-2 small em PyTorch a partir de weights.bin (mesmos pesos do Bend).

Uso: gpt2_ref.py "prompt" [n_tokens]   ->  ids do prompt, top-5 logits do 1º passo, ids gerados (guloso), texto.
"""
import os, sys
import numpy as np
import torch
import tiktoken
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "demos/gpt2/data")


def load():
    raw = np.fromfile(f"{D}/weights.bin", dtype="<f4")
    pos = 0
    def take(*shape):
        nonlocal pos
        n = int(np.prod(shape)); a = torch.from_numpy(raw[pos:pos + n].copy()).reshape(*shape); pos += n
        return a
    p = {"wte": take(50257, 768), "wpe": take(1024, 768), "layers": []}
    for _ in range(12):
        p["layers"].append(dict(ln1_g=take(768), ln1_b=take(768), aw=take(768, 2304), ab=take(2304),
                                pw=take(768, 768), pb=take(768), ln2_g=take(768), ln2_b=take(768),
                                fw=take(768, 3072), fb=take(3072), mw=take(3072, 768), mb=take(768)))
    p["lnf_g"] = take(768); p["lnf_b"] = take(768)
    assert pos == raw.size
    return p


def forward(p, ids):
    t = len(ids)
    x = p["wte"][ids] + p["wpe"][:t]
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool))
    for L in p["layers"]:
        h = F.layer_norm(x, (768,), L["ln1_g"], L["ln1_b"], eps=1e-5)
        qkv = h @ L["aw"] + L["ab"]
        q, k, v = qkv.split(768, dim=1)
        heads = []
        for i in range(12):
            sl = slice(i * 64, (i + 1) * 64)
            s = (q[:, sl] @ k[:, sl].T) / 8.0
            s = s.masked_fill(~mask, float("-inf"))
            heads.append(torch.softmax(s, dim=1) @ v[:, sl])
        x = x + torch.cat(heads, dim=1) @ L["pw"] + L["pb"]
        h = F.layer_norm(x, (768,), L["ln2_g"], L["ln2_b"], eps=1e-5)
        x = x + F.gelu(h @ L["fw"] + L["fb"], approximate="tanh") @ L["mw"] + L["mb"]
    x = F.layer_norm(x, (768,), p["lnf_g"], p["lnf_b"], eps=1e-5)
    return x @ p["wte"].T


def main():
    prompt = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    enc = tiktoken.get_encoding("gpt2")
    p = load()
    ids = enc.encode(prompt)
    print("prompt_ids", ids)
    with torch.no_grad():
        first = forward(p, ids)[-1]
        top = torch.topk(first, 5)
        print("top5_ids", top.indices.tolist())
        print("top5_logits", " ".join(f"{v:.4f}" for v in top.values.tolist()))
        gen = []
        cur = list(ids)
        for _ in range(n):
            nxt = int(forward(p, cur)[-1].argmax())
            gen.append(nxt); cur.append(nxt)
    print("gerados", gen)
    print("texto", repr(enc.decode(cur)))


if __name__ == "__main__":
    main()
