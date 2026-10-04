"""Referência em Python do tokenizer BPE do bend-ml (mesmo algoritmo, mesmo desempate).

Tokens são inteiros: 0..255 são bytes; 256+k é o token criado pela regra k.
Uma tabela é uma lista de pares (a, b), da regra mais antiga para a mais nova.
"""


def merge(ids, pair, new):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train(data: bytes, n_merges: int):
    ids = list(data)
    table = []
    for k in range(n_merges):
        counts = {}  # dict mantém a ordem de primeira aparição
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        best = None
        for pair, c in counts.items():
            if best is None or c > counts[best]:  # estritamente maior: o primeiro vence empates
                best = pair
        table.append(best)
        ids = merge(ids, best, 256 + k)
    return table


def encode(table, data: bytes):
    ids = list(data)
    for k, pair in enumerate(table):
        ids = merge(ids, pair, 256 + k)
    return ids


def decode(table, ids):
    vocab = {i: bytes([i]) for i in range(256)}
    for k, (a, b) in enumerate(table):
        vocab[256 + k] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids)
