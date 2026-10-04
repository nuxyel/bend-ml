"""Python reference for the bend-ml BPE tokenizer (same algorithm, same tie-break).

Tokens are integers: 0..255 are bytes; 256+k is the token created by rule k.
A table is a list of pairs (a, b), from the oldest rule to the newest.
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
        counts = {}  # dict keeps the order of first appearance
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        best = None
        for pair, c in counts.items():
            if best is None or c > counts[best]:  # strictly greater: the first one wins ties
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
