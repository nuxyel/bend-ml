"""MLP 784-128-10 on MNIST in PyTorch (CPU), twin of demos/mnist/train.bend.

Same initial weights (text files written by `make_init`), same batch order (no
shuffling), same learning rate, plain SGD, mean cross-entropy loss. Prints one line per
epoch: mean training loss, test accuracy, seconds.
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "demos/mnist/data")
INIT = os.path.join(ROOT, "demos/mnist/data/init")


def read_idx(path, images):
    raw = open(path, "rb").read()
    if images:
        n = int.from_bytes(raw[4:8], "big")
        return np.frombuffer(raw, np.uint8, offset=16).reshape(n, 784)
    return np.frombuffer(raw, np.uint8, offset=8)


def make_init(seed=0):
    os.makedirs(INIT, exist_ok=True)
    rng = np.random.default_rng(seed)
    def u(i, o):  # same range as PyTorch's default nn.Linear: U(-1/sqrt(i), 1/sqrt(i))
        k = 1.0 / np.sqrt(i)
        return rng.uniform(-k, k, (i, o)).astype(np.float32)
    parts = {"w1": u(784, 128), "b1": np.zeros(128, np.float32), "w2": u(128, 10), "b2": np.zeros(10, np.float32)}
    for k, v in parts.items():
        np.savetxt(os.path.join(INIT, k + ".txt"), v.reshape(1, -1), fmt="%.9g")
    return parts


def load_init():
    return {k: np.loadtxt(os.path.join(INIT, k + ".txt"), dtype=np.float32, ndmin=1).reshape(*shape)
            for k, shape in [("w1", (784, 128)), ("b1", (128,)), ("w2", (128, 10)), ("b2", (10,))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = the whole epoch")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--make-init", action="store_true")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.make_init or not os.path.exists(os.path.join(INIT, "w1.txt")):
        make_init()
    p = {k: torch.tensor(v, requires_grad=True) for k, v in load_init().items()}
    Xtr = torch.from_numpy(read_idx(f"{DATA}/train-images-idx3-ubyte", True).astype(np.float32) / 255.0)
    ytr = torch.from_numpy(read_idx(f"{DATA}/train-labels-idx1-ubyte", False).astype(np.int64))
    Xte = torch.from_numpy(read_idx(f"{DATA}/t10k-images-idx3-ubyte", True).astype(np.float32) / 255.0)
    yte = torch.from_numpy(read_idx(f"{DATA}/t10k-labels-idx1-ubyte", False).astype(np.int64))
    nb = len(Xtr) // a.bs
    if a.max_batches:
        nb = min(nb, a.max_batches)

    def fwd(x):
        return torch.relu(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]

    print(f"torch {torch.__version__} threads={torch.get_num_threads()} bs={a.bs} lr={a.lr} batches/epoch={nb}")
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); tot = 0.0
        for b in range(nb):
            x = Xtr[b * a.bs:(b + 1) * a.bs]; y = ytr[b * a.bs:(b + 1) * a.bs]
            loss = F.cross_entropy(fwd(x), y)
            for q in p.values(): q.grad = None
            loss.backward()
            with torch.no_grad():
                for q in p.values(): q -= a.lr * q.grad
            tot += loss.item()
        dt = time.time() - t0
        with torch.no_grad():
            acc = (fwd(Xte).argmax(1) == yte).float().mean().item()
        print(f"epoch {ep} train_loss={tot / nb:.6f} test_acc={acc:.4f} seconds={dt:.2f}")


if __name__ == "__main__":
    main()
