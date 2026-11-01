"""The GPT-2 demo's weights as loaded, bit for bit: an FNV-1a hash of every weight against a pinned value.

demos/gpt2/fast.bend is built with its generation replaced by a hash of the loaded Model: every cell of every Array
(the whole capacity, so the padding too, and each Array's size), in band order, and every number of every list, as
their 32-bit patterns. It runs on 1 thread (one band per matrix) and on 16 (2^4 bands, 2^5 for wte), whose layouts
differ, so each has its own pinned hash. A change to the loader that keeps these hashes loads the same weights.

  reference/.venv/bin/python reference/test_gpt2_load.py [fast.bend]    # needs reference/gpt2_prep.py --split
"""
import os, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
# the loader of 2026-11-01 (devel 5a041a7), on the weights written by reference/gpt2_prep.py --split
PINNED = {1: 2905384170, 16: 138541988}

HELPERS = '''
# ---- added by reference/test_gpt2_load.py: an FNV-1a hash of every weight's bits ----
def xbits(f: F32) -> U32:
  match f:
    case F32{w}:
      U32{w}

def hmix(+h: U32, +w: U32) -> U32:
  (U32.xor(h, w) * 16777619 : U32)

def harr(n: Nat, +i: U32, +h: U32, r: Array<F32> & F32) -> U32:
  match n r:
    case 0n Tuple{a, x}:
      h
    case 1n+p Tuple{a, +x}:
      harr(p, (i + 1 : U32), hmix(h, xbits(x)), Array.get(F32, a, (i + 1 : U32)))

def hsz(+h: U32, r: Array<F32> & U32) -> U32:
  match r:
    case Tuple{a, +n}:
      harr(U32.to_nat(n), 0, hmix(h, n), Array.get(F32, a, 0))

def hmat(-r: Nat, -c: Nat, +h: U32, m: TA.Mat<r, c>) -> U32:
  match m:
    case TA.Mat{a}:
      hsz(h, Array.size(F32, a))

def hbands(-r: Nat, -c: Nat, b: TA.Bands<r, c>, +h: U32) -> U32:
  match b:
    case TA.BLeaf{m}:
      hmat(r, c, h, m)
    case TA.BNode{x, y}:
      hbands(Nat.sub(r, TA.half(r)), c, y, hbands(TA.half(r), c, x, h))

def hlist(xs: List<&2, F32>, +h: U32) -> U32:
  match xs:
    case Nil{}:
      hmix(h, 7)
    case Con{x, t}:
      hlist(t, hmix(h, xbits(x)))

def hlayer(+h: U32, l: Layer) -> U32:
  match l:
    case Layer{ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb}:
      hlist(mb, hbands(768n, 3072n, mw, hlist(fb, hbands(3072n, 768n, fw, hlist(ln2b, hlist(ln2g, hlist(pb, hbands(768n, 768n, pw, hlist(ab, hbands(2304n, 768n, aw, hlist(ln1b, hlist(ln1g, h))))))))))))

def hlayers(ls: List<&1, Layer>, +h: U32) -> U32:
  match ls:
    case Nil{}:
      h
    case Con{l, t}:
      hlayers(t, hlayer(h, l))

def hmodel(m: Model) -> U32:
  match m:
    case Model{wte, wpe, layers, lnfg, lnfb}:
      hlist(lnfb, hlist(lnfg, hlayers(layers, hmat(1024n, 768n, hbands(50257n, 768n, wte, 2166136261), wpe))))
'''


def sub(t, a, b):
    if t.count(a) != 1:
        sys.exit(f"test_gpt2_load: expected one {a!r} in the source")
    return t.replace(a, b)


def hash_source(t):
    """fast.bend's source with the generation replaced by a print of the loaded weights' hash."""
    marker = "# refuse to run on a malformed merges table"
    t = sub(t, marker, HELPERS + "\n" + marker)
    return sub(t, "        generate(model, mb, pb, prompt, n)\n", "        IO.print(\"hash \" ++ U32.show(hmodel(model)))\n")


def build(src_text, exe):
    """Builds a variant of fast.bend; it must sit in demos/gpt2 for its relative imports."""
    tmp = os.path.join(ROOT, "demos/gpt2", f"_t{os.getpid()}_{os.path.basename(exe)}.bend")
    try:
        open(tmp, "w").write(src_text)
        r = subprocess.run([BEND, tmp, "-o", exe], capture_output=True, text=True, cwd=ROOT, env=ENV)
        if r.returncode != 0 or not os.path.exists(exe):
            sys.exit("test_gpt2_load: build failed\n" + r.stdout + r.stderr)
    finally:
        os.remove(tmp)


def run_hash(exe, threads):
    out = subprocess.run([exe, "Hi", "1", "+RTS", "--threads", str(threads), "-RTS"], capture_output=True, text=True,
                         cwd=ROOT, env=ENV).stdout
    line = [l for l in out.splitlines() if l.startswith("hash ")]
    return int(line[0].split()[1]) if line else None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "demos/gpt2/fast.bend")
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "gpt2_hash")
        build(hash_source(open(path).read()), exe)
        bad = 0
        for th, want in PINNED.items():
            got = run_hash(exe, th)
            print(f"{th} thread(s): hash {got} (pinned {want}){'' if got == want else '  DIFFERENT'}")
            bad += got != want
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
