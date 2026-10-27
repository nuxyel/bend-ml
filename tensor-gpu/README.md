# bend-ml-tensor-gpu

The six products of [`bend-ml-tensor-array`](../tensor-array) (`matmul`, `matmul_nt`, `matmul_tn` and their accumulating
forms), with the same arguments, result types and **bits**, each run as one `!` call (a bang) so that it runs on the GPU
when there is one; and, since 0.1.1.0, bangs over any leaf code written in plain Bend (`run1`..`run4`), for the rest of
a training step (element-wise layers, norms, attention, the optimizer). It is **not proved**, and says so: like [`bend-ml-tensor-fork`](../tensor-fork), it shares arrays
between the leaves of a fork tree through Base's `@unsafe` `Array.fork`.

- Bend: **2.0.39** · License: MIT · Depends on `bend-ml-tensor-array@0.1.18.0`. Versions on BendHub: 0.1.0.0, the
  products alone (hash `0xb00297062e21c56b88008f0c5387c0a0`); 0.1.1.0, with `run1`..`run4` (`0x92c8fe8f8abfa0e0018d7f1c3ddd061c`), both on tensor-array 0.1.12.0; 0.1.1.1, the same code on tensor-array
  0.1.13.0; 0.1.2.0, `matmul_precise` (tensor-array 0.1.14.0's opt-in precise product, a bang a block, its slices copied on the
  host) on tensor-array 0.1.14.0; 0.1.2.1, the same product with every block's bang starting at its place in A and B
  (no copy, no pages pulled back): a SmolLM2-sized block step on the GPU 975 -> ~695 ms for 3 (2026-10-18); 0.1.3.0,
  tensor-array 0.1.15.0's order in 4 x 2 tiles (not recommended, see tensor-array's README, 0.1.16.0); 0.1.4.0, 0.1.2.1's
  code on tensor-array 0.1.16.0, with its bits (2026-10-25); 0.1.5.0, the random fills and the offset fills (below) on
  tensor-array 0.1.18.0, whose products are 0.1.16.0's, so the products keep their bits (hash
  `0x6538b6a56d11b5a4d65ef7d8b18c1f9f`, 2026-10-27).
- `bend tensor-gpu/main.bend` → `SOME PROOFS FAIL`, naming the 62 defs that rely on `@unsafe` code. All of it is Bend:
  no foreign code, no CUDA written by hand.
- Tests: `tests/bits.bend`, against tensor-array, bit for bit, on the GPU, on 1 and 16 CPU threads and on the JS lane;
  `tests/kernels.bend`, `run1`..`run4` and `run1_new`..`run4_new` against their leaf run once over every cell, on the
  GPU and on 1 and 16 CPU threads; `tests/fill.bend`, the random fills (below) against their cell functions, bit for
  bit, on every lane, with the offset fills against their cell functions and against slices of a full fill, their
  distance to libm and their moments; `reference/test_rng.py`, a NumPy float32 twin of the fills and of two offset
  slices, word for word.
- Written by [`scripts/gen_tensor_gpu.py`](../scripts/gen_tensor_gpu.py) (4 x 4 tiles): a tile carries 64 sums through
  its loop, which is tedious to write by hand.

```python
import Base
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import bend-ml-tensor-gpu@0.1.5.0/main.bend as TG

def prod(x: TA.Mat<256n, 576n>, w: TA.Mat<1536n, 576n>) -> TA.MMulNT<256n, 576n, 1536n>:
  TG.matmul_nt(256n, 576n, 1536n, 0n, x, w)   # instead of TA.Mat.matmul_nt(...); par is ignored
```

Build with CUDA to get the GPU: `CUDA_HOME=... bend x.bend -o x` writes `x` and `x.gpu`. Without CUDA, or with
`./x +RTS --gpu off -RTS`, the same bangs run on the CPU threads, with the same bits.

## How it works

Each product is one bang over a balanced fork tree. Leaf t computes a 4 x 4 tile of C (tiles in row-major order; leaves
past the last tile write nothing). Every node shares A, B and C (no copy), as upstream's one-block pattern does
(bendlang/bend `tests/run/stencil3d.bend`). A plain product makes its C inside the bang: made on the host, a new C often
reused a block whose pages were on the device, and zeroing it there pulled them back over PCIe (9.2 → 6.0 ms on a
256 x 576 x 1536 product).

The bits: tensor-array's dot product keeps four partial sums. Product p goes to sum p mod 4, in increasing p, each sum
starting at +0.0, and the four are combined at the end in an order that depends on k mod 4 (`TA.t8_comb`). Each leaf does
the same for each of its 16 cells, four steps of k a turn; the steps past k in the last turn add +0.0, which leaves a sum
unchanged, since a sum that starts at +0.0 is never -0.0. An accumulating product adds a cell to C once, with
tensor-array's own `cwr`. Bend builds the device code without fused multiply-add. So every cell has tensor-array's bits,
on any tile size and any number of leaves; `tests/bits.bend` checks 2,958 products (n, m = 1..9 with k = 0, 1, 3, 8, 9,
17, and bigger shapes, in the three layouts and both modes), the six public products and three big ones, with -0, ±inf,
NaN and a subnormal mixed into the inputs.

## Bangs over any leaf code (0.1.1.0)

`runK` and `runK_new` (K = 1..4 outputs) run one bang over 2^d leaves. Leaf t of l = 2^d gets t, l, a record `e` of
scalars (any `Data` type, the same for every leaf), four input arrays and its K outputs, and returns the outputs. The
leaf is a template (`~f`) written in plain Bend: it takes every array as an ordinary argument, reads it with `Array.get`
and may write the outputs, so the code that uses it needs no `@unsafe`; the sharing is inside the package, where every
node of the tree holds a handle to each array and joins the outputs.

```python
type EN is Data:
  EN{n: U32}

# out[i] = 2 * x[i] for the cells t, t + l, .. below n
def dbl(k: Nat, +i: U32, +l: U32, o: Array<F32>, r: Array<F32> & F32) -> Array<F32>:
  match k r:
    case 0n Tuple{a, +x}: o
    case 1n+q Tuple{a, +x}: dbl(q, (i + l : U32), l, Array.set(F32, o, i, (2.0 * x : F32)), Array.get(F32, a, (i + l : U32)))

def leaf(+t: U32, +l: U32, +e: EN, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: Array<F32>) -> Array<F32>:
  match e:
    case EN{+n}: dbl(TG.share(t, n, l), t, l, o, Array.get(F32, x1, t))

# x of n cells into a new array of 2^c cells; returns R1{x, the three unused inputs, the output}
def twice(+n: Nat, +c: Nat, x: Array<F32>) -> TG.R1:
  TG.run1_new(~EN, ~leaf, TG.depth(n), EN{U32.from_nat(n)}, x, TG.none(Unit{}), TG.none(Unit{}), TG.none(Unit{}), c)
```

- `runK(~E, ~f, d, e, x1, x2, x3, x4, o1, .., oK)` writes the outputs it is given, in place (an optimizer's weights and
  moments); `runK_new(~E, ~f, d, e, x1, x2, x3, x4, c1, .., cK)` makes them inside the bang with 2^c cells, since an
  array made on the host pulls device pages back. Both return `RK{x1, x2, x3, x4, outputs}` (`O2`..`O4` for several).
- `depth(n)`: a tree for n cells (2^14 leaves once there are that many). `share(t, n, l)`: how many of the cells t, t +
  l, .. lie below n, so a leaf takes its cells round-robin and neighbouring lanes read neighbouring cells. `none(Unit{})`:
  an array of one cell for an unused input. Compute sizes, depths and capacities on the host and pass them in.
- What is not checked, as for the products: that a leaf writes only its own cells (two leaves writing one cell race)
  and never an input, and that leaves past the work return their outputs unchanged. The leaves run in a run-time order
  inside each group of 128, so a leaf's result must not depend on another's.
- The bits are the leaf's own code, the same on the GPU as on the CPU threads, except where CUDA's math functions
  differ from the host's (an `exp` on the device takes the float overload; the known limit of [bendlang/bend#1305](https://github.com/bendlang/bend/issues/1305)).
- BendTorch's GPU layers are written over these.

## Random fills (0.1.5.0)

`fill_uniform(s, n, c)` and `fill_gauss(s, n, c)` make n uniforms or Gaussians of a counter s (a `U32`, such as the
training step) in a new array of 2^c cells, in one bang; `uniform_mat(s, r, m)` and `gauss_mat(s, r, m)` give them as an
r x m `Mat`. Cell i is a pure function of (s, i), so a lane computes its cells without reading anything, and the array
is made on the device.

- The uniform is mlbend-generative 0.1.0.0's: its mix32 over `s * 0x9E3779B9 + i + 0x1234567A`, the low 24 bits plus
  one over 2^24, in (0, 1]. `rng_uniform(s, i)` gives that package's words (the test checks 2^20 of them).
- A Gaussian is Box-Muller over the pair (2p, 2p + 1): cell 2p is r cos(2π u_b) and cell 2p + 1 is r sin(2π u_b), with
  r = sqrt(-2 ln u_a), u_a the uniform of index 2p and u_b of 2p + 1. A lane makes a pair, so each pair costs one ln,
  one sqrt and one sine and cosine.
- ln, sin and cos are written in Bend with `+ - * /` and the exponent's bits, not with `F32.log`/`sin`/`cos`, because
  CUDA's float functions give other bits than the host's. `+ - * /` and `sqrt` are IEEE's on every lane and Bend fuses
  no multiply-add, so the fills give the same words on the GPU, on 1 and 16 CPU threads and on the JS lane. They are
  close to libm, not equal: at most 2 ulp for ln, 1 for sin and cos, and 1.3e-6 between a Gaussian and Box-Muller with
  libm's functions over 2^18 cells (most of it from writing 2π as `6.2831855` there). A twin in NumPy's float32 that
  does the same operations in the same order gives the same words (`reference/test_rng.py`).
- The cell functions (`rng_uniform(s, i)`, `rng_gauss2(s, p)` for a pair, `rng_gauss(s, i)` for one cell) are plain
  Bend, so a leaf of your own (`runK`) can make the noise where it is used, for instance x_t = a x_0 + b ε in one bang,
  with no array of noise in between.
- Offset fills, for a batch made in slices: `fill_uniform_at(s, i0, n, c)` and `fill_gauss_at(s, i0, n, c)` make the
  cells i0 .. i0 + n - 1 of counter s, and `uniform_mat_at(s, i0, r, m)` and `gauss_mat_at(s, i0, r, m)` give them as
  an r x m `Mat`. Output cell k is `rng_uniform(s, i0 + k)` or `rng_gauss(s, i0 + k)`, so a slice has the same words
  as those cells of a full fill: rows b .. b + r - 1 of an R x m `gauss_mat(s, R, m)` are `gauss_mat_at(s, b * m, r, m)`.
  The Gaussians' pairs stay global, not per slice: with an odd i0 the first cell is the sine half of the pair
  (i0 - 1, i0), and when the last index i0 + n - 1 is even the last cell is the cosine half of its pair. A lane still
  makes a whole pair and writes the halves that fall in the slice. The plain fills are the case i0 = 0, with the same
  words as before. i0 is a `U32`, like the cell functions' index, and i0 + k wraps mod 2^32.
- Moments of 2^20 Gaussians of counter 1: mean 0.00093, variance 0.99982, and 0.6831, 0.9543 and 0.9973 of them inside
  1, 2 and 3 (the normal's 0.6827, 0.9545, 0.9973). This is a sanity check, not a test of the generator's quality.

```python
eps = TG.gauss_mat(step, 256n, 3072n)   # TA.Mat<256n, 3072n>, cell i = TG.rng_gauss(step, i)
part = TG.gauss_mat_at(step, 113664, 16n, 3072n)   # i0 = 37 * 3072: rows 37 .. 52 of eps, the same words
```

## Speed (first measurement)

RTX 4050 Laptop, driver 615.78.08, CUDA 12.9, against tensor-fork on 16 CPU threads (Core Ultra 7 155H), ms a product,
medians of 3 alternated rounds, 2026-10-18 (an idle machine; the final numbers come with bend-ml's next benchmark run):

| product (BendTorch's linear layers at SmolLM2-135M's sizes) | tensor-fork, CPU | tensor-gpu, GPU |
|---|---:|---:|
| `x·Wᵀ`, 256 x 576 → 1536 (`matmul_nt`) | 9.6 | 6.05 |
| `dy·W`, 256 x 1536 → 576 (`matmul`) | 10.0 | 4.05 |
| `dW += dyᵀ·x`, 1536 x 256 → 576 (`matmul_tn_acc`) | 8.0 | 2.85 |
| 512 x 4096 → 512 (`matmul`) | 36.6 | 8.2 |

What it costs and what is left:

- Keeping tensor-array's four partial sums costs a lot of registers: the device program uses 241 a lane with 4 x 4
  tiles (2 blocks of 128 lanes per SM). A kernel that keeps one sum a cell ran the 512 x 4096 x 512 product in 3.5 ms
  instead of 8.2, with different bits.
- Each bang has a fixed cost of about 0.8-1.4 ms (the runtime grows the tree on one group of lanes first), and arrays
  the host touched between bangs move over PCIe. Small products are faster on the CPU.
- The leaves' order on the lanes is dealt at run time inside each group of 128, so the loads are not fully coalesced
  (16 sectors a request for `matmul_nt`). Upstream issues are drafted with the measurements.
