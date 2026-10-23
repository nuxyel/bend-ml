# bend-ml-tensor-fork

The six products of [`bend-ml-tensor-array`](../tensor-array) (`matmul`, `matmul_nt`, `matmul_tn` and their accumulating
forms), with the same arguments, result types and bits, but with the parallel blocks **sharing** their operands instead
of copying them, and `par1`..`par4`, which run any kernel of yours over blocks of a range on up to four shared arrays.
It is faster wherever the work splits into blocks; it is **not proved**, and says so.

- Bend: **2.0.38** · License: MIT · Depends on `bend-ml-tensor-array@0.1.14.0`.
- `bend tensor-fork/main.bend` → `SOME PROOFS FAIL`, naming the 32 defs that rely on `@unsafe` code (below). All of it is
  Bend: no foreign code.
- Tests: `tests/bits.bend`, against tensor-array, bit for bit (below).

```python
import Base
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import bend-ml-tensor-fork@0.1.2.0/main.bend as TF

def prod(a: TA.Mat<100n, 784n>, w: TA.Mat<784n, 128n>) -> TA.MMul<100n, 784n, 128n>:
  TF.matmul(100n, 784n, 128n, 3n, a, w)   # instead of TA.Mat.matmul(100n, 784n, 128n, 3n, a, w)
```

Blocks of any kernel: `par2(~E, ~A, ~B, ~f, e, d, cnt, a, b)` calls `f(e, i0, k, B2{a, b})` on 2^d blocks `i0 .. i0+k-1` of
`0 .. cnt-1`, in a tree of parallel lets, every block holding a handle to `a` and to `b` (no copy); `e` is a reusable
record (`Data`) with whatever else `f` needs. `f` must write only inside its block, and no block may read what another
writes; then the arrays hold the same bits at every depth and thread count as `f` over the whole range.

```python
def scale_leaf(+s: F32, +i0: U32, +k: Nat, a: Array<F32>) -> Array<F32>:
  my_scale_loop(k, i0, s, a)              # a[i] = a[i] * s for i = i0 .. i0+k-1
TF.par1(~F32, ~F32, ~scale_leaf, 0.5, 4n, n, a)   # 16 blocks
```

## Why it is a separate package

An `Array` in Bend has one owner, which is what lets a product write into it in place, and what the proofs of
bend-ml's other packages rely on. Base's `Array.fork` gives two handles to one array, so two parallel blocks can read A
and B and write their own parts of C without a copy. Base marks it `@unsafe`: the checker cannot prove that two blocks
never write the same cell. Every def that reaches it is reported, and the package's check fails. So it lives here, next
to [`bend-ml-binary-read`](../binary-read) (the other package outside the proofs, for foreign code), and no package with
laws imports it. Use tensor-array when you want every product proved; use this one where the speed matters and the
bit tests are enough for you.

## How it works

A product that splits (tensor-array's `split_depth`, two or more rows) packs A and B once into cells of 8, as
tensor-array's sequential path does. Every node of the fork tree shares the two packed operands and C, and each leaf
runs tensor-array's own cells (`t8_rows`, `t8_cols`) straight into its rows (or columns) of C. tensor-array's parallel
path instead copies an operand at every node and hands each block's cells back as a list that is then written into C.
A product that does not split, or that has one row, runs tensor-array's sequential code. The sums are the same sums in
the same order, so the bits are tensor-array's at depth 0, at any depth and thread count.

## Speed

One epoch of bend-ml's MNIST demo (`demos/mnist/fast.bend`, SGD at lr 0.1) with its seven products taken from this
package instead of tensor-array, 5 alternated rounds, medians, Core Ultra 7 155H, idle (2026-10-16):

| | tensor-array 0.1.12.0 | tensor-fork 0.1.0.0 |
|---|---|---|
| 16 threads | 4.4 s | **3.0 s** |
| 1 thread (pinned to a P-core) | 12.0 s | **6.2 s** |

The same loss (0.52047706) and the same 9129 test hits in every run. One thread gains as much because the demo's
products split by their shape, not by the thread count: on one thread tensor-array still pays its copies and lists,
and this package pays nothing.

Per product, 16 threads, depth 4 against what tensor-array callers get: 1.7x at MNIST's 100 x 784 x 128, 2.5x and 2.8x
at its gradients, 1.5-1.7x at SmolLM2-sized products of 128 and 512 rows, 3.3x at a weight gradient over 128 tokens
(1536 x 128 by 128 x 576). Products too small to split are tensor-array's, unchanged. On one thread with no split, the
two are the same code.

0.1.1.0, inside a training step of one decoder block at SmolLM2-135M's sizes (512 rows; BendTorch's `demo/smol_block.bend`,
16 threads, 3 runs each): the packing in blocks and the output arrays sized without `cap_depth` took the step from
562 to 527 ms per 3 steps (6%); at 360M's sizes 1303 to 1238 ms (5%). One thread: the same.

Two leaves computing 2 x 2 tiles of C (two lines of A against two of B per pass over k, the same sums in the same order,
the same bits) were tried and not adopted: 11% slower on one P-core and 4-25% slower inside a 16-thread product. The
cells were not limited by reads; each dot is a chain of scalar F32 additions in a fixed order.

A matrix · vector over the same sharing was not adopted: it beat `Bands.matvec` in isolation at short rows but not
inside a whole generation (SmolLM2-135M 10% faster, 360M even, 1.7B 13.5% slower). The GPU (a bang over the same tree)
runs it with the same bits but 5-10x slower than the CPU on this machine.

## What is checked, and what is not

`tests/bits.bend` (`make check`, on the C lane at 1 and 16 threads; `make check-full` adds the JS lane):

- `gemm_any` against tensor-array's at depth 0 in every layout (nn, nt, tn) and both modes (C = A · B and C + A · B),
  on tensor-array's `parseq` grid (n, m = 1..9, k = 0, 1, 3, 8, 9, 17, depths 0..4) and six larger shapes at depths
  0..6: 14,832 products, 0 differing words in C, and 0 in A and B (the shared operands come back as they went in). The
  inputs mix in -0, infinities, NaN and a subnormal.
- The six public products at par 4 against tensor-array's at par 0 on 128 x 100 by 100 x 784: 0 differing words.
- `matmul`, `matmul_nt` and `matmul_tn_acc` at par 6 on 512 x 576 by 576 x 1536 (A and B big enough to pack in blocks):
  0 differing words.
- `par1`..`par4` with a kernel over blocks of `0 .. cnt-1` at depths 0..6 and cnt = 0, 1, 7, 100, 1000, against the same
  kernel in one loop: 0 differing words in every array, the ones only read included.
- Deliberate faults it catches: a right half one row short (5,282 differing words), a right half of columns one
  column late (60,557), a block of rows written at row 0 (22,740); a right block of `blocks2` starting at the left
  block's start (765,956 words in the products, and in `par2`), the packing's blocks written from cell 0 (766,974), a
  right block of `blocks4` one half short (141).

Not proved: that no leaf writes A or B, and that the leaves' blocks of C are disjoint. The blocks are the halves of the
fork tree, as in tensor-array's own split, and the test samples both. `make check` also pins the list of `@unsafe`
defs, so a new one cannot slip in unnoticed:

`blocks1`..`blocks4`, `join2`..`join4`, `par1`..`par4` (the blocks), `pk_par` (packing in blocks), `rpar`, `cpar`,
`t8_tree`, `t8_go`, `gemm_fork` (the product's fork tree), and `gemm_rows`, `gemm_any2`, `gemm_any`, `matmul`,
`matmul_acc`, `matmul_nt`, `matmul_nt_acc`, `matmul_tn`, `matmul_tn_acc` (what reaches it).

## Versions

- `0.1.0.0` (2026-10-16): the six products over the shared fork tree, on tensor-array 0.1.12.0.
- `0.1.1.0` (2026-10-17): `par1`..`par4`; A and B packed in blocks when they are big; output arrays sized without
  `TA.cap_depth`. The products' bits are unchanged.
- `0.1.1.1` (2026-10-18): on tensor-array 0.1.13.0 (its `cap_depth` compares natively), so that a program can take
  tensor-fork, tensor-gpu and tensor-array 0.1.13.0 together. The same code and bits.
- `0.1.2.0` (2026-10-18): `matmul_precise`, tensor-array 0.1.14.0's opt-in precise product (k in blocks of 1024)
  with this package's products a block: the same bits as tensor-array's (`tests/precise.bend`). On tensor-array 0.1.14.0;
  the other products unchanged.
