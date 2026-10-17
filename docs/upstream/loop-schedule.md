# Draft issue for bendlang/bend: the same dot-product loop compiles 10-20% apart depending on code around it

Status: draft, not posted. Reproduced on Bend 2.0.37, clang 22, Core Ultra 7 155H (2026-10-16).

---

**Title:** C backend: a memory-bound loop's speed depends on unrelated code (flipping `BLK_SHR` in a program that never forks makes it 10% faster)

## Summary

`matvec_schedule.bend` multiplies a 50,257 x 768 matrix by a vector, on one thread, through `bend-ml-tensor-array`
from BendHub (no fork anywhere, so the emitted C has `#define BLK_SHR 0`). If I take that C, change only that line to
`#define BLK_SHR 1`, and compile both with the same `clang -std=c11 -O3`, the second binary is 10-12% faster on the
same products, with the same result. It runs more instructions, not fewer: clang schedules the hot loop differently.

I'm reporting it because it looks like a performance knob that nobody turns on purpose: the speed of a memory-bound
loop depends on what else got compiled next to it. In a variant of the same program the gap was 21%, and in a larger
program (a Llama forward pass) removing an unrelated field moved one-thread time by 4-6% with the hot loop's
instructions unchanged, only shifted 16 bytes (aligning loops with `-falign-loops=64` closed most of that one).

## Repro

`matvec_schedule.bend` and `matvec_schedule.sh` (attached). The script emits the C, makes the flipped copy, builds both
and times them pinned to one P-core:

```
round 1 op 0  as built: ms=757 checksum=90.09511  BLK_SHR 1: ms=677 checksum=90.09511
round 1 op 1  as built: ms=781 checksum=90.09511  BLK_SHR 1: ms=696 checksum=90.09511
round 2 op 0  as built: ms=762 checksum=90.09511  BLK_SHR 1: ms=680 checksum=90.09511
round 2 op 1  as built: ms=791 checksum=90.09511  BLK_SHR 1: ms=696 checksum=90.09511
round 3 op 0  as built: ms=751 checksum=90.09511  BLK_SHR 1: ms=680 checksum=90.09511
round 3 op 1  as built: ms=794 checksum=90.09511  BLK_SHR 1: ms=706 checksum=90.09511
```

(op 0: `Mat.matmul_nt` with one row; op 1: `Bands.matvec`; 38 products each.)

`perf stat` (op 0, one thread):

| | as built | `BLK_SHR 1` |
|---|---|---|
| instructions | 19.25 G | 20.79 G |
| cycles | 4.13 G | 3.58 G |
| LLC load misses | 404 k | 29.5 k |

Both binaries spend their time in the same function, `spin_47` (the dot product with four partial sums). `perf
annotate`, instructions over 2% of the loop's samples:

```
as built                                         BLK_SHR 1
movss  (%r11,%rcx,4),%xmm3    4.77               movss  (%r10,%r15,4),%xmm2   10.08
mulss  (%rbx,%r10,4),%xmm3    3.24               mulss  (%r10,%r8,4),%xmm4    15.27
addss  %xmm6,%xmm3           41.75               addss  %xmm7,%xmm4            7.17
```

As built, the loop waits on one `addss` right after its load (42% of the samples); with the flag, both loads index
off one base register (the heap) and the wait is spread. Aligning loops to 64 bytes does not change this case. A
smaller program with the same loop shape, written with Base only, does not show the gap: it depends on the code
around the loop.

## What I'd expect

Not necessarily a fix in Bend: this is clang's scheduling of the C that Bend emits. But since the emitted C decides
it, maybe the backend can steer it, for instance by emitting a loop's array accesses from one hoisted base pointer
(the form the faster build ended up with), so a memory-bound loop's speed stops depending on unrelated code.

## Why it matters to me

I measure every change in bend-ml and llama.bend against the previous build. A 10-20% swing from code placement is the
size of the effects I'm trying to measure, so I now check every small gap with counters, the hot loop's disassembly
and a rebuilt binary before reporting it.
