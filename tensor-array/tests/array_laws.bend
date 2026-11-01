# tensor-array's three laws about Base's Array, used: get_after_set and get_other_set on concrete arrays, where
# ASet.Apart computes to Unit, and what main prints. Run it with `bend` (the JavaScript lane): the C runtime stops
# with "runtime fail-stop" on the hand-built unbalanced tree of main's third value.
import Base
import ../main.bend as TA

# the laws, used on an array of 8 F32s (index 13 wraps to 5)
def use_it(a: Array<F32>) -> {Pair.snd(Array<F32>, F32, Array.get(F32, Array.set(F32, a, 13, 1.5), 13)) == 1.5 : F32}:
  TA.get_after_set(F32, a, 13, 1.5)

# get_other_set on the same 8 cells (13 wraps to 5, read at 2), and on a tree whose high half a is any
# array: there Apart computes to Unit, so Unit{} proves it
def use_other() -> {Pair.snd(Array<F32>, F32, Array.get(F32, Array.set(F32, [0.0 : F32*8n], 13, 1.5), 2)) == Pair.snd(Array<F32>, F32, Array.get(F32, [0.0 : F32*8n], 2)) : F32}:
  TA.get_other_set(F32, [0.0 : F32*8n], 13, 2, 1.5, Unit{})

def use_other2(a: Array<F32>) -> {Pair.snd(Array<F32>, F32, Array.get(F32, Array.set(F32, ANode{ANode{ALeaf{0.0}, ALeaf{1.0}}, a}, 0, 1.5), 3)) == Pair.snd(Array<F32>, F32, Array.get(F32, ANode{ANode{ALeaf{0.0}, ALeaf{1.0}}, a}, 3)) : F32}:
  TA.get_other_set(F32, ANode{ANode{ALeaf{0.0}, ALeaf{1.0}}, a}, 0, 3, 1.5, Unit{})

def second(r: Array<F32> & F32) -> F32:
  (b, x) = r
  x

# 1.5 (get_after_set), 2.5 (get_other_set: the write at 13 leaves cell 2 alone), and 9.0: on the
# unbalanced tree a write at 2 reaches the leaf a read at 3 finds, though 2 and 3 differ after the mask
def main() -> F32 & F32 & F32:
  (second(Array.get(F32, Array.set(F32, [0.0 : F32*8n], 13, 1.5), 13)),
   second(Array.get(F32, Array.set(F32, Array.set(F32, [0.0 : F32*8n], 2, 2.5), 13, 1.5), 2)),
   second(Array.get(F32, Array.set(F32, ANode{ANode{ALeaf{0.0}, ALeaf{1.0}}, ALeaf{2.0}}, 2, 9.0), 3)))
