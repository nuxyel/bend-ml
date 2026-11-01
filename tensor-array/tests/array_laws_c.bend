# tensor-array's three laws about Base's Array, used on balanced trees only, so the program runs on the C lane as well
# as on the JavaScript one. tests/array_laws.bend's third value is a hand-built unbalanced tree, which the compiled
# runtime refuses ("runtime fail-stop": it keeps an Array as one flat block, and the two halves of a node must have the
# same length); this file keeps every tree perfect, so both lanes must print the same four numbers.
import Base
import ../main.bend as TA

# get_after_set on an array of 8 F32s (index 13 wraps to 5)
def use_it(a: Array<F32>) -> {Pair.snd(Array<F32>, F32, Array.get(F32, Array.set(F32, a, 13, 1.5), 13)) == 1.5 : F32}:
  TA.get_after_set(F32, a, 13, 1.5)

# size_set: a write keeps the size
def use_size(a: Array<F32>) -> {TA.ASet.sz(F32, Array.set(F32, a, 13, 1.5)) == TA.ASet.sz(F32, a) : U32}:
  TA.size_set(F32, a, 13, 1.5)

# get_other_set on 8 cells built by Array.new's literal (13 wraps to 5, read at 2): Apart computes to Unit
def use_other() -> {Pair.snd(Array<F32>, F32, Array.get(F32, Array.set(F32, [0.0 : F32*8n], 13, 1.5), 2)) == Pair.snd(Array<F32>, F32, Array.get(F32, [0.0 : F32*8n], 2)) : F32}:
  TA.get_other_set(F32, [0.0 : F32*8n], 13, 2, 1.5, Unit{})

# get_other_set on a hand-built perfect tree of 4 leaves: a write at 2 leaves cell 3 alone
def use_other4() -> {Pair.snd(Array<F32>, F32, Array.get(F32, Array.set(F32, ANode{ANode{ALeaf{0.0}, ALeaf{1.0}}, ANode{ALeaf{2.0}, ALeaf{3.0}}}, 2, 9.0), 3)) == Pair.snd(Array<F32>, F32, Array.get(F32, ANode{ANode{ALeaf{0.0}, ALeaf{1.0}}, ANode{ALeaf{2.0}, ALeaf{3.0}}}, 3)) : F32}:
  TA.get_other_set(F32, ANode{ANode{ALeaf{0.0}, ALeaf{1.0}}, ANode{ALeaf{2.0}, ALeaf{3.0}}}, 2, 3, 9.0, Unit{})

def second(r: Array<F32> & F32) -> F32:
  (b, x) = r
  x

# 1.5 (get_after_set), 2.5 (get_other_set: the write at 13 leaves cell 2 alone), then on the hand-built perfect tree of
# 4 leaves: 3.0 (a write at 2 leaves cell 3 alone) and 9.0 (a read at 2 sees the write)
def main() -> F32 & F32 & F32 & F32:
  (second(Array.get(F32, Array.set(F32, [0.0 : F32*8n], 13, 1.5), 13)),
   second(Array.get(F32, Array.set(F32, Array.set(F32, [0.0 : F32*8n], 2, 2.5), 13, 1.5), 2)),
   second(Array.get(F32, Array.set(F32, ANode{ANode{ALeaf{0.0}, ALeaf{1.0}}, ANode{ALeaf{2.0}, ALeaf{3.0}}}, 2, 9.0), 3)),
   second(Array.get(F32, Array.set(F32, ANode{ANode{ALeaf{0.0}, ALeaf{1.0}}, ANode{ALeaf{2.0}, ALeaf{3.0}}}, 2, 9.0), 2)))
