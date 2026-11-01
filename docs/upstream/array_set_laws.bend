# Three laws about Base's Array, proved outside Base with Bend 2.0.39 (`bend array_set_laws.bend --verdict`):
# - size_set: Array.set keeps Array.size;
# - get_after_set: what Array.set writes at i, Array.get reads back at i;
# - get_other_set: a write at i leaves alone the cell that a read at k reaches, when the two descents part.
# The first two hold for every i (the two descents wrap i the same way) and for every tree Base can build,
# balanced or not, since a write replaces one leaf and leaves the tree's shape alone. "The size" is the
# second component of the pair Array.size returns, read with Base's Pair.snd (sz below); the value is
# Pair.snd of the pair Array.get returns.
#
# The third law's hypothesis says that the descent Array.set makes for i and the descent Array.get makes
# for k take different branches at some node before either reaches a leaf (Apart below). It is not "i and
# k differ after the mask", because that is false on an unbalanced tree: Array.size follows the leftmost
# path, so in ANode{ANode{ALeaf, ALeaf}, ALeaf} (size 4) the indices 2 and 3 differ but both reach the
# last leaf, and a write at 2 changes what a read at 3 returns (main prints it: 9.0). On a perfect tree of
# 2^d leaves, which Array.new and literals build, the descents part exactly when the masked indices differ;
# I have not proved that equivalence in Bend, since it needs lemmas about U32 bits. For a concrete tree
# Apart computes to Unit, so Unit{} proves it (use_other, and use_other2 with a subtree left abstract).
#
# Why the proofs are shaped this way: an Array is affine, and so is it inside a proof, so a proof may use a
# given array once. The obvious lemmas each need the same subtree again (one to say Array.size of it is
# (it, its size), one for the induction step, one to open the pair the descent returns), so they cannot
# be combined. Instead, one pass over the tree (facts) proves all six facts the step needs at once; each
# fact is an equation, and equations can be reused (+). A second pass (copy) hands back the same array
# and its size as a plain value, which the top-level proof needs to pick the first branch. The third law
# uses one pass too (both): it proves the six facts and four more about the read along k, and on the half
# the write does not enter it runs a smaller pass (side).
#
# The trailing Unit in Facts, Copy, Side and Reads works around bendlang/bend#1077: a + binder on the last
# field of a nested pair fails to check, so no fact sits in the last field.

import Base

# sz(T, a): the size of a as a plain U32, the second component of the pair Array.size returns
def sz(-T: Data, a: Array<T>) -> U32:
  Pair.snd(Array<T>, U32, Array.size(T, a))

# the three results the descents build: the swapped pair, the written array, and the read pair
def G(-T: Data, a: Array<T>, +n: U32, +j: U32, +v: T, +z: Bool) -> Array<T> & T:
  Array.swap.go(T, a, n, j, v, z)

def NA(-T: Data, a: Array<T>, +n: U32, +j: U32, +v: T, +z: Bool) -> Array<T>:
  Array.set.fin(T, G(T, a, n, j, v, z))

def K(-T: Data, a: Array<T>, +n: U32, +j: U32, +v: T, +z: Bool) -> Array<T> & T:
  Array.get.go(T, NA(T, a, n, j, v, z), n, j, z)

# Facts: what one pass over the tree a proves, for any n, j, v, z (n need not be a's real size):
# 1. Array.size(a) is the pair (a, sz(a));
# 2. the pair the write's descent returns is (the written array, the old value);
# 3. Array.size of the written array is the pair (written array, its sz);
# 4. the written array has a's sz;
# 5. reading the written array along the same path gives v;
# 6. the pair the read returns is (its first component, its second).
def Facts(-T: Data, -a: Array<T>, -n: U32, -j: U32, -v: T, -z: Bool) -> Type:
  {(a, sz(T, a)) == Array.size(T, a) : Array<T> & U32} & {(NA(T, a, n, j, v, z), Pair.snd(Array<T>, T, G(T, a, n, j, v, z))) == G(T, a, n, j, v, z) : Array<T> & T} & {(NA(T, a, n, j, v, z), sz(T, NA(T, a, n, j, v, z))) == Array.size(T, NA(T, a, n, j, v, z)) : Array<T> & U32} & {sz(T, NA(T, a, n, j, v, z)) == sz(T, a) : U32} & {Pair.snd(Array<T>, T, K(T, a, n, j, v, z)) == v : T} & {(Pair.fst(Array<T>, T, K(T, a, n, j, v, z)), Pair.snd(Array<T>, T, K(T, a, n, j, v, z))) == K(T, a, n, j, v, z) : Array<T> & T} & Unit

# Array.size hands back the array it measured, next to sz(T, a)
def size_pair(-T: Data, a: Array<T>) -> {(a, sz(T, a)) == Array.size(T, a) : Array<T> & U32}:
  match a:
    case ALeaf{x}:
      {==}
    case ANode{xs, ys}:
      %size_pair(T, xs) : {(ANode{xs, ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}
      {==}

def node_t(-T: Data, -xs: Array<T>, -ys: Array<T>, +n: U32, +j: U32, +v: T, f: Facts(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))))) -> Facts(T, ANode{xs, ys}, n, j, v, True{}):
  match f:
    case Tuple{+c1, r1}:
      match r1:
        case Tuple{+c2, r2}:
          match r2:
            case Tuple{+c3, r3}:
              match r3:
                case Tuple{+c4, r4}:
                  match r4:
                    case Tuple{+c5, r5}:
                      match r5:
                        case Tuple{+c6, u}:
                          ({(%c1 : {(ANode{xs, ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}; {==}) : {(ANode{xs, ys}, sz(T, ANode{xs, ys})) == Array.size(T, ANode{xs, ys}) : Array<T> & U32}}, {(%c2 : {(Array.set.fin(T, Array.swap.lo(T, ys, _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _))) == Array.swap.lo(T, ys, _) : Array<T> & T}; {==}) : {(NA(T, ANode{xs, ys}, n, j, v, True{}), Pair.snd(Array<T>, T, G(T, ANode{xs, ys}, n, j, v, True{}))) == G(T, ANode{xs, ys}, n, j, v, True{}) : Array<T> & T}}, {(%c2 : {(Array.set.fin(T, Array.swap.lo(T, ys, _)), sz(T, Array.set.fin(T, Array.swap.lo(T, ys, _)))) == Array.size(T, Array.set.fin(T, Array.swap.lo(T, ys, _))) : Array<T> & U32}; %c3 : {(ANode{Array.set.fin(T, G(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))))), ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}; {==}) : {(NA(T, ANode{xs, ys}, n, j, v, True{}), sz(T, NA(T, ANode{xs, ys}, n, j, v, True{}))) == Array.size(T, NA(T, ANode{xs, ys}, n, j, v, True{})) : Array<T> & U32}}, {(%c2 : {sz(T, Array.set.fin(T, Array.swap.lo(T, ys, _))) == sz(T, ANode{xs, ys}) : U32}; %c3 : {Pair.snd(Array<T>, U32, Array.size.node(T, ys, _)) == sz(T, ANode{xs, ys}) : U32}; %c1 : {U32.shl(sz(T, Array.set.fin(T, G(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))))))) == Pair.snd(Array<T>, U32, Array.size.node(T, ys, _)) : U32}; %c4 : {U32.shl(sz(T, Array.set.fin(T, G(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))))))) == U32.shl(_) : U32}; {==}) : {sz(T, NA(T, ANode{xs, ys}, n, j, v, True{})) == sz(T, ANode{xs, ys}) : U32}}, {(%c2 : {Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, j, True{})) == v : T}; %c6 : {Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _)) == v : T}; c5) : {Pair.snd(Array<T>, T, K(T, ANode{xs, ys}, n, j, v, True{})) == v : T}}, {(%c2 : {(Pair.fst(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, j, True{})), Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, j, True{}))) == Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, j, True{}) : Array<T> & T}; %c6 : {(Pair.fst(Array<T>, T, Array.swap.lo(T, ys, _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _))) == Array.swap.lo(T, ys, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, K(T, ANode{xs, ys}, n, j, v, True{})), Pair.snd(Array<T>, T, K(T, ANode{xs, ys}, n, j, v, True{}))) == K(T, ANode{xs, ys}, n, j, v, True{}) : Array<T> & T}}, Unit{})

def node_f(-T: Data, -xs: Array<T>, -ys: Array<T>, +n: U32, +j: U32, +v: T, +s: {(xs, sz(T, xs)) == Array.size(T, xs) : Array<T> & U32}, f: Facts(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))))) -> Facts(T, ANode{xs, ys}, n, j, v, False{}):
  match f:
    case Tuple{+c1, r1}:
      match r1:
        case Tuple{+c2, r2}:
          match r2:
            case Tuple{+c3, r3}:
              match r3:
                case Tuple{+c4, r4}:
                  match r4:
                    case Tuple{+c5, r5}:
                      match r5:
                        case Tuple{+c6, u}:
                          ({(%s : {(ANode{xs, ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}; {==}) : {(ANode{xs, ys}, sz(T, ANode{xs, ys})) == Array.size(T, ANode{xs, ys}) : Array<T> & U32}}, {(%c2 : {(Array.set.fin(T, Array.swap.hi(T, xs, _)), Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _))) == Array.swap.hi(T, xs, _) : Array<T> & T}; {==}) : {(NA(T, ANode{xs, ys}, n, j, v, False{}), Pair.snd(Array<T>, T, G(T, ANode{xs, ys}, n, j, v, False{}))) == G(T, ANode{xs, ys}, n, j, v, False{}) : Array<T> & T}}, {(%c2 : {(Array.set.fin(T, Array.swap.hi(T, xs, _)), sz(T, Array.set.fin(T, Array.swap.hi(T, xs, _)))) == Array.size(T, Array.set.fin(T, Array.swap.hi(T, xs, _))) : Array<T> & U32}; %s : {(ANode{xs, Array.set.fin(T, G(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n)))))}, Pair.snd(Array<T>, U32, Array.size.node(T, Array.set.fin(T, G(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))))), _))) == Array.size.node(T, Array.set.fin(T, G(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))))), _) : Array<T> & U32}; {==}) : {(NA(T, ANode{xs, ys}, n, j, v, False{}), sz(T, NA(T, ANode{xs, ys}, n, j, v, False{}))) == Array.size(T, NA(T, ANode{xs, ys}, n, j, v, False{})) : Array<T> & U32}}, {(%c2 : {sz(T, Array.set.fin(T, Array.swap.hi(T, xs, _))) == sz(T, ANode{xs, ys}) : U32}; %s : {Pair.snd(Array<T>, U32, Array.size.node(T, Array.set.fin(T, G(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))))), _)) == Pair.snd(Array<T>, U32, Array.size.node(T, ys, _)) : U32}; {==}) : {sz(T, NA(T, ANode{xs, ys}, n, j, v, False{})) == sz(T, ANode{xs, ys}) : U32}}, {(%c2 : {Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, j, False{})) == v : T}; %c6 : {Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _)) == v : T}; c5) : {Pair.snd(Array<T>, T, K(T, ANode{xs, ys}, n, j, v, False{})) == v : T}}, {(%c2 : {(Pair.fst(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, j, False{})), Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, j, False{}))) == Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, j, False{}) : Array<T> & T}; %c6 : {(Pair.fst(Array<T>, T, Array.swap.hi(T, xs, _)), Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _))) == Array.swap.hi(T, xs, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, K(T, ANode{xs, ys}, n, j, v, False{})), Pair.snd(Array<T>, T, K(T, ANode{xs, ys}, n, j, v, False{}))) == K(T, ANode{xs, ys}, n, j, v, False{}) : Array<T> & T}}, Unit{})

def facts(-T: Data, a: Array<T>, +n: U32, +j: U32, +v: T, +z: Bool) -> Facts(T, a, n, j, v, z):
  match a z:
    case ALeaf{x} _:
      ({==}, {==}, {==}, {==}, {==}, {==}, Unit{})
    case ANode{xs, ys} True{}:
      node_t(T, xs, ys, n, j, v, facts(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n)))))
    case ANode{xs, ys} False{}:
      node_f(T, xs, ys, n, j, v, size_pair(T, xs), facts(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n)))))

# copy(T, a): one pass over a hands back the same array b, its size n as a value, and the facts
# that tie them to a: b is a, n is sz(T, b), and Array.size(b) is the pair (b, sz(T, b))
def Copy(-T: Data, -a: Array<T>) -> Type:
  &b: Array<T> -> &n: U32 -> {b == a : Array<T>} & {n == sz(T, b) : U32} & {(b, sz(T, b)) == Array.size(T, b) : Array<T> & U32} & Unit

def copy_node(-T: Data, -xs: Array<T>, ys: Array<T>, r: Copy(T, xs)) -> Copy(T, ANode{xs, ys}):
  match r:
    case Tuple{b, r1}:
      match r1:
        case Tuple{+m, r2}:
          match r2:
            case Tuple{+p, r3}:
              match r3:
                case Tuple{+q, r4}:
                  match r4:
                    case Tuple{+s, u}:
                      (ANode{b, ys}, U32.shl(m),
                        {(%p : {ANode{b, ys} == ANode{_, ys} : Array<T>}; {==}) : {ANode{b, ys} == ANode{xs, ys} : Array<T>}},
                        {(%s : {U32.shl(m) == Pair.snd(Array<T>, U32, Array.size.node(T, ys, _)) : U32}; %q : {U32.shl(m) == U32.shl(_) : U32}; {==}) : {U32.shl(m) == sz(T, ANode{b, ys}) : U32}},
                        {(%s : {(ANode{b, ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}; {==}) : {(ANode{b, ys}, sz(T, ANode{b, ys})) == Array.size(T, ANode{b, ys}) : Array<T> & U32}},
                        Unit{})

def copy(-T: Data, a: Array<T>) -> Copy(T, a):
  match a:
    case ALeaf{x}:
      (ALeaf{x}, 1, {==}, {==}, {==}, Unit{})
    case ANode{xs, ys}:
      copy_node(T, xs, ys, copy(T, xs))

# the masked index and the first branch, as Array.swap.at and Array.get.at compute them from a size n
def J(+i: U32, +n: U32) -> U32:
  U32.and(i, U32.sub(n, 1))

def Z(+i: U32, +n: U32) -> Bool:
  U32.is_lt(J(i, n), U32.shr(n))

def close_size(-T: Data, -b: Array<T>, +n: U32, +i: U32, +v: T, +q: {n == sz(T, b) : U32}, f: Facts(T, b, n, J(i, n), v, Z(i, n))) -> {sz(T, Array.set(T, b, i, v)) == sz(T, b) : U32}:
  match f:
    case Tuple{+c1, r1}:
      match r1:
        case Tuple{+c2, r2}:
          match r2:
            case Tuple{+c3, r3}:
              match r3:
                case Tuple{+c4, r4}:
                  %c1 : {sz(T, Array.set.fin(T, Array.swap.at(T, i, v, _))) == sz(T, b) : U32}
                  %q : {sz(T, Array.set.fin(T, Array.swap.go(T, b, _, J(i, _), v, Z(i, _)))) == _ : U32}
                  Equal.trans(U32, sz(T, NA(T, b, n, J(i, n), v, Z(i, n))), sz(T, b), n, c4, Equal.sym(U32, n, sz(T, b), q))

def close_get(-T: Data, -b: Array<T>, +n: U32, +i: U32, +v: T, +q: {n == sz(T, b) : U32}, f: Facts(T, b, n, J(i, n), v, Z(i, n))) -> {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, b, i, v), i)) == v : T}:
  match f:
    case Tuple{+c1, r1}:
      match r1:
        case Tuple{+c2, r2}:
          match r2:
            case Tuple{+c3, r3}:
              match r3:
                case Tuple{+c4, r4}:
                  match r4:
                    case Tuple{+c5, r5}:
                      %c1 : {Pair.snd(Array<T>, T, Array.get.at(T, i, Array.size(T, Array.set.fin(T, Array.swap.at(T, i, v, _))))) == v : T}
                      %q : {Pair.snd(Array<T>, T, Array.get.at(T, i, Array.size(T, Array.set.fin(T, Array.swap.go(T, b, _, J(i, _), v, Z(i, _)))))) == v : T}
                      %c3 : {Pair.snd(Array<T>, T, Array.get.at(T, i, _)) == v : T}
                      %Equal.trans(U32, n, sz(T, b), sz(T, NA(T, b, n, J(i, n), v, Z(i, n))), q, Equal.sym(U32, sz(T, NA(T, b, n, J(i, n), v, Z(i, n))), sz(T, b), c4)) : {Pair.snd(Array<T>, T, Array.get.go(T, NA(T, b, n, J(i, n), v, Z(i, n)), _, J(i, _), Z(i, _))) == v : T}
                      c5

def top_size(-T: Data, -a: Array<T>, +i: U32, +v: T, r: Copy(T, a)) -> {sz(T, Array.set(T, a, i, v)) == sz(T, a) : U32}:
  match r:
    case Tuple{b, r1}:
      match r1:
        case Tuple{+n, r2}:
          match r2:
            case Tuple{+p, r3}:
              match r3:
                case Tuple{+q, r4}:
                  %p : {sz(T, Array.set(T, _, i, v)) == sz(T, _) : U32}
                  close_size(T, b, n, i, v, q, facts(T, b, n, J(i, n), v, Z(i, n)))

def top_get(-T: Data, -a: Array<T>, +i: U32, +v: T, r: Copy(T, a)) -> {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, a, i, v), i)) == v : T}:
  match r:
    case Tuple{b, r1}:
      match r1:
        case Tuple{+n, r2}:
          match r2:
            case Tuple{+p, r3}:
              match r3:
                case Tuple{+q, r4}:
                  %p : {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, _, i, v), i)) == v : T}
                  close_get(T, b, n, i, v, q, facts(T, b, n, J(i, n), v, Z(i, n)))

# LAW: writing v at i keeps the array's size (any i: the index wraps around)
law size_set:
  for -T: Data
  for a: Array<T>
  for +i: U32
  for +v: T
  {sz(T, Array.set(T, a, i, v)) == sz(T, a) : U32}

def size_set(T, a, i, v):
  top_size(T, a, i, v, copy(T, a))

# LAW: what Array.set writes at i, Array.get reads at i (any i: both wrap it the same way)
law get_after_set:
  for -T: Data
  for a: Array<T>
  for +i: U32
  for +v: T
  {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, a, i, v), i)) == v : T}

def get_after_set(T, a, i, v):
  top_get(T, a, i, v, copy(T, a))

# Apart(T, a, n, j, z, k, w): the write's descent (index j, first branch z) and the read's descent (k, w),
# both started from the size n, take different branches at some node of a before either reaches a leaf.
# It is Empty when both reach the same leaf, and Unit at the node where they split.
def Apart(-T: Data, a: Array<T>, +n: U32, +j: U32, +z: Bool, +k: U32, +w: Bool) -> Type:
  match a z w:
    case ALeaf{x} _ _:
      Empty
    case ANode{xs, ys} True{} True{}:
      Apart(T, xs, U32.shr(n), j, U32.is_lt(j, U32.shr(U32.shr(n))), k, U32.is_lt(k, U32.shr(U32.shr(n))))
    case ANode{xs, ys} False{} False{}:
      Apart(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))), U32.sub(k, U32.shr(n)), U32.is_lt(U32.sub(k, U32.shr(n)), U32.shr(U32.shr(n))))
    case ANode{xs, ys} True{} False{}:
      Unit
    case ANode{xs, ys} False{} True{}:
      Unit

# Side: what a pass over a subtree the write does not enter proves: Array.size(a) is (a, sz(a)), and the
# pair a read along (n, k, w) returns is (its first component, its second)
def Side(-T: Data, -a: Array<T>, -n: U32, -k: U32, -w: Bool) -> Type:
  {(a, sz(T, a)) == Array.size(T, a) : Array<T> & U32} & {(Pair.fst(Array<T>, T, Array.get.go(T, a, n, k, w)), Pair.snd(Array<T>, T, Array.get.go(T, a, n, k, w))) == Array.get.go(T, a, n, k, w) : Array<T> & T} & Unit

def side_node(-T: Data, -xs: Array<T>, -ys: Array<T>, +n: U32, +k: U32, +w: Bool, sx: Side(T, xs, U32.shr(n), k, U32.is_lt(k, U32.shr(U32.shr(n)))), sy: Side(T, ys, U32.shr(n), U32.sub(k, U32.shr(n)), U32.is_lt(U32.sub(k, U32.shr(n)), U32.shr(U32.shr(n))))) -> Side(T, ANode{xs, ys}, n, k, w):
  match w:
    case True{}:
      match sx:
        case Tuple{+s1, r1}:
          match r1:
            case Tuple{+g1, u1}:
              match sy:
                case Tuple{+s2, r2}:
                  match r2:
                    case Tuple{+g2, u2}:
                      ({(%s1 : {(ANode{xs, ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}; {==}) : {(ANode{xs, ys}, sz(T, ANode{xs, ys})) == Array.size(T, ANode{xs, ys}) : Array<T> & U32}}, {(%g1 : {(Pair.fst(Array<T>, T, Array.swap.lo(T, ys, _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _))) == Array.swap.lo(T, ys, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, True{})), Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, True{}))) == Array.get.go(T, ANode{xs, ys}, n, k, True{}) : Array<T> & T}}, Unit{})
    case False{}:
      match sx:
        case Tuple{+s1, r1}:
          match r1:
            case Tuple{+g1, u1}:
              match sy:
                case Tuple{+s2, r2}:
                  match r2:
                    case Tuple{+g2, u2}:
                      ({(%s1 : {(ANode{xs, ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}; {==}) : {(ANode{xs, ys}, sz(T, ANode{xs, ys})) == Array.size(T, ANode{xs, ys}) : Array<T> & U32}}, {(%g2 : {(Pair.fst(Array<T>, T, Array.swap.hi(T, xs, _)), Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _))) == Array.swap.hi(T, xs, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, False{})), Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, False{}))) == Array.get.go(T, ANode{xs, ys}, n, k, False{}) : Array<T> & T}}, Unit{})

def side(-T: Data, a: Array<T>, +n: U32, +k: U32, +w: Bool) -> Side(T, a, n, k, w):
  match a:
    case ALeaf{x}:
      ({==}, {==}, Unit{})
    case ANode{xs, ys}:
      # bend-lint: disable-next bend-lint-rules/performance/missed_fork -- a proof by one pass over the tree not-expected-because: proofs are checked, never run
      side_node(T, xs, ys, n, k, w, side(T, xs, U32.shr(n), k, U32.is_lt(k, U32.shr(U32.shr(n)))), side(T, ys, U32.shr(n), U32.sub(k, U32.shr(n)), U32.is_lt(U32.sub(k, U32.shr(n)), U32.shr(U32.shr(n)))))

# Reads: what one pass over the tree a proves about a write along (n, j, v, z) and a read along (n, k, w):
# 1. the pair the write's descent returns is (the written array, the old value);
# 2. the read of a returns a pair (first component, second);
# 3. so does the read of the written array;
# 4. if the descents split, the read finds the same value in the written array as in a.
def Reads(-T: Data, -a: Array<T>, -n: U32, -j: U32, -v: T, -z: Bool, -k: U32, -w: Bool) -> Type:
  {(NA(T, a, n, j, v, z), Pair.snd(Array<T>, T, G(T, a, n, j, v, z))) == G(T, a, n, j, v, z) : Array<T> & T} & {(Pair.fst(Array<T>, T, Array.get.go(T, a, n, k, w)), Pair.snd(Array<T>, T, Array.get.go(T, a, n, k, w))) == Array.get.go(T, a, n, k, w) : Array<T> & T} & {(Pair.fst(Array<T>, T, Array.get.go(T, NA(T, a, n, j, v, z), n, k, w)), Pair.snd(Array<T>, T, Array.get.go(T, NA(T, a, n, j, v, z), n, k, w))) == Array.get.go(T, NA(T, a, n, j, v, z), n, k, w) : Array<T> & T} & (Apart(T, a, n, j, z, k, w) -> {Pair.snd(Array<T>, T, Array.get.go(T, NA(T, a, n, j, v, z), n, k, w)) == Pair.snd(Array<T>, T, Array.get.go(T, a, n, k, w)) : T}) & Unit

# the written low half WL and the written high half WH of a node of size n
def WL(-T: Data, xs: Array<T>, +n: U32, +j: U32, +v: T) -> Array<T>:
  NA(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))))

def WH(-T: Data, ys: Array<T>, +n: U32, +j: U32, +v: T) -> Array<T>:
  NA(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))))

# the read's descent into the low half and into the high half of a node of size n
def RL(-T: Data, xs: Array<T>, +n: U32, +k: U32) -> Array<T> & T:
  Array.get.go(T, xs, U32.shr(n), k, U32.is_lt(k, U32.shr(U32.shr(n))))

def RH(-T: Data, ys: Array<T>, +n: U32, +k: U32) -> Array<T> & T:
  Array.get.go(T, ys, U32.shr(n), U32.sub(k, U32.shr(n)), U32.is_lt(U32.sub(k, U32.shr(n)), U32.shr(U32.shr(n))))

def no_leaf(-T: Data, -x: T, -n: U32, -j: U32, -v: T, -z: Bool, -k: U32, -w: Bool, h: Apart(T, ALeaf{x}, n, j, z, k, w)) -> {Pair.snd(Array<T>, T, Array.get.go(T, NA(T, ALeaf{x}, n, j, v, z), n, k, w)) == Pair.snd(Array<T>, T, Array.get.go(T, ALeaf{x}, n, k, w)) : T}:
  match h:

# the write goes into the low half xs
def reads_t(-T: Data, -xs: Array<T>, -ys: Array<T>, +n: U32, +j: U32, +v: T, +k: U32, +w: Bool, +g: {(Pair.fst(Array<T>, T, RH(T, ys, n, k)), Pair.snd(Array<T>, T, RH(T, ys, n, k))) == RH(T, ys, n, k) : Array<T> & T}, r: Reads(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))), k, U32.is_lt(k, U32.shr(U32.shr(n))))) -> Reads(T, ANode{xs, ys}, n, j, v, True{}, k, w):
  match w:
    case True{}:
      match r:
        case Tuple{+d, r1}:
          match r1:
            case Tuple{+e1, r2}:
              match r2:
                case Tuple{+e2, r3}:
                  match r3:
                    case Tuple{e3, u}:
                      ({(%d : {(Array.set.fin(T, Array.swap.lo(T, ys, _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _))) == Array.swap.lo(T, ys, _) : Array<T> & T}; {==}) : {(NA(T, ANode{xs, ys}, n, j, v, True{}), Pair.snd(Array<T>, T, G(T, ANode{xs, ys}, n, j, v, True{}))) == G(T, ANode{xs, ys}, n, j, v, True{}) : Array<T> & T}},
                        {(%e1 : {(Pair.fst(Array<T>, T, Array.swap.lo(T, ys, _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _))) == Array.swap.lo(T, ys, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, True{})), Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, True{}))) == Array.get.go(T, ANode{xs, ys}, n, k, True{}) : Array<T> & T}},
                        {(%d : {(Pair.fst(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, k, True{})), Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, k, True{}))) == Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, k, True{}) : Array<T> & T}; %e2 : {(Pair.fst(Array<T>, T, Array.swap.lo(T, ys, _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _))) == Array.swap.lo(T, ys, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, NA(T, ANode{xs, ys}, n, j, v, True{}), n, k, True{})), Pair.snd(Array<T>, T, Array.get.go(T, NA(T, ANode{xs, ys}, n, j, v, True{}), n, k, True{}))) == Array.get.go(T, NA(T, ANode{xs, ys}, n, j, v, True{}), n, k, True{}) : Array<T> & T}},
                        (h => (%d : {Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, k, True{})) == Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, True{})) : T}; %e2 : {Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _)) == Pair.snd(Array<T>, T, Array.swap.lo(T, ys, RL(T, xs, n, k))) : T}; %e1 : {Pair.snd(Array<T>, T, RL(T, WL(T, xs, n, j, v), n, k)) == Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _)) : T}; e3(h))),
                        Unit{})
    case False{}:
      match r:
        case Tuple{+d, r1}:
          match r1:
            case Tuple{+e1, r2}:
              match r2:
                case Tuple{+e2, r3}:
                  match r3:
                    case Tuple{e3, u}:
                      ({(%d : {(Array.set.fin(T, Array.swap.lo(T, ys, _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _))) == Array.swap.lo(T, ys, _) : Array<T> & T}; {==}) : {(NA(T, ANode{xs, ys}, n, j, v, True{}), Pair.snd(Array<T>, T, G(T, ANode{xs, ys}, n, j, v, True{}))) == G(T, ANode{xs, ys}, n, j, v, True{}) : Array<T> & T}},
                        {(%g : {(Pair.fst(Array<T>, T, Array.swap.hi(T, xs, _)), Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _))) == Array.swap.hi(T, xs, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, False{})), Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, False{}))) == Array.get.go(T, ANode{xs, ys}, n, k, False{}) : Array<T> & T}},
                        {(%d : {(Pair.fst(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, k, False{})), Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, k, False{}))) == Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, k, False{}) : Array<T> & T}; %g : {(Pair.fst(Array<T>, T, Array.swap.hi(T, WL(T, xs, n, j, v), _)), Pair.snd(Array<T>, T, Array.swap.hi(T, WL(T, xs, n, j, v), _))) == Array.swap.hi(T, WL(T, xs, n, j, v), _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, NA(T, ANode{xs, ys}, n, j, v, True{}), n, k, False{})), Pair.snd(Array<T>, T, Array.get.go(T, NA(T, ANode{xs, ys}, n, j, v, True{}), n, k, False{}))) == Array.get.go(T, NA(T, ANode{xs, ys}, n, j, v, True{}), n, k, False{}) : Array<T> & T}},
                        (h => (%d : {Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, k, False{})) == Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, False{})) : T}; %g : {Pair.snd(Array<T>, T, Array.swap.hi(T, WL(T, xs, n, j, v), _)) == Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _)) : T}; {==})),
                        Unit{})

# the write goes into the high half ys
def reads_f(-T: Data, -xs: Array<T>, -ys: Array<T>, +n: U32, +j: U32, +v: T, +k: U32, +w: Bool, +g: {(Pair.fst(Array<T>, T, RL(T, xs, n, k)), Pair.snd(Array<T>, T, RL(T, xs, n, k))) == RL(T, xs, n, k) : Array<T> & T}, r: Reads(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))), U32.sub(k, U32.shr(n)), U32.is_lt(U32.sub(k, U32.shr(n)), U32.shr(U32.shr(n))))) -> Reads(T, ANode{xs, ys}, n, j, v, False{}, k, w):
  match w:
    case True{}:
      match r:
        case Tuple{+d, r1}:
          match r1:
            case Tuple{+e1, r2}:
              match r2:
                case Tuple{+e2, r3}:
                  match r3:
                    case Tuple{e3, u}:
                      ({(%d : {(Array.set.fin(T, Array.swap.hi(T, xs, _)), Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _))) == Array.swap.hi(T, xs, _) : Array<T> & T}; {==}) : {(NA(T, ANode{xs, ys}, n, j, v, False{}), Pair.snd(Array<T>, T, G(T, ANode{xs, ys}, n, j, v, False{}))) == G(T, ANode{xs, ys}, n, j, v, False{}) : Array<T> & T}},
                        {(%g : {(Pair.fst(Array<T>, T, Array.swap.lo(T, ys, _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _))) == Array.swap.lo(T, ys, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, True{})), Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, True{}))) == Array.get.go(T, ANode{xs, ys}, n, k, True{}) : Array<T> & T}},
                        {(%d : {(Pair.fst(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, k, True{})), Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, k, True{}))) == Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, k, True{}) : Array<T> & T}; %g : {(Pair.fst(Array<T>, T, Array.swap.lo(T, WH(T, ys, n, j, v), _)), Pair.snd(Array<T>, T, Array.swap.lo(T, WH(T, ys, n, j, v), _))) == Array.swap.lo(T, WH(T, ys, n, j, v), _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, NA(T, ANode{xs, ys}, n, j, v, False{}), n, k, True{})), Pair.snd(Array<T>, T, Array.get.go(T, NA(T, ANode{xs, ys}, n, j, v, False{}), n, k, True{}))) == Array.get.go(T, NA(T, ANode{xs, ys}, n, j, v, False{}), n, k, True{}) : Array<T> & T}},
                        (h => (%d : {Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, k, True{})) == Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, True{})) : T}; %g : {Pair.snd(Array<T>, T, Array.swap.lo(T, WH(T, ys, n, j, v), _)) == Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _)) : T}; {==})),
                        Unit{})
    case False{}:
      match r:
        case Tuple{+d, r1}:
          match r1:
            case Tuple{+e1, r2}:
              match r2:
                case Tuple{+e2, r3}:
                  match r3:
                    case Tuple{e3, u}:
                      ({(%d : {(Array.set.fin(T, Array.swap.hi(T, xs, _)), Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _))) == Array.swap.hi(T, xs, _) : Array<T> & T}; {==}) : {(NA(T, ANode{xs, ys}, n, j, v, False{}), Pair.snd(Array<T>, T, G(T, ANode{xs, ys}, n, j, v, False{}))) == G(T, ANode{xs, ys}, n, j, v, False{}) : Array<T> & T}},
                        {(%e1 : {(Pair.fst(Array<T>, T, Array.swap.hi(T, xs, _)), Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _))) == Array.swap.hi(T, xs, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, False{})), Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, False{}))) == Array.get.go(T, ANode{xs, ys}, n, k, False{}) : Array<T> & T}},
                        {(%d : {(Pair.fst(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, k, False{})), Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, k, False{}))) == Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, k, False{}) : Array<T> & T}; %e2 : {(Pair.fst(Array<T>, T, Array.swap.hi(T, xs, _)), Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _))) == Array.swap.hi(T, xs, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, Array.get.go(T, NA(T, ANode{xs, ys}, n, j, v, False{}), n, k, False{})), Pair.snd(Array<T>, T, Array.get.go(T, NA(T, ANode{xs, ys}, n, j, v, False{}), n, k, False{}))) == Array.get.go(T, NA(T, ANode{xs, ys}, n, j, v, False{}), n, k, False{}) : Array<T> & T}},
                        (h => (%d : {Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, k, False{})) == Pair.snd(Array<T>, T, Array.get.go(T, ANode{xs, ys}, n, k, False{})) : T}; %e2 : {Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _)) == Pair.snd(Array<T>, T, Array.swap.hi(T, xs, RH(T, ys, n, k))) : T}; %e1 : {Pair.snd(Array<T>, T, RH(T, WH(T, ys, n, j, v), n, k)) == Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _)) : T}; e3(h))),
                        Unit{})

# Both: the six facts of the first two laws and the four of the third, from one pass
def Both(-T: Data, -a: Array<T>, -n: U32, -j: U32, -v: T, -z: Bool, -k: U32, -w: Bool) -> Type:
  Facts(T, a, n, j, v, z) & Reads(T, a, n, j, v, z, k, w)

def both_t(-T: Data, -xs: Array<T>, -ys: Array<T>, +n: U32, +j: U32, +v: T, +k: U32, +w: Bool, sy: Side(T, ys, U32.shr(n), U32.sub(k, U32.shr(n)), U32.is_lt(U32.sub(k, U32.shr(n)), U32.shr(U32.shr(n)))), f: Both(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))), k, U32.is_lt(k, U32.shr(U32.shr(n))))) -> Both(T, ANode{xs, ys}, n, j, v, True{}, k, w):
  match sy:
    case Tuple{+s, r1}:
      match r1:
        case Tuple{+g, u}:
          match f:
            case Tuple{o, r}:
              (node_t(T, xs, ys, n, j, v, o), reads_t(T, xs, ys, n, j, v, k, w, g, r))

def both_f(-T: Data, -xs: Array<T>, -ys: Array<T>, +n: U32, +j: U32, +v: T, +k: U32, +w: Bool, sx: Side(T, xs, U32.shr(n), k, U32.is_lt(k, U32.shr(U32.shr(n)))), f: Both(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))), U32.sub(k, U32.shr(n)), U32.is_lt(U32.sub(k, U32.shr(n)), U32.shr(U32.shr(n))))) -> Both(T, ANode{xs, ys}, n, j, v, False{}, k, w):
  match sx:
    case Tuple{+s, r1}:
      match r1:
        case Tuple{+g, u}:
          match f:
            case Tuple{o, r}:
              (node_f(T, xs, ys, n, j, v, s, o), reads_f(T, xs, ys, n, j, v, k, w, g, r))

def both(-T: Data, a: Array<T>, +n: U32, +j: U32, +v: T, +z: Bool, +k: U32, +w: Bool) -> Both(T, a, n, j, v, z, k, w):
  match a z:
    case ALeaf{x} _:
      (({==}, {==}, {==}, {==}, {==}, {==}, Unit{}), ({==}, {==}, {==}, (h => no_leaf(T, x, n, j, v, z, k, w, h)), Unit{}))
    case ANode{xs, ys} True{}:
      both_t(T, xs, ys, n, j, v, k, w, side(T, ys, U32.shr(n), U32.sub(k, U32.shr(n)), U32.is_lt(U32.sub(k, U32.shr(n)), U32.shr(U32.shr(n)))), both(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))), k, U32.is_lt(k, U32.shr(U32.shr(n)))))
    case ANode{xs, ys} False{}:
      both_f(T, xs, ys, n, j, v, k, w, side(T, xs, U32.shr(n), k, U32.is_lt(k, U32.shr(U32.shr(n)))), both(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))), U32.sub(k, U32.shr(n)), U32.is_lt(U32.sub(k, U32.shr(n)), U32.shr(U32.shr(n)))))


def close_other(-T: Data, -b: Array<T>, +n: U32, +i: U32, +k: U32, +v: T, +q: {n == sz(T, b) : U32}, f: Both(T, b, n, J(i, n), v, Z(i, n), J(k, n), Z(k, n))) -> Apart(T, b, sz(T, b), J(i, sz(T, b)), Z(i, sz(T, b)), J(k, sz(T, b)), Z(k, sz(T, b))) -> {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, b, i, v), k)) == Pair.snd(Array<T>, T, Array.get(T, b, k)) : T}:
  match f:
    case Tuple{o, r}:
      match o:
        case Tuple{+c1, o1}:
          match o1:
            case Tuple{+c2, o2}:
              match o2:
                case Tuple{+c3, o3}:
                  match o3:
                    case Tuple{+c4, o4}:
                      match r:
                        case Tuple{+d, r1}:
                          match r1:
                            case Tuple{+e1, r2}:
                              match r2:
                                case Tuple{+e2, r3}:
                                  match r3:
                                    case Tuple{e3, u}:
                                      %c1 : (Apart(T, b, sz(T, b), J(i, sz(T, b)), Z(i, sz(T, b)), J(k, sz(T, b)), Z(k, sz(T, b))) -> {Pair.snd(Array<T>, T, Array.get.at(T, k, Array.size(T, Array.set.fin(T, Array.swap.at(T, i, v, _))))) == Pair.snd(Array<T>, T, Array.get.at(T, k, _)) : T})
                                      %q : (Apart(T, b, _, J(i, _), Z(i, _), J(k, _), Z(k, _)) -> {Pair.snd(Array<T>, T, Array.get.at(T, k, Array.size(T, Array.set.fin(T, Array.swap.go(T, b, _, J(i, _), v, Z(i, _)))))) == Pair.snd(Array<T>, T, Array.get.go(T, b, _, J(k, _), Z(k, _))) : T})
                                      %c3 : (Apart(T, b, n, J(i, n), Z(i, n), J(k, n), Z(k, n)) -> {Pair.snd(Array<T>, T, Array.get.at(T, k, _)) == Pair.snd(Array<T>, T, Array.get.go(T, b, n, J(k, n), Z(k, n))) : T})
                                      %Equal.trans(U32, n, sz(T, b), sz(T, NA(T, b, n, J(i, n), v, Z(i, n))), q, Equal.sym(U32, sz(T, NA(T, b, n, J(i, n), v, Z(i, n))), sz(T, b), c4)) : (Apart(T, b, n, J(i, n), Z(i, n), J(k, n), Z(k, n)) -> {Pair.snd(Array<T>, T, Array.get.go(T, NA(T, b, n, J(i, n), v, Z(i, n)), _, J(k, _), Z(k, _))) == Pair.snd(Array<T>, T, Array.get.go(T, b, n, J(k, n), Z(k, n))) : T})
                                      e3


def top_other(-T: Data, -a: Array<T>, +i: U32, +k: U32, +v: T, r: Copy(T, a)) -> Apart(T, a, sz(T, a), J(i, sz(T, a)), Z(i, sz(T, a)), J(k, sz(T, a)), Z(k, sz(T, a))) -> {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, a, i, v), k)) == Pair.snd(Array<T>, T, Array.get(T, a, k)) : T}:
  match r:
    case Tuple{b, r1}:
      match r1:
        case Tuple{+n, r2}:
          match r2:
            case Tuple{+p, r3}:
              match r3:
                case Tuple{+q, r4}:
                  %p : (Apart(T, _, sz(T, _), J(i, sz(T, _)), Z(i, sz(T, _)), J(k, sz(T, _)), Z(k, sz(T, _))) -> {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, _, i, v), k)) == Pair.snd(Array<T>, T, Array.get(T, _, k)) : T})
                  close_other(T, b, n, i, k, v, q, both(T, b, n, J(i, n), v, Z(i, n), J(k, n), Z(k, n)))

# LAW: a write at i leaves alone the cell a read at k reaches, when the two descents part before a leaf
# (any i and k: both wrap them; h is what makes them different cells, see the top of the file)
law get_other_set:
  for -T: Data
  for a: Array<T>
  for +i: U32
  for +k: U32
  for +v: T
  for h: Apart(T, a, sz(T, a), J(i, sz(T, a)), Z(i, sz(T, a)), J(k, sz(T, a)), Z(k, sz(T, a)))
  {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, a, i, v), k)) == Pair.snd(Array<T>, T, Array.get(T, a, k)) : T}

def get_other_set(T, a, i, k, v, h):
  top_other(T, a, i, k, v, copy(T, a))(h)

# the laws, used on an array of 8 F32s (index 13 wraps to 5)
def use_it(a: Array<F32>) -> {Pair.snd(Array<F32>, F32, Array.get(F32, Array.set(F32, a, 13, 1.5), 13)) == 1.5 : F32}:
  get_after_set(F32, a, 13, 1.5)

# get_other_set on the same 8 cells (13 wraps to 5, read at 2), and on a tree whose high half a is any
# array: there Apart computes to Unit, so Unit{} proves it
def use_other() -> {Pair.snd(Array<F32>, F32, Array.get(F32, Array.set(F32, [0.0 : F32*8n], 13, 1.5), 2)) == Pair.snd(Array<F32>, F32, Array.get(F32, [0.0 : F32*8n], 2)) : F32}:
  get_other_set(F32, [0.0 : F32*8n], 13, 2, 1.5, Unit{})

def use_other2(a: Array<F32>) -> {Pair.snd(Array<F32>, F32, Array.get(F32, Array.set(F32, ANode{ANode{ALeaf{0.0}, ALeaf{1.0}}, a}, 0, 1.5), 3)) == Pair.snd(Array<F32>, F32, Array.get(F32, ANode{ANode{ALeaf{0.0}, ALeaf{1.0}}, a}, 3)) : F32}:
  get_other_set(F32, ANode{ANode{ALeaf{0.0}, ALeaf{1.0}}, a}, 0, 3, 1.5, Unit{})

def second(r: Array<F32> & F32) -> F32:
  (b, x) = r
  x

# 1.5 (get_after_set), 2.5 (get_other_set: the write at 13 leaves cell 2 alone), and 9.0: on the
# unbalanced tree a write at 2 reaches the leaf a read at 3 finds, though 2 and 3 differ after the mask
def main() -> F32 & F32 & F32:
  (second(Array.get(F32, Array.set(F32, [0.0 : F32*8n], 13, 1.5), 13)),
   second(Array.get(F32, Array.set(F32, Array.set(F32, [0.0 : F32*8n], 2, 2.5), 13, 1.5), 2)),
   second(Array.get(F32, Array.set(F32, ANode{ANode{ALeaf{0.0}, ALeaf{1.0}}, ALeaf{2.0}}, 2, 9.0), 3)))
