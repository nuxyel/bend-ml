# bend-ml-nat-lemmas: Nat and List lemmas, proved, that Bend's Base does not have.
#
#   import bend-ml-nat-lemmas@0.1.0.0/main.bend as NL
#   NL.add_comm(a, b)   NL.mul_assoc(a, b, c)   NL.product_append(xs, ys) ...
#
# Each `law` states a fact; the `def` right below it is the proof, checked by the kernel.
# How to read a proof: it is a function whose TYPE is the statement.
#   - `match` does case analysis (the number is 0, or it is 1 + p).
#   - A call to the function itself is the induction hypothesis ("I already know
#     it holds for p, so I prove it for 1 + p").
#   - `%e : P` rewrites the goal using the equality `e`.
#   - `{==}` closes the goal when both sides already compute to the same thing.
# A parameter carries `+` when the proof uses it more than once.

import Base

# Product of the elements of a list of Nat; the empty list is 1.
# It is the "number of elements" of a tensor whose shape is the list.
def product(xs: List<&2, Nat>) -> Nat:
  match xs:
    case Nil{}:
      1n
    case Con{h, t}:
      Nat.mul(h, product(t))


law add_zero:
  for a: Nat
  {Nat.add(a, 0n) == a : Nat}

# a + 0 == a
# Case 0: 0 + 0 computes to 0. Step: uses the hypothesis for p.
def add_zero(a):
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_zero(p) : {1n+Nat.add(p, 0n) == 1n+_ : Nat}
      {==}

law add_succ:
  for  a: Nat
  for -b: Nat
  {1n+Nat.add(a, b) == Nat.add(a, 1n+b) : Nat}

# 1 + (a + b) == a + (1 + b)
def add_succ(a, b):
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_succ(p, b) : {2n+Nat.add(p, b) == 1n+_ : Nat}
      {==}

law add_comm:
  for  a: Nat
  for +b: Nat
  {Nat.add(a, b) == Nat.add(b, a) : Nat}

# a + b == b + a
def add_comm(a, b):
  match a:
    case 0n:
      %add_zero(b) : {_ == Nat.add(b, 0n) : Nat}
      {==}
    case 1n+p:
      %add_succ(b, p) : {1n+Nat.add(p, b) == _ : Nat}
      %add_comm(p, b) : {1n+Nat.add(p, b) == 1n+_ : Nat}
      {==}

law add_assoc:
  for  a: Nat
  for -b: Nat
  for -c: Nat
  {Nat.add(a, Nat.add(b, c)) == Nat.add(Nat.add(a, b), c) : Nat}

# a + (b + c) == (a + b) + c
def add_assoc(a, b, c):
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_assoc(p, b, c) : {1n+Nat.add(p, Nat.add(b, c)) == 1n+_ : Nat}
      {==}

law mul_zero:
  for a: Nat
  {Nat.mul(a, 0n) == 0n : Nat}

# a * 0 == 0
def mul_zero(a):
  match a:
    case 0n:
      {==}
    case 1n+p:
      mul_zero(p)

# x + (y + z) == y + (x + z): swaps the middle terms of a triple sum.
# (Auxiliary lemma; not a published law.)
def add_swap(x: Nat, +y: Nat, +z: Nat) -> {Nat.add(x, Nat.add(y, z)) == Nat.add(y, Nat.add(x, z)) : Nat}:
  match x:
    case 0n:
      {==}
    case 1n+q:
      %add_succ(y, Nat.add(q, z)) : {1n+Nat.add(q, Nat.add(y, z)) == _ : Nat}
      %add_swap(q, y, z) : {1n+Nat.add(q, Nat.add(y, z)) == 1n+_ : Nat}
      {==}

law mul_succ:
  for +a: Nat
  for +b: Nat
  {Nat.mul(a, 1n+b) == Nat.add(a, Nat.mul(a, b)) : Nat}

# a * (1 + b) == a + a * b
def mul_succ(a, b):
  match a:
    case 0n:
      {==}
    case 1n++p:
      %Equal.sym(Nat, Nat.mul(p, 1n+b), Nat.add(p, Nat.mul(p, b)), mul_succ(p, b)) : {1n+Nat.add(b, _) == 1n+Nat.add(p, Nat.add(b, Nat.mul(p, b))) : Nat}
      %add_swap(b, p, Nat.mul(p, b)) : {1n+Nat.add(b, Nat.add(p, Nat.mul(p, b))) == 1n+_ : Nat}
      {==}

law mul_comm:
  for +a: Nat
  for +b: Nat
  {Nat.mul(a, b) == Nat.mul(b, a) : Nat}

# a * b == b * a
def mul_comm(a, b):
  match a:
    case 0n:
      %mul_zero(b) : {_ == Nat.mul(b, 0n) : Nat}
      {==}
    case 1n++p:
      %Equal.sym(Nat, Nat.mul(b, 1n+p), Nat.add(b, Nat.mul(b, p)), mul_succ(b, p)) : {Nat.add(b, Nat.mul(p, b)) == _ : Nat}
      %mul_comm(p, b) : {Nat.add(b, Nat.mul(p, b)) == Nat.add(b, _) : Nat}
      {==}

law mul_dist:
  for  a: Nat
  for -b: Nat
  for +c: Nat
  {Nat.add(Nat.mul(a, c), Nat.mul(b, c)) == Nat.mul(Nat.add(a, b), c) : Nat}

# a * c + b * c == (a + b) * c
def mul_dist(a, b, c):
  match a:
    case 0n:
      {==}
    case 1n+p:
      %mul_dist(p, b, c) : {Nat.add(Nat.add(c, Nat.mul(p, c)), Nat.mul(b, c)) == Nat.add(c, _) : Nat}
      %add_assoc(c, Nat.mul(p, c), Nat.mul(b, c)) : {_ == Nat.add(c, Nat.add(Nat.mul(p, c), Nat.mul(b, c))) : Nat}
      {==}

law mul_assoc:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  {Nat.mul(a, Nat.mul(b, c)) == Nat.mul(Nat.mul(a, b), c) : Nat}

# a * (b * c) == (a * b) * c
# Step: (1+p)*(b*c) = b*c + p*(b*c). The hypothesis says p*(b*c) == (p*b)*c, and
# distributivity says b*c + (p*b)*c == (b + p*b)*c, which is the right-hand side.
def mul_assoc(a, b, c):
  match a:
    case 0n:
      {==}
    case 1n++p:
      %Equal.sym(Nat, Nat.mul(p, Nat.mul(b, c)), Nat.mul(Nat.mul(p, b), c), mul_assoc(p, b, c)) : {Nat.add(Nat.mul(b, c), _) == Nat.mul(Nat.add(b, Nat.mul(p, b)), c) : Nat}
      mul_dist(b, Nat.mul(p, b), c)

law mul_one_l:
  for +a: Nat
  {Nat.mul(1n, a) == a : Nat}

# 1 * a == a
# 1 * a computes to a + 0, and we already proved that a + 0 == a.
def mul_one_l(a):
  add_zero(a)

law mul_one_r:
  for +a: Nat
  {Nat.mul(a, 1n) == a : Nat}

# a * 1 == a
def mul_one_r(a):
  match a:
    case 0n:
      {==}
    case 1n++p:
      %mul_one_r(p) : {1n+Nat.mul(p, 1n) == 1n+_ : Nat}
      {==}

law append_nil:
  for -A: Data
  for  xs: List<&2, A>
  {List.append(&2, A, xs, Nil{}) == xs : List<&2, A>}

# xs ++ [] == xs
def append_nil(A, xs):
  match xs:
    case Nil{}:
      {==}
    case Con{h, t}:
      %append_nil(A, t) : {Con{h, List.append(&2, A, t, Nil{})} == Con{h, _} : List<&2, A>}
      {==}

law append_assoc:
  for -A: Data
  for  xs: List<&2, A>
  for  ys: List<&2, A>
  for  zs: List<&2, A>
  {List.append(&2, A, List.append(&2, A, xs, ys), zs) == List.append(&2, A, xs, List.append(&2, A, ys, zs)) : List<&2, A>}

# (xs ++ ys) ++ zs == xs ++ (ys ++ zs)
def append_assoc(A, xs, ys, zs):
  match xs:
    case Nil{}:
      {==}
    case Con{h, t}:
      %append_assoc(A, t, ys, zs) : {Con{h, List.append(&2, A, List.append(&2, A, t, ys), zs)} == Con{h, _} : List<&2, A>}
      {==}

law length_append:
  for -A: Data
  for  xs: List<&2, A>
  for  ys: List<&2, A>
  {List.length(&2, A, List.append(&2, A, xs, ys)) == Nat.add(List.length(&2, A, xs), List.length(&2, A, ys)) : Nat}

# length (xs ++ ys) == length xs + length ys
def length_append(A, xs, ys):
  match xs:
    case Nil{}:
      {==}
    case Con{h, t}:
      %length_append(A, t, ys) : {1n+List.length(&2, A, List.append(&2, A, t, ys)) == 1n+_ : Nat}
      {==}

law product_append:
  for +xs: List<&2, Nat>
  for +ys: List<&2, Nat>
  {product(List.append(&2, Nat, xs, ys)) == Nat.mul(product(xs), product(ys)) : Nat}

# product (xs ++ ys) == product xs * product ys
# Nil: product ys == 1 * product ys (by the mul_one_l law, read backwards).
# Con: product(h :: t ++ ys) = h * product(t ++ ys); the hypothesis replaces that with
# h * (product t * product ys), and mul_assoc closes it.
def product_append(xs, ys):
  match xs:
    case Nil{}:
      Equal.sym(Nat, Nat.mul(1n, product(ys)), product(ys), mul_one_l(product(ys)))
    case Con{+h, +t}:
      %Equal.sym(Nat, product(List.append(&2, Nat, t, ys)), Nat.mul(product(t), product(ys)), product_append(t, ys)) : {Nat.mul(h, _) == Nat.mul(Nat.mul(h, product(t)), product(ys)) : Nat}
      mul_assoc(h, product(t), product(ys))
