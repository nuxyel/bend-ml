# bend-ml-bpe-tokenizer: byte-level BPE with a proved roundtrip.
#
#   import bend-ml-bpe-tokenizer@0.1.2.0/main.bend as BPE
#
# A token is a raw byte (B{n}) or the result of a merge rule (M{id}).
# The table is a list of rules from the oldest to the newest, like GPT-2's
# merges.txt: the rule with id k joins the pair (a, b) into the token M{k}.
# `encode` applies the rules in order; `decode` expands each token back
# into bytes. The laws are at the end of the file, each with its proof.
#
# How to read the proofs: a proof in Bend is a function whose TYPE is the statement.
# `match` does case analysis, the recursive call is the induction hypothesis,
# `%e : P` rewrites the goal with the equality `e`, `{==}` closes when the
# two sides are already the same term. `+x` marks a variable that may be
# used more than once.

import Base
import bend-ml-nat-lemmas@0.1.0.0/main.bend as NL

# Token: a raw byte (B) or the result of a merge rule, by rule id (M).
type Tk is Data:
  B{n: Nat}
  M{k: Nat}

# Merge rule: joins the pair (a, b) into the token M{id}.
type Rule is Data:
  Rule{id: Nat, a: Tk, b: Tk}

def Tk.eq(+x: Tk, +y: Tk) -> Bool:
  match x y:
    case B{+m} B{+n}:
      Nat.is_eq(m, n)
    case M{+i} M{+j}:
      Nat.is_eq(i, j)
    case B{m} M{j}:
      False{}
    case M{i} B{n}:
      False{}

# does xs start with exactly the pair (a, b)?
def peek(+a: Tk, +b: Tk, xs: List<&2, Tk>) -> Bool:
  match xs:
    case Nil{}:
      False{}
    case Con{+x, t}:
      match t:
        case Nil{}:
          False{}
        case Con{+y, u}:
          Bool.and(Tk.eq(x, a), Tk.eq(y, b))

# Replaces each occurrence (non-overlapping, left to right) of the pair
# (a, b) with the token c. `hit` is always `peek(a, b, xs)`: the caller computes it,
# because a match can only open parameters.
def merge.go(xs: List<&2, Tk>, hit: Bool, +a: Tk, +b: Tk, +c: Tk) -> List<&2, Tk>:
  match xs hit:
    case Con{x, Con{y, +t}} True{}:
      c <> merge.go(t, peek(a, b, t), a, b, c)
    case Con{x, Nil{}} True{}:
      x <> Nil{}
    case Con{x, +t} False{}:
      x <> merge.go(t, peek(a, b, t), a, b, c)
    case Nil{} _:
      Nil{}

def merge(+a: Tk, +b: Tk, +c: Tk, +xs: List<&2, Tk>) -> List<&2, Tk>:
  merge.go(xs, peek(a, b, xs), a, b, c)

# applies the rules from the oldest to the newest; `done` keeps the ones already applied
# (newest first), which is exactly the table that decode needs.
def encode.go(todo: List<&2, Rule>, done: List<&2, Rule>, xs: List<&2, Tk>) -> List<&2, Tk>:
  match todo:
    case Nil{}:
      xs
    case Con{+r, rest}:
      match r:
        case Rule{+i, +a, +b}:
          encode.go(rest, Rule{i, a, b} <> done, merge(a, b, M{i}, xs))

# The table goes from the oldest rule to the newest (like merges.txt).
def encode(+table: List<&2, Rule>, xs: List<&2, Tk>) -> List<&2, Tk>:
  encode.go(table, Nil{}, xs)

# is the newest rule in the table the one that creates the token tk?
def hits(table: List<&2, Rule>, +tk: Tk) -> Bool:
  match table:
    case Nil{}:
      False{}
    case Con{Rule{+i, a, b}, rest}:
      match tk:
        case B{n}:
          False{}
        case M{+j}:
          Nat.is_eq(i, j)

# Expansion of a token into bytes. The table has the newest rule first and
# a rule only refers to older tokens, so expanding a and b looks only at
# the rest of the table. `hit` is always `hits(table, tk)`.
def exp.go(table: List<&2, Rule>, hit: Bool, tk: Tk) -> List<&2, Nat>:
  match table hit tk:
    case Nil{} _ B{n}:
      n <> Nil{}
    case Nil{} _ M{j}:
      Nil{}
    case Con{r, rest} _ B{n}:
      n <> Nil{}
    case Con{Rule{i, +a, +b}, +rest} True{} M{j}:
      List.append(&2, Nat, exp.go(rest, hits(rest, a), a), exp.go(rest, hits(rest, b), b))
    case Con{r, +rest} False{} M{+j}:
      exp.go(rest, hits(rest, M{j}), M{j})

def exp(+table: List<&2, Rule>, +tk: Tk) -> List<&2, Nat>:
  exp.go(table, hits(table, tk), tk)

# decodes with the already reversed table (newest first)
def dec(+table: List<&2, Rule>, ids: List<&2, Tk>) -> List<&2, Nat>:
  match ids:
    case Nil{}:
      Nil{}
    case Con{+x, t}:
      List.append(&2, Nat, exp(table, x), dec(table, t))

def decode(+table: List<&2, Rule>, ids: List<&2, Tk>) -> List<&2, Nat>:
  dec(List.reverse(&2, Rule, table), ids)

# bytes -> initial tokens
def lift(bs: List<&2, Nat>) -> List<&2, Tk>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{n, t}:
      B{n} <> lift(t)

# ---- predicates used in the laws (all computable) ----

# does some rule in the table have the id i?
def has(+D: List<&2, Rule>, +i: Nat) -> Bool:
  match D:
    case Nil{}:
      False{}
    case Con{Rule{+j, a, b}, t}:
      Bool.or(Nat.is_eq(i, j), has(t, i))

# is the token a byte, or was it created by a rule in the table?
def kn(+D: List<&2, Rule>, tk: Tk) -> Bool:
  match tk:
    case B{n}:
      True{}
    case M{+j}:
      has(D, j)

def knl(+D: List<&2, Rule>, xs: List<&2, Tk>) -> Bool:
  match xs:
    case Nil{}:
      True{}
    case Con{x, t}:
      Bool.and(kn(D, x), knl(D, t))

# well-formed table: no id repeats
def wf.go(todo: List<&2, Rule>, +done: List<&2, Rule>) -> Bool:
  match todo:
    case Nil{}:
      True{}
    case Con{Rule{+i, +a, +b}, rest}:
      Bool.and(Bool.not(has(done, i)), wf.go(rest, Rule{i, a, b} <> done))

def wf(table: List<&2, Rule>) -> Bool:
  wf.go(table, Nil{})

# =====================================================================
# Training (needs no proof: it generates tables; a table is valid if wf(table))
# =====================================================================
# ---------------------------------------------------------------
# training: count adjacent pairs, merge the most frequent one, repeat
# (tie-break: the first pair that appeared, as in minbpe)
# ---------------------------------------------------------------
type Cnt is Data:
  Cnt{a: Tk, b: Tk, n: Nat}

# is the first Cnt of cs the pair (a, b)?
def same_head(cs: List<&2, Cnt>, +a: Tk, +b: Tk) -> Bool:
  match cs:
    case Nil{}:
      False{}
    case Con{Cnt{+x, +y, k}, rest}:
      Bool.and(Tk.eq(x, a), Tk.eq(y, b))

# adds 1 to the counter of the pair (a, b), or creates it at the end of the list
def bump.go(cs: List<&2, Cnt>, hit: Bool, +a: Tk, +b: Tk) -> List<&2, Cnt>:
  match cs hit:
    case Nil{} _:
      Cnt{a, b, 1n} <> Nil{}
    case Con{Cnt{x, y, +k}, rest} True{}:
      Cnt{x, y, 1n+k} <> rest
    case Con{c, +rest} False{}:
      c <> bump.go(rest, same_head(rest, a, b), a, b)

def bump(+cs: List<&2, Cnt>, +a: Tk, +b: Tk) -> List<&2, Cnt>:
  bump.go(cs, same_head(cs, a, b), a, b)

def count.go(xs: List<&2, Tk>, cs: List<&2, Cnt>) -> List<&2, Cnt>:
  match xs:
    case Nil{}:
      cs
    case Con{+x, +t}:
      match t:
        case Nil{}:
          cs
        case Con{+y, u}:
          count.go(Con{y, u}, bump(cs, x, y))

def pick2(better: Bool, c: Cnt, cur: Cnt) -> Cnt:
  match better:
    case True{}:
      c
    case False{}:
      cur

def pick(c: Cnt, cur: Cnt) -> Cnt:
  match c cur:
    case Cnt{+a, +b, +k} Cnt{+x, +y, +m}:
      pick2(Nat.is_lt(m, k), Cnt{a, b, k}, Cnt{x, y, m})

def best.go(cs: List<&2, Cnt>, cur: Cnt) -> Cnt:
  match cs:
    case Nil{}:
      cur
    case Con{c, rest}:
      best.go(rest, pick(c, cur))

def best(cs: List<&2, Cnt>) -> Maybe<&1, Cnt>:
  match cs:
    case Nil{}:
      None{}
    case Con{c, rest}:
      Some{best.go(rest, c)}

# up to n merges, ids from `next` on; m is the current best pair (best(count(xs)))
def train.go(n: Nat, +next: Nat, +xs: List<&2, Tk>, m: Maybe<&1, Cnt>) -> List<&2, Rule>:
  match n m:
    case 0n _:
      Nil{}
    case 1n+p None{}:
      Nil{}
    case 1n+p Some{Cnt{+a, +b, k}}:
      +ys = merge(a, b, M{next}, xs)
      Rule{next, a, b} <> train.go(p, 1n+next, ys, best(count.go(ys, Nil{})))

# the table from the oldest rule to the newest
def train(n: Nat, +next: Nat, +xs: List<&2, Tk>) -> List<&2, Rule>:
  train.go(n, next, xs, best(count.go(xs, Nil{})))

# =====================================================================
# Auxiliary lemmas for the proofs
# =====================================================================

# a type chosen by a Bool: rewriting through it refutes True == False
def BD(b: Bool, t: Type, f: Type) -> Type:
  match b:
    case True{}:
      t
    case False{}:
      f

def nat_eq_sound(a: Nat, b: Nat, e: {True{} == Nat.is_eq(a, b) : Bool}) -> {a == b : Nat}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      %e : BD(_, Unit, {0n == 1n+q : Nat})
      Unit{}
    case 1n+p 0n:
      %e : BD(_, Unit, {1n+p == 0n : Nat})
      Unit{}
    case 1n+p 1n+q:
      %nat_eq_sound(p, q, e) : {1n+p == 1n+_ : Nat}
      {==}

def nat_eq_refl(a: Nat) -> {True{} == Nat.is_eq(a, a) : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      nat_eq_refl(p)

def and_true(p: Bool, q: Bool, e: {True{} == Bool.and(p, q) : Bool}) -> {True{} == p : Bool} & {True{} == q : Bool}:
  match p:
    case True{}:
      ({==}, e)
    case False{}:
      %e : BD(_, Unit, {True{} == False{} : Bool} & {True{} == q : Bool})
      Unit{}

def and_intro(p: Bool, q: Bool, ep: {True{} == p : Bool}, eq: {True{} == q : Bool}) -> {True{} == Bool.and(p, q) : Bool}:
  match p:
    case True{}:
      eq
    case False{}:
      %ep : BD(_, Unit, {True{} == False{} : Bool})
      Unit{}

def or_left(p: Bool, q: Bool, ep: {True{} == p : Bool}) -> {True{} == Bool.or(p, q) : Bool}:
  match p:
    case True{}:
      {==}
    case False{}:
      %ep : BD(_, Unit, {True{} == q : Bool})
      Unit{}

def or_right(p: Bool, q: Bool, eq: {True{} == q : Bool}) -> {True{} == Bool.or(p, q) : Bool}:
  match p:
    case True{}:
      {==}
    case False{}:
      eq

# not p is true  ->  p is false
def not_true(p: Bool, e: {True{} == Bool.not(p) : Bool}) -> {False{} == p : Bool}:
  match p:
    case True{}:
      %e : BD(_, Unit, {False{} == True{} : Bool})
      Unit{}
    case False{}:
      {==}

def tk_eq_sound(+x: Tk, +y: Tk, e: {True{} == Tk.eq(x, y) : Bool}) -> {x == y : Tk}:
  match x y:
    case B{+m} B{+n}:
      %nat_eq_sound(m, n, e) : {B{m} == B{_} : Tk}
      {==}
    case M{+i} M{+j}:
      %nat_eq_sound(i, j, e) : {M{i} == M{_} : Tk}
      {==}
    case B{m} M{j}:
      %e : BD(_, Unit, {B{m} == M{j} : Tk})
      Unit{}
    case M{i} B{n}:
      %e : BD(_, Unit, {M{i} == B{n} : Tk})
      Unit{}

# Adding the (new) rule i on top of the table does not change the expansion of a
# token M{j} that the old table already knows (j != i, because i is new).
def ext_m(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk, +j: Nat, h: Bool, eh: {h == Nat.is_eq(i, j) : Bool}, nf: {True{} == Bool.not(has(D, i)) : Bool}, kt: {True{} == has(D, j) : Bool}) -> {exp(Rule{i, a, b} <> D, M{j}) == exp(D, M{j}) : List<&2, Nat>}:
  match h:
    case True{}:
      # h = True means i == j; but j is known (has D j) and i is not (not has D i)
      +ij = nat_eq_sound(i, j, eh)
      +hij = Equal.cong(Nat, Bool, k => has(D, k), i, j, ij)
      +f1 = Equal.trans(Bool, False{}, has(D, i), has(D, j), not_true(has(D, i), nf), hij)
      +tf = Equal.trans(Bool, True{}, has(D, j), False{}, kt, Equal.sym(Bool, False{}, has(D, j), f1))
      %tf : BD(_, Unit, {exp.go(Rule{i, a, b} <> D, Cmp.is_eq(Nat.cmp(i, j)), M{j}) == exp.go(D, hits(D, M{j}), M{j}) : List<&2, Nat>})
      Unit{}
    case False{}:
      %eh : {exp.go(Rule{i, a, b} <> D, _, M{j}) == exp(D, M{j}) : List<&2, Nat>}
      {==}

def and_true_l(p: Bool, q: Bool, e: {True{} == Bool.and(p, q) : Bool}) -> {True{} == p : Bool}:
  match p:
    case True{}:
      {==}
    case False{}:
      %e : BD(_, Unit, {True{} == False{} : Bool})
      Unit{}

def and_true_r(p: Bool, q: Bool, e: {True{} == Bool.and(p, q) : Bool}) -> {True{} == q : Bool}:
  match p:
    case True{}:
      e
    case False{}:
      %e : BD(_, Unit, {True{} == q : Bool})
      Unit{}

# the same for any known token (a byte never depends on the table)
def ext_tok(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk, t: Tk, kt: {True{} == kn(D, t) : Bool}, nf: {True{} == Bool.not(has(D, i)) : Bool}) -> {exp(Rule{i, a, b} <> D, t) == exp(D, t) : List<&2, Nat>}:
  match D t:
    case Nil{} B{n}:
      {==}
    case Con{Rule{j, a2, b2}, rest} B{n}:
      {==}
    case _ M{+j}:
      ext_m(D, i, a, b, j, Nat.is_eq(i, j), {==}, nf, kt)

def ext_dec.step(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk, +x: Tk, +t: List<&2, Tk>, k1: {True{} == kn(D, x) : Bool}, nf: {True{} == Bool.not(has(D, i)) : Bool}, ih: {dec(Rule{i, a, b} <> D, t) == dec(D, t) : List<&2, Nat>}) -> {dec(Rule{i, a, b} <> D, x <> t) == dec(D, x <> t) : List<&2, Nat>}:
  %ext_tok(D, i, a, b, x, k1, nf) : {List.append(&2, Nat, exp(Rule{i, a, b} <> D, x), dec(Rule{i, a, b} <> D, t)) == List.append(&2, Nat, _, dec(D, t)) : List<&2, Nat>}
  %ih : {List.append(&2, Nat, exp(Rule{i, a, b} <> D, x), dec(Rule{i, a, b} <> D, t)) == List.append(&2, Nat, exp(Rule{i, a, b} <> D, x), _) : List<&2, Nat>}
  {==}

# decoding a list of known tokens does not change when the new rule is added
def ext_dec(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk, xs: List<&2, Tk>, +kx: {True{} == knl(D, xs) : Bool}, +nf: {True{} == Bool.not(has(D, i)) : Bool}) -> {dec(Rule{i, a, b} <> D, xs) == dec(D, xs) : List<&2, Nat>}:
  match xs:
    case Nil{}:
      {==}
    case Con{+x, +t}:
      ext_dec.step(D, i, a, b, x, t, and_true_l(kn(D, x), knl(D, t), kx), nf, ext_dec(D, i, a, b, t, and_true_r(kn(D, x), knl(D, t), kx), nf))

# The new token M{i} expands to the expansion of a followed by that of b
# (looking only at the old table D).
def exp_new(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk, h: Bool, eh: {h == Nat.is_eq(i, i) : Bool}) -> {exp(Rule{i, a, b} <> D, M{i}) == List.append(&2, Nat, exp(D, a), exp(D, b)) : List<&2, Nat>}:
  match h:
    case True{}:
      %eh : {exp.go(Rule{i, a, b} <> D, _, M{i}) == List.append(&2, Nat, exp(D, a), exp(D, b)) : List<&2, Nat>}
      {==}
    case False{}:
      +f = Equal.trans(Bool, True{}, Nat.is_eq(i, i), False{}, nat_eq_refl(i), Equal.sym(Bool, False{}, Nat.is_eq(i, i), eh))
      %f : BD(_, Unit, {exp(Rule{i, a, b} <> D, M{i}) == List.append(&2, Nat, exp(D, a), exp(D, b)) : List<&2, Nat>})
      Unit{}

# Central step: in place of a b t we get M{i} followed by m, where m decodes
# like t. Both sides decode the same because M{i} expands to exp(a) ++ exp(b).
def mp_core(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk, +t: List<&2, Tk>, +m: List<&2, Tk>, ka: {True{} == kn(D, a) : Bool}, kb: {True{} == kn(D, b) : Bool}, +nf: {True{} == Bool.not(has(D, i)) : Bool}, ihm: {dec(Rule{i, a, b} <> D, m) == dec(Rule{i, a, b} <> D, t) : List<&2, Nat>}) -> {dec(Rule{i, a, b} <> D, M{i} <> m) == dec(Rule{i, a, b} <> D, a <> b <> t) : List<&2, Nat>}:
  %Equal.sym(List<&2, Nat>, exp(Rule{i, a, b} <> D, M{i}), List.append(&2, Nat, exp(D, a), exp(D, b)), exp_new(D, i, a, b, Nat.is_eq(i, i), {==})) : {List.append(&2, Nat, _, dec(Rule{i, a, b} <> D, m)) == List.append(&2, Nat, exp(Rule{i, a, b} <> D, a), List.append(&2, Nat, exp(Rule{i, a, b} <> D, b), dec(Rule{i, a, b} <> D, t))) : List<&2, Nat>}
  %Equal.sym(List<&2, Nat>, dec(Rule{i, a, b} <> D, m), dec(Rule{i, a, b} <> D, t), ihm) : {List.append(&2, Nat, List.append(&2, Nat, exp(D, a), exp(D, b)), _) == List.append(&2, Nat, exp(Rule{i, a, b} <> D, a), List.append(&2, Nat, exp(Rule{i, a, b} <> D, b), dec(Rule{i, a, b} <> D, t))) : List<&2, Nat>}
  %Equal.sym(List<&2, Nat>, exp(Rule{i, a, b} <> D, a), exp(D, a), ext_tok(D, i, a, b, a, ka, nf)) : {List.append(&2, Nat, List.append(&2, Nat, exp(D, a), exp(D, b)), dec(Rule{i, a, b} <> D, t)) == List.append(&2, Nat, _, List.append(&2, Nat, exp(Rule{i, a, b} <> D, b), dec(Rule{i, a, b} <> D, t))) : List<&2, Nat>}
  %Equal.sym(List<&2, Nat>, exp(Rule{i, a, b} <> D, b), exp(D, b), ext_tok(D, i, a, b, b, kb, nf)) : {List.append(&2, Nat, List.append(&2, Nat, exp(D, a), exp(D, b)), dec(Rule{i, a, b} <> D, t)) == List.append(&2, Nat, exp(D, a), List.append(&2, Nat, _, dec(Rule{i, a, b} <> D, t))) : List<&2, Nat>}
  NL.append_assoc(Nat, exp(D, a), exp(D, b), dec(Rule{i, a, b} <> D, t))

# if x == a and x is known, a is known too
def kn_sub(+D: List<&2, Rule>, +x: Tk, +a: Tk, ex: {x == a : Tk}, kx: {True{} == kn(D, x) : Bool}) -> {True{} == kn(D, a) : Bool}:
  %ex : {True{} == kn(D, _) : Bool}
  kx

# x stays out of the rewrite: if the rest decodes the same, x <> rest does too
def mp_false(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk, +x: Tk, +t: List<&2, Tk>, +mg: List<&2, Tk>, ih: {dec(Rule{i, a, b} <> D, mg) == dec(Rule{i, a, b} <> D, t) : List<&2, Nat>}) -> {dec(Rule{i, a, b} <> D, x <> mg) == dec(Rule{i, a, b} <> D, x <> t) : List<&2, Nat>}:
  %ih : {List.append(&2, Nat, exp(Rule{i, a, b} <> D, x), dec(Rule{i, a, b} <> D, mg)) == List.append(&2, Nat, exp(Rule{i, a, b} <> D, x), _) : List<&2, Nat>}
  {==}

# the pair is x y: the tokens were equal to a and b
def mp_hit(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk, +x: Tk, +y: Tk, +t: List<&2, Tk>, +m: List<&2, Tk>, +ex: {x == a : Tk}, +ey: {y == b : Tk}, +kxx: {True{} == kn(D, x) : Bool}, +kyy: {True{} == kn(D, y) : Bool}, +nf: {True{} == Bool.not(has(D, i)) : Bool}, ihm: {dec(Rule{i, a, b} <> D, m) == dec(Rule{i, a, b} <> D, t) : List<&2, Nat>}) -> {dec(Rule{i, a, b} <> D, M{i} <> m) == dec(Rule{i, a, b} <> D, x <> y <> t) : List<&2, Nat>}:
  %Equal.sym(Tk, x, a, ex) : {dec(Rule{i, a, b} <> D, M{i} <> m) == dec(Rule{i, a, b} <> D, _ <> y <> t) : List<&2, Nat>}
  %Equal.sym(Tk, y, b, ey) : {dec(Rule{i, a, b} <> D, M{i} <> m) == dec(Rule{i, a, b} <> D, a <> _ <> t) : List<&2, Nat>}
  mp_core(D, i, a, b, t, m, kn_sub(D, x, a, ex, kxx), kn_sub(D, y, b, ey, kyy), nf, ihm)

# merge.go does not change what decode returns (with the new rule on top of the table)
def mp(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk, xs: List<&2, Tk>, hit: Bool, +eh: {hit == peek(a, b, xs) : Bool}, +kx: {True{} == knl(D, xs) : Bool}, +nf: {True{} == Bool.not(has(D, i)) : Bool}) -> {dec(Rule{i, a, b} <> D, merge.go(xs, hit, a, b, M{i})) == dec(Rule{i, a, b} <> D, xs) : List<&2, Nat>}:
  match xs hit:
    case Nil{} _:
      {==}
    case Con{x, Nil{}} True{}:
      %eh : BD(_, Unit, {dec(Rule{i, a, b} <> D, merge.go(x <> Nil{}, True{}, a, b, M{i})) == dec(Rule{i, a, b} <> D, x <> Nil{}) : List<&2, Nat>})
      Unit{}
    case Con{+x, Con{+y, +t}} True{}:
      mp_hit(D, i, a, b, x, y, t, merge.go(t, peek(a, b, t), a, b, M{i}), tk_eq_sound(x, a, and_true_l(Tk.eq(x, a), Tk.eq(y, b), eh)), tk_eq_sound(y, b, and_true_r(Tk.eq(x, a), Tk.eq(y, b), eh)), and_true_l(kn(D, x), Bool.and(kn(D, y), knl(D, t)), kx), and_true_l(kn(D, y), knl(D, t), and_true_r(kn(D, x), Bool.and(kn(D, y), knl(D, t)), kx)), nf, mp(D, i, a, b, t, peek(a, b, t), {==}, and_true_r(kn(D, y), knl(D, t), and_true_r(kn(D, x), Bool.and(kn(D, y), knl(D, t)), kx)), nf))
    case Con{+x, +t} False{}:
      mp_false(D, i, a, b, x, t, merge.go(t, peek(a, b, t), a, b, M{i}), mp(D, i, a, b, t, peek(a, b, t), {==}, and_true_r(kn(D, x), knl(D, t), kx), nf))

# ---- known tokens stay known ----

# adding a rule to the table only increases what is known
def kn_mono(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk, t: Tk, kt: {True{} == kn(D, t) : Bool}) -> {True{} == kn(Rule{i, a, b} <> D, t) : Bool}:
  match t:
    case B{n}:
      {==}
    case M{+j}:
      or_right(Nat.is_eq(j, i), has(D, j), kt)

# the token created by the new rule is known
def kn_new(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk) -> {True{} == kn(Rule{i, a, b} <> D, M{i}) : Bool}:
  or_left(Nat.is_eq(i, i), has(D, i), nat_eq_refl(i))

def kp(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk, xs: List<&2, Tk>, hit: Bool, +eh: {hit == peek(a, b, xs) : Bool}, +kx: {True{} == knl(D, xs) : Bool}) -> {True{} == knl(Rule{i, a, b} <> D, merge.go(xs, hit, a, b, M{i})) : Bool}:
  match xs hit:
    case Nil{} _:
      {==}
    case Con{x, Nil{}} True{}:
      %eh : BD(_, Unit, {True{} == knl(Rule{i, a, b} <> D, merge.go(x <> Nil{}, True{}, a, b, M{i})) : Bool})
      Unit{}
    case Con{+x, Con{+y, +t}} True{}:
      and_intro(kn(Rule{i, a, b} <> D, M{i}), knl(Rule{i, a, b} <> D, merge.go(t, peek(a, b, t), a, b, M{i})), kn_new(D, i, a, b), kp(D, i, a, b, t, peek(a, b, t), {==}, and_true_r(kn(D, y), knl(D, t), and_true_r(kn(D, x), Bool.and(kn(D, y), knl(D, t)), kx))))
    case Con{+x, +t} False{}:
      and_intro(kn(Rule{i, a, b} <> D, x), knl(Rule{i, a, b} <> D, merge.go(t, peek(a, b, t), a, b, M{i})), kn_mono(D, i, a, b, x, and_true_l(kn(D, x), knl(D, t), kx)), kp(D, i, a, b, t, peek(a, b, t), {==}, and_true_r(kn(D, x), knl(D, t), kx)))

# ---- induction over the rules ----

# every emitted token is known to the complete (reversed) table
def enc_known(todo: List<&2, Rule>, +done: List<&2, Rule>, +xs: List<&2, Tk>, +wfe: {True{} == wf.go(todo, done) : Bool}, +kx: {True{} == knl(done, xs) : Bool}) -> {True{} == knl(List.reverse.go(&2, Rule, todo, done), encode.go(todo, done, xs)) : Bool}:
  match todo:
    case Nil{}:
      kx
    case Con{Rule{+i, +a, +b}, +rest}:
      enc_known(rest, Rule{i, a, b} <> done, merge.go(xs, peek(a, b, xs), a, b, M{i}), and_true_r(Bool.not(has(done, i)), wf.go(rest, Rule{i, a, b} <> done), wfe), kp(done, i, a, b, xs, peek(a, b, xs), {==}, kx))

# applying all the rules does not change what decode returns
def enc_dec(todo: List<&2, Rule>, +done: List<&2, Rule>, +xs: List<&2, Tk>, +wfe: {True{} == wf.go(todo, done) : Bool}, +kx: {True{} == knl(done, xs) : Bool}) -> {dec(List.reverse.go(&2, Rule, todo, done), encode.go(todo, done, xs)) == dec(done, xs) : List<&2, Nat>}:
  match todo:
    case Nil{}:
      {==}
    case Con{Rule{+i, +a, +b}, +rest}:
      Equal.trans(List<&2, Nat>, dec(List.reverse.go(&2, Rule, rest, Rule{i, a, b} <> done), encode.go(rest, Rule{i, a, b} <> done, merge.go(xs, peek(a, b, xs), a, b, M{i}))), dec(Rule{i, a, b} <> done, xs), dec(done, xs), Equal.trans(List<&2, Nat>, dec(List.reverse.go(&2, Rule, rest, Rule{i, a, b} <> done), encode.go(rest, Rule{i, a, b} <> done, merge.go(xs, peek(a, b, xs), a, b, M{i}))), dec(Rule{i, a, b} <> done, merge.go(xs, peek(a, b, xs), a, b, M{i})), dec(Rule{i, a, b} <> done, xs), enc_dec(rest, Rule{i, a, b} <> done, merge.go(xs, peek(a, b, xs), a, b, M{i}), and_true_r(Bool.not(has(done, i)), wf.go(rest, Rule{i, a, b} <> done), wfe), kp(done, i, a, b, xs, peek(a, b, xs), {==}, kx)), mp(done, i, a, b, xs, peek(a, b, xs), {==}, kx, and_true_l(Bool.not(has(done, i)), wf.go(rest, Rule{i, a, b} <> done), wfe))), ext_dec(done, i, a, b, xs, kx, and_true_l(Bool.not(has(done, i)), wf.go(rest, Rule{i, a, b} <> done), wfe)))

# ---- base case and final laws ----

# with no rules, the bytes decode to themselves
def lift_dec(bs: List<&2, Nat>) -> {dec(Nil{}, lift(bs)) == bs : List<&2, Nat>}:
  match bs:
    case Nil{}:
      {==}
    case Con{+n, +t}:
      %lift_dec(t) : {n <> dec(Nil{}, lift(t)) == n <> _ : List<&2, Nat>}
      {==}

def lift_known(bs: List<&2, Nat>) -> {True{} == knl(Nil{}, lift(bs)) : Bool}:
  match bs:
    case Nil{}:
      {==}
    case Con{n, t}:
      lift_known(t)

# =====================================================================
# LAWS (the package laws, explained in plain language in the README)
# =====================================================================

# LAW 1 (roundtrip): with a well-formed table (all rule ids
# distinct), decoding what was encoded returns the original bytes.
law roundtrip:
  for +table: List<&2, Rule>
  for +bs: List<&2, Nat>
  for +w: {True{} == wf(table) : Bool}
  {decode(table, encode(table, lift(bs))) == bs : List<&2, Nat>}

def roundtrip(table, bs, w):
  Equal.trans(List<&2, Nat>, decode(table, encode(table, lift(bs))), dec(Nil{}, lift(bs)), bs, enc_dec(table, Nil{}, lift(bs), w, lift_known(bs)), lift_dec(bs))

# LAW 2 (vocabulary): every token emitted by encode is a raw byte or was
# created by one of the table's rules; an unknown id never appears.
law vocab_bound:
  for +table: List<&2, Rule>
  for +bs: List<&2, Nat>
  for +w: {True{} == wf(table) : Bool}
  {True{} == knl(List.reverse(&2, Rule, table), encode(table, lift(bs))) : Bool}

def vocab_bound(table, bs, w):
  enc_known(table, Nil{}, lift(bs), w, lift_known(bs))

# LAW 3 (concatenation): decoding two token lists together is decoding
# each one and joining the bytes.
law dec_append:
  for +D: List<&2, Rule>
  for xs: List<&2, Tk>
  for +ys: List<&2, Tk>
  {dec(D, List.append(&2, Tk, xs, ys)) == List.append(&2, Nat, dec(D, xs), dec(D, ys)) : List<&2, Nat>}

def dec_append(D, xs, ys):
  match xs:
    case Nil{}:
      {==}
    case Con{+x, +t}:
      %Equal.sym(List<&2, Nat>, dec(D, List.append(&2, Tk, t, ys)), List.append(&2, Nat, dec(D, t), dec(D, ys)), dec_append(D, t, ys)) : {List.append(&2, Nat, exp(D, x), _) == List.append(&2, Nat, List.append(&2, Nat, exp(D, x), dec(D, t)), dec(D, ys)) : List<&2, Nat>}
      Equal.sym(List<&2, Nat>, List.append(&2, Nat, List.append(&2, Nat, exp(D, x), dec(D, t)), dec(D, ys)), List.append(&2, Nat, exp(D, x), List.append(&2, Nat, dec(D, t), dec(D, ys))), NL.append_assoc(Nat, exp(D, x), dec(D, t), dec(D, ys)))

# =====================================================================
# LAW 4: train generates well-formed tables (closes the roundtrip gap)
# =====================================================================
#
# train creates rules with ids next, next+1, next+2, ... None repeats, so the
# table satisfies wf, which is the hypothesis of the roundtrip law. Proof by induction
# over train.go, keeping the invariant "all ids already used are smaller
# than the next id".

# n < 1 + n
def lt_succ(n: Nat) -> {True{} == Nat.is_lt(n, 1n+n) : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      lt_succ(p)

# j < i  ->  i != j
def lt_ne(j: Nat, i: Nat, e: {True{} == Nat.is_lt(j, i) : Bool}) -> {False{} == Nat.is_eq(i, j) : Bool}:
  match j i:
    case 0n 0n:
      %e : BD(_, Unit, {False{} == Nat.is_eq(0n, 0n) : Bool})
      Unit{}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      %e : BD(_, Unit, {False{} == Nat.is_eq(0n, 1n+p) : Bool})
      Unit{}
    case 1n+p 1n+q:
      lt_ne(p, q, e)

# j < i  ->  j < 1 + i
def lt_up(j: Nat, i: Nat, e: {True{} == Nat.is_lt(j, i) : Bool}) -> {True{} == Nat.is_lt(j, 1n+i) : Bool}:
  match j i:
    case 0n 0n:
      %e : BD(_, Unit, {True{} == Nat.is_lt(0n, 1n) : Bool})
      Unit{}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      %e : BD(_, Unit, {True{} == Nat.is_lt(1n+p, 1n) : Bool})
      Unit{}
    case 1n+p 1n+q:
      lt_up(p, q, e)

# all ids in the rule list are smaller than i
def below(D: List<&2, Rule>, +i: Nat) -> Bool:
  match D:
    case Nil{}:
      True{}
    case Con{Rule{+j, a, b}, t}:
      Bool.and(Nat.is_lt(j, i), below(t, i))

def below_has(D: List<&2, Rule>, +i: Nat, +e: {True{} == below(D, i) : Bool}) -> {False{} == has(D, i) : Bool}:
  match D:
    case Nil{}:
      {==}
    case Con{Rule{+j, a, b}, +t}:
      %lt_ne(j, i, and_true_l(Nat.is_lt(j, i), below(t, i), e)) : {False{} == Bool.or(_, has(t, i)) : Bool}
      below_has(t, i, and_true_r(Nat.is_lt(j, i), below(t, i), e))

def below_up(D: List<&2, Rule>, +i: Nat, +e: {True{} == below(D, i) : Bool}) -> {True{} == below(D, 1n+i) : Bool}:
  match D:
    case Nil{}:
      {==}
    case Con{Rule{+j, a, b}, +t}:
      and_intro(Nat.is_lt(j, 1n+i), below(t, 1n+i), lt_up(j, i, and_true_l(Nat.is_lt(j, i), below(t, i), e)), below_up(t, i, and_true_r(Nat.is_lt(j, i), below(t, i), e)))

# adding the rule with id `next` (greater than all those in `done`) keeps the table "below 1+next"
def below_cons(+a: Tk, +b: Tk, +D: List<&2, Rule>, +i: Nat, +e: {True{} == below(D, i) : Bool}) -> {True{} == below(Rule{i, a, b} <> D, 1n+i) : Bool}:
  and_intro(Nat.is_lt(i, 1n+i), below(D, 1n+i), lt_succ(i), below_up(D, i, e))

# not (has D i) when all the ids in D are smaller than i
def below_fresh(+D: List<&2, Rule>, +i: Nat, +e: {True{} == below(D, i) : Bool}) -> {True{} == Bool.not(has(D, i)) : Bool}:
  %below_has(D, i, e) : {True{} == Bool.not(_) : Bool}
  {==}

# train.go generates rules with ids next, next+1, ...: none repeats, whatever the corpus
def tw(n: Nat, +next: Nat, +xs: List<&2, Tk>, m: Maybe<&1, Cnt>, +done: List<&2, Rule>, +bl: {True{} == below(done, next) : Bool}) -> {True{} == wf.go(train.go(n, next, xs, m), done) : Bool}:
  match n m:
    case 0n _:
      {==}
    case 1n++p None{}:
      {==}
    case 1n++p Some{Cnt{+a, +b, k}}:
      and_intro(Bool.not(has(done, next)), wf.go(train.go(p, 1n+next, merge(a, b, M{next}, xs), best(count.go(merge(a, b, M{next}, xs), Nil{}))), Rule{next, a, b} <> done), below_fresh(done, next, bl), tw(p, 1n+next, merge(a, b, M{next}, xs), best(count.go(merge(a, b, M{next}, xs), Nil{})), Rule{next, a, b} <> done, below_cons(a, b, done, next, bl)))

# LAW (train_wf): the table that train returns is always well formed, so the roundtrip holds for it
law train_wf:
  for n: Nat
  for +next: Nat
  for +xs: List<&2, Tk>
  {True{} == wf(train(n, next, xs)) : Bool}

def train_wf(n, next, xs):
  tw(n, next, xs, best(count.go(xs, Nil{})), Nil{}, {==})

# LAW 5 (roundtrip_trained): train a table on any corpus, with as many merges as you like:
# decoding what it encodes always returns the original bytes. No hypothesis at all.
law roundtrip_trained:
  for +n: Nat
  for +corpus: List<&2, Nat>
  for +bs: List<&2, Nat>
  {decode(train(n, 0n, lift(corpus)), encode(train(n, 0n, lift(corpus)), lift(bs))) == bs : List<&2, Nat>}

def roundtrip_trained(n, corpus, bs):
  roundtrip(train(n, 0n, lift(corpus)), bs, train_wf(n, 0n, lift(corpus)))
