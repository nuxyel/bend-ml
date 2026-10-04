# GPT-2 tokenizer in Bend: pre-tokenizer (the GPT-2 regex over bytes) + the 50,000
# merge rules, applied with the bend-ml-bpe-tokenizer package (proved roundtrip).
#
# The pre-tokenizer decodes UTF-8 and classifies every code point like the regex does
# (\s, \p{L}, \p{N}); the classes of code points >= 128 come from unicode.bend, which
# reference/gen_unicode_table.py generates with the `regex` module. Known limits: code points
# above U+1FFFF count as letters, and so do CJK ideographs and Hangul (they are letters in
# Unicode, so this is exact); invalid UTF-8 bytes count as letters.

import Base
import bend-ml-bpe-tokenizer@0.1.2.0/main.bend as BPE
import ./unicode.bend as U

# ---- reading files ----

def rb_done(m: File & Result<&1, &1, U32 & String, List<&2, U32>>) -> IO(List<&2, U32>):
  (f, r) = m
  do IO<List<&2, U32>>:
    bs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
    File.close(f)
    return bs

def read_bytes(path: String) -> IO(List<&2, U32>):
  do IO<List<&2, U32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_bytes(f, 400000000)
    rb_done(m)

def id_tok2(small: Bool, +n: Nat) -> BPE.Tk:
  match small:
    case True{}:
      BPE.B{n}
    case False{}:
      BPE.M{Nat.sub(n, 256n)}

# numeric id in the bpe/ scheme (byte = 0..255, merge k = 256+k) -> token

def id_tok(+n: Nat) -> BPE.Tk:
  id_tok2(Nat.is_lt(n, 256n), n)

# ---- integers in text: any byte that is not a digit separates ----

def digit2(lo: Bool, hi: Bool, +code: U32) -> Maybe<&2, Nat>:
  match lo hi:
    case True{} True{}:
      Some{U32.to_nat(U32.sub(code, 48))}
    case _ _:
      None{}

def digit(+code: U32) -> Maybe<&2, Nat>:
  digit2(U32.is_le(48, code), U32.is_le(code, 57), code)

def head_digit(xs: List<&2, U32>) -> Maybe<&2, Nat>:
  match xs:
    case Nil{}:
      None{}
    case Con{+c, t}:
      digit(c)

def flush(have: Bool, +cur: Nat, acc: List<&2, Nat>) -> List<&2, Nat>:
  match have:
    case True{}:
      cur <> acc
    case False{}:
      acc

def nums.go(xs: List<&2, U32>, d: Maybe<&2, Nat>, +cur: Nat, have: Bool, acc: List<&2, Nat>) -> List<&2, Nat>:
  match xs d:
    case Nil{} _:
      List.reverse(&2, Nat, flush(have, cur, acc))
    case Con{c, +t} Some{+v}:
      nums.go(t, head_digit(t), Nat.add(Nat.mul(cur, 10n), v), True{}, acc)
    case Con{c, +t} None{}:
      nums.go(t, head_digit(t), 0n, False{}, flush(have, cur, acc))

def nums(+xs: List<&2, U32>) -> List<&2, Nat>:
  nums.go(xs, head_digit(xs), 0n, False{}, Nil{})

# pairs of ids become rules with sequential ids

def rules.go(ns: List<&2, Nat>, +k: Nat) -> List<&2, BPE.Rule>:
  match ns:
    case Nil{}:
      Nil{}
    case Con{+a, t}:
      match t:
        case Nil{}:
          Nil{}
        case Con{+b, u}:
          BPE.Rule{k, id_tok(a), id_tok(b)} <> rules.go(u, 1n+k)

# ---- runtime check of the merges table read from a file ----
# rules.go gives the rules the sequential ids 0, 1, 2..., so BPE.wf (distinct ids) holds
# by construction; what a malformed file can still break is the shape of the table.
# It must have an even number of ids, and rule k may only refer to ids that exist
# before it (bytes < 256, or merges 256..256+k-1). The check is linear in the table size.
def ids_ok(ns: List<&2, Nat>, +k: Nat) -> Bool:
  match ns:
    case Nil{}:
      True{}
    case Con{+a, t}:
      match t:
        case Nil{}:
          False{}
        case Con{+b, u}:
          Bool.and(Bool.and(Nat.is_lt(a, Nat.add(256n, k)), Nat.is_lt(b, Nat.add(256n, k))), ids_ok(u, 1n+k))

def toks_of_ids(ns: List<&2, Nat>) -> List<&2, BPE.Tk>:
  match ns:
    case Nil{}:
      Nil{}
    case Con{n, t}:
      id_tok(n) <> toks_of_ids(t)


# ---- table and permutation of the GPT-2 base ids ----

def nth(xs: List<&2, Nat>, n: Nat) -> Nat:
  match xs n:
    case Con{h, t} 0n:
      h
    case Con{h, t} 1n+p:
      nth(t, p)
    case Nil{} _:
      0n

# GPT-2 id of a token: bytes go through perm; merges are 256+k

def gid(+perm: List<&2, Nat>, +t: BPE.Tk) -> Nat:
  match t:
    case BPE.B{+n}:
      nth(perm, n)
    case BPE.M{+k}:
      Nat.add(256n, k)

def gids(+perm: List<&2, Nat>, xs: List<&2, BPE.Tk>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      gid(perm, h) <> gids(perm, t)

# ---- pre-tokenizer: the GPT-2 regex over UTF-8 ----
#   's|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+
# The text is first TAGGED: every byte b becomes b + 256*class + 2048*continuation, where
# the class is the one of its code point (0 letter, 1 digit, 2 other, 3 whitespace; 4 means
# end of list) and continuation marks the 2nd, 3rd and 4th byte of a multibyte character.
# The pre-tokenizer then works on the classes; the tags are removed before the BPE.

def sel(b: Bool, x: Nat, y: Nat) -> Nat:
  match b:
    case True{}:
      x
    case False{}:
      y

def in_range(+c: U32, +lo: U32, +hi: U32) -> Bool:
  Bool.and(U32.is_le(lo, c), U32.is_le(c, hi))

# class of an ASCII byte
def class_ascii(+c: U32) -> Nat:
  sel(Bool.or(in_range(c, 65, 90), in_range(c, 97, 122)), 0n, sel(in_range(c, 48, 57), 1n, sel(Bool.or(Bool.or(U32.is_eq(c, 32), in_range(c, 9, 13)), in_range(c, 28, 31)), 3n, 2n)))

# class of a code point >= 128: the first range of the sorted table that does not end before it
def after_h(+cp: U32, t: List<&2, U32>) -> Bool:
  match t:
    case Con{lo, Con{+hi, rest}}:
      U32.is_lt(hi, cp)
    case _:
      False{}

def inside_h(+cp: U32, t: List<&2, U32>) -> Bool:
  match t:
    case Con{+lo, Con{+hi, rest}}:
      in_range(cp, lo, hi)
    case _:
      False{}

def cls_go(+cp: U32, tbl: List<&2, U32>, aft: Bool, ins: Bool) -> Nat:
  match tbl aft ins:
    case Con{lo, Con{hi, Con{c, +rest}}} True{} _:
      cls_go(cp, rest, after_h(cp, rest), inside_h(cp, rest))
    case Con{lo, Con{hi, Con{+c, rest}}} False{} True{}:
      U32.to_nat(c)
    case _ _ _:
      0n

def cls_cp(+tbl: List<&2, U32>, +cp: U32) -> Nat:
  cls_go(cp, tbl, after_h(cp, tbl), inside_h(cp, tbl))

def class_cp(+tbl: List<&2, U32>, +cp: U32) -> Nat:
  sel(U32.is_lt(cp, 128), class_ascii(cp), cls_cp(tbl, cp))

# length of the UTF-8 character that starts with byte c (1 for stray or invalid bytes)
def clen(+c: U32) -> Nat:
  sel(U32.is_lt(c, 192), 1n, sel(U32.is_lt(c, 224), 2n, sel(U32.is_lt(c, 240), 3n, sel(U32.is_lt(c, 248), 4n, 1n))))

def byte_at(xs: List<&2, U32>, i: Nat) -> U32:
  match xs i:
    case Con{h, t} 0n:
      h
    case Con{h, t} 1n+p:
      byte_at(t, p)
    case Nil{} _:
      0

def cont_bits(+x: U32) -> U32:
  U32.mod(x, 64)

# the code point of the character of n bytes at the start of xs
def cp_of(+n: Nat, +xs: List<&2, U32>) -> U32:
  match n:
    case 2n:
      U32.add(U32.mul(U32.mod(byte_at(xs, 0n), 32), 64), cont_bits(byte_at(xs, 1n)))
    case 3n:
      U32.add(U32.add(U32.mul(U32.mod(byte_at(xs, 0n), 16), 4096), U32.mul(cont_bits(byte_at(xs, 1n)), 64)), cont_bits(byte_at(xs, 2n)))
    case 4n:
      U32.add(U32.add(U32.add(U32.mul(U32.mod(byte_at(xs, 0n), 8), 262144), U32.mul(cont_bits(byte_at(xs, 1n)), 4096)), U32.mul(cont_bits(byte_at(xs, 2n)), 64)), cont_bits(byte_at(xs, 3n)))
    case _:
      byte_at(xs, 0n)

def tagged(+b: U32, +cls: Nat, +cont: Bool) -> U32:
  U32.add(U32.add(b, U32.mul(U32.from_nat(cls), 256)), U32.mul(U32.from_nat(sel(cont, 1n, 0n)), 2048))

# tags the first n bytes of xs (the 1st is not a continuation), then continues with `rest`
def emit(n: Nat, +cls: Nat, +cont: Bool, xs: List<&2, U32>, rest: List<&2, U32>) -> List<&2, U32>:
  match n xs:
    case 1n+p Con{b, t}:
      tagged(b, cls, cont) <> emit(p, cls, True{}, t, rest)
    case _ _:
      rest

def n_of(+xs: List<&2, U32>) -> Nat:
  clen(byte_at(xs, 0n))

def cls_head(+tbl: List<&2, U32>, +xs: List<&2, U32>) -> Nat:
  class_cp(tbl, cp_of(n_of(xs), xs))

# fuel >= number of bytes left; n and cls describe the character at the start of xs
def tag(fuel: Nat, +tbl: List<&2, U32>, +xs: List<&2, U32>, +n: Nat, +cls: Nat) -> List<&2, U32>:
  match fuel xs:
    case 1n+p Con{b, t}:
      emit(n, cls, False{}, xs, tag(p, tbl, List.drop(&2, U32, xs, n), n_of(List.drop(&2, U32, xs, n)), cls_head(tbl, List.drop(&2, U32, xs, n))))
    case _ _:
      Nil{}

def tag_all(+tbl: List<&2, U32>, +xs: List<&2, U32>) -> List<&2, U32>:
  tag(List.length(&2, U32, xs), tbl, xs, n_of(xs), cls_head(tbl, xs))

def class_of(+c: U32) -> Nat:
  U32.to_nat(U32.mod(U32.div(c, 256), 8))

def head_class(xs: List<&2, U32>) -> Nat:
  match xs:
    case Nil{}:
      4n
    case Con{+c, t}:
      class_of(c)

def head_u32(xs: List<&2, U32>) -> U32:
  match xs:
    case Nil{}:
      0
    case Con{c, t}:
      U32.mod(c, 256)

def tail_l(xs: List<&2, U32>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{c, t}:
      t

# length (2 or 3) of the contraction at the start of xs, or 0

def contr2(+q: Bool, +b: U32, +c: U32) -> Nat:
  sel(Bool.and(q, Bool.or(Bool.or(U32.is_eq(b, 115), U32.is_eq(b, 116)), Bool.or(U32.is_eq(b, 109), U32.is_eq(b, 100)))), 2n, sel(Bool.and(q, Bool.or(Bool.or(Bool.and(U32.is_eq(b, 114), U32.is_eq(c, 101)), Bool.and(U32.is_eq(b, 118), U32.is_eq(c, 101))), Bool.and(U32.is_eq(b, 108), U32.is_eq(c, 108)))), 3n, 0n))

def contr_len(+xs: List<&2, U32>) -> Nat:
  contr2(U32.is_eq(head_u32(xs), 39), head_u32(tail_l(xs)), head_u32(tail_l(tail_l(xs))))

# splits off the longest prefix of class `cls`: (prefix, rest)

def span(xs: List<&2, U32>, ok: Bool, +cls: Nat, acc: List<&2, U32>) -> List<&2, U32> & List<&2, U32>:
  match xs ok:
    case Con{c, +t} True{}:
      span(t, Nat.is_eq(head_class(t), cls), cls, c <> acc)
    case _ _:
      (List.reverse(&2, U32, acc), xs)

def span_cls(+xs: List<&2, U32>, +cls: Nat) -> List<&2, U32> & List<&2, U32>:
  span(xs, Nat.is_eq(head_class(xs), cls), cls, Nil{})

def prepend(c: U32, p: List<&2, U32> & List<&2, U32>) -> List<&2, U32> & List<&2, U32>:
  match p:
    case Tuple{a, b}:
      (c <> a, b)

# whitespace: \s+(?!\S) leaves the last space for the next token if something follows

# number of bytes of the last character of xs (1 + trailing continuation bytes)
def lcl(xs: List<&2, U32>, cur: Nat) -> Nat:
  match xs:
    case Nil{}:
      cur
    case Con{+h, t}:
      lcl(t, sel(U32.is_eq(U32.div(h, 2048), 1), Nat.add(1n, cur), 1n))

def ws4(long: Bool, +run: List<&2, U32>, +rest: List<&2, U32>, +n: Nat, +m: Nat) -> List<&2, U32> & List<&2, U32>:
  match long:
    case False{}:
      (run, rest)
    case True{}:
      (List.take(&2, U32, run, Nat.sub(n, m)), List.append(&2, U32, List.drop(&2, U32, run, Nat.sub(n, m)), rest))

def ws3(+run: List<&2, U32>, +rest: List<&2, U32>, ends: Bool, +n: Nat) -> List<&2, U32> & List<&2, U32>:
  match ends:
    case True{}:
      (run, rest)
    case False{}:
      ws4(Nat.is_lt(lcl(run, 0n), n), run, rest, n, lcl(run, 0n))

def ws2(p: List<&2, U32> & List<&2, U32>) -> List<&2, U32> & List<&2, U32>:
  match p:
    case Tuple{+run, +rest}:
      ws3(run, rest, Nat.is_eq(head_class(rest), 4n), List.length(&2, U32, run))

def ws(xs: List<&2, U32>) -> List<&2, U32> & List<&2, U32>:
  ws2(span_cls(xs, 3n))

# 0 contraction, 1 run of its own class, 2 space + run, 3 run of whitespace

def kind_of(n_contr: Nat, cc: Nat, is_sp: Bool, hc: Nat) -> Nat:
  sel(Nat.is_lt(0n, n_contr), 0n, sel(Bool.not(Nat.is_eq(cc, 3n)), 1n, sel(Bool.and(is_sp, Nat.is_lt(hc, 3n)), 2n, 3n)))

def nt(kind: Nat, +xs: List<&2, U32>, +n_contr: Nat) -> List<&2, U32> & List<&2, U32>:
  match kind:
    case 0n:
      (List.take(&2, U32, xs, n_contr), List.drop(&2, U32, xs, n_contr))
    case 1n:
      span_cls(xs, head_class(xs))
    case 2n:
      prepend(head_u32(xs), span_cls(tail_l(xs), head_class(tail_l(xs))))
    case 1n+k:
      ws(xs)

def next_token(+xs: List<&2, U32>) -> List<&2, U32> & List<&2, U32>:
  match xs:
    case Nil{}:
      (Nil{}, Nil{})
    case Con{+c, t}:
      nt(kind_of(contr_len(xs), class_of(c), U32.is_eq(U32.mod(c, 256), 32), head_class(t)), xs, contr_len(xs))

# splits the text into words; `fuel` >= number of tokens; pr = next_token(xs)

def split(fuel: Nat, xs: List<&2, U32>, pr: List<&2, U32> & List<&2, U32>) -> List<&2, List<&2, U32>>:
  match fuel pr:
    case 0n _:
      Nil{}
    case 1n+p Tuple{tok, +rest}:
      match tok:
        case Nil{}:
          Nil{}
        case Con{h, t}:
          Con{h, t} <> split(p, rest, next_token(rest))

def pretok(+xs: List<&2, U32>) -> List<&2, List<&2, U32>>:
  split(List.length(&2, U32, xs), xs, next_token(xs))

# ---- encoding the whole text ----

def nats_of(xs: List<&2, U32>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      U32.to_nat(U32.mod(h, 256)) <> nats_of(t)

def encode_words(+table: List<&2, BPE.Rule>, +perm: List<&2, Nat>, ws: List<&2, List<&2, U32>>) -> List<&2, Nat>:
  match ws:
    case Nil{}:
      Nil{}
    case Con{w, t}:
      List.append(&2, Nat, gids(perm, BPE.encode(table, BPE.lift(nats_of(w)))), encode_words(table, perm, t))

def encode_text(+table: List<&2, BPE.Rule>, +perm: List<&2, Nat>, +bs: List<&2, U32>) -> List<&2, Nat>:
  encode_words(table, perm, pretok(tag_all(U.non_letters(), bs)))

# ---- test CLI: ids of the text in a file ----

def show_nats(xs: List<&2, Nat>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      Nat.show(h) ++ " " ++ show_nats(t)

def go_ok(ok: Bool, +ns: List<&2, Nat>, +pm: List<&2, U32>, tx: List<&2, U32>) -> IO(Unit):
  match ok:
    case True{}:
      IO.print(show_nats(encode_text(rules.go(ns, 0n), nums(pm), tx)))
    case False{}:
      IO.print("invalid merges table")

def go(+m: List<&2, U32>, +pm: List<&2, U32>, tx: List<&2, U32>) -> IO(Unit):
  go_ok(ids_ok(nums(m), 0n), nums(m), pm, tx)

def run3(a: String, b: String, c: String) -> IO(Unit):
  do IO<Unit>:
    m : List<&2, U32> <- read_bytes(a)
    pm : List<&2, U32> <- read_bytes(b)
    tx : List<&2, U32> <- read_bytes(c)
    go(m, pm, tx)

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{a, Con{b, Con{c, Nil{}}}}}:
      run3(a, b, c)
    case _:
      IO.print("usage: tok.bend merges_num.txt perm.txt text")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
