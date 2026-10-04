# Tokenizer do GPT-2 em Bend: pré-tokenizador (a regex do GPT-2 sobre bytes) + as 50 000
# regras de merge, aplicadas com o pacote bend-ml-bpe-tokenizer (roundtrip provado).
#
# Limite conhecido: o pré-tokenizador classifica bytes ASCII exatamente; todo byte >= 128
# (UTF-8 multibyte) conta como LETRA. Isso casa com a regex para letras acentuadas e de
# outros alfabetos, mas não separa símbolos Unicode que não são letras (travessão, emoji).

import Base
import bend-ml-bpe-tokenizer@0.1.0.0/main.bend as BPE

# ---- leitura de arquivos ----

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

# id numérico no esquema do bpe/ (byte = 0..255, fusão k = 256+k) -> token

def id_tok(+n: Nat) -> BPE.Tk:
  id_tok2(Nat.is_lt(n, 256n), n)

# ---- inteiros em texto: qualquer byte que não é dígito separa ----

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

# pares de ids viram regras com id sequencial

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

def toks_of_ids(ns: List<&2, Nat>) -> List<&2, BPE.Tk>:
  match ns:
    case Nil{}:
      Nil{}
    case Con{n, t}:
      id_tok(n) <> toks_of_ids(t)


# ---- tabela e permutação de ids base do GPT-2 ----

def nth(xs: List<&2, Nat>, n: Nat) -> Nat:
  match xs n:
    case Con{h, t} 0n:
      h
    case Con{h, t} 1n+p:
      nth(t, p)
    case Nil{} _:
      0n

# id GPT-2 de um token: bytes passam por perm; fusões são 256+k

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

# ---- pré-tokenizador: a regex do GPT-2 sobre bytes ----
#   's|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+
# classes: 0 letra (ASCII ou byte >= 128), 1 dígito, 2 outro, 3 espaço, 4 fim da lista

def sel(b: Bool, x: Nat, y: Nat) -> Nat:
  match b:
    case True{}:
      x
    case False{}:
      y

def in_range(+c: U32, +lo: U32, +hi: U32) -> Bool:
  Bool.and(U32.is_le(lo, c), U32.is_le(c, hi))

def class_of(+c: U32) -> Nat:
  sel(U32.is_le(128, c), 0n, sel(Bool.or(in_range(c, 65, 90), in_range(c, 97, 122)), 0n, sel(in_range(c, 48, 57), 1n, sel(Bool.or(U32.is_eq(c, 32), in_range(c, 9, 13)), 3n, 2n))))

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
      c

def tail_l(xs: List<&2, U32>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{c, t}:
      t

# tamanho (2 ou 3) da contração no início de xs, ou 0

def contr2(+q: Bool, +b: U32, +c: U32) -> Nat:
  sel(Bool.and(q, Bool.or(Bool.or(U32.is_eq(b, 115), U32.is_eq(b, 116)), Bool.or(U32.is_eq(b, 109), U32.is_eq(b, 100)))), 2n, sel(Bool.and(q, Bool.or(Bool.or(Bool.and(U32.is_eq(b, 114), U32.is_eq(c, 101)), Bool.and(U32.is_eq(b, 118), U32.is_eq(c, 101))), Bool.and(U32.is_eq(b, 108), U32.is_eq(c, 108)))), 3n, 0n))

def contr_len(+xs: List<&2, U32>) -> Nat:
  contr2(U32.is_eq(head_u32(xs), 39), head_u32(tail_l(xs)), head_u32(tail_l(tail_l(xs))))

# separa o maior prefixo da classe `cls`: (prefixo, resto)

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

# espaços: \s+(?!\S) deixa o último espaço para o próximo token se vier algo depois

def ws4(long: Bool, +run: List<&2, U32>, +rest: List<&2, U32>, +n: Nat) -> List<&2, U32> & List<&2, U32>:
  match long:
    case False{}:
      (run, rest)
    case True{}:
      (List.take(&2, U32, run, Nat.sub(n, 1n)), List.append(&2, U32, List.drop(&2, U32, run, Nat.sub(n, 1n)), rest))

def ws3(+run: List<&2, U32>, +rest: List<&2, U32>, ends: Bool, +n: Nat) -> List<&2, U32> & List<&2, U32>:
  match ends:
    case True{}:
      (run, rest)
    case False{}:
      ws4(Nat.is_lt(1n, n), run, rest, n)

def ws2(p: List<&2, U32> & List<&2, U32>) -> List<&2, U32> & List<&2, U32>:
  match p:
    case Tuple{+run, +rest}:
      ws3(run, rest, Nat.is_eq(head_class(rest), 4n), List.length(&2, U32, run))

def ws(xs: List<&2, U32>) -> List<&2, U32> & List<&2, U32>:
  ws2(span_cls(xs, 3n))

# 0 contração, 1 corrida da própria classe, 2 espaço + corrida, 3 corrida de espaços

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
      nt(kind_of(contr_len(xs), class_of(c), U32.is_eq(c, 32), head_class(t)), xs, contr_len(xs))

# parte o texto em palavras; `fuel` >= número de tokens; pr = next_token(xs)

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

# ---- codificação do texto inteiro ----

def nats_of(xs: List<&2, U32>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      U32.to_nat(h) <> nats_of(t)

def encode_words(+table: List<&2, BPE.Rule>, +perm: List<&2, Nat>, ws: List<&2, List<&2, U32>>) -> List<&2, Nat>:
  match ws:
    case Nil{}:
      Nil{}
    case Con{w, t}:
      List.append(&2, Nat, gids(perm, BPE.encode(table, BPE.lift(nats_of(w)))), encode_words(table, perm, t))

def encode_text(+table: List<&2, BPE.Rule>, +perm: List<&2, Nat>, bs: List<&2, U32>) -> List<&2, Nat>:
  encode_words(table, perm, pretok(bs))

# ---- CLI de teste: ids do texto em um arquivo ----

def show_nats(xs: List<&2, Nat>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      Nat.show(h) ++ " " ++ show_nats(t)

def go(+m: List<&2, U32>, +pm: List<&2, U32>, tx: List<&2, U32>) -> IO(Unit):
  IO.print(show_nats(encode_text(rules.go(nums(m), 0n), nums(pm), tx)))

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
      IO.print("uso: tok.bend merges_num.txt perm.txt texto")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
