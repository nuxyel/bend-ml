# bend-ml-bpe-tokenizer: BPE byte-level com roundtrip provado.
#
#   import bend-ml-bpe-tokenizer@0.1.0.0/main.bend as BPE
#
# Um token é um byte cru (B{n}) ou o resultado de uma regra de merge (M{id}).
# A tabela é uma lista de regras da mais antiga para a mais nova, como o
# merges.txt do GPT-2: a regra de id k junta o par (a, b) no token M{k}.
# `encode` aplica as regras em ordem; `decode` expande cada token de volta
# para bytes. As leis estão no fim do arquivo, cada uma com sua prova.
#
# Como ler as provas: uma prova em Bend é uma função cujo TIPO é a afirmação.
# `match` faz análise de casos, a chamada recursiva é a hipótese de indução,
# `%e : P` reescreve o objetivo com a igualdade `e`, `{==}` fecha quando os
# dois lados já são o mesmo termo. `+x` marca uma variável que pode ser
# usada mais de uma vez.

import Base
import bend-ml-nat-lemmas@0.1.0.0/main.bend as NL

# Token: um byte cru (B) ou o resultado de uma regra de merge, pelo id da regra (M).
type Tk is Data:
  B{n: Nat}
  M{k: Nat}

# Regra de merge: junta o par (a, b) no token M{id}.
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

# o início de xs é exatamente o par (a, b)?
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

# Troca cada ocorrência (não sobreposta, da esquerda para a direita) do par
# (a, b) pelo token c. `hit` é sempre `peek(a, b, xs)`: o chamador calcula,
# porque um match só abre parâmetros.
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

# aplica as regras da mais antiga para a mais nova; `done` guarda as já aplicadas
# (a mais nova primeiro), que é exatamente a tabela que o decode precisa.
def encode.go(todo: List<&2, Rule>, done: List<&2, Rule>, xs: List<&2, Tk>) -> List<&2, Tk>:
  match todo:
    case Nil{}:
      xs
    case Con{+r, rest}:
      match r:
        case Rule{+i, +a, +b}:
          encode.go(rest, Rule{i, a, b} <> done, merge(a, b, M{i}, xs))

# A tabela vem da regra mais antiga para a mais nova (como o merges.txt).
def encode(+table: List<&2, Rule>, xs: List<&2, Tk>) -> List<&2, Tk>:
  encode.go(table, Nil{}, xs)

# a regra mais nova da tabela é a que cria o token tk?
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

# Expansão de um token em bytes. A tabela tem a regra mais nova primeiro e
# uma regra só se refere a tokens mais antigos, então expandir a e b olha
# apenas o resto da tabela. `hit` é sempre `hits(table, tk)`.
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

# decodifica com a tabela já invertida (mais nova primeiro)
def dec(+table: List<&2, Rule>, ids: List<&2, Tk>) -> List<&2, Nat>:
  match ids:
    case Nil{}:
      Nil{}
    case Con{+x, t}:
      List.append(&2, Nat, exp(table, x), dec(table, t))

def decode(+table: List<&2, Rule>, ids: List<&2, Tk>) -> List<&2, Nat>:
  dec(List.reverse(&2, Rule, table), ids)

# bytes -> tokens iniciais
def lift(bs: List<&2, Nat>) -> List<&2, Tk>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{n, t}:
      B{n} <> lift(t)

# ---- predicados usados nas leis (todos computáveis) ----

# alguma regra da tabela tem o id i?
def has(+D: List<&2, Rule>, +i: Nat) -> Bool:
  match D:
    case Nil{}:
      False{}
    case Con{Rule{+j, a, b}, t}:
      Bool.or(Nat.is_eq(i, j), has(t, i))

# o token é um byte, ou foi criado por uma regra da tabela?
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

# tabela bem formada: nenhum id se repete
def wf.go(todo: List<&2, Rule>, +done: List<&2, Rule>) -> Bool:
  match todo:
    case Nil{}:
      True{}
    case Con{Rule{+i, +a, +b}, rest}:
      Bool.and(Bool.not(has(done, i)), wf.go(rest, Rule{i, a, b} <> done))

def wf(table: List<&2, Rule>) -> Bool:
  wf.go(table, Nil{})

# =====================================================================
# Treino (não precisa de prova: gera tabelas; a tabela vale se wf(table))
# =====================================================================
# ---------------------------------------------------------------
# treino: conta pares adjacentes, funde o mais frequente, repete
# (desempate: o primeiro par que apareceu, como no minbpe)
# ---------------------------------------------------------------
type Cnt is Data:
  Cnt{a: Tk, b: Tk, n: Nat}

# o primeiro Cnt de cs é o par (a, b)?
def same_head(cs: List<&2, Cnt>, +a: Tk, +b: Tk) -> Bool:
  match cs:
    case Nil{}:
      False{}
    case Con{Cnt{+x, +y, k}, rest}:
      Bool.and(Tk.eq(x, a), Tk.eq(y, b))

# soma 1 ao contador do par (a, b), ou o cria no fim da lista
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

# até n fusões, ids a partir de `next`; m é o melhor par atual (best(count(xs)))
def train.go(n: Nat, +next: Nat, +xs: List<&2, Tk>, m: Maybe<&1, Cnt>) -> List<&2, Rule>:
  match n m:
    case 0n _:
      Nil{}
    case 1n+p None{}:
      Nil{}
    case 1n+p Some{Cnt{+a, +b, k}}:
      +ys = merge(a, b, M{next}, xs)
      Rule{next, a, b} <> train.go(p, 1n+next, ys, best(count.go(ys, Nil{})))

# a tabela da regra mais antiga para a mais nova
def train(n: Nat, +next: Nat, +xs: List<&2, Tk>) -> List<&2, Rule>:
  train.go(n, next, xs, best(count.go(xs, Nil{})))

# =====================================================================
# Lemas auxiliares das provas
# =====================================================================

# um tipo escolhido por um Bool: reescrever através dele refuta True == False
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

# not p é verdadeiro  ->  p é falso
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

# Acrescentar a regra i (nova) ao topo da tabela não muda a expansão de um
# token M{j} que a tabela antiga já conhece (j != i, porque i é novo).
def ext_m(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk, +j: Nat, h: Bool, eh: {h == Nat.is_eq(i, j) : Bool}, nf: {True{} == Bool.not(has(D, i)) : Bool}, kt: {True{} == has(D, j) : Bool}) -> {exp(Rule{i, a, b} <> D, M{j}) == exp(D, M{j}) : List<&2, Nat>}:
  match h:
    case True{}:
      # h = True quer dizer i == j; mas j é conhecido (has D j) e i não (not has D i)
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

# o mesmo para qualquer token conhecido (um byte nunca depende da tabela)
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

# decodificar uma lista de tokens conhecidos não muda ao acrescentar a regra nova
def ext_dec(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk, xs: List<&2, Tk>, +kx: {True{} == knl(D, xs) : Bool}, +nf: {True{} == Bool.not(has(D, i)) : Bool}) -> {dec(Rule{i, a, b} <> D, xs) == dec(D, xs) : List<&2, Nat>}:
  match xs:
    case Nil{}:
      {==}
    case Con{+x, +t}:
      ext_dec.step(D, i, a, b, x, t, and_true_l(kn(D, x), knl(D, t), kx), nf, ext_dec(D, i, a, b, t, and_true_r(kn(D, x), knl(D, t), kx), nf))

# O token novo M{i} expande para a expansão de a seguida da de b
# (olhando só a tabela antiga D).
def exp_new(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk, h: Bool, eh: {h == Nat.is_eq(i, i) : Bool}) -> {exp(Rule{i, a, b} <> D, M{i}) == List.append(&2, Nat, exp(D, a), exp(D, b)) : List<&2, Nat>}:
  match h:
    case True{}:
      %eh : {exp.go(Rule{i, a, b} <> D, _, M{i}) == List.append(&2, Nat, exp(D, a), exp(D, b)) : List<&2, Nat>}
      {==}
    case False{}:
      +f = Equal.trans(Bool, True{}, Nat.is_eq(i, i), False{}, nat_eq_refl(i), Equal.sym(Bool, False{}, Nat.is_eq(i, i), eh))
      %f : BD(_, Unit, {exp(Rule{i, a, b} <> D, M{i}) == List.append(&2, Nat, exp(D, a), exp(D, b)) : List<&2, Nat>})
      Unit{}

# Passo central: no lugar de a b t aparece M{i} seguido de m, onde m decodifica
# como t. Os dois lados decodificam igual porque M{i} expande para exp(a) ++ exp(b).
def mp_core(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk, +t: List<&2, Tk>, +m: List<&2, Tk>, ka: {True{} == kn(D, a) : Bool}, kb: {True{} == kn(D, b) : Bool}, +nf: {True{} == Bool.not(has(D, i)) : Bool}, ihm: {dec(Rule{i, a, b} <> D, m) == dec(Rule{i, a, b} <> D, t) : List<&2, Nat>}) -> {dec(Rule{i, a, b} <> D, M{i} <> m) == dec(Rule{i, a, b} <> D, a <> b <> t) : List<&2, Nat>}:
  %Equal.sym(List<&2, Nat>, exp(Rule{i, a, b} <> D, M{i}), List.append(&2, Nat, exp(D, a), exp(D, b)), exp_new(D, i, a, b, Nat.is_eq(i, i), {==})) : {List.append(&2, Nat, _, dec(Rule{i, a, b} <> D, m)) == List.append(&2, Nat, exp(Rule{i, a, b} <> D, a), List.append(&2, Nat, exp(Rule{i, a, b} <> D, b), dec(Rule{i, a, b} <> D, t))) : List<&2, Nat>}
  %Equal.sym(List<&2, Nat>, dec(Rule{i, a, b} <> D, m), dec(Rule{i, a, b} <> D, t), ihm) : {List.append(&2, Nat, List.append(&2, Nat, exp(D, a), exp(D, b)), _) == List.append(&2, Nat, exp(Rule{i, a, b} <> D, a), List.append(&2, Nat, exp(Rule{i, a, b} <> D, b), dec(Rule{i, a, b} <> D, t))) : List<&2, Nat>}
  %Equal.sym(List<&2, Nat>, exp(Rule{i, a, b} <> D, a), exp(D, a), ext_tok(D, i, a, b, a, ka, nf)) : {List.append(&2, Nat, List.append(&2, Nat, exp(D, a), exp(D, b)), dec(Rule{i, a, b} <> D, t)) == List.append(&2, Nat, _, List.append(&2, Nat, exp(Rule{i, a, b} <> D, b), dec(Rule{i, a, b} <> D, t))) : List<&2, Nat>}
  %Equal.sym(List<&2, Nat>, exp(Rule{i, a, b} <> D, b), exp(D, b), ext_tok(D, i, a, b, b, kb, nf)) : {List.append(&2, Nat, List.append(&2, Nat, exp(D, a), exp(D, b)), dec(Rule{i, a, b} <> D, t)) == List.append(&2, Nat, exp(D, a), List.append(&2, Nat, _, dec(Rule{i, a, b} <> D, t))) : List<&2, Nat>}
  NL.append_assoc(Nat, exp(D, a), exp(D, b), dec(Rule{i, a, b} <> D, t))

# se x == a e x é conhecido, a também é
def kn_sub(+D: List<&2, Rule>, +x: Tk, +a: Tk, ex: {x == a : Tk}, kx: {True{} == kn(D, x) : Bool}) -> {True{} == kn(D, a) : Bool}:
  %ex : {True{} == kn(D, _) : Bool}
  kx

# x fica de fora do rewrite: se o resto decodifica igual, x <> resto também
def mp_false(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk, +x: Tk, +t: List<&2, Tk>, +mg: List<&2, Tk>, ih: {dec(Rule{i, a, b} <> D, mg) == dec(Rule{i, a, b} <> D, t) : List<&2, Nat>}) -> {dec(Rule{i, a, b} <> D, x <> mg) == dec(Rule{i, a, b} <> D, x <> t) : List<&2, Nat>}:
  %ih : {List.append(&2, Nat, exp(Rule{i, a, b} <> D, x), dec(Rule{i, a, b} <> D, mg)) == List.append(&2, Nat, exp(Rule{i, a, b} <> D, x), _) : List<&2, Nat>}
  {==}

# o par é x y: os tokens eram iguais a a e b
def mp_hit(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk, +x: Tk, +y: Tk, +t: List<&2, Tk>, +m: List<&2, Tk>, +ex: {x == a : Tk}, +ey: {y == b : Tk}, +kxx: {True{} == kn(D, x) : Bool}, +kyy: {True{} == kn(D, y) : Bool}, +nf: {True{} == Bool.not(has(D, i)) : Bool}, ihm: {dec(Rule{i, a, b} <> D, m) == dec(Rule{i, a, b} <> D, t) : List<&2, Nat>}) -> {dec(Rule{i, a, b} <> D, M{i} <> m) == dec(Rule{i, a, b} <> D, x <> y <> t) : List<&2, Nat>}:
  %Equal.sym(Tk, x, a, ex) : {dec(Rule{i, a, b} <> D, M{i} <> m) == dec(Rule{i, a, b} <> D, _ <> y <> t) : List<&2, Nat>}
  %Equal.sym(Tk, y, b, ey) : {dec(Rule{i, a, b} <> D, M{i} <> m) == dec(Rule{i, a, b} <> D, a <> _ <> t) : List<&2, Nat>}
  mp_core(D, i, a, b, t, m, kn_sub(D, x, a, ex, kxx), kn_sub(D, y, b, ey, kyy), nf, ihm)

# merge.go não muda o que o decode devolve (com a regra nova no topo da tabela)
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

# ---- tokens conhecidos continuam conhecidos ----

# acrescentar uma regra à tabela só aumenta o que é conhecido
def kn_mono(+D: List<&2, Rule>, +i: Nat, +a: Tk, +b: Tk, t: Tk, kt: {True{} == kn(D, t) : Bool}) -> {True{} == kn(Rule{i, a, b} <> D, t) : Bool}:
  match t:
    case B{n}:
      {==}
    case M{+j}:
      or_right(Nat.is_eq(j, i), has(D, j), kt)

# o token criado pela regra nova é conhecido
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

# ---- indução sobre as regras ----

# todo token emitido é conhecido pela tabela completa (invertida)
def enc_known(todo: List<&2, Rule>, +done: List<&2, Rule>, +xs: List<&2, Tk>, +wfe: {True{} == wf.go(todo, done) : Bool}, +kx: {True{} == knl(done, xs) : Bool}) -> {True{} == knl(List.reverse.go(&2, Rule, todo, done), encode.go(todo, done, xs)) : Bool}:
  match todo:
    case Nil{}:
      kx
    case Con{Rule{+i, +a, +b}, +rest}:
      enc_known(rest, Rule{i, a, b} <> done, merge.go(xs, peek(a, b, xs), a, b, M{i}), and_true_r(Bool.not(has(done, i)), wf.go(rest, Rule{i, a, b} <> done), wfe), kp(done, i, a, b, xs, peek(a, b, xs), {==}, kx))

# aplicar todas as regras não muda o que o decode devolve
def enc_dec(todo: List<&2, Rule>, +done: List<&2, Rule>, +xs: List<&2, Tk>, +wfe: {True{} == wf.go(todo, done) : Bool}, +kx: {True{} == knl(done, xs) : Bool}) -> {dec(List.reverse.go(&2, Rule, todo, done), encode.go(todo, done, xs)) == dec(done, xs) : List<&2, Nat>}:
  match todo:
    case Nil{}:
      {==}
    case Con{Rule{+i, +a, +b}, +rest}:
      Equal.trans(List<&2, Nat>, dec(List.reverse.go(&2, Rule, rest, Rule{i, a, b} <> done), encode.go(rest, Rule{i, a, b} <> done, merge.go(xs, peek(a, b, xs), a, b, M{i}))), dec(Rule{i, a, b} <> done, xs), dec(done, xs), Equal.trans(List<&2, Nat>, dec(List.reverse.go(&2, Rule, rest, Rule{i, a, b} <> done), encode.go(rest, Rule{i, a, b} <> done, merge.go(xs, peek(a, b, xs), a, b, M{i}))), dec(Rule{i, a, b} <> done, merge.go(xs, peek(a, b, xs), a, b, M{i})), dec(Rule{i, a, b} <> done, xs), enc_dec(rest, Rule{i, a, b} <> done, merge.go(xs, peek(a, b, xs), a, b, M{i}), and_true_r(Bool.not(has(done, i)), wf.go(rest, Rule{i, a, b} <> done), wfe), kp(done, i, a, b, xs, peek(a, b, xs), {==}, kx)), mp(done, i, a, b, xs, peek(a, b, xs), {==}, kx, and_true_l(Bool.not(has(done, i)), wf.go(rest, Rule{i, a, b} <> done), wfe))), ext_dec(done, i, a, b, xs, kx, and_true_l(Bool.not(has(done, i)), wf.go(rest, Rule{i, a, b} <> done), wfe)))

# ---- caso base e leis finais ----

# sem regras, os bytes decodificam para eles mesmos
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
# LAWS (as leis do pacote, em português no README)
# =====================================================================

# LEI 1 (roundtrip): com uma tabela bem formada (ids de regra todos
# diferentes), decodificar o que foi codificado devolve os bytes originais.
law roundtrip:
  for +table: List<&2, Rule>
  for +bs: List<&2, Nat>
  for +w: {True{} == wf(table) : Bool}
  {decode(table, encode(table, lift(bs))) == bs : List<&2, Nat>}

def roundtrip(table, bs, w):
  Equal.trans(List<&2, Nat>, decode(table, encode(table, lift(bs))), dec(Nil{}, lift(bs)), bs, enc_dec(table, Nil{}, lift(bs), w, lift_known(bs)), lift_dec(bs))

# LEI 2 (vocabulário): todo token emitido por encode é um byte cru ou foi
# criado por uma das regras da tabela; nunca aparece um id desconhecido.
law vocab_bound:
  for +table: List<&2, Rule>
  for +bs: List<&2, Nat>
  for +w: {True{} == wf(table) : Bool}
  {True{} == knl(List.reverse(&2, Rule, table), encode(table, lift(bs))) : Bool}

def vocab_bound(table, bs, w):
  enc_known(table, Nil{}, lift(bs), w, lift_known(bs))

# LEI 3 (concatenação): decodificar duas listas de tokens juntas é decodificar
# cada uma e juntar os bytes.
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
