# bend-ml-tensor-array: tensores sobre Array<F32> plano, com a shape no tipo.
#
#   import bend-ml-tensor-array@0.1.0.0/main.bend as TA
#
# Mat<r, c> guarda r*c números num Array<F32> linha a linha (índice i*c + j). As
# dimensões são parâmetros de tipo apagados: a mesma garantia do bend-ml-tensor
# (produto com dimensões erradas não compila), mas ~50x mais rápido que listas:
# Array.get/set por índice, e produtos em blocos de linhas em paralelo.
#
# Um Array é afim (um só dono): toda operação que lê um operando o devolve junto
# com o resultado (como Array.get faz), em registros MM, AR, ... Use Mat.clone
# quando precisar de duas cópias.
#
# O que NÃO é provado: a invariante "o Array tem capacidade >= r*c" vale porque
# os construtores (Mat.zeros, Mat.of_list) alocam o tamanho certo; e F32 é
# validado por testes contra o PyTorch (reference/test_tensor_array.py), não por prova.

import Base

# ---------------------------------------------------------------------
# tipos de apoio
# ---------------------------------------------------------------------

type R is Type:
  R{a: Array<F32>, b: Array<F32>, x: F32}

type G is Type:
  G{a: Array<F32>, b: Array<F32>, c: Array<F32>}

# menor d com 2^d >= n (n >= 2)

def dotg(k: Nat, +ia: U32, +ib: U32, +sa: U32, +sb: U32, acc: F32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> R:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}:
      R{a, b, acc}
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      dotg(p, (ia + sa : U32), (ib + sb : U32), sa, sb, (acc + (x * y : F32) : F32), Array.get(F32, a, (ia + sa : U32)), Array.get(F32, b, (ib + sb : U32)))

# a linha i de C, colunas j.. ; r é o produto da coluna atual (já calculado)

def gcols(mleft: Nat, +kn: Nat, +ia0: U32, +ib0: U32, +sa: U32, +sb: U32, +bcol: U32, +cidx: U32, c: Array<F32>, r: R) -> G:
  match mleft r:
    case 0n R{a, b, +x}:
      G{a, b, Array.set(F32, c, cidx, x)}
    case 1n+p R{a, b, +x}:
      gcols(p, kn, ia0, (ib0 + bcol : U32), sa, sb, bcol, (cidx + 1 : U32), Array.set(F32, c, cidx, x), dotg(kn, ia0, (ib0 + bcol : U32), sa, sb, 0.0, Array.get(F32, a, ia0), Array.get(F32, b, (ib0 + bcol : U32))))

def grows(nleft: Nat, +mm1: Nat, +kn: Nat, +i: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +mu: U32, g: G) -> G:
  match nleft g:
    case 0n G{a, b, c}:
      G{a, b, c}
    case 1n+p G{a, b, c}:
      grows(p, mm1, kn, (i + 1 : U32), arow, sa, sb, bcol, mu, gcols(mm1, kn, (i * arow : U32), 0, sa, sb, bcol, (i * mu : U32), c, dotg(kn, (i * arow : U32), 0, sa, sb, 0.0, Array.get(F32, a, (i * arow : U32)), Array.get(F32, b, 0))))

# C (n x m) = A · B con k passos; c é o array de saída (com capacidade n*m)

def gemm(+n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  grows(n, Nat.sub(m, 1n), k, 0, arow, sa, sb, bcol, U32.from_nat(m), G{a, b, c})


# ---------------------------------------------------------------------
# gemm paralelo: 2^d blocos de linhas de C, cada um com a sua cópia de A e B,
# devolvendo listas que são escritas em C no fim
# ---------------------------------------------------------------------

type S3 is Type:
  S3{a: Array<F32>, b: Array<F32>, out: List<&2, F32>}

def lcols(mleft: Nat, +kn: Nat, +ia0: U32, +ib0: U32, +sa: U32, +sb: U32, +bcol: U32, out: List<&2, F32>, r: R) -> S3:
  match mleft r:
    case 0n R{a, b, +x}:
      S3{a, b, x <> out}
    case 1n+p R{a, b, +x}:
      lcols(p, kn, ia0, (ib0 + bcol : U32), sa, sb, bcol, x <> out, dotg(kn, ia0, (ib0 + bcol : U32), sa, sb, 0.0, Array.get(F32, a, ia0), Array.get(F32, b, (ib0 + bcol : U32))))

def lrows(nleft: Nat, +mm1: Nat, +kn: Nat, +i: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, st: S3) -> S3:
  match nleft st:
    case 0n S3{a, b, out}:
      S3{a, b, out}
    case 1n+p S3{a, b, out}:
      lrows(p, mm1, kn, (i + 1 : U32), arow, sa, sb, bcol, lcols(mm1, kn, (i * arow : U32), 0, sa, sb, bcol, out, dotg(kn, (i * arow : U32), 0, sa, sb, 0.0, Array.get(F32, a, (i * arow : U32)), Array.get(F32, b, 0))))

def lblock_fin(st: S3) -> List<&2, F32>:
  match st:
    case S3{a, b, out}:
      List.reverse(&2, F32, out)

def lblock(cnt: Nat, +mm1: Nat, +kn: Nat, +i: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>) -> List<&2, F32>:
  lblock_fin(lrows(cnt, mm1, kn, i, arow, sa, sb, bcol, S3{a, b, Nil{}}))

def lpar(d: Nat, +cnt: Nat, +mm1: Nat, +kn: Nat, +i: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, ra: Array<F32> & Array<F32>, rb: Array<F32> & Array<F32>) -> List<&2, F32>:
  match d ra rb:
    case 0n Tuple{a, a2} Tuple{b, b2}:
      lblock(cnt, mm1, kn, i, arow, sa, sb, bcol, a, b)
    case 1n++q Tuple{a, a2} Tuple{b, b2}:
      x y = lpar(q, Nat.div(cnt, 2n), mm1, kn, i, arow, sa, sb, bcol, Array.clone(F32, a), Array.clone(F32, b)) lpar(q, Nat.sub(cnt, Nat.div(cnt, 2n)), mm1, kn, (i + U32.from_nat(Nat.div(cnt, 2n)) : U32), arow, sa, sb, bcol, Array.clone(F32, a2), Array.clone(F32, b2))
      List.append(&2, F32, x, y)

def write_l(xs: List<&2, F32>, +idx: U32, c: Array<F32>) -> Array<F32>:
  match xs:
    case Nil{}:
      c
    case Con{h, t}:
      write_l(t, (idx + 1 : U32), Array.set(F32, c, idx, h))

# c[idx..] += lista

def gp3(a: Array<F32>, b: Array<F32>, c: Array<F32>, xs: List<&2, F32>) -> G:
  G{a, b, write_l(xs, 0, c)}

def gp2(d: Nat, +n: Nat, +mm1: Nat, +kn: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, c: Array<F32>, pa: Array<F32> & Array<F32>, pb: Array<F32> & Array<F32>) -> G:
  match pa pb:
    case Tuple{a1, a2} Tuple{b1, b2}:
      gp3(a1, b1, c, lpar(d, n, mm1, kn, 0, arow, sa, sb, bcol, Array.clone(F32, a2), Array.clone(F32, b2)))

# C (n x m) = A · B, com 2^d blocos de linhas em paralelo; devolve A, B (intactos) e C

def gemm_par(d: Nat, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  gp2(d, n, Nat.sub(m, 1n), k, arow, sa, sb, bcol, c, Array.clone(F32, a), Array.clone(F32, b))

# ---------------------------------------------------------------------
# operações elementwise (em lugar, por índice)
# ---------------------------------------------------------------------

def depth(n: Nat) -> Nat:
  Nat.add(U32.log2(U32.from_nat(Nat.sub(n, 1n))), 1n)

# ---------------------------------------------------------------------
# gemm: C[i][j] = soma_p A[i*arow + p*sa] * B[p*sb + j*bcol]
# (nn: arow=K, sa=1, sb=M, bcol=1 | nt: B guardada m x k: sb=1, bcol=K | tn: A guardada k x n: arow=1, sa=N)
# ---------------------------------------------------------------------

# acc += A[ia + t*sa] * B[ib + t*sb] por k passos; ra, rb já trazem o primeiro par

type P2 is Type:
  P2{a: Array<F32>, b: Array<F32>}

type AL is Type:
  AL{c: Array<F32>, l: List<&2, F32>}

# lista de m elementos a partir de idx (lê com Array.get, devolvendo o array)

def read_l(mleft: Nat, +idx: U32, acc: List<&2, F32>, rc: Array<F32> & F32) -> AL:
  match mleft rc:
    case 0n Tuple{c, +v}:
      AL{c, List.reverse(&2, F32, v <> acc)}
    case 1n+p Tuple{c, +v}:
      read_l(p, (idx + 1 : U32), v <> acc, Array.get(F32, c, (idx + 1 : U32)))

# escreve a lista a partir de idx

def addl(xs: List<&2, F32>, +idx: U32, rc: Array<F32> & F32) -> Array<F32>:
  match xs rc:
    case Nil{} Tuple{c, v}:
      c
    case Con{+h, t} Tuple{c, +v}:
      addl(t, (idx + 1 : U32), Array.get(F32, Array.set(F32, c, idx, (v + h : F32)), (idx + 1 : U32)))

# soma bias (lista de m) a cada uma das n linhas de c (n x m)

def bias_rows(nleft: Nat, +i: U32, +mu: U32, +bl: List<&2, F32>, c: Array<F32>) -> Array<F32>:
  match nleft:
    case 0n:
      c
    case 1n+p:
      bias_rows(p, (i + 1 : U32), mu, bl, addl(bl, (i * mu : U32), Array.get(F32, c, (i * mu : U32))))

# relu em lugar nos primeiros n elementos

def relu_l(nleft: Nat, +t: U32, rc: Array<F32> & F32) -> Array<F32>:
  match nleft rc:
    case 0n Tuple{c, v}:
      c
    case 1n+p Tuple{c, +v}:
      relu_l(p, (t + 1 : U32), Array.get(F32, Array.set(F32, c, t, F32.max(v, 0.0)), (t + 1 : U32)))

def pos2(b: Bool) -> F32:
  match b:
    case True{}:
      1.0
    case False{}:
      0.0

def pos(+y: F32) -> F32:
  pos2(F32.is_gt(y, 0.0))

# d[t] *= (h[t] > 0): gradiente da relu; devolve os dois arrays

def mask_l(nleft: Nat, +t: U32, rd: Array<F32> & F32, rh: Array<F32> & F32) -> P2:
  match nleft rd rh:
    case 0n Tuple{d, x} Tuple{h, y}:
      P2{d, h}
    case 1n+p Tuple{d, +x} Tuple{h, +y}:
      mask_l(p, (t + 1 : U32), Array.get(F32, Array.set(F32, d, t, (x * pos(y) : F32)), (t + 1 : U32)), Array.get(F32, h, (t + 1 : U32)))

# w[t] -= lr * dw[t]

def maxl2(xs: List<&2, F32>, cur: F32) -> F32:
  match xs:
    case Nil{}:
      cur
    case Con{+h, t}:
      maxl2(t, F32.max(cur, h))

def maxl(xs: List<&2, F32>) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{+h, t}:
      maxl2(t, h)

def expl(xs: List<&2, F32>, +m: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      F32.exp((h - m : F32)) <> expl(t, m)

def suml(xs: List<&2, F32>, acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{+h, t}:
      suml(t, (acc + h : F32))

def divl(xs: List<&2, F32>, +s: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      (h / s : F32) <> divl(t, s)

def softmax2(+es: List<&2, F32>) -> List<&2, F32>:
  divl(es, suml(es, 0.0))

def softmax(+xs: List<&2, F32>) -> List<&2, F32>:
  softmax2(expl(xs, maxl(xs)))

# p[label]

def sgd_l(nleft: Nat, +t: U32, +lr: F32, rw: Array<F32> & F32, rd: Array<F32> & F32) -> P2:
  match nleft rw rd:
    case 0n Tuple{w, x} Tuple{d, y}:
      P2{w, d}
    case 1n+p Tuple{w, +x} Tuple{d, +y}:
      sgd_l(p, (t + 1 : U32), lr, Array.get(F32, Array.set(F32, w, t, (x - (lr * y : F32) : F32)), (t + 1 : U32)), Array.get(F32, d, (t + 1 : U32)))

# ---------------------------------------------------------------------
# softmax + entropia cruzada por linha (listas de 10 elementos: custo desprezível)
# ---------------------------------------------------------------------

def pick(ps: List<&2, F32>, n: Nat) -> F32:
  match ps n:
    case Con{+h, t} 0n:
      h
    case Con{h, t} 1n+q:
      pick(t, q)
    case Nil{} _:
      0.0

# (p - onehot(label)) / n

def hit(b: Bool) -> F32:
  match b:
    case True{}:
      1.0
    case False{}:
      0.0

def grad_l(ps: List<&2, F32>, +label: Nat, +j: Nat, +nf: F32) -> List<&2, F32>:
  match ps:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      (((h - hit(Nat.is_eq(label, j)) : F32) / nf) : F32) <> grad_l(t, label, 1n+j, nf)

type L2 is Type:
  L2{loss: F32, c: Array<F32>}

def ce_row2(+ps: List<&2, F32>, +label: Nat, +nf: F32, +idx: U32, loss: F32, c: Array<F32>) -> L2:
  L2{(loss - F32.log(F32.max(pick(ps, label), 0.000000000001)) : F32), write_l(grad_l(ps, label, 0n, nf), idx, c)}

def ce_row(al: AL, +label: Nat, +nf: F32, +idx: U32, loss: F32) -> L2:
  match al:
    case AL{c, l}:
      ce_row2(softmax(l), label, nf, idx, loss, c)

# percorre as linhas de z2 (n x m): r traz a perda acumulada e o array; ao fim, o array virou o gradiente

type Mat<-r: Nat, -c: Nat> is Type:
  Mat{d: Array<F32>}

# menor d com 2^d >= n (n >= 2); n < 2 vale 0

def ce_rows(labels: List<&2, Nat>, +mm1: Nat, +mu: U32, +nf: F32, +i: U32, r: L2) -> L2:
  match labels r:
    case Nil{} L2{loss, c}:
      L2{loss, c}
    case Con{+y, t} L2{loss, c}:
      ce_rows(t, mm1, mu, nf, (i + 1 : U32), ce_row(read_l(mm1, (i * mu : U32), Nil{}, Array.get(F32, c, (i * mu : U32))), y, nf, (i * mu : U32), loss))

# ---- apoio: acertos por linha ----

type CR is Type:
  CR{c: Array<F32>, n: Nat}

def gt_head(xs: List<&2, F32>, +v: F32) -> Bool:
  match xs:
    case Nil{}:
      False{}
    case Con{+h, t}:
      F32.is_gt(h, v)

def argmax2(xs: List<&2, F32>, g: Bool, +i: Nat, +bi: Nat, +bv: F32) -> Nat:
  match xs g:
    case Nil{} _:
      bi
    case Con{+h, +t} True{}:
      argmax2(t, gt_head(t, h), 1n+i, i, h)
    case Con{h, +t} False{}:
      argmax2(t, gt_head(t, bv), 1n+i, bi, bv)

def argmax(xs: List<&2, F32>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case Con{+h, +t}:
      argmax2(t, gt_head(t, h), 1n, 0n, h)

def one(b: Bool) -> Nat:
  match b:
    case True{}:
      1n
    case False{}:
      0n

def count_rows3(+y: Nat, n: Nat, al: AL) -> CR:
  match al:
    case AL{c, l}:
      CR{c, Nat.add(n, one(Nat.is_eq(argmax(l), y)))}

def count_rows(labels: List<&2, Nat>, +mm1: Nat, +mu: U32, +i: U32, r: CR) -> CR:
  match labels r:
    case Nil{} CR{c, n}:
      CR{c, n}
    case Con{+y, t} CR{c, n}:
      count_rows(t, mm1, mu, (i + 1 : U32), count_rows3(y, n, read_l(mm1, (i * mu : U32), Nil{}, Array.get(F32, c, (i * mu : U32)))))


# =====================================================================
# API tipada: Mat<r, c> (as dimensões entram no tipo)
# =====================================================================

def cap_depth2(small: Bool, n: Nat) -> Nat:
  match small:
    case True{}:
      0n
    case False{}:
      depth(n)

def cap_depth(+n: Nat) -> Nat:
  cap_depth2(Nat.is_lt(n, 2n), n)

# r x c de zeros, com a capacidade exata (potência de 2 mínima)

def Mat.zeros(+r: Nat, +c: Nat) -> Mat<r, c>:
  Mat{Array.new(F32, cap_depth(Nat.mul(r, c)), 0.0)}

def Mat.fill(+r: Nat, +c: Nat, +v: F32) -> Mat<r, c>:
  Mat{Array.new(F32, cap_depth(Nat.mul(r, c)), v)}

def fill_list(xs: List<&2, F32>, +i: U32, a: Array<F32>) -> Array<F32>:
  match xs:
    case Nil{}:
      a
    case Con{h, t}:
      fill_list(t, (i + 1 : U32), Array.set(F32, a, i, h))

def len_is(xs: List<&2, F32>, n: Nat) -> Bool:
  match xs n:
    case Nil{} 0n:
      True{}
    case Con{h, t} 1n+p:
      len_is(t, p)
    case _ _:
      False{}

def Mat.of_list2(-r: Nat, -c: Nat, +rr: Nat, +cc: Nat, ok: Bool, xs: List<&2, F32>) -> Maybe<&1, Mat<r, c>>:
  match ok:
    case True{}:
      Some{Mat{fill_list(xs, 0, Array.new(F32, cap_depth(Nat.mul(rr, cc)), 0.0))}}
    case False{}:
      None{}

# Mat<r, c> a partir de r*c números em ordem de linha; None se o tamanho não bate

def Mat.of_list(+r: Nat, +c: Nat, +xs: List<&2, F32>) -> Maybe<&1, Mat<r, c>>:
  Mat.of_list2(r, c, r, c, len_is(xs, Nat.mul(r, c)), xs)

# lê os n primeiros números, devolvendo a matriz

type ML<-r: Nat, -c: Nat> is Type:
  ML{m: Mat<r, c>, l: List<&2, F32>}

def Mat.to_list2(-r: Nat, -c: Nat, al: AL) -> ML<r, c>:
  match al:
    case AL{d, l}:
      ML{Mat{d}, l}

# duas cópias independentes

def Mat.to_list(+r: Nat, +c: Nat, m: Mat<r, c>) -> ML<r, c>:
  match m:
    case Mat{d}:
      Mat.to_list2(r, c, read_l(Nat.sub(Nat.mul(r, c), 1n), 0, Nil{}, Array.get(F32, d, 0)))

type MM2<-r: Nat, -c: Nat> is Type:
  MM2{a: Mat<r, c>, b: Mat<r, c>}

def Mat.clone2(-r: Nat, -c: Nat, p: Array<F32> & Array<F32>) -> MM2<r, c>:
  match p:
    case Tuple{x, y}:
      MM2{Mat{x}, Mat{y}}

# ---- produtos: C (n x m) = A · B, com 2^par blocos de linhas em paralelo ----

# C = A(n x k) · B(k x m)

def Mat.clone(-r: Nat, -c: Nat, m: Mat<r, c>) -> MM2<r, c>:
  match m:
    case Mat{d}:
      Mat.clone2(r, c, Array.clone(F32, d))

type MMul<-n: Nat, -k: Nat, -m: Nat> is Type:
  MMul{a: Mat<n, k>, b: Mat<k, m>, c: Mat<n, m>}

# C = A(n x k) · Bᵀ, com B guardada m x k

type MMulNT<-n: Nat, -k: Nat, -m: Nat> is Type:
  MMulNT{a: Mat<n, k>, b: Mat<m, k>, c: Mat<n, m>}

# C = Aᵀ · B, com A guardada k x n

type MMulTN<-n: Nat, -k: Nat, -m: Nat> is Type:
  MMulTN{a: Mat<k, n>, b: Mat<k, m>, c: Mat<n, m>}

def gemm_any(d: Nat, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> G:
  match d:
    case 0n:
      gemm(n, m, k, arow, sa, sb, bcol, a, b, c)
    case 1n+q:
      gemm_par(1n+q, n, m, k, arow, sa, sb, bcol, a, b, c)

def mmul_fin(-n: Nat, -k: Nat, -m: Nat, g: G) -> MMul<n, k, m>:
  match g:
    case G{a, b, c}:
      MMul{Mat{a}, Mat{b}, Mat{c}}

def mmul_nt_fin(-n: Nat, -k: Nat, -m: Nat, g: G) -> MMulNT<n, k, m>:
  match g:
    case G{a, b, c}:
      MMulNT{Mat{a}, Mat{b}, Mat{c}}

def mmul_tn_fin(-n: Nat, -k: Nat, -m: Nat, g: G) -> MMulTN<n, k, m>:
  match g:
    case G{a, b, c}:
      MMulTN{Mat{a}, Mat{b}, Mat{c}}

def mm_a(+n: Nat, +k: Nat, +m: Nat, d: Nat, a: Array<F32>, b: Array<F32>) -> MMul<n, k, m>:
  mmul_fin(n, k, m, gemm_any(d, n, m, k, U32.from_nat(k), 1, U32.from_nat(m), 1, a, b, Array.new(F32, cap_depth(Nat.mul(n, m)), 0.0)))

# C = A · B   (n x k) · (k x m); par = log2 do número de blocos paralelos (0 = sequencial)

def Mat.matmul(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: Mat<n, k>, b: Mat<k, m>) -> MMul<n, k, m>:
  match a b:
    case Mat{x} Mat{y}:
      mm_a(n, k, m, par, x, y)

def mm_nt(+n: Nat, +k: Nat, +m: Nat, d: Nat, a: Array<F32>, b: Array<F32>) -> MMulNT<n, k, m>:
  mmul_nt_fin(n, k, m, gemm_any(d, n, m, k, U32.from_nat(k), 1, 1, U32.from_nat(k), a, b, Array.new(F32, cap_depth(Nat.mul(n, m)), 0.0)))

# C = A · Bᵀ   (n x k) · (m x k)ᵀ: para pesos guardados com uma linha por saída

def Mat.matmul_nt(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: Mat<n, k>, b: Mat<m, k>) -> MMulNT<n, k, m>:
  match a b:
    case Mat{x} Mat{y}:
      mm_nt(n, k, m, par, x, y)

def mm_tn(+n: Nat, +k: Nat, +m: Nat, d: Nat, a: Array<F32>, b: Array<F32>) -> MMulTN<n, k, m>:
  mmul_tn_fin(n, k, m, gemm_any(d, n, m, k, 1, U32.from_nat(n), U32.from_nat(m), 1, a, b, Array.new(F32, cap_depth(Nat.mul(n, m)), 0.0)))

# C = Aᵀ · B   (k x n)ᵀ · (k x m): gradiente de pesos, X^T · dY

def Mat.matmul_tn(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: Mat<k, n>, b: Mat<k, m>) -> MMulTN<n, k, m>:
  match a b:
    case Mat{x} Mat{y}:
      mm_tn(n, k, m, par, x, y)

# ---- operações elementwise e do treino ----

# soma o bias (uma linha, m números) a cada linha de C (n x m)

type MBias<-n: Nat, -m: Nat> is Type:
  MBias{c: Mat<n, m>, b: Mat<1n, m>}

def add_row2(-n: Nat, -m: Nat, +nn: Nat, +mm: Nat, cd: Array<F32>, al: AL) -> MBias<n, m>:
  match al:
    case AL{bd, bl}:
      MBias{Mat{bias_rows(nn, 0, U32.from_nat(mm), bl, cd)}, Mat{bd}}

def Mat.add_row(+n: Nat, +m: Nat, c: Mat<n, m>, b: Mat<1n, m>) -> MBias<n, m>:
  match c b:
    case Mat{cd} Mat{bd}:
      add_row2(n, m, n, m, cd, read_l(Nat.sub(m, 1n), 0, Nil{}, Array.get(F32, bd, 0)))

# par de matrizes do mesmo formato (o resultado de uma operação que lê dois operandos)

type MPair<-r: Nat, -c: Nat> is Type:
  MPair{a: Mat<r, c>, b: Mat<r, c>}

def mpair_fin(-r: Nat, -c: Nat, p: P2) -> MPair<r, c>:
  match p:
    case P2{a, b}:
      MPair{Mat{a}, Mat{b}}

def relu_fin(-r: Nat, -c: Nat, d: Array<F32>) -> Mat<r, c>:
  Mat{d}

def Mat.relu(+r: Nat, +c: Nat, a: Mat<r, c>) -> Mat<r, c>:
  match a:
    case Mat{d}:
      relu_fin(r, c, relu_l(Nat.mul(r, c), 0, Array.get(F32, d, 0)))

# (dy * (h > 0), h): gradiente que atravessa a relu

def Mat.relu_bwd(+r: Nat, +c: Nat, dy: Mat<r, c>, h: Mat<r, c>) -> MPair<r, c>:
  match dy h:
    case Mat{d} Mat{x}:
      mpair_fin(r, c, mask_l(Nat.mul(r, c), 0, Array.get(F32, d, 0), Array.get(F32, x, 0)))

# (w - lr * dw, dw): um passo de SGD

def Mat.sgd(+r: Nat, +c: Nat, +lr: F32, w: Mat<r, c>, dw: Mat<r, c>) -> MPair<r, c>:
  match w dw:
    case Mat{x} Mat{y}:
      mpair_fin(r, c, sgd_l(Nat.mul(r, c), 0, lr, Array.get(F32, x, 0), Array.get(F32, y, 0)))

# (a + b, b): soma elementwise (a - (-1)*b é exatamente a + b)

def Mat.add(+r: Nat, +c: Nat, a: Mat<r, c>, b: Mat<r, c>) -> MPair<r, c>:
  match a b:
    case Mat{x} Mat{y}:
      mpair_fin(r, c, sgd_l(Nat.mul(r, c), 0, (0.0 - 1.0 : F32), Array.get(F32, x, 0), Array.get(F32, y, 0)))

# soma de cada coluna (n x m -> 1 x m), devolvendo a matriz

type MColSum<-n: Nat, -m: Nat> is Type:
  MColSum{a: Mat<n, m>, s: Mat<1n, m>}

def colsum_fin(-n: Nat, -m: Nat, g: G) -> MColSum<n, m>:
  match g:
    case G{ones, a, s}:
      MColSum{Mat{a}, Mat{s}}

def Mat.col_sums(+n: Nat, +m: Nat, a: Mat<n, m>) -> MColSum<n, m>:
  match a:
    case Mat{d}:
      colsum_fin(n, m, gemm(1n, m, n, 0, 1, U32.from_nat(m), 1, Array.new(F32, cap_depth(n), 1.0), d, Array.new(F32, cap_depth(m), 0.0)))

# entropia cruzada média e seu gradiente nos logits: (softmax - one-hot) / n.
# labels tem uma entrada por linha (n); o gradiente substitui os logits.

type MCE<-n: Nat, -c: Nat> is Type:
  MCE{loss: F32, grad: Mat<n, c>}

def ce_fin(-n: Nat, -c: Nat, +nf: F32, r: L2) -> MCE<n, c>:
  match r:
    case L2{loss, d}:
      MCE{(loss / nf : F32), Mat{d}}

def Mat.softmax_ce(+n: Nat, +c: Nat, z: Mat<n, c>, labels: List<&2, Nat>) -> MCE<n, c>:
  match z:
    case Mat{d}:
      ce_fin(n, c, F32.from_nat(n), ce_rows(labels, Nat.sub(c, 1n), U32.from_nat(c), F32.from_nat(n), 0, L2{0.0, d}))

# acertos de argmax por linha contra labels

type MHits<-n: Nat, -c: Nat> is Type:
  MHits{z: Mat<n, c>, hits: Nat}

def hits_fin(-n: Nat, -c: Nat, r: CR) -> MHits<n, c>:
  match r:
    case CR{d, k}:
      MHits{Mat{d}, k}

def Mat.count_correct(+n: Nat, +c: Nat, z: Mat<n, c>, labels: List<&2, Nat>) -> MHits<n, c>:
  match z:
    case Mat{d}:
      hits_fin(n, c, count_rows(labels, Nat.sub(c, 1n), U32.from_nat(c), 0, CR{d, 0n}))

# uma linha como lista (devolvendo a matriz) e o contrário

type MRow<-r: Nat, -c: Nat> is Type:
  MRow{m: Mat<r, c>, row: List<&2, F32>}

def row_fin(-r: Nat, -c: Nat, al: AL) -> MRow<r, c>:
  match al:
    case AL{d, l}:
      MRow{Mat{d}, l}

def Mat.read_row(+r: Nat, +c: Nat, m: Mat<r, c>, +i: U32) -> MRow<r, c>:
  match m:
    case Mat{d}:
      row_fin(r, c, read_l(Nat.sub(c, 1n), (i * U32.from_nat(c) : U32), Nil{}, Array.get(F32, d, (i * U32.from_nat(c) : U32))))

def Mat.write_row(-r: Nat, +c: Nat, m: Mat<r, c>, +i: U32, xs: List<&2, F32>) -> Mat<r, c>:
  match m:
    case Mat{d}:
      Mat{write_l(xs, (i * U32.from_nat(c) : U32), d)}
