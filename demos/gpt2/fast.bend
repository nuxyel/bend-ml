# GPT-2 small (124M) em Bend com bend-ml-tensor-array: os produtos matriz·vetor (124 M
# mult-soma por token) rodam sobre Array<F32> com blocos paralelos; atenção, LayerNorm
# e GELU ficam em listas (pequenos). Formatos das matrizes conferidos pelo tipo.
#
#   bend demos/gpt2/fast.bend -o /tmp/gpt2_fast && /tmp/gpt2_fast "The capital of France is" 8
#
# Rodar da raiz do repositório, depois de `reference/gpt2_prep.py --split`.

import Base
import bend-ml-tensor-array@0.1.1.0/main.bend as TA
import bend-ml-tensor@0.1.1.0/main.bend as T
import bend-ml-bpe-tokenizer@0.1.1.0/main.bend as BPE
import ./load.bend as L
import ./tok.bend as K

# níveis de paralelismo dos produtos: 2^PAR blocos; 0 = sequencial. Em matriz·vetor cada peso é
# lido uma vez só, e copiar a matriz para cada bloco custa mais que calcular: par=3 é ~10x PIOR
# que 0 (NOTES.md, exp. 9).
def par() -> Nat:
  0n

def slice64(xs: List<&2, F32>, off: Nat) -> List<&2, F32>:
  List.take(&2, F32, List.drop(&2, F32, xs, off), 64n)

def scores_l(+qh: List<&2, F32>, ks: List<&2, List<&2, F32>>, +off: Nat) -> List<&2, F32>:
  match ks:
    case Nil{}:
      Nil{}
    case Con{k, t}:
      (T.dot.go(qh, slice64(k, off), 0.0) * 0.125 : F32) <> scores_l(qh, t, off)

def weighted(w: List<&2, F32>, vs: List<&2, List<&2, F32>>, +off: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match w vs:
    case Con{+a, wt} Con{v, vt}:
      weighted(wt, vt, off, T.zip_with.add(acc, T.scale.l(a, slice64(v, off))))
    case _ _:
      acc

def head_out(qh: List<&2, F32>, ks: List<&2, List<&2, F32>>, +vs: List<&2, List<&2, F32>>, +off: Nat) -> List<&2, F32>:
  weighted(T.softmax.l(scores_l(qh, ks, off)), vs, off, T.fill.l(64n, 0.0))

def attn(h: Nat, +off: Nat, +q: List<&2, F32>, +ks: List<&2, List<&2, F32>>, +vs: List<&2, List<&2, F32>>) -> List<&2, F32>:
  match h:
    case 0n:
      Nil{}
    case 1n+p:
      List.append(&2, F32, head_out(slice64(q, off), ks, vs, off), attn(p, Nat.add(off, 64n), q, ks, vs))

def gt_head(xs: List<&2, F32>, +v: F32) -> Bool:
  match xs:
    case Nil{}:
      False{}
    case Con{+h, t}:
      F32.is_gt(h, v)

def argmax.go(xs: List<&2, F32>, g: Bool, +i: Nat, +bi: Nat, +bv: F32) -> Nat:
  match xs g:
    case Nil{} _:
      bi
    case Con{+h, +t} True{}:
      argmax.go(t, gt_head(t, h), 1n+i, i, h)
    case Con{h, +t} False{}:
      argmax.go(t, gt_head(t, bv), 1n+i, bi, bv)

def argmax(xs: List<&2, F32>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case Con{+h, +t}:
      argmax.go(t, gt_head(t, h), 1n, 0n, h)

def selb(b: Bool, x: List<&2, U32>, y: List<&2, U32>) -> List<&2, U32>:
  match b:
    case True{}:
      x
    case False{}:
      y

def enc1(+cp: U32) -> List<&2, U32>:
  selb(U32.is_lt(cp, 128), cp <> Nil{}, selb(U32.is_lt(cp, 2048), (192 + U32.div(cp, 64) : U32) <> ((128 + U32.mod(cp, 64) : U32) <> Nil{}), selb(U32.is_lt(cp, 65536), (224 + U32.div(cp, 4096) : U32) <> ((128 + U32.mod(U32.div(cp, 64), 64) : U32) <> ((128 + U32.mod(cp, 64) : U32) <> Nil{})), (240 + U32.div(cp, 262144) : U32) <> ((128 + U32.mod(U32.div(cp, 4096), 64) : U32) <> ((128 + U32.mod(U32.div(cp, 64), 64) : U32) <> ((128 + U32.mod(cp, 64) : U32) <> Nil{}))))))

def utf8_of(cs: List<&2, Char>) -> List<&2, U32>:
  match cs:
    case Nil{}:
      Nil{}
    case Con{Chr{cp}, t}:
      List.append(&2, U32, enc1(cp), utf8_of(t))

def u32s(xs: List<&2, Nat>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{n, t}:
      U32.from_nat(n) <> u32s(t)

def index_sel(b: Bool, x: Nat, y: Nat) -> Nat:
  match b:
    case True{}:
      x
    case False{}:
      y

def index_of(perm: List<&2, Nat>, +id: Nat, +i: Nat) -> Nat:
  match perm:
    case Nil{}:
      0n
    case Con{p, t}:
      index_sel(Nat.is_eq(p, id), i, index_of(t, id, 1n+i))

def tk_sel(b: Bool, x: BPE.Tk, y: BPE.Tk) -> BPE.Tk:
  match b:
    case True{}:
      x
    case False{}:
      y

def tk_of_gid(+perm: List<&2, Nat>, +id: Nat) -> BPE.Tk:
  tk_sel(Nat.is_lt(id, 256n), BPE.B{index_of(perm, id, 0n)}, BPE.M{Nat.sub(id, 256n)})

def tks_of(+perm: List<&2, Nat>, ids: List<&2, Nat>) -> List<&2, BPE.Tk>:
  match ids:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      tk_of_gid(perm, h) <> tks_of(perm, t)

def head_b(xs: List<&2, U32>) -> U32:
  match xs:
    case Nil{}:
      0
    case Con{c, t}:
      c

def ulen(+b: U32) -> Nat:
  K.sel(U32.is_lt(b, 192), 1n, K.sel(U32.is_lt(b, 224), 2n, K.sel(U32.is_lt(b, 240), 3n, 4n)))

def selu(b: Bool, x: U32, y: U32) -> U32:
  match b:
    case True{}:
      x
    case False{}:
      y

def cont(+b: U32) -> Bool:
  Bool.and(U32.is_le(128, b), U32.is_le(b, 191))

def cp_of(+b0: U32, +b1: U32, +b2: U32, +b3: U32, n: Nat) -> U32:
  match n:
    case 1n:
      selu(U32.is_lt(b0, 128), b0, 63)
    case 2n:
      selu(cont(b1), (U32.sub(b0, 192) * 64 + U32.sub(b1, 128) : U32), 63)
    case 3n:
      selu(Bool.and(cont(b1), cont(b2)), (U32.sub(b0, 224) * 4096 + U32.sub(b1, 128) * 64 + U32.sub(b2, 128) : U32), 63)
    case _:
      selu(Bool.and(cont(b1), Bool.and(cont(b2), cont(b3))), (U32.sub(b0, 240) * 262144 + U32.sub(b1, 128) * 4096 + U32.sub(b2, 128) * 64 + U32.sub(b3, 128) : U32), 63)

def drop_n(xs: List<&2, U32>, n: Nat) -> List<&2, U32>:
  List.drop(&2, U32, xs, n)

def tail1(xs: List<&2, U32>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{c, t}:
      t

def utf8_chars(fuel: Nat, bs: List<&2, U32>) -> List<&2, Char>:
  match fuel bs:
    case 1n+p Con{+b0, +t}:
      Chr{cp_of(b0, head_b(t), head_b(tail1(t)), head_b(tail1(tail1(t))), ulen(b0))} <> utf8_chars(p, drop_n(t, Nat.sub(ulen(b0), 1n)))
    case _ _:
      Nil{}

def text_of2(+bs: List<&2, U32>) -> String:
  String.from_list(utf8_chars(List.length(&2, U32, bs), bs))

def text_of(+table: List<&2, BPE.Rule>, +perm: List<&2, Nat>, ids: List<&2, Nat>) -> String:
  text_of2(u32s(BPE.decode(table, tks_of(perm, ids))))

# ---------------------------------------------------------------------
# modelo
# ---------------------------------------------------------------------

type Layer is Type:
  Layer{ln1g: List<&2, F32>, +ln1b: List<&2, F32>, aw: TA.Mat<2304n, 768n>, +ab: List<&2, F32>, pw: TA.Mat<768n, 768n>, +pb: List<&2, F32>, +ln2g: List<&2, F32>, +ln2b: List<&2, F32>, fw: TA.Mat<3072n, 768n>, +fb: List<&2, F32>, mw: TA.Mat<768n, 3072n>, +mb: List<&2, F32>}

type Model is Type:
  Model{wte: TA.Mat<50257n, 768n>, wpe: TA.Mat<1024n, 768n>, layers: List<&1, Layer>, +lnfg: List<&2, F32>, +lnfb: List<&2, F32>}

# cache de chaves e valores de uma camada (uma linha de 768 números por posição)

type KV is Data:
  KV{k: List<&2, List<&2, F32>>, v: List<&2, List<&2, F32>>}

type BR is Type:
  BR{x: List<&2, F32>, kv: KV, l: Layer}

def nm(+i: Nat, f: String) -> String:
  "h" ++ Nat.show(i) ++ "." ++ f

def get_list(name: String, +n: Nat) -> IO(List<&2, F32>):
  L.read_floats("demos/gpt2/data/w/" ++ name ++ ".bin", n)

def get_mat(name: String, +r: Nat, +c: Nat) -> IO(TA.Mat<r, c>):
  do IO<TA.Mat<r, c>>:
    fl : List<&2, F32> <- L.read_floats("demos/gpt2/data/w/" ++ name ++ ".bin", Nat.mul(r, c))
    return TA.Mat.from_list(r, c, fl)

def load_layer(+i: Nat) -> IO(Layer):
  do IO<Layer>:
    ln1g : List<&2, F32> <- get_list(nm(i, "ln1_g"), 768n)
    ln1b : List<&2, F32> <- get_list(nm(i, "ln1_b"), 768n)
    aw : TA.Mat<2304n, 768n> <- get_mat(nm(i, "aw"), 2304n, 768n)
    ab : List<&2, F32> <- get_list(nm(i, "ab"), 2304n)
    pw : TA.Mat<768n, 768n> <- get_mat(nm(i, "pw"), 768n, 768n)
    pb : List<&2, F32> <- get_list(nm(i, "pb"), 768n)
    ln2g : List<&2, F32> <- get_list(nm(i, "ln2_g"), 768n)
    ln2b : List<&2, F32> <- get_list(nm(i, "ln2_b"), 768n)
    fw : TA.Mat<3072n, 768n> <- get_mat(nm(i, "fw"), 3072n, 768n)
    fb : List<&2, F32> <- get_list(nm(i, "fb"), 3072n)
    mw : TA.Mat<768n, 3072n> <- get_mat(nm(i, "mw"), 768n, 3072n)
    mb : List<&2, F32> <- get_list(nm(i, "mb"), 768n)
    return Layer{ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb}

def load_layers(k: Nat, +i: Nat, acc: List<&1, Layer>) -> IO(List<&1, Layer>):
  match k:
    case 0n:
      IO.pure(List<&1, Layer>, List.reverse(&1, Layer, acc))
    case 1n+p:
      do IO<List<&1, Layer>>:
        l : Layer <- load_layer(i)
        load_layers(p, 1n+i, l <> acc)

def load_model() -> IO(Model):
  do IO<Model>:
    wte : TA.Mat<50257n, 768n> <- get_mat("wte", 50257n, 768n)
    wpe : TA.Mat<1024n, 768n> <- get_mat("wpe", 1024n, 768n)
    layers : List<&1, Layer> <- load_layers(12n, 0n, Nil{})
    lnfg : List<&2, F32> <- get_list("lnf_g", 768n)
    lnfb : List<&2, F32> <- get_list("lnf_b", 768n)
    return Model{wte, wpe, layers, lnfg, lnfb}

# ---------------------------------------------------------------------
# um bloco do transformador para uma posição
# ---------------------------------------------------------------------

def ln(+xs: List<&2, F32>, +g: List<&2, F32>, +b: List<&2, F32>) -> List<&2, F32>:
  T.layernorm.l(xs, 768.0, 0.00001, g, b)

# 6. segundo produto do MLP e saída do bloco

def k6b(+x1: List<&2, F32>, kv: KV, +ln1g: List<&2, F32>, +ln1b: List<&2, F32>, aw: TA.Mat<2304n, 768n>, +ab: List<&2, F32>, pw: TA.Mat<768n, 768n>, +pb: List<&2, F32>, +ln2g: List<&2, F32>, +ln2b: List<&2, F32>, fw: TA.Mat<3072n, 768n>, +fb: List<&2, F32>, mw: TA.Mat<768n, 3072n>, +mb: List<&2, F32>, r: TA.ML<1n, 768n>) -> BR:
  match r:
    case TA.ML{m, y}:
      BR{T.zip_with.add(x1, T.zip_with.add(y, mb)), kv, Layer{ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb}}

# 5. primeiro produto do MLP (768 -> 3072), bias e GELU

def k6(+x1: List<&2, F32>, kv: KV, +ln1g: List<&2, F32>, +ln1b: List<&2, F32>, aw: TA.Mat<2304n, 768n>, +ab: List<&2, F32>, pw: TA.Mat<768n, 768n>, +pb: List<&2, F32>, +ln2g: List<&2, F32>, +ln2b: List<&2, F32>, fw: TA.Mat<3072n, 768n>, +fb: List<&2, F32>, +mb: List<&2, F32>, r: TA.MMulNT<1n, 3072n, 768n>) -> BR:
  match r:
    case TA.MMulNT{g, mw, c}:
      k6b(x1, kv, ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb, TA.Mat.to_list(1n, 768n, c))

def k5b(+x1: List<&2, F32>, kv: KV, +ln1g: List<&2, F32>, +ln1b: List<&2, F32>, aw: TA.Mat<2304n, 768n>, +ab: List<&2, F32>, pw: TA.Mat<768n, 768n>, +pb: List<&2, F32>, +ln2g: List<&2, F32>, +ln2b: List<&2, F32>, fw: TA.Mat<3072n, 768n>, +fb: List<&2, F32>, mw: TA.Mat<768n, 3072n>, +mb: List<&2, F32>, r: TA.ML<1n, 3072n>) -> BR:
  match r:
    case TA.ML{m, y}:
      k6(x1, kv, ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mb, TA.Mat.matmul_nt(1n, 3072n, 768n, par(), TA.Mat.from_list(1n, 3072n, T.gelu.l(T.zip_with.add(y, fb))), mw))

def k5(+x1: List<&2, F32>, kv: KV, +ln1g: List<&2, F32>, +ln1b: List<&2, F32>, aw: TA.Mat<2304n, 768n>, +ab: List<&2, F32>, pw: TA.Mat<768n, 768n>, +pb: List<&2, F32>, +ln2g: List<&2, F32>, +ln2b: List<&2, F32>, +fb: List<&2, F32>, mw: TA.Mat<768n, 3072n>, +mb: List<&2, F32>, r: TA.MMulNT<1n, 768n, 3072n>) -> BR:
  match r:
    case TA.MMulNT{h, fw, c}:
      k5b(x1, kv, ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb, TA.Mat.to_list(1n, 3072n, c))

# 4. projeção da atenção, resíduo, segundo LayerNorm

def k4c(+x1: List<&2, F32>, kv: KV, +ln1g: List<&2, F32>, +ln1b: List<&2, F32>, aw: TA.Mat<2304n, 768n>, +ab: List<&2, F32>, pw: TA.Mat<768n, 768n>, +pb: List<&2, F32>, +ln2g: List<&2, F32>, +ln2b: List<&2, F32>, fw: TA.Mat<3072n, 768n>, +fb: List<&2, F32>, mw: TA.Mat<768n, 3072n>, +mb: List<&2, F32>) -> BR:
  k5(x1, kv, ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fb, mw, mb, TA.Mat.matmul_nt(1n, 768n, 3072n, par(), TA.Mat.from_list(1n, 768n, ln(x1, ln2g, ln2b)), fw))

def k4b(+x: List<&2, F32>, kv: KV, +ln1g: List<&2, F32>, +ln1b: List<&2, F32>, aw: TA.Mat<2304n, 768n>, +ab: List<&2, F32>, pw: TA.Mat<768n, 768n>, +pb: List<&2, F32>, +ln2g: List<&2, F32>, +ln2b: List<&2, F32>, fw: TA.Mat<3072n, 768n>, +fb: List<&2, F32>, mw: TA.Mat<768n, 3072n>, +mb: List<&2, F32>, r: TA.ML<1n, 768n>) -> BR:
  match r:
    case TA.ML{m, y}:
      k4c(T.zip_with.add(x, T.zip_with.add(y, pb)), kv, ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb)

def k4(+x: List<&2, F32>, kv: KV, +ln1g: List<&2, F32>, +ln1b: List<&2, F32>, aw: TA.Mat<2304n, 768n>, +ab: List<&2, F32>, +pb: List<&2, F32>, +ln2g: List<&2, F32>, +ln2b: List<&2, F32>, fw: TA.Mat<3072n, 768n>, +fb: List<&2, F32>, mw: TA.Mat<768n, 3072n>, +mb: List<&2, F32>, r: TA.MMulNT<1n, 768n, 768n>) -> BR:
  match r:
    case TA.MMulNT{att, pw, c}:
      k4b(x, kv, ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb, TA.Mat.to_list(1n, 768n, c))

# 3. atenção sobre o cache

def k3b(+x: List<&2, F32>, +ln1g: List<&2, F32>, +ln1b: List<&2, F32>, aw: TA.Mat<2304n, 768n>, +ab: List<&2, F32>, pw: TA.Mat<768n, 768n>, +pb: List<&2, F32>, +ln2g: List<&2, F32>, +ln2b: List<&2, F32>, fw: TA.Mat<3072n, 768n>, +fb: List<&2, F32>, mw: TA.Mat<768n, 3072n>, +mb: List<&2, F32>, +q: List<&2, F32>, +nk: List<&2, List<&2, F32>>, +nv: List<&2, List<&2, F32>>) -> BR:
  k4(x, KV{nk, nv}, ln1g, ln1b, aw, ab, pb, ln2g, ln2b, fw, fb, mw, mb, TA.Mat.matmul_nt(1n, 768n, 768n, par(), TA.Mat.from_list(1n, 768n, attn(12n, 0n, q, nk, nv)), pw))

# 2. qkv: q (768), k (768), v (768)

def k3(+x: List<&2, F32>, kv: KV, +ln1g: List<&2, F32>, +ln1b: List<&2, F32>, aw: TA.Mat<2304n, 768n>, +ab: List<&2, F32>, pw: TA.Mat<768n, 768n>, +pb: List<&2, F32>, +ln2g: List<&2, F32>, +ln2b: List<&2, F32>, fw: TA.Mat<3072n, 768n>, +fb: List<&2, F32>, mw: TA.Mat<768n, 3072n>, +mb: List<&2, F32>, +q: List<&2, F32>, +kr: List<&2, F32>, +vr: List<&2, F32>) -> BR:
  match kv:
    case KV{ks, vs}:
      k3b(x, ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb, q, List.append(&2, List<&2, F32>, ks, kr <> Nil{}), List.append(&2, List<&2, F32>, vs, vr <> Nil{}))

def k2b(+x: List<&2, F32>, kv: KV, +ln1g: List<&2, F32>, +ln1b: List<&2, F32>, aw: TA.Mat<2304n, 768n>, +ab: List<&2, F32>, pw: TA.Mat<768n, 768n>, +pb: List<&2, F32>, +ln2g: List<&2, F32>, +ln2b: List<&2, F32>, fw: TA.Mat<3072n, 768n>, +fb: List<&2, F32>, mw: TA.Mat<768n, 3072n>, +mb: List<&2, F32>, +qkv: List<&2, F32>) -> BR:
  k3(x, kv, ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb, List.take(&2, F32, qkv, 768n), List.take(&2, F32, List.drop(&2, F32, qkv, 768n), 768n), List.drop(&2, F32, qkv, 1536n))

def k2(+x: List<&2, F32>, kv: KV, +ln1g: List<&2, F32>, +ln1b: List<&2, F32>, +ab: List<&2, F32>, pw: TA.Mat<768n, 768n>, +pb: List<&2, F32>, +ln2g: List<&2, F32>, +ln2b: List<&2, F32>, fw: TA.Mat<3072n, 768n>, +fb: List<&2, F32>, mw: TA.Mat<768n, 3072n>, +mb: List<&2, F32>, aw: TA.Mat<2304n, 768n>, r: TA.ML<1n, 2304n>) -> BR:
  match r:
    case TA.ML{m, y}:
      k2b(x, kv, ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb, T.zip_with.add(y, ab))

def k1(+x: List<&2, F32>, kv: KV, +ln1g: List<&2, F32>, +ln1b: List<&2, F32>, +ab: List<&2, F32>, pw: TA.Mat<768n, 768n>, +pb: List<&2, F32>, +ln2g: List<&2, F32>, +ln2b: List<&2, F32>, fw: TA.Mat<3072n, 768n>, +fb: List<&2, F32>, mw: TA.Mat<768n, 3072n>, +mb: List<&2, F32>, r: TA.MMulNT<1n, 768n, 2304n>) -> BR:
  match r:
    case TA.MMulNT{h, aw, c}:
      k2(x, kv, ln1g, ln1b, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb, aw, TA.Mat.to_list(1n, 2304n, c))

def block(l: Layer, kv: KV, +x: List<&2, F32>) -> BR:
  match l:
    case Layer{+ln1g, +ln1b, aw, +ab, pw, +pb, +ln2g, +ln2b, fw, +fb, mw, +mb}:
      k1(x, kv, ln1g, ln1b, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb, TA.Mat.matmul_nt(1n, 768n, 2304n, par(), TA.Mat.from_list(1n, 768n, ln(x, ln1g, ln1b)), aw))

# ---------------------------------------------------------------------
# camadas, embedding e logits
# ---------------------------------------------------------------------

type RunOut is Type:
  RunOut{x: List<&2, F32>, layers: List<&1, Layer>, cs: List<&2, KV>}

# pend é o resultado do bloco anterior

def run(ls: List<&1, Layer>, cs: List<&2, KV>, al: List<&1, Layer>, ac: List<&2, KV>, pend: BR) -> RunOut:
  match ls cs pend:
    case Con{l, lt} Con{c, ct} BR{x, kv, l0}:
      run(lt, ct, l0 <> al, kv <> ac, block(l, c, x))
    case _ _ BR{x, kv, l0}:
      RunOut{x, List.reverse(&1, Layer, l0 <> al), List.reverse(&2, KV, kv <> ac)}

def start(ls: List<&1, Layer>, cs: List<&2, KV>, x: List<&2, F32>) -> RunOut:
  match ls cs:
    case Con{l, lt} Con{c, ct}:
      run(lt, ct, Nil{}, Nil{}, block(l, c, x))
    case _ _:
      RunOut{x, Nil{}, Nil{}}

type Out is Type:
  Out{id: Nat, lg: F32, cs: List<&2, KV>, m: Model}

def f5(+id_row: List<&2, F32>, wte: TA.Mat<50257n, 768n>, wpe: TA.Mat<1024n, 768n>, layers: List<&1, Layer>, +lnfg: List<&2, F32>, +lnfb: List<&2, F32>, cs: List<&2, KV>) -> Out:
  Out{argmax(id_row), T.max.l(id_row), cs, Model{wte, wpe, layers, lnfg, lnfb}}

def f4b(wte: TA.Mat<50257n, 768n>, wpe: TA.Mat<1024n, 768n>, layers: List<&1, Layer>, +lnfg: List<&2, F32>, +lnfb: List<&2, F32>, cs: List<&2, KV>, r: TA.ML<1n, 50257n>) -> Out:
  match r:
    case TA.ML{m, row}:
      f5(row, wte, wpe, layers, lnfg, lnfb, cs)

def f4(wpe: TA.Mat<1024n, 768n>, layers: List<&1, Layer>, +lnfg: List<&2, F32>, +lnfb: List<&2, F32>, cs: List<&2, KV>, r: TA.MMulNT<1n, 768n, 50257n>) -> Out:
  match r:
    case TA.MMulNT{h, wte, c}:
      f4b(wte, wpe, layers, lnfg, lnfb, cs, TA.Mat.to_list(1n, 50257n, c))

def f3(wpe: TA.Mat<1024n, 768n>, +lnfg: List<&2, F32>, +lnfb: List<&2, F32>, wte: TA.Mat<50257n, 768n>, r: RunOut) -> Out:
  match r:
    case RunOut{x, layers, cs}:
      f4(wpe, layers, lnfg, lnfb, cs, TA.Mat.matmul_nt(1n, 768n, 50257n, par(), TA.Mat.from_list(1n, 768n, ln(x, lnfg, lnfb)), wte))

def f3b(wpe: TA.Mat<1024n, 768n>, +lnfg: List<&2, F32>, +lnfb: List<&2, F32>, wte: TA.Mat<50257n, 768n>, r: RunOut) -> Out:
  f3(wpe, lnfg, lnfb, wte, r)

def f2(wte: TA.Mat<50257n, 768n>, layers: List<&1, Layer>, +lnfg: List<&2, F32>, +lnfb: List<&2, F32>, cs: List<&2, KV>, row: List<&2, F32>, r: TA.MRow<1024n, 768n>) -> Out:
  match r:
    case TA.MRow{wpe, prow}:
      f3b(wpe, lnfg, lnfb, wte, start(layers, cs, T.zip_with.add(row, prow)))

def f1(wpe: TA.Mat<1024n, 768n>, layers: List<&1, Layer>, +lnfg: List<&2, F32>, +lnfb: List<&2, F32>, cs: List<&2, KV>, +pos: U32, r: TA.MRow<50257n, 768n>) -> Out:
  match r:
    case TA.MRow{wte, row}:
      f2(wte, layers, lnfg, lnfb, cs, row, TA.Mat.read_row(1024n, 768n, wpe, pos))

# uma posição: devolve o id mais provável, o seu logit, os novos caches e o modelo

def fwd(m: Model, cs: List<&2, KV>, tok: Nat, pos: Nat) -> Out:
  match m:
    case Model{wte, wpe, layers, lnfg, lnfb}:
      f1(wpe, layers, lnfg, lnfb, cs, U32.from_nat(pos), TA.Mat.read_row(50257n, 768n, wte, U32.from_nat(tok)))

# ---------------------------------------------------------------------
# geração gulosa (alimenta o prompt, depois o próprio token mais provável)
# ---------------------------------------------------------------------

type St is Data:
  St{cs: List<&2, KV>, tok: Nat, pos: Nat, queue: List<&2, Nat>, left: Nat, gen: List<&2, Nat>}

type SM is Type:
  SM{st: St, m: Model}

def tsec2(+dt: Nat) -> String:
  Nat.show(Nat.div(dt, 1000n)) ++ "." ++ Nat.show(Nat.div(Nat.mod(dt, 1000n), 100n)) ++ "s"

def tsec(t1: Nat, t0: Nat) -> String:
  tsec2(Nat.sub(t1, t0))

def adv2(queue: List<&2, Nat>, left: Nat, +pos: Nat, gen: List<&2, Nat>, +t0: Nat, r: Out) -> IO(SM):
  match queue r:
    case Con{q, qt} Out{id, lg, cs, m}:
      IO.pure(SM, SM{St{cs, q, 1n+pos, qt, left, gen}, m})
    case Nil{} Out{+id, +lg, cs, m}:
      do IO<SM>:
        t1 : Nat <- IO.now()
        IO.print("  id " ++ Nat.show(id) ++ "  logit " ++ F32.show(lg) ++ "  pos " ++ Nat.show(pos) ++ "  t=" ++ tsec(t1, t0))
        return SM{St{cs, id, 1n+pos, Nil{}, Nat.sub(left, 1n), id <> gen}, m}

def adv(m: Model, st: St, t0: Nat) -> IO(SM):
  match st:
    case St{cs, tok, +pos, queue, left, gen}:
      adv2(queue, left, pos, gen, t0, fwd(m, cs, tok, pos))

def loop(fuel: Nat, +t0: Nat, sm: SM) -> IO(St):
  match fuel sm:
    case 0n SM{st, m}:
      IO.pure(St, st)
    case 1n+p SM{st, m}:
      do IO<St>:
        s2 : SM <- adv(m, st, t0)
        loop(p, t0, s2)

# ---------------------------------------------------------------------
# programa principal
# ---------------------------------------------------------------------

def fin(+table: List<&2, BPE.Rule>, +perm: List<&2, Nat>, ids: List<&2, Nat>, st: St) -> IO(Unit):
  match st:
    case St{cs, tok, pos, queue, left, gen}:
      IO.print("texto: " ++ text_of(table, perm, List.append(&2, Nat, ids, List.reverse(&2, Nat, gen))))

def start2(model: Model, +table: List<&2, BPE.Rule>, +perm: List<&2, Nat>, +ids: List<&2, Nat>, +n: Nat) -> IO(Unit):
  match ids:
    case Nil{}:
      IO.print("prompt vazio")
    case Con{t0, qt}:
      do IO<Unit>:
        t0ms : Nat <- IO.now()
        IO.print("prompt ids: " ++ K.show_nats(ids))
        st : St <- loop(Nat.add(List.length(&2, Nat, ids), Nat.sub(n, 1n)), t0ms, SM{St{List.replicate(KV, 12n, KV{Nil{}, Nil{}}), t0, 0n, qt, n, Nil{}}, model})
        fin(table, perm, ids, st)

def generate(model: Model, +mb: List<&2, U32>, +pb: List<&2, U32>, prompt: String, n: Nat) -> IO(Unit):
  start2(model, K.rules.go(K.nums(mb), 0n), K.nums(pb), K.encode_text(K.rules.go(K.nums(mb), 0n), K.nums(pb), utf8_of(String.to_list(prompt))), n)

def run1(prompt: String, mn: Maybe<&2, Nat>) -> IO(Unit):
  match mn:
    case None{}:
      IO.print("uso: gpt2_fast \"prompt\" n_tokens")
    case Some{n}:
      do IO<Unit>:
        mb : List<&2, U32> <- K.read_bytes("demos/gpt2/data/merges_num.txt")
        pb : List<&2, U32> <- K.read_bytes("demos/gpt2/data/perm.txt")
        t0 : Nat <- IO.now()
        model : Model <- load_model()
        t1 : Nat <- IO.now()
        IO.print("pesos carregados em " ++ Nat.show(Nat.div(Nat.sub(t1, t0), 1000n)) ++ "s")
        generate(model, mb, pb, prompt, n)

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{prompt, Con{n, Nil{}}}}:
      run1(prompt, Nat.read(n))
    case _:
      IO.print("uso: gpt2_fast \"prompt\" n_tokens")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
