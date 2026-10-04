# GPT-2 small (124M) in Bend: inference with a KV cache, greedy generation.
#
#   bend demos/gpt2/gpt2.bend -o gpt2 && ./gpt2 "The capital of France is" 8
#
# Run from the repository root, after `reference/gpt2_prep.py --split`.
# The matrix shapes enter the types (bend-ml-tensor); the tokenizer is
# bend-ml-bpe-tokenizer with GPT-2's table of 50,000 rules.

import Base
import bend-ml-tensor@0.1.2.0/main.bend as T
import bend-ml-bpe-tokenizer@0.1.2.0/main.bend as BPE
import ./load.bend as L
import ./tok.bend as K

# ---------------------------------------------------------------------
# model
# ---------------------------------------------------------------------

type Layer is Data:
  Layer{ln1g: T.Vec<768n>, ln1b: T.Vec<768n>, aw: T.Mat<2304n, 768n>, ab: T.Vec<2304n>, pw: T.Mat<768n, 768n>, pb: T.Vec<768n>, ln2g: T.Vec<768n>, ln2b: T.Vec<768n>, fw: T.Mat<3072n, 768n>, fb: T.Vec<3072n>, mw: T.Mat<768n, 3072n>, mb: T.Vec<768n>}

type Model is Data:
  Model{wte: T.Mat<50257n, 768n>, wpe: T.Mat<1024n, 768n>, layers: List<&2, Layer>, lnfg: T.Vec<768n>, lnfb: T.Vec<768n>}

# key and value cache of one layer (one row of 768 numbers per position)

type KV is Data:
  KV{k: List<&2, List<&2, F32>>, v: List<&2, List<&2, F32>>}

type BR is Data:
  BR{x: T.Mat<1n, 768n>, kv: KV}

def nm(+i: Nat, f: String) -> String:
  "h" ++ Nat.show(i) ++ "." ++ f

def load_layer(+i: Nat) -> IO(Layer):
  do IO<Layer>:
    ln1g : T.Vec<768n> <- L.get_vec(nm(i, "ln1_g"), 768n)
    ln1b : T.Vec<768n> <- L.get_vec(nm(i, "ln1_b"), 768n)
    aw : T.Mat<2304n, 768n> <- L.get_mat(nm(i, "aw"), 2304n, 768n)
    ab : T.Vec<2304n> <- L.get_vec(nm(i, "ab"), 2304n)
    pw : T.Mat<768n, 768n> <- L.get_mat(nm(i, "pw"), 768n, 768n)
    pb : T.Vec<768n> <- L.get_vec(nm(i, "pb"), 768n)
    ln2g : T.Vec<768n> <- L.get_vec(nm(i, "ln2_g"), 768n)
    ln2b : T.Vec<768n> <- L.get_vec(nm(i, "ln2_b"), 768n)
    fw : T.Mat<3072n, 768n> <- L.get_mat(nm(i, "fw"), 3072n, 768n)
    fb : T.Vec<3072n> <- L.get_vec(nm(i, "fb"), 3072n)
    mw : T.Mat<768n, 3072n> <- L.get_mat(nm(i, "mw"), 768n, 3072n)
    mb : T.Vec<768n> <- L.get_vec(nm(i, "mb"), 768n)
    return Layer{ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb}

def load_layers(k: Nat, +i: Nat, acc: List<&2, Layer>) -> IO(List<&2, Layer>):
  match k:
    case 0n:
      IO.pure(List<&2, Layer>, List.reverse(&2, Layer, acc))
    case 1n+p:
      do IO<List<&2, Layer>>:
        l : Layer <- load_layer(i)
        load_layers(p, 1n+i, l <> acc)

def load_model() -> IO(Model):
  do IO<Model>:
    wte : T.Mat<50257n, 768n> <- L.get_mat("wte", 50257n, 768n)
    wpe : T.Mat<1024n, 768n> <- L.get_mat("wpe", 1024n, 768n)
    layers : List<&2, Layer> <- load_layers(12n, 0n, Nil{})
    lnfg : T.Vec<768n> <- L.get_vec("lnf_g", 768n)
    lnfb : T.Vec<768n> <- L.get_vec("lnf_b", 768n)
    return Model{wte, wpe, layers, lnfg, lnfb}

# ---------------------------------------------------------------------
# attention for one position: q (768) against the cached keys/values
# ---------------------------------------------------------------------

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

# 12 heads of 64: concatenates the outputs

def attn(h: Nat, +off: Nat, +q: List<&2, F32>, +ks: List<&2, List<&2, F32>>, +vs: List<&2, List<&2, F32>>) -> List<&2, F32>:
  match h:
    case 0n:
      Nil{}
    case 1n+p:
      List.append(&2, F32, head_out(slice64(q, off), ks, vs, off), attn(p, Nat.add(off, 64n), q, ks, vs))

# ---------------------------------------------------------------------
# one transformer block for one position (each step receives the previous one)
# ---------------------------------------------------------------------

def mat1(r: List<&2, F32>) -> T.Mat<1n, 768n>:
  T.Mat{r <> Nil{}}

def row_of(-c: Nat, m: T.Mat<1n, c>) -> List<&2, F32>:
  match m:
    case T.Mat{Con{r, t}}:
      r
    case T.Mat{Nil{}}:
      Nil{}

def blk_e(+kv: KV, +x1: T.Mat<1n, 768n>, g: T.Mat<1n, 768n>) -> BR:
  BR{T.Mat.add(1n, 768n, x1, g), kv}

def blk_d(+l: Layer, +kv: KV, +x1: T.Mat<1n, 768n>, +h2: T.Mat<1n, 768n>) -> BR:
  match l:
    case Layer{ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb}:
      blk_e(kv, x1, T.Mat.add_row(1n, 768n, T.Mat.matmul_t(1n, 3072n, 768n, T.Mat.gelu(1n, 3072n, T.Mat.add_row(1n, 3072n, T.Mat.matmul_t(1n, 768n, 3072n, h2, fw), fb)), mw), mb))

def blk_c(+l: Layer, +kv: KV, +x: T.Mat<1n, 768n>, +att: T.Mat<1n, 768n>) -> BR:
  match l:
    case Layer{ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb}:
      blk_d(Layer{ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb}, kv, T.Mat.add(1n, 768n, x, T.Mat.add_row(1n, 768n, T.Mat.matmul_t(1n, 768n, 768n, att, pw), pb)), T.Mat.layernorm(1n, 768n, 0.00001, ln2g, ln2b, T.Mat.add(1n, 768n, x, T.Mat.add_row(1n, 768n, T.Mat.matmul_t(1n, 768n, 768n, att, pw), pb))))

# qkv: one row of 2304 = q (768), k (768), v (768)

def blk_b2(+l: Layer, +x: T.Mat<1n, 768n>, +nk: List<&2, List<&2, F32>>, +nv: List<&2, List<&2, F32>>, +q: List<&2, F32>) -> BR:
  blk_c(l, KV{nk, nv}, x, mat1(attn(12n, 0n, q, nk, nv)))

def blk_b(+l: Layer, +c: KV, +x: T.Mat<1n, 768n>, +qkv: List<&2, F32>) -> BR:
  match c:
    case KV{ks, vs}:
      blk_b2(l, x, List.append(&2, List<&2, F32>, ks, List.take(&2, F32, List.drop(&2, F32, qkv, 768n), 768n) <> Nil{}), List.append(&2, List<&2, F32>, vs, List.drop(&2, F32, qkv, 1536n) <> Nil{}), List.take(&2, F32, qkv, 768n))

def blk_a(+l: Layer, +c: KV, +x: T.Mat<1n, 768n>, h: T.Mat<1n, 768n>) -> BR:
  match l:
    case Layer{ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb}:
      blk_b(Layer{ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb}, c, x, row_of(2304n, T.Mat.add_row(1n, 2304n, T.Mat.matmul_t(1n, 768n, 2304n, h, aw), ab)))

def block(+l: Layer, c: KV, +x: T.Mat<1n, 768n>) -> BR:
  match l:
    case Layer{ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb}:
      blk_a(Layer{ln1g, ln1b, aw, ab, pw, pb, ln2g, ln2b, fw, fb, mw, mb}, c, x, T.Mat.layernorm(1n, 768n, 0.00001, ln1g, ln1b, x))

# ---------------------------------------------------------------------
# embedding, 12 layers, logits
# ---------------------------------------------------------------------

def nth_row(rows: List<&2, List<&2, F32>>, i: Nat) -> List<&2, F32>:
  match rows i:
    case Con{r, t} 0n:
      r
    case Con{r, t} 1n+p:
      nth_row(t, p)
    case Nil{} _:
      Nil{}

def embed(+wte: T.Mat<50257n, 768n>, wpe: T.Mat<1024n, 768n>, tok: Nat, pos: Nat) -> T.Mat<1n, 768n>:
  mat1(T.zip_with.add(nth_row(T.Mat.to_rows(50257n, 768n, wte), tok), nth_row(T.Mat.to_rows(1024n, 768n, wpe), pos)))

# walks the layers; pend is the result of the previous block (x, cache)

def run(ls: List<&2, Layer>, cs: List<&2, KV>, acc: List<&2, KV>, pend: BR) -> T.Mat<1n, 768n> & List<&2, KV>:
  match ls cs pend:
    case Con{l, lt} Con{c, ct} BR{x, kv}:
      run(lt, ct, kv <> acc, block(l, c, x))
    case _ _ BR{x, kv}:
      (x, List.reverse(&2, KV, kv <> acc))

def start(ls: List<&2, Layer>, cs: List<&2, KV>, x: T.Mat<1n, 768n>) -> T.Mat<1n, 768n> & List<&2, KV>:
  match ls cs:
    case Con{l, lt} Con{c, ct}:
      run(lt, ct, Nil{}, block(l, c, x))
    case _ _:
      (x, Nil{})

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

type Out is Data:
  Out{id: Nat, lg: F32, cs: List<&2, KV>}

def lo3(+row: List<&2, F32>, cs: List<&2, KV>) -> Out:
  Out{argmax(row), T.max.l(row), cs}

def lo2(lg: T.Mat<1n, 50257n>, cs: List<&2, KV>) -> Out:
  match lg:
    case T.Mat{Con{r, t}}:
      lo3(r, cs)
    case T.Mat{Nil{}}:
      Out{0n, 0.0, cs}

def logits_out(+wte: T.Mat<50257n, 768n>, +lnfg: T.Vec<768n>, +lnfb: T.Vec<768n>, r: T.Mat<1n, 768n> & List<&2, KV>) -> Out:
  match r:
    case Tuple{x, cs}:
      lo2(T.Mat.matmul_t(1n, 768n, 50257n, T.Mat.layernorm(1n, 768n, 0.00001, lnfg, lnfb, x), wte), cs)

# one position: returns the most likely id, its logit and the new caches

def fwd(m: Model, cs: List<&2, KV>, tok: Nat, pos: Nat) -> Out:
  match m:
    case Model{+wte, wpe, layers, +lnfg, +lnfb}:
      logits_out(wte, lnfg, lnfb, start(layers, cs, embed(wte, wpe, tok, pos)))

# ---------------------------------------------------------------------
# greedy generation (feeds the prompt, then its own most likely token)
# ---------------------------------------------------------------------

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

type St is Data:
  St{cs: List<&2, KV>, tok: Nat, pos: Nat, queue: List<&2, Nat>, left: Nat, gen: List<&2, Nat>}

def adv2(queue: List<&2, Nat>, left: Nat, +pos: Nat, gen: List<&2, Nat>, +t0: Nat, r: Out) -> IO(St):
  match queue r:
    case Con{q, qt} Out{id, lg, cs}:
      IO.pure(St, St{cs, q, 1n+pos, qt, left, gen})
    case Nil{} Out{+id, +lg, cs}:
      do IO<St>:
        t1 : Nat <- IO.now()
        IO.print("  id " ++ Nat.show(id) ++ "  logit " ++ F32.show(lg) ++ "  pos " ++ Nat.show(pos) ++ "  t=" ++ Nat.show(Nat.div(Nat.sub(t1, t0), 1000n)) ++ "s")
        return St{cs, id, 1n+pos, Nil{}, Nat.sub(left, 1n), id <> gen}

def adv(+m: Model, st: St, t0: Nat) -> IO(St):
  match st:
    case St{cs, tok, +pos, queue, left, gen}:
      adv2(queue, left, pos, gen, t0, fwd(m, cs, tok, pos))

def loop(fuel: Nat, +m: Model, +t0: Nat, st: St) -> IO(St):
  match fuel:
    case 0n:
      IO.pure(St, st)
    case 1n+p:
      do IO<St>:
        s2 : St <- adv(m, st, t0)
        loop(p, m, t0, s2)

# ---------------------------------------------------------------------
# text <-> bytes (UTF-8)
# ---------------------------------------------------------------------

def u32s(xs: List<&2, Nat>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{n, t}:
      U32.from_nat(n) <> u32s(t)

# GPT-2 ids -> bpe/ token

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

# code point -> UTF-8 bytes

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
# main program
# ---------------------------------------------------------------------

def fin(+table: List<&2, BPE.Rule>, +perm: List<&2, Nat>, ids: List<&2, Nat>, st: St) -> IO(Unit):
  match st:
    case St{cs, tok, pos, queue, left, gen}:
      IO.print("text: " ++ text_of(table, perm, List.append(&2, Nat, ids, List.reverse(&2, Nat, gen))))

def start2(+model: Model, +table: List<&2, BPE.Rule>, +perm: List<&2, Nat>, +ids: List<&2, Nat>, +n: Nat) -> IO(Unit):
  match ids:
    case Nil{}:
      IO.print("empty prompt")
    case Con{t0, qt}:
      do IO<Unit>:
        t0ms : Nat <- IO.now()
        IO.print("prompt ids: " ++ K.show_nats(ids))
        st : St <- loop(Nat.add(List.length(&2, Nat, ids), Nat.sub(n, 1n)), model, t0ms, St{List.replicate(KV, 12n, KV{Nil{}, Nil{}}), t0, 0n, qt, n, Nil{}})
        fin(table, perm, ids, st)

def generate(model: Model, +mb: List<&2, U32>, +pb: List<&2, U32>, prompt: String, n: Nat) -> IO(Unit):
  start2(model, K.rules.go(K.nums(mb), 0n), K.nums(pb), K.encode_text(K.rules.go(K.nums(mb), 0n), K.nums(pb), utf8_of(String.to_list(prompt))), n)

def run1(prompt: String, mn: Maybe<&2, Nat>) -> IO(Unit):
  match mn:
    case None{}:
      IO.print("usage: gpt2 \"prompt\" n_tokens")
    case Some{n}:
      do IO<Unit>:
        mb : List<&2, U32> <- K.read_bytes("demos/gpt2/data/merges_num.txt")
        pb : List<&2, U32> <- K.read_bytes("demos/gpt2/data/perm.txt")
        t0 : Nat <- IO.now()
        model : Model <- load_model()
        t1 : Nat <- IO.now()
        IO.print("weights loaded in " ++ Nat.show(Nat.div(Nat.sub(t1, t0), 1000n)) ++ "s")
        generate(model, mb, pb, prompt, n)

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{prompt, Con{n, Nil{}}}}:
      run1(prompt, Nat.read(n))
    case _:
      IO.print("usage: gpt2 \"prompt\" n_tokens")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
