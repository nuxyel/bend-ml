"""Writes tensor-gpu/main.bend, bend-ml-tensor-gpu: tensor-array's six products on the GPU with tensor-array's bits.

  python scripts/gen_tensor_gpu.py [TM TN] [out]      # tile of C a leaf computes (default 4 4)

Each leaf of one bang computes a TM x TN tile of C with four partial sums a cell, in tensor-array's order: product p of
a dot goes to sum p mod 4, in increasing p, each sum starting at +0.0, and the four are combined by TA.t8_comb with
rot = k mod 4 (tensor-array's dotr and t8_dot). The loop takes four steps of k a turn; the last turn's steps past k add
+0.0, which leaves a sum unchanged (a sum that starts at +0.0 is never -0.0). An accumulating product adds the cell to C
once with TA.cwr. So every cell has the bits tensor-array computes, whatever the tile or the number of leaves.
The code is generated because a tile of TM x TN cells carries 4 * TM * TN sums through the loop.
"""
import sys

TM, TN = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (4, 4)
OUT = sys.argv[3] if len(sys.argv) > 3 else "tensor-gpu/main.bend"
G = 4  # steps of k a turn
cells = [(r, c) for r in range(TM) for c in range(TN)]
S = [f"s{q}_{r}_{c}" for (r, c) in cells for q in range(4)]  # sums, cell-major
X = [f"x{g}_{r}" for g in range(G) for r in range(TM)]  # A values: step g, row r
Y = [f"y{g}_{c}" for g in range(G) for c in range(TN)]  # B values: step g, column c
V = X + Y


def sum_name(q, r, c):
    return f"s{q}_{r}_{c}"


def j(xs, sep=", "):
    return sep.join(xs)


o = []
p = o.append
p(f"""# bend-ml-tensor-gpu: tensor-array's products on the GPU, with tensor-array's bits.
#
#   import ../tensor-array/main.bend as TA
#   import bend-ml-tensor-gpu@0.1.5.0/main.bend as TG
#   TG.matmul(n, k, m, par, a, b)      # the same arguments, result type and bits as TA.Mat.matmul
#
# The six products (matmul, matmul_nt, matmul_tn and their accumulating forms) take tensor-array's Mat values and return
# its result types; par is ignored (the GPU decides the split). Each product is one `!` call (a bang): a balanced fork
# tree whose leaves each compute a {TM} x {TN} tile of C, every node sharing A, B and C through Base's Array.fork (two
# handles to one block, no copy), as upstream's one-block pattern does (bendlang/bend tests/run/stencil3d.bend). On a
# machine without a GPU, or with `+RTS --gpu off -RTS`, the same bang runs on the CPU threads.
#
# The bits: a cell of C is computed in tensor-array's order. Product p of a dot goes to partial sum p mod 4, in
# increasing p, each sum starting at +0.0, and the four sums are combined by TA.t8_comb with rot = k mod 4, as
# tensor-array's dotr and t8_dot do; an accumulating product adds the cell to C once with TA.cwr. The loop takes four
# steps of k a turn; the steps past k in the last turn add +0.0, which leaves a sum unchanged (a sum that starts at +0.0
# is never -0.0). Bend builds the device code with --fmad=false, so no multiply-add is fused. tests/bits.bend checks the
# bits against tensor-array on many shapes, on the GPU and the CPU.
#
# Counter-based random numbers (fill_uniform, fill_gauss, uniform_mat, gauss_mat and their _at forms; the last section):
# n uniforms or Gaussians of a counter made in one bang, each a pure function of (counter, index), with the same bits on
# every lane.
#
# Not proved, and marked so: Array.fork and Array.join are @unsafe in Base, so the defs below that call them, or take
# a shared array, are @unsafe, and `bend main.bend` names every def that relies on them. Nothing checks that a leaf
# writes only its own tile of C, or never writes A or B; the tiles are disjoint by construction (leaf t writes tile t).
# All of it is Bend: no foreign code. Written by bend-ml's scripts/gen_tensor_gpu.py ({TM} x {TN} tiles).
import Base
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

# the sizes and strides of one product: C[i][j] = sum_p A[oa + i*arow + p*sa] * B[ob + p*sb + j*bcol], n x m, k steps
# (oa, ob: 0 except for matmul_precise's blocks)
type E is Data:
  E{{n: U32, m: U32, k: U32, tcols: U32, tiles: U32, arow: U32, sa: U32, sb: U32, bcol: U32, kg: Nat, rot: Nat, rem: U32, ac: Bool, oa: U32, ob: U32}}

# a product that counts only when its step is inside k (the last turn's steps past k add +0.0)
def pk(ok: Bool, +x: F32) -> F32:
  match ok:
    case True{{}}: x
    case False{{}}: 0.0

# the values of one turn: A's {TM} rows and B's {TN} columns at four consecutive steps, read through the arrays in order
type GR is Type:
  GR{{a: Array<F32>, b: Array<F32>, {j(v + ': F32' for v in V)}}}

""")

# getter chain: gx_i reads value i (of V), carrying values 0..i-1; the A values come from a, the B values from b
n_v = len(V)
for i in reversed(range(n_v)):
    known = [f"+{V[t]}: F32" for t in range(i)]
    head = f"def gr_{i}(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, {j(known) + ', ' if known else ''}"
    is_a = i < len(X)
    if is_a:
        head += f"rv: Array<F32> & F32, b: Array<F32>) -> GR:\n"
        body = "  match rv:\n    case Tuple{a, +" + V[i] + "}:\n"
    else:
        head += f"a: Array<F32>, rv: Array<F32> & F32) -> GR:\n"
        body = "  match rv:\n    case Tuple{b, +" + V[i] + "}:\n"
    vals = j(V[: i + 1])
    if i == n_v - 1:
        body += f"      GR{{a, b, {vals}}}\n"
    else:
        nxt = V[i + 1]
        args = f"ia, ib, arow, sa, sb, bcol, {vals}"
        if i + 1 < len(X):  # next is an A value: step g, row r
            g, r = divmod(i + 1, TM)
            body += f"      gr_{i + 1}({args}, Array.get(F32, a, (ia + ((sa * {g} : U32) + (arow * {r} : U32) : U32) : U32)), b)\n"
        else:
            g, c = divmod(i + 1 - len(X), TN)
            src = "a" if is_a else "a"
            if is_a:  # switching from a to b: a is the matched array, b the parameter
                body += f"      gr_{i + 1}({args}, a, Array.get(F32, b, (ib + ((sb * {g} : U32) + (bcol * {c} : U32) : U32) : U32)))\n"
            else:
                body += f"      gr_{i + 1}({args}, a, Array.get(F32, b, (ib + ((sb * {g} : U32) + (bcol * {c} : U32) : U32) : U32)))\n"
    o.append(head + body)
p(f"""# the turn whose A values start at ia and B values at ib
def turn(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>) -> GR:
  gr_0(ia, ib, arow, sa, sb, bcol, Array.get(F32, a, ia), b)

# the tile's cells after the four sums are combined, row by row
type T is Type:
  T{{a: Array<F32>, b: Array<F32>, {j(f'c{r}_{c}: F32' for (r, c) in cells)}}}

""")


def upd(guard):
    out = []
    for (r, c) in cells:
        for q in range(4):
            prod = f"(x{q}_{r} * y{q}_{c} : F32)"
            if guard:
                prod = f"pk(U32.is_lt({q}, rem), {prod})"
            out.append(f"({sum_name(q, r, c)} + {prod} : F32)")
    return out


sp = j("+" + s + ": F32" for s in S)
vp = j("+" + v for v in V)
fin = []
for (r, c) in cells:
    a = upd(True)[(r * TN + c) * 4: (r * TN + c) * 4 + 4]
    fin.append(f"TA.t8_comb(rot, {j(a)})")
p(f"""# kg more full turns; g holds the current turn's values. The last call (kg = 0) adds the turn that holds the steps
# past the last full turn (rem = k mod 4 of them), counting only those, and combines the sums.
@unsafe
def mm(kg: Nat, +ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +rem: U32, +rot: Nat, {sp}, g: GR) -> T:
  match kg g:
    case 0n GR{{a, b, {vp}}}:
      T{{a, b, {j(fin)}}}
    case 1n+q GR{{a, b, {vp}}}:
      mm(q, (ia + (sa * 4 : U32) : U32), (ib + (sb * 4 : U32) : U32), arow, sa, sb, bcol, rem, rot, {j(upd(False))}, turn((ia + (sa * 4 : U32) : U32), (ib + (sb * 4 : U32) : U32), arow, sa, sb, bcol, a, b))

# writes cell (i, j) of C when it is inside n x m
def wr(ok: Bool, +ac: Bool, +idx: U32, +x: F32, c: Array<F32>) -> Array<F32>:
  match ok:
    case True{{}}: TA.cwr(ac, idx, x, c)
    case False{{}}: c

""")
expr = "c"
for (r, c) in cells:
    ok = f"Bool.and(U32.is_lt((i0 + {r} : U32), n), U32.is_lt((j0 + {c} : U32), m))"
    idx = f"(((i0 + {r} : U32) * m : U32) + (j0 + {c} : U32) : U32)"
    expr = f"wr({ok}, ac, {idx}, c{r}_{c}, {expr})"
p(f"""# the tile at rows i0.., columns j0.. of C, from the loop's result; A and B are dropped (the caller holds them)
@unsafe
def put(+ac: Bool, +n: U32, +m: U32, +i0: U32, +j0: U32, c: Array<F32>, t: T) -> Array<F32>:
  match t:
    case T{{a, b, {j('+' + f'c{r}_{c}' for (r, c) in cells)}}}:
      {expr}

@unsafe
def tile(+e: E, +i0: U32, +j0: U32, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> Array<F32>:
  match e:
    case E{{+n, +m, +k, +tcols, +tiles, +arow, +sa, +sb, +bcol, +kg, +rot, +rem, +ac, +oa, +ob}}:
      put(ac, n, m, i0, j0, c, mm(kg, (oa + (i0 * arow : U32) : U32), (ob + (j0 * bcol : U32) : U32), arow, sa, sb, bcol, rem, rot, {j('0.0' for _ in S)}, turn((oa + (i0 * arow : U32) : U32), (ob + (j0 * bcol : U32) : U32), arow, sa, sb, bcol, a, b)))

# leaf t computes tile t (row-major over the tiles); leaves past the last tile write nothing
@unsafe
def leaf_if(ok: Bool, +e: E, +t: U32, +tcols: U32, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> Array<F32>:
  match ok:
    case True{{}}: tile(e, (U32.div(t, tcols) * {TM} : U32), (U32.mod(t, tcols) * {TN} : U32), a, b, c)
    case False{{}}: c

@unsafe
def leaf(+e: E, +t: U32, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> Array<F32>:
  match e:
    case E{{+n, +m, +k, +tcols, +tiles, +arow, +sa, +sb, +bcol, +kg, +rot, +rem, +ac, +oa, +ob}}:
      leaf_if(U32.is_lt(t, tiles), e, t, tcols, a, b, c)

@unsafe
def tree(+e: E, d: Nat, +t: U32, +a: Array<F32>, +b: Array<F32>, +c: Array<F32>) -> Array<F32>:
  match d:
    case 0n: leaf(e, t, a, b, c)
    case 1n++q:
      l r = tree(e, q, (t * 2 : U32), a, b, c) tree(e, q, ((t * 2 : U32) + 1 : U32), a, b, c)
      Array.join(F32, l, r)

# C made inside the bang: the root of the ! call allocates it on the device. Made on the host instead, a new C often
# reuses a block whose pages sit on the device, so zeroing it on the host pulls them back over PCIe and the bang moves
# them again (a 256 x 1536 product: 9.2 ms against 6.0, measured 2026-10-18). The leaves write every cell of C.
@unsafe
def tree_new(+e: E, d: Nat, +a: Array<F32>, +b: Array<F32>, +cd: Nat) -> Array<F32>:
  tree(e, d, 0, a, b, Array.new(F32, cd, 0.0))

@unsafe
def bang_new(+e: E, d: Nat, +a: Array<F32>, +b: Array<F32>, +cd: Nat) -> Array<F32>:
  tree_new!(e, d, a, b, cd)

# one bang over the tiles; A and B are shared with every leaf and come back unchanged
@unsafe
def bang(+e: E, d: Nat, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> Array<F32>:
  tree!(e, d, 0, a, b, c)

type ABC is Type:
  ABC{{a: Array<F32>, b: Array<F32>, c: Array<F32>}}

@unsafe
def run_e(+e: E, +tiles: U32, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> ABC:
  ABC{{a, b, bang(e, TA.cap_depth(U32.to_nat(tiles)), a, b, c)}}

# the product's sizes and strides
def e_ofo(+ac: Bool, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +oa: U32, +ob: U32) -> E:
  E{{U32.from_nat(n), U32.from_nat(m), U32.from_nat(k), U32.from_nat(Nat.div(Nat.add(m, {TN - 1}n), {TN}n)), U32.from_nat(Nat.mul(Nat.div(Nat.add(n, {TM - 1}n), {TM}n), Nat.div(Nat.add(m, {TN - 1}n), {TN}n))), arow, sa, sb, bcol, Nat.div(k, 4n), Nat.mod(k, 4n), U32.from_nat(Nat.mod(k, 4n)), ac, oa, ob}}

def e_of(+ac: Bool, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32) -> E:
  e_ofo(ac, n, m, k, arow, sa, sb, bcol, 0, 0)

def tiles(+n: Nat, +m: Nat) -> Nat:
  Nat.mul(Nat.div(Nat.add(n, {TM - 1}n), {TM}n), Nat.div(Nat.add(m, {TN - 1}n), {TN}n))

# C (n x m) = A · B with k steps into a new C made on the device
@unsafe
def gemm_new(+n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +a: Array<F32>, +b: Array<F32>) -> ABC:
  ABC{{a, b, bang_new(e_of(False{{}}, n, m, k, arow, sa, sb, bcol), TA.cap_depth(tiles(n, m)), a, b, TA.cap_depth(Nat.mul(n, m)))}}

# C (n x m) = A · B with k steps, or C + A · B when ac, C[i][j] = sum_p A[i*arow + p*sa] * B[p*sb + j*bcol]
@unsafe
def gemm(+ac: Bool, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> ABC:
  run_e(e_of(ac, n, m, k, arow, sa, sb, bcol), U32.from_nat(tiles(n, m)), a, b, c)

def mm_fin(-n: Nat, -k: Nat, -m: Nat, g: ABC) -> TA.MMul<n, k, m>:
  match g:
    case ABC{{a, b, c}}: TA.MMul{{TA.Mat{{a}}, TA.Mat{{b}}, TA.Mat{{c}}}}

def mm_nt_fin(-n: Nat, -k: Nat, -m: Nat, g: ABC) -> TA.MMulNT<n, k, m>:
  match g:
    case ABC{{a, b, c}}: TA.MMulNT{{TA.Mat{{a}}, TA.Mat{{b}}, TA.Mat{{c}}}}

def mm_tn_fin(-n: Nat, -k: Nat, -m: Nat, g: ABC) -> TA.MMulTN<n, k, m>:
  match g:
    case ABC{{a, b, c}}: TA.MMulTN{{TA.Mat{{a}}, TA.Mat{{b}}, TA.Mat{{c}}}}

# C = A · B   (n x k) · (k x m), tensor-array's TA.Mat.matmul bits
@unsafe
def matmul(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<k, m>) -> TA.MMul<n, k, m>:
  match a b:
    case TA.Mat{{x}} TA.Mat{{y}}:
      mm_fin(n, k, m, gemm_new(n, m, k, U32.from_nat(k), 1, U32.from_nat(m), 1, x, y))

# C + A · B, into C (n x m)
@unsafe
def matmul_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<k, m>, c: TA.Mat<n, m>) -> TA.MMul<n, k, m>:
  match a b c:
    case TA.Mat{{x}} TA.Mat{{y}} TA.Mat{{z}}:
      mm_fin(n, k, m, gemm(True{{}}, n, m, k, U32.from_nat(k), 1, U32.from_nat(m), 1, x, y, z))

# C = A · Bᵀ   (n x k) · (m x k)ᵀ: for weights stored with one row per output
@unsafe
def matmul_nt(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<m, k>) -> TA.MMulNT<n, k, m>:
  match a b:
    case TA.Mat{{x}} TA.Mat{{y}}:
      mm_nt_fin(n, k, m, gemm_new(n, m, k, U32.from_nat(k), 1, 1, U32.from_nat(k), x, y))

# C + A · Bᵀ, into C (n x m)
@unsafe
def matmul_nt_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<m, k>, c: TA.Mat<n, m>) -> TA.MMulNT<n, k, m>:
  match a b c:
    case TA.Mat{{x}} TA.Mat{{y}} TA.Mat{{z}}:
      mm_nt_fin(n, k, m, gemm(True{{}}, n, m, k, U32.from_nat(k), 1, 1, U32.from_nat(k), x, y, z))

# C = Aᵀ · B   (k x n)ᵀ · (k x m): weight gradient, Xᵀ · dY
@unsafe
def matmul_tn(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<k, n>, b: TA.Mat<k, m>) -> TA.MMulTN<n, k, m>:
  match a b:
    case TA.Mat{{x}} TA.Mat{{y}}:
      mm_tn_fin(n, k, m, gemm_new(n, m, k, 1, U32.from_nat(n), U32.from_nat(m), 1, x, y))

# C + Aᵀ · B, into C (n x m): dW += Xᵀ · dY
@unsafe
def matmul_tn_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<k, n>, b: TA.Mat<k, m>, c: TA.Mat<n, m>) -> TA.MMulTN<n, k, m>:
  match a b c:
    case TA.Mat{{x}} TA.Mat{{y}} TA.Mat{{z}}:
      mm_tn_fin(n, k, m, gemm(True{{}}, n, m, k, 1, U32.from_nat(n), U32.from_nat(m), 1, x, y, z))
""")

# ---- bangs over any leaf code (0.1.1.0): run1..run4, run1_new..run4_new ----
OT = {1: "Array<F32>", 2: "O2", 3: "O3", 4: "O4"}
FN = "@+t: U32 -> @+l: U32 -> @+e: E -> Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> {o} -> {o}"
XS = "+x1: Array<F32>, +x2: Array<F32>, +x3: Array<F32>, +x4: Array<F32>"
XL = "x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>"
p("""
# =====================================================================================================================
# Bangs over any leaf code (0.1.1.0). runK and runK_new (K = 1..4 outputs) run one bang over 2^d leaves: leaf t of
# l = 2^d gets t, l, a record e of scalars (the same for every leaf), four input arrays and the K outputs, and returns
# the outputs. The leaf code is a template (~f), written by the caller in plain Bend: it takes every array as an ordinary
# argument (it reads them with Array.get and may write the outputs), so it needs no @unsafe; the sharing is here, where
# every node of the tree holds a handle to each array (Array.fork in Base, upstream's one-block pattern) and joins the
# outputs (Array.join). runK takes the outputs from the caller and writes them in place (an optimizer's weights and
# moments); runK_new makes them inside the bang with 2^c1.. cells, since an array made on the host pulls device pages
# back (see bang_new above). Both return the inputs unchanged with the outputs (RK). An unused input is any array, such
# as none(Unit{}).
#
# Not checked, as for the products: that a leaf writes only its own cells of the outputs (two leaves writing one cell
# race) and never an input. Leaves past the work must return their outputs unchanged; share(t, n, l) is how many of the
# cells t, t + l, t + 2l, .. lie below n (leaf t's cells when n cells are dealt round-robin, so neighbouring lanes read
# neighbouring cells), and depth(n) is a tree for n cells: 2^14 leaves once there are that many, else the fewest that
# cover them. The leaves run in a run-time order inside each group of 128 (bendlang/bend's CUDA lowering), so a leaf's
# result must not depend on another's. Every number a leaf computes is the caller's own code, so the bits are the
# caller's: the same on the GPU as on the CPU threads, except where CUDA's math functions differ from the host's (an exp
# on the device takes the float overload; the known limit of bendlang/bend#1305).
# =====================================================================================================================

def depth2(big: Bool, +n: Nat) -> Nat:
  match big:
    case True{}: 14n
    case False{}: TA.cap_depth(n)

# leaves for n cells: 2^14 when there are that many, else the fewest powers of two that cover them
def depth(+n: Nat) -> Nat:
  depth2(Nat.is_le(16384n, n), n)

# how many of the cells t, t + l, t + 2l, .. lie below n
def share(+t: U32, +n: U32, +l: U32) -> Nat:
  U32.to_nat(Bool.pick(U32, U32.is_lt(t, n), U32.div(((n - t : U32) + (l - 1 : U32) : U32), l), 0))

# an array of one cell, for an unused input
def none(u: Unit) -> Array<F32>:
  Array.new(F32, 0n, 0.0)

type O2 is Type:
  O2{a: Array<F32>, b: Array<F32>}

type O3 is Type:
  O3{a: Array<F32>, b: Array<F32>, c: Array<F32>}

type O4 is Type:
  O4{a: Array<F32>, b: Array<F32>, c: Array<F32>, d: Array<F32>}

type R1 is Type:
  R1{x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, a: Array<F32>}

type R2 is Type:
  R2{x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: O2}

type R3 is Type:
  R3{x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: O3}

type R4 is Type:
  R4{x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: O4}

@unsafe
def join2(x: O2, y: O2) -> O2:
  match x y:
    case O2{a, b} O2{a2, b2}: O2{Array.join(F32, a, a2), Array.join(F32, b, b2)}

@unsafe
def join3(x: O3, y: O3) -> O3:
  match x y:
    case O3{a, b, c} O3{a2, b2, c2}: O3{Array.join(F32, a, a2), Array.join(F32, b, b2), Array.join(F32, c, c2)}

@unsafe
def join4(x: O4, y: O4) -> O4:
  match x y:
    case O4{a, b, c, d} O4{a2, b2, c2, d2}: O4{Array.join(F32, a, a2), Array.join(F32, b, b2), Array.join(F32, c, c2), Array.join(F32, d, d2)}
""")
for k in range(1, 5):
    o_t, f_t = OT[k], FN.format(o=OT[k])
    tpl = f"~E: Data, ~f: {f_t}"
    ta = "~E, ~f"
    names = ["a", "b", "c", "d"][:k]
    outs_s = ", ".join(f"+o{n}: Array<F32>" for n in names)
    outs_l = ", ".join(f"o{n}: Array<F32>" for n in names)
    oargs = ", ".join(f"o{n}" for n in names)
    pack = "o" + names[0] if k == 1 else f"O{k}{{{oargs}}}"
    caps = ", ".join(f"+c{n}: Nat" for n in names)
    cargs = ", ".join(f"c{n}" for n in names)
    news = ", ".join(f"Array.new(F32, c{n}, 0.0)" for n in names)
    join = "Array.join(F32, x, y)" if k == 1 else f"join{k}(x, y)"
    rk = f"R{k}"
    p(f"""
@unsafe
def tree{k}({tpl}, +d: Nat, +t: U32, +l: U32, +e: E, {XS}, {outs_s}) -> {o_t}:
  match d:
    case 0n: f(t, l, e, x1, x2, x3, x4, {pack})
    case 1n+q:
      x y = tree{k}({ta}, q, (t * 2 : U32), l, e, x1, x2, x3, x4, {oargs}) tree{k}({ta}, q, ((t * 2 : U32) + 1 : U32), l, e, x1, x2, x3, x4, {oargs})
      {join}
""")
    p(f"""
@unsafe
def root{k}({tpl}, +d: Nat, +e: E, {XS}, {outs_l}) -> {o_t}:
  tree{k}({ta}, d, 0, U32.pow(2, d), e, x1, x2, x3, x4, {oargs})
""")
    p(f"""
@unsafe
def root{k}_new({tpl}, +d: Nat, +e: E, {XS}, {caps}) -> {o_t}:
  root{k}({ta}, d, e, x1, x2, x3, x4, {news})

@unsafe
def go{k}({tpl}, +d: Nat, +e: E, {XS}, {outs_l}) -> {rk}:
  {rk}{{x1, x2, x3, x4, root{k}!({ta}, d, e, x1, x2, x3, x4, {oargs})}}

@unsafe
def go{k}_new({tpl}, +d: Nat, +e: E, {XS}, {caps}) -> {rk}:
  {rk}{{x1, x2, x3, x4, root{k}_new!({ta}, d, e, x1, x2, x3, x4, {cargs})}}

# one bang over 2^d leaves with {k} output{'s' if k > 1 else ''} written in place
@unsafe
def run{k}({tpl}, +d: Nat, +e: E, {XL}, {outs_l}) -> {rk}:
  go{k}({ta}, d, e, x1, x2, x3, x4, {oargs})

# one bang over 2^d leaves with {k} new output{'s' if k > 1 else ''} of 2^c.. cells, made on the device
@unsafe
def run{k}_new({tpl}, +d: Nat, +e: E, {XL}, {caps}) -> {rk}:
  go{k}_new({ta}, d, e, x1, x2, x3, x4, {cargs})
""")
p("""
# ---- tensor-array 0.1.14.0's precise product on the GPU (opt-in) ----
# TA.Mat.matmul_precise's sums, a bang a block of 1024 steps that starts at the block's place in A and B (oa, ob; no
# copy): the first block's bang makes C on the device, each next one adds its cells into C as the accumulating kernel
# does (once, as tensor-array's cwr), so every cell is ((x0 + x1) + x2) .., the same bits as tensor-array's
# (tests/precise.bend). Only plain (not accumulating) products, as tensor-array's matmul_precise.

def blk_len(+bi: U32, +k: U32) -> Nat:
  U32.to_nat(Bool.pick(U32, U32.is_lt((k - (bi * 1024 : U32) : U32), 1024), (k - (bi * 1024 : U32) : U32), 1024))

# the blocks after the first: nb more, from block bi, each added into C
@unsafe
def blocks_acc(nb: Nat, +bi: U32, +n: Nat, +m: Nat, +k: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> Array<F32>:
  match nb:
    case 0n: c
    case 1n++q: blocks_acc(q, (bi + 1 : U32), n, m, k, arow, sa, sb, bcol, a, b, bang(e_ofo(True{}, n, m, blk_len(bi, k), arow, sa, sb, bcol, ((bi * 1024 : U32) * sa : U32), ((bi * 1024 : U32) * sb : U32)), TA.cap_depth(tiles(n, m)), a, b, c))

@unsafe
def precise_c(+n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +a: Array<F32>, +b: Array<F32>) -> Array<F32>:
  blocks_acc(Nat.sub(U32.to_nat(U32.div((U32.from_nat(k) + 1023 : U32), 1024)), 1n), 1, n, m, U32.from_nat(k), arow, sa, sb, bcol, a, b, bang_new(e_ofo(False{}, n, m, 1024n, arow, sa, sb, bcol, 0, 0), TA.cap_depth(tiles(n, m)), a, b, TA.cap_depth(Nat.mul(n, m))))

@unsafe
def mm_precise3(+n: Nat, +k: Nat, +m: Nat, +a: Array<F32>, +b: Array<F32>) -> TA.MMul<n, k, m>:
  mm_fin(n, k, m, ABC{a, b, precise_c(n, m, k, U32.from_nat(k), 1, U32.from_nat(m), 1, a, b)})

@unsafe
def mm_precise2(long: Bool, +n: Nat, +k: Nat, +m: Nat, +par: Nat, a: TA.Mat<n, k>, b: TA.Mat<k, m>) -> TA.MMul<n, k, m>:
  match long a b:
    case True{} TA.Mat{x} TA.Mat{y}: mm_precise3(n, k, m, x, y)
    case False{} a b: matmul(n, k, m, par, a, b)

# C = A · B with k summed in blocks of 1024: TA.Mat.matmul_precise's bits (opt-in)
@unsafe
def matmul_precise(+n: Nat, +k: Nat, +m: Nat, +par: Nat, a: TA.Mat<n, k>, b: TA.Mat<k, m>) -> TA.MMul<n, k, m>:
  mm_precise2(Nat.is_lt(1024n, k), n, k, m, par, a, b)
""")
p("""
# =====================================================================================================================
# Counter-based random fills (prototype). fill_uniform and fill_gauss make n numbers of a counter s in one bang, each
# number a pure function of (s, its index), so a lane computes its cells without reading anything; uniform_mat and
# gauss_mat give them as a tensor-array Mat. The _at forms (fill_uniform_at, fill_gauss_at, uniform_mat_at, gauss_mat_at)
# make the cells i0 .. i0 + n - 1 of the same counter, so a batch can be made in slices: their cell k is the cell
# function at i0 + k, and the Gaussians' pairs stay global (2p, 2p + 1), so an odd i0 starts with the sine half of the
# pair (i0 - 1, i0). The plain forms are the case i0 = 0. Indexes are U32 and wrap: i0 + k is taken mod 2^32. The
# functions behind them (rng_uniform, rng_gauss2, rng_gauss) are plain
# Bend that a caller's own leaf (runK) can call too, to make noise where it is used (x_t = a * x_0 + b * noise in one
# bang) instead of in an array of its own.
#
# The counter hash and the uniform are mlbend-generative 0.1.0.0's (its mix32 over s * 0x9E3779B9 + i + 0x1234567A,
# the low 24 bits plus one over 2^24, a uniform in (0, 1]), so rng_uniform gives that package's words. A Gaussian is
# Box-Muller over the pair (2p, 2p + 1): cell 2p is r cos(2 pi u_b), cell 2p + 1 is r sin(2 pi u_b), with
# r = sqrt(-2 ln u_a), u_a the uniform of index 2p and u_b of 2p + 1, so each pair costs one ln, one sqrt and one sine
# and cosine. ln, sin and cos are written here with + - * / and the exponent's bits, not F32.log/sin/cos: CUDA's float
# functions give other bits than the host's (bendlang/bend#1305), while + - * / and sqrt are
# IEEE's on every lane and Bend fuses no multiply-add, so these numbers have the same bits on the GPU, on the CPU
# threads and on the JS lane. They are within a few ulp of the libm values (tests/fill.bend measures it), not equal.
# =====================================================================================================================

def rng_word(x: F32) -> U32:
  match x:
    case F32{w}: U32{w}

def rng_f32(x: U32) -> F32:
  match x:
    case U32{w}: F32{w}

# mlbend-generative's mix32: xor-shift 16, multiply, xor-shift 13, multiply, xor-shift 16, over U32 that wraps
def rng_mix(+z: U32) -> U32:
  +a = U32.xor(z, U32.shrn(z, 16n))
  +b = (a * 2246822519 : U32)
  +c = U32.xor(b, U32.shrn(b, 13n))
  +e = (c * 3266489917 : U32)
  U32.xor(e, U32.shrn(e, 16n))

# the hash of index i of counter s
def rng_hash(+s: U32, +i: U32) -> U32:
  rng_mix(((s * 2654435769 : U32) + (i + 305419898 : U32) : U32))

# the uniform of index i of counter s, in (0, 1]: the hash's low 24 bits plus one, over 2^24 (exact)
def rng_uniform(+s: U32, +i: U32) -> F32:
  (U32.to_f32((U32.and(rng_hash(s, i), 16777215) + 1 : U32)) / 16777216.0 : F32)

# ln m for m in [sqrt(1/2), sqrt(2)], from s = (m - 1) / (m + 1): ln m = 2 (s + s^3/3 + s^5/5 + s^7/7 + s^9/9)
def rng_ln_m(+ex: F32, +m: F32) -> F32:
  +s = ((m - 1.0 : F32) / (m + 1.0 : F32) : F32)
  +z = (s * s : F32)
  +c3 = (0.14285715 + (z * 0.11111111 : F32) : F32)
  +c2 = (0.2 + (z * c3 : F32) : F32)
  +p = (z * (0.33333334 + (z * c2 : F32) : F32) : F32)
  +t = (s + s : F32)
  ((ex * 0.693145751953125 : F32) + ((t + (t * p : F32) : F32) + (ex * 0.0000014286067653 : F32) : F32) : F32)

# ln x for a positive normal x: x = m * 2^ex with m in [sqrt(1/2), sqrt(2)) read from the bits (a mantissa above
# sqrt(2)'s, 0x3504F3, is halved and the exponent raised by one); ex * ln 2 in two parts, the first exact
def rng_ln_w(+w: U32) -> F32:
  +big = U32.is_lt(3474675, U32.and(w, 8388607))
  +mb = U32.or(U32.and(w, 8388607), Bool.pick(U32, big, 1056964608, 1065353216))
  rng_ln_m((U32.to_f32(U32.shrn(w, 23n)) - Bool.pick(F32, big, 126.0, 127.0) : F32), rng_f32(mb))

def rng_ln(x: F32) -> F32:
  rng_ln_w(rng_word(x))

# sin x and cos x for x in [0, pi/4], Taylor to x^9 and x^10 (the next terms are below 2^-28)
def rng_sin(+x: F32) -> F32:
  +z = (x * x : F32)
  +c4 = (0.0001984127 - (z * 0.0000027557319 : F32) : F32)
  +c3 = (0.008333334 - (z * c4 : F32) : F32)
  +c2 = (0.16666667 - (z * c3 : F32) : F32)
  (x - ((x * z : F32) * c2 : F32) : F32)

def rng_cos(+x: F32) -> F32:
  +z = (x * x : F32)
  +c5 = (0.0000248015873 - (z * 0.00000027557319 : F32) : F32)
  +c4 = (0.0013888889 - (z * c5 : F32) : F32)
  +c3 = (0.041666668 - (z * c4 : F32) : F32)
  +c2 = (0.5 - (z * c3 : F32) : F32)
  (1.0 - (z * c2 : F32) : F32)

# the result of rng_sc: sin and cos of one angle
type SC is Data:
  SC{s: F32, c: F32}

# sin and cos of a quarter turn's first eighth (x) or second (pi/2 - x, the two swapped), then turned by q quarters
def rng_sc_q(+q: U32, +sx: F32, +cx: F32) -> SC:
  SC{Bool.pick(F32, U32.is_eq(q, 0), sx, Bool.pick(F32, U32.is_eq(q, 1), cx, Bool.pick(F32, U32.is_eq(q, 2), F32.neg(sx), F32.neg(cx)))), Bool.pick(F32, U32.is_eq(q, 0), cx, Bool.pick(F32, U32.is_eq(q, 1), F32.neg(sx), Bool.pick(F32, U32.is_eq(q, 2), F32.neg(cx), sx)))}

def rng_sc_x(+q: U32, +hi: Bool, +x: F32) -> SC:
  +a = rng_sin(x)
  +b = rng_cos(x)
  rng_sc_q(q, Bool.pick(F32, hi, b, a), Bool.pick(F32, hi, a, b))

# sin and cos of 2 pi v / 2^24 (v < 2^24): the quarter q is v's top 2 bits, w the rest; past the eighth (2^21) the angle
# is taken from the quarter's end (2^22 - w, exact), so the polynomial only sees [0, pi/4]
def rng_sc_w(+q: U32, +w: U32) -> SC:
  +hi = U32.is_lt(2097152, w)
  rng_sc_x(q, hi, (U32.to_f32(Bool.pick(U32, hi, (4194304 - w : U32), w)) * 0.00000037450702 : F32))

def rng_sc(+v: U32) -> SC:
  rng_sc_w(U32.shrn(v, 22n), U32.and(v, 4194303))

def rng_pair(+r: F32, a: SC) -> F32 & F32:
  match a:
    case SC{+s, +c}: Tuple{(r * c : F32), (r * s : F32)}

# the Gaussians of cells 2p and 2p + 1 of counter s (Box-Muller). The angle's 24 bits are u_b's (its hash plus one,
# mod 2^24), so the angle is 2 pi u_b exactly before rounding
def rng_gauss2(+s: U32, +p: U32) -> F32 & F32:
  +i = (p + p : U32)
  rng_pair(F32.sqrt((0.0 - (2.0 * rng_ln(rng_uniform(s, i)) : F32) : F32)), rng_sc(U32.and((rng_hash(s, (i + 1 : U32)) + 1 : U32), 16777215)))

def rng_pick(+odd: Bool, g: F32 & F32) -> F32:
  match g:
    case Tuple{+a, +b}: Bool.pick(F32, odd, b, a)

# the Gaussian of cell i alone (fill_gauss's cell i)
def rng_gauss(+s: U32, +i: U32) -> F32:
  rng_pick(U32.is_eq(U32.and(i, 1), 1), rng_gauss2(s, U32.shrn(i, 1n)))

# the counter, the first index and the number of cells, the same for every leaf
type FE is Data:
  FE{s: U32, i0: U32, n: U32}

def fu_go(k: Nat, +i: U32, +l: U32, +s: U32, +i0: U32, o: Array<F32>) -> Array<F32>:
  match k:
    case 0n: o
    case 1n+q: fu_go(q, (i + l : U32), l, s, i0, Array.set(F32, o, i, rng_uniform(s, (i0 + i : U32))))

# leaf t writes the cells t, t + l, .. below n, cell k the uniform of index i0 + k
def fu_leaf(+t: U32, +l: U32, +e: FE, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: Array<F32>) -> Array<F32>:
  match e:
    case FE{+s, +i0, +n}: fu_go(share(t, n, l), t, l, s, i0, o)

# writes cell i when it is below n (the cell before the first of an odd i0, which wraps to 2^32 - 1, and the cell after
# the last are not written)
def fg_if(ok: Bool, +i: U32, +x: F32, o: Array<F32>) -> Array<F32>:
  match ok:
    case True{}: Array.set(F32, o, i, x)
    case False{}: o

def fg_put(+i: U32, +n: U32, g: F32 & F32, o: Array<F32>) -> Array<F32>:
  match g:
    case Tuple{+a, +b}: fg_if(U32.is_lt((i + 1 : U32), n), (i + 1 : U32), b, fg_if(U32.is_lt(i, n), i, a, o))

# pair j of the slice is the global pair q + j (q = i0 / 2), cells 2j - h and 2j - h + 1 of the output (h = i0 mod 2)
def fg_go(k: Nat, +j: U32, +l: U32, +s: U32, +q: U32, +h: U32, +n: U32, o: Array<F32>) -> Array<F32>:
  match k:
    case 0n: o
    case 1n+r: fg_go(r, (j + l : U32), l, s, q, h, n, fg_put(((j + j : U32) - h : U32), n, rng_gauss2(s, (q + j : U32)), o))

# the number of global pairs that cells i0 .. i0 + n - 1 touch: (i0 mod 2 + n + 1) / 2
def fg_pairs(+i0: U32, +n: U32) -> U32:
  U32.shrn(((U32.and(i0, 1) + n : U32) + 1 : U32), 1n)

# leaf t writes the pairs t, t + l, .. of the slice's pairs, each into the cells of it that lie in the slice
def fg_leaf(+t: U32, +l: U32, +e: FE, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: Array<F32>) -> Array<F32>:
  match e:
    case FE{+s, +i0, +n}: fg_go(share(t, fg_pairs(i0, n), l), t, l, s, U32.shrn(i0, 1n), U32.and(i0, 1), n, o)

def r1_out(r: R1) -> Array<F32>:
  match r:
    case R1{x1, x2, x3, x4, a}: a

# n uniforms of counter s from index i0, cell k = rng_uniform(s, i0 + k) (mod 2^32), in a new array of 2^c cells made
# on the device (one bang)
@unsafe
def fill_uniform_at(+s: U32, +i0: U32, +n: Nat, +c: Nat) -> Array<F32>:
  r1_out(run1_new(~FE, ~fu_leaf, depth(n), FE{s, i0, U32.from_nat(n)}, none(Unit{}), none(Unit{}), none(Unit{}), none(Unit{}), c))

# n Gaussians of counter s from index i0, cell k = rng_gauss(s, i0 + k) (mod 2^32), in a new array of 2^c cells made
# on the device (one bang, a lane a global pair of cells; an odd i0 starts with the sine half of the pair (i0 - 1, i0))
@unsafe
def fill_gauss_at(+s: U32, +i0: U32, +n: Nat, +c: Nat) -> Array<F32>:
  r1_out(run1_new(~FE, ~fg_leaf, depth(U32.to_nat(fg_pairs(i0, U32.from_nat(n)))), FE{s, i0, U32.from_nat(n)}, none(Unit{}), none(Unit{}), none(Unit{}), none(Unit{}), c))

# n uniforms of counter s, cell i = rng_uniform(s, i), in a new array of 2^c cells made on the device (one bang)
@unsafe
def fill_uniform(+s: U32, +n: Nat, +c: Nat) -> Array<F32>:
  fill_uniform_at(s, 0, n, c)

# n Gaussians of counter s, cell i = rng_gauss(s, i), in a new array of 2^c cells made on the device (one bang, a lane a
# pair of cells)
@unsafe
def fill_gauss(+s: U32, +n: Nat, +c: Nat) -> Array<F32>:
  fill_gauss_at(s, 0, n, c)

# an r x m Mat of uniforms of counter s, row-major (cell i * m + j = rng_uniform(s, i * m + j))
@unsafe
def uniform_mat(+s: U32, +r: Nat, +m: Nat) -> TA.Mat<r, m>:
  TA.Mat{fill_uniform(s, Nat.mul(r, m), TA.cap_depth(Nat.mul(r, m)))}

# an r x m Mat of Gaussians of counter s, row-major (cell i * m + j = rng_gauss(s, i * m + j))
@unsafe
def gauss_mat(+s: U32, +r: Nat, +m: Nat) -> TA.Mat<r, m>:
  TA.Mat{fill_gauss(s, Nat.mul(r, m), TA.cap_depth(Nat.mul(r, m)))}

# an r x m Mat of uniforms of counter s from index i0 (cell i * m + j = rng_uniform(s, i0 + i * m + j)); rows b .. b + r - 1
# of an R x m uniform_mat are uniform_mat_at(s, b * m, r, m)
@unsafe
def uniform_mat_at(+s: U32, +i0: U32, +r: Nat, +m: Nat) -> TA.Mat<r, m>:
  TA.Mat{fill_uniform_at(s, i0, Nat.mul(r, m), TA.cap_depth(Nat.mul(r, m)))}

# an r x m Mat of Gaussians of counter s from index i0 (cell i * m + j = rng_gauss(s, i0 + i * m + j)); rows
# b .. b + r - 1 of an R x m gauss_mat are gauss_mat_at(s, b * m, r, m)
@unsafe
def gauss_mat_at(+s: U32, +i0: U32, +r: Nat, +m: Nat) -> TA.Mat<r, m>:
  TA.Mat{fill_gauss_at(s, i0, Nat.mul(r, m), TA.cap_depth(Nat.mul(r, m)))}
""")
open(OUT, "w").write("\n".join(o) if False else "".join(x if x.endswith("\n") else x + "\n" for x in o))
print(f"{OUT}: {TM}x{TN} tiles, {len(S)} sums, {len(V)} values a turn")
