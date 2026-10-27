# run1..run4 and run1_new..run4_new (bangs over any leaf code) against the same leaf run once on the host over every
# cell (t = 0, l = 1), bit for bit. Each output a, b, c, d is s * x + o for its input x1, x2, x3, x4 (o: its old value,
# or 0 for a new output), on n = 1, 7, 1000, 16389 (more cells than the 2^14 leaves) and 70000 cells, with the leaves'
# trees from depth(n). Prints, per n: the differing words in the outputs of the eight runs, then in the inputs (the
# shared arrays must come back as they went in): "0 0" five times.
import Base
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ../main.bend as TG

def val(+i: U32, +seed: U32) -> F32:
  +h = U32.mod(((i * 2654435761 : U32) + (seed * 40503 : U32) : U32), 1000)
  Bool.pick(F32, U32.is_eq(h, 7), (0.0 - 0.0 : F32), Bool.pick(F32, U32.is_eq(h, 13), (1.0 / 0.0 : F32), Bool.pick(F32, U32.is_eq(h, 17), (0.0 / 0.0 : F32), (((U32.to_f32(h) / 250.0 : F32) - 2.0 : F32)))))

def fill(k: Nat, +i: U32, +seed: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n: a
    case 1n+p: fill(p, (i + 1 : U32), seed, Array.set(F32, a, i, val(i, seed)))

def arr(+cnt: Nat, +seed: U32) -> Array<F32>:
  fill(cnt, 0, seed, Array.new(F32, TA.cap_depth(cnt), 0.0))

def bits(x: F32) -> U32:
  match x:
    case F32{w}: U32{w}

def cmp(k: Nat, +i: U32, acc: U32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> U32:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}: acc
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      cmp(p, (i + 1 : U32), (acc + Bool.pick(U32, Bool.or(U32.is_eq(bits(x), bits(y)), Bool.and(F32.is_ne(x, x), F32.is_ne(y, y))), 0, 1) : U32), Array.get(F32, a, (i + 1 : U32)), Array.get(F32, b, (i + 1 : U32)))

def diff(+cnt: Nat, x: Array<F32>, y: Array<F32>) -> U32:
  cmp(cnt, 0, 0, Array.get(F32, x, 0), Array.get(F32, y, 0))

type Ev is Data:
  Ev{n: U32, s: F32}

# o[i] = s * x[i] + o[i] for the cells i, i + l, .. (k of them)
def ax(k: Nat, +i: U32, +l: U32, +s: F32, ro: Array<F32> & F32, rx: Array<F32> & F32) -> Array<F32>:
  match k ro rx:
    case 0n Tuple{o, +y} Tuple{x, +v}: o
    case 1n+q Tuple{o, +y} Tuple{x, +v}:
      +u = (i + l : U32)
      ax(q, u, l, s, Array.get(F32, Array.set(F32, o, i, ((s * v : F32) + y : F32)), u), Array.get(F32, x, u))

def one(+t: U32, +l: U32, +e: Ev, o: Array<F32>, x: Array<F32>) -> Array<F32>:
  match e:
    case Ev{+n, +s}: ax(TG.share(t, n, l), t, l, s, Array.get(F32, o, t), Array.get(F32, x, t))

def leaf1(+t: U32, +l: U32, +e: Ev, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: Array<F32>) -> Array<F32>:
  one(t, l, e, o, x1)

def leaf2(+t: U32, +l: U32, +e: Ev, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: TG.O2) -> TG.O2:
  match o:
    case TG.O2{a, b}: TG.O2{one(t, l, e, a, x1), one(t, l, e, b, x2)}

def leaf3(+t: U32, +l: U32, +e: Ev, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: TG.O3) -> TG.O3:
  match o:
    case TG.O3{a, b, c}: TG.O3{one(t, l, e, a, x1), one(t, l, e, b, x2), one(t, l, e, c, x3)}

def leaf4(+t: U32, +l: U32, +e: Ev, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>, o: TG.O4) -> TG.O4:
  match o:
    case TG.O4{a, b, c, d}: TG.O4{one(t, l, e, a, x1), one(t, l, e, b, x2), one(t, l, e, c, x3), one(t, l, e, d, x4)}

def ev(+n: Nat) -> Ev:
  Ev{U32.from_nat(n), 1.5}

def zs(+n: Nat) -> Array<F32>:
  Array.new(F32, TA.cap_depth(n), 0.0)

def o2_4(o: TG.O2, c: Array<F32>, d: Array<F32>) -> TG.O4:
  match o:
    case TG.O2{a, b}: TG.O4{a, b, c, d}

def o3_4(o: TG.O3, d: Array<F32>) -> TG.O4:
  match o:
    case TG.O3{a, b, c}: TG.O4{a, b, c, d}

# the reference: the leaf once on the host, every cell
def ref(+n: Nat, +k: Nat, +inplace: Bool) -> TG.O4:
  match k inplace:
    case 1n True{}: TG.O4{leaf1(0, 1, ev(n), arr(n, 1), arr(n, 2), arr(n, 3), arr(n, 4), arr(n, 11)), zs(n), zs(n), zs(n)}
    case 1n False{}: TG.O4{leaf1(0, 1, ev(n), arr(n, 1), arr(n, 2), arr(n, 3), arr(n, 4), zs(n)), zs(n), zs(n), zs(n)}
    case 2n True{}: o2_4(leaf2(0, 1, ev(n), arr(n, 1), arr(n, 2), arr(n, 3), arr(n, 4), TG.O2{arr(n, 11), arr(n, 12)}), zs(n), zs(n))
    case 2n False{}: o2_4(leaf2(0, 1, ev(n), arr(n, 1), arr(n, 2), arr(n, 3), arr(n, 4), TG.O2{zs(n), zs(n)}), zs(n), zs(n))
    case 3n True{}: o3_4(leaf3(0, 1, ev(n), arr(n, 1), arr(n, 2), arr(n, 3), arr(n, 4), TG.O3{arr(n, 11), arr(n, 12), arr(n, 13)}), zs(n))
    case 3n False{}: o3_4(leaf3(0, 1, ev(n), arr(n, 1), arr(n, 2), arr(n, 3), arr(n, 4), TG.O3{zs(n), zs(n), zs(n)}), zs(n))
    case _ True{}: leaf4(0, 1, ev(n), arr(n, 1), arr(n, 2), arr(n, 3), arr(n, 4), TG.O4{arr(n, 11), arr(n, 12), arr(n, 13), arr(n, 14)})
    case _ False{}: leaf4(0, 1, ev(n), arr(n, 1), arr(n, 2), arr(n, 3), arr(n, 4), TG.O4{zs(n), zs(n), zs(n), zs(n)})

# a run's outputs (padded to four) and the differing words of its inputs against arr(n, 1..4)
type Got is Type:
  Got{o: TG.O4, bad: U32}

def ins(+n: Nat, x1: Array<F32>, x2: Array<F32>, x3: Array<F32>, x4: Array<F32>) -> U32:
  (((diff(n, x1, arr(n, 1)) + diff(n, x2, arr(n, 2)) : U32) + (diff(n, x3, arr(n, 3)) + diff(n, x4, arr(n, 4)) : U32)) : U32)

def g1(+n: Nat, r: TG.R1) -> Got:
  match r:
    case TG.R1{x1, x2, x3, x4, a}: Got{TG.O4{a, zs(n), zs(n), zs(n)}, ins(n, x1, x2, x3, x4)}

def g2(+n: Nat, r: TG.R2) -> Got:
  match r:
    case TG.R2{x1, x2, x3, x4, o}: Got{o2_4(o, zs(n), zs(n)), ins(n, x1, x2, x3, x4)}

def g3(+n: Nat, r: TG.R3) -> Got:
  match r:
    case TG.R3{x1, x2, x3, x4, o}: Got{o3_4(o, zs(n)), ins(n, x1, x2, x3, x4)}

def g4(+n: Nat, r: TG.R4) -> Got:
  match r:
    case TG.R4{x1, x2, x3, x4, o}: Got{o, ins(n, x1, x2, x3, x4)}

@unsafe
def got(+n: Nat, +k: Nat, +inplace: Bool) -> Got:
  got2(n, k, inplace, TG.depth(n), TA.cap_depth(n))

@unsafe
def got2(+n: Nat, +k: Nat, +inplace: Bool, +d: Nat, +c: Nat) -> Got:
  match k inplace:
    case 1n True{}: g1(n, TG.run1(~Ev, ~leaf1, d, ev(n), arr(n, 1), arr(n, 2), arr(n, 3), arr(n, 4), arr(n, 11)))
    case 1n False{}: g1(n, TG.run1_new(~Ev, ~leaf1, d, ev(n), arr(n, 1), arr(n, 2), arr(n, 3), arr(n, 4), c))
    case 2n True{}: g2(n, TG.run2(~Ev, ~leaf2, d, ev(n), arr(n, 1), arr(n, 2), arr(n, 3), arr(n, 4), arr(n, 11), arr(n, 12)))
    case 2n False{}: g2(n, TG.run2_new(~Ev, ~leaf2, d, ev(n), arr(n, 1), arr(n, 2), arr(n, 3), arr(n, 4), c, c))
    case 3n True{}: g3(n, TG.run3(~Ev, ~leaf3, d, ev(n), arr(n, 1), arr(n, 2), arr(n, 3), arr(n, 4), arr(n, 11), arr(n, 12), arr(n, 13)))
    case 3n False{}: g3(n, TG.run3_new(~Ev, ~leaf3, d, ev(n), arr(n, 1), arr(n, 2), arr(n, 3), arr(n, 4), c, c, c))
    case _ True{}: g4(n, TG.run4(~Ev, ~leaf4, d, ev(n), arr(n, 1), arr(n, 2), arr(n, 3), arr(n, 4), arr(n, 11), arr(n, 12), arr(n, 13), arr(n, 14)))
    case _ False{}: g4(n, TG.run4_new(~Ev, ~leaf4, d, ev(n), arr(n, 1), arr(n, 2), arr(n, 3), arr(n, 4), c, c, c, c))

def d4(+n: Nat, x: TG.O4, y: TG.O4) -> U32:
  match x y:
    case TG.O4{a, b, c, d} TG.O4{a2, b2, c2, d2}: (((diff(n, a, a2) + diff(n, b, b2) : U32) + (diff(n, c, c2) + diff(n, d, d2) : U32)) : U32)

type P2 is Data:
  P2{o: U32, i: U32}

def one_run(+n: Nat, +k: Nat, +inplace: Bool, g: Got) -> P2:
  match g:
    case Got{o, +bad}: P2{d4(n, o, ref(n, k, inplace)), bad}

def add3(+a: U32, +b: U32, x: P2, y: P2) -> P2:
  match x y:
    case P2{+xo, +xi} P2{+yo, +yi}: P2{((a + xo : U32) + yo : U32), ((b + xi : U32) + yi : U32)}

@unsafe
def runs(k: Nat, +n: Nat, acc: P2) -> P2:
  match k acc:
    case 0n _: acc
    case 1n++q P2{+a, +b}: runs(q, n, add3(a, b, one_run(n, 1n+q, True{}, got(n, 1n+q, True{})), one_run(n, 1n+q, False{}, got(n, 1n+q, False{}))))

def show2(r: P2) -> String:
  match r:
    case P2{+o, +i}: U32.show(o) ++ " " ++ U32.show(i)

@unsafe
def line(+n: Nat) -> String:
  show2(runs(4n, n, P2{0, 0}))

@unsafe
def test_main() -> IO(Unit):
  do IO<Unit>:
    a : Unit <- IO.print(line(1n))
    b : Unit <- IO.print(line(7n))
    c : Unit <- IO.print(line(1000n))
    d : Unit <- IO.print(line(16389n))
    IO.print(line(70000n))

# This program takes no arguments. Since Bend 2.0.38 a runtime option goes between +RTS and -RTS, and one written
# without them arrives here as an argument while the program runs on the default thread count, so a check of 1 and 16
# threads would pass without testing them: any argument is an error.
def alone(args: List<String>, go: IO(Unit)) -> IO(Unit):
  match args:
    case Nil{}:
      go
    case Con{p, Nil{}}:
      go
    case Con{p, Con{a, t}}:
      do IO<Unit>:
        IO.print_err("unexpected argument " ++ a ++ ": this program takes none (runtime options go between +RTS and -RTS)")
        IO.exit(2)

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    alone(args, test_main())
