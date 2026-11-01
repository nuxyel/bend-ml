# softmax cross-entropy over NR rows of NC logits: tensor-fork's TF.softmax_ce with the rows in 2^PAR blocks
# (PAR = 0 runs tensor-array's TA.Mat.softmax_ce itself). bench/ce_fork.py fills in NR, NC, PAR and MODE:
#   MODE 0: the fill only (the logits made and one cell read), the cost the other modes share
#   MODE 1: the fill and softmax_ce; prints the loss word and three gradient words
#   MODE 2: as 1, then the sum of every gradient word (for the bits, not timed)
import Base
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ../tensor-fork/main.bend as TF

def val(+i: U32) -> F32:
  (((U32.to_f32(U32.mod((i * 2654435761 : U32), 1000)) / 250.0 : F32)) - 2.0 : F32)

def fill(k: Nat, +i: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n: a
    case 1n+p: fill(p, (i + 1 : U32), Array.set(F32, a, i, val(i)))

def labels(k: Nat, +i: U32, +c: U32) -> List<&2, Nat>:
  match k:
    case 0n: Nil{}
    case 1n+p: Con{U32.to_nat(U32.mod((i * 2246822519 : U32), c)), labels(p, (i + 1 : U32), c)}

def bits(x: F32) -> U32:
  match x:
    case F32{w}: U32{w}

def wsum(k: Nat, +i: U32, acc: U32, r: Array<F32> & F32) -> U32:
  match k r:
    case 0n Tuple{a, x}: acc
    case 1n+p Tuple{a, +x}: wsum(p, (i + 1 : U32), (acc + bits(x) : U32), Array.get(F32, a, (i + 1 : U32)))

def w(r: Array<F32> & F32) -> String:
  match r:
    case Tuple{a, +x}: U32.show(bits(x))

def w2(+l: U32, r: Array<F32> & F32) -> String:
  match r:
    case Tuple{a, +x}: U32.show(bits(x)) ++ " " ++ w(Array.get(F32, a, l))

def w3(+m: U32, +l: U32, r: Array<F32> & F32) -> String:
  match r:
    case Tuple{a, +x}: U32.show(bits(x)) ++ " " ++ w2(l, Array.get(F32, a, m))

def show3(all: Bool, +cnt: Nat, a: Array<F32>) -> String:
  match all:
    case True{}: "sum " ++ U32.show(wsum(cnt, 0, 0, Array.get(F32, a, 0)))
    case False{}: w3(U32.from_nat(Nat.div(cnt, 2n)), U32.from_nat(Nat.sub(cnt, 1n)), Array.get(F32, a, 0))

def out(-n: Nat, -c: Nat, +mode: Nat, +cnt: Nat, r: TA.MCE<n, c>) -> String:
  match r:
    case TA.MCE{+loss, TA.Mat{a}}: "loss " ++ U32.show(bits(loss)) ++ " grad " ++ show3(Nat.is_eq(mode, 2n), cnt, a)

def go(fill_only: Bool, +mode: Nat, +n: Nat, +c: Nat, +par: Nat, z: Array<F32>) -> String:
  match fill_only:
    case True{}: "fill " ++ w(Array.get(F32, z, U32.from_nat(Nat.sub(Nat.mul(n, c), 1n))))
    case False{}: out(n, c, mode, Nat.mul(n, c), TF.softmax_ce(n, c, par, TA.Mat{z}, labels(n, 0, U32.from_nat(c))))

def main() -> IO(Unit):
  IO.print(go(Nat.is_eq(MODE, 0n), MODE, NR, NC, PAR, fill(Nat.mul(NR, NC), 0, Array.new(F32, TA.cap_depth(Nat.mul(NR, NC)), 0.0))))
