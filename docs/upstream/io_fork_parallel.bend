# Four IO.fork computations of a flat loop, seeded at run time so nothing is folded at compile time.
# Bend 2.0.35: 1.86 s with --threads 1, 1.73 s with --threads 16, user time equal to real time.
#
#   bend io_fork_parallel.bend -o f && time ./f --threads 1 && time ./f --threads 16
import Base

def work(n: Nat, acc: U32) -> U32:
  match n:
    case 0n:
      acc
    case 1n+p:
      work(p, (acc * 1103515245 + 12345 : U32))

def job(+seed: U32) -> IO(U32):
  IO.pure(U32, work(4000000000n, seed))

def main() -> IO(Unit):
  do IO<Unit>:
    +t : Nat <- IO.now()
    a : Chan(U32) <- IO.fork(U32, job((U32.from_nat(t) + 1 : U32)))
    b : Chan(U32) <- IO.fork(U32, job((U32.from_nat(t) + 2 : U32)))
    c : Chan(U32) <- IO.fork(U32, job((U32.from_nat(t) + 3 : U32)))
    d : Chan(U32) <- IO.fork(U32, job((U32.from_nat(t) + 4 : U32)))
    x : U32 <- IO.join(U32, a)
    y : U32 <- IO.join(U32, b)
    z : U32 <- IO.join(U32, c)
    w : U32 <- IO.join(U32, d)
    IO.print(U32.show((x + y + z + w : U32)))
