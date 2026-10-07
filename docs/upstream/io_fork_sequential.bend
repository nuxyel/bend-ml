# Baseline for io_fork_parallel.bend: the same four jobs, run one after another, no fork.

import Base

def work(n: Nat, acc: U32) -> U32:
  match n:
    case 0n:
      acc
    case 1n+p:
      work(p, (acc * 1103515245 + 12345 : U32))

# the forked computation needs the value for its own effect, so it cannot leave the work to the joiner
def job(+seed: U32) -> IO(U32):
  do IO<U32>:
    +now : Nat <- IO.now()
    +v : U32 = work(4000000000n, (seed + U32.from_nat(now) : U32))
    IO.print(U32.show(v))
    return v

def main() -> IO(Unit):
  do IO<Unit>:
    +t : Nat <- IO.now()
    x : U32 <- job((U32.from_nat(t) + 1 : U32))
    y : U32 <- job((U32.from_nat(t) + 2 : U32))
    z : U32 <- job((U32.from_nat(t) + 3 : U32))
    w : U32 <- job((U32.from_nat(t) + 4 : U32))
    IO.print(U32.show((x + y + z + w : U32)))
