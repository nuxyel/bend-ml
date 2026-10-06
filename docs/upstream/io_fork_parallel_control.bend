# Control for io_fork_parallel.bend: the same 4 loops as two parallel lets of two calls.
# Bend 2.0.35: 1.69 s with --threads 1, 0.85 s with --threads 16 (2x: each let runs 2 calls at once).

import Base

def work(n: Nat, acc: U32) -> U32:
  match n:
    case 0n:
      acc
    case 1n+p:
      work(p, (acc * 1103515245 + 12345 : U32))

def four(+s: U32) -> U32:
  a b = work(4000000000n, (s + 1 : U32)) work(4000000000n, (s + 2 : U32))
  c d = work(4000000000n, (s + 3 : U32)) work(4000000000n, (s + 4 : U32))
  (a + b + c + d : U32)

def main() -> IO(Unit):
  do IO<Unit>:
    t : Nat <- IO.now()
    IO.print(U32.show(four(U32.from_nat(t))))
