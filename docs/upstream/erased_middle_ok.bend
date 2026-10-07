# erased_last_bad.bend with -z moved before the runtime parameter i: the fork comes back.
#
#   bend erased_middle_ok.bend -o mid && time ./mid --threads 1 && time ./mid --threads 16
import Base

def work(n: Nat, acc: U32) -> U32:
  match n:
    case 0n:
      acc
    case 1n+p:
      work(p, (acc * 1103515245 + 12345 : U32))

def tree(+d: Nat, -z: Nat, +i: U32) -> U32:
  match d:
    case 0n:
      work(600000000n, i)
    case 1n+q:
      a b = tree(q, z, (i * 2 : U32)) tree(q, z, (i * 2 + 1 : U32))
      (a + b : U32)

def main() -> U32:
  tree(4n, 0n, 1)
