# The same program with one extra ERASED parameter, -z, at the end of tree's parameter list,
# passed unchanged. The parallel let is compiled as two sequential calls: --threads 16 takes
# as long as --threads 1. Moving -z before the runtime parameter i (erased_middle_ok.bend),
# or adding any runtime parameter after it, restores the fork.
#
#   bend erased_last_bad.bend -o bad && time ./bad --threads 1 && time ./bad --threads 16
import Base

def work(n: Nat, acc: U32) -> U32:
  match n:
    case 0n:
      acc
    case 1n+p:
      work(p, (acc * 1103515245 + 12345 : U32))

def tree(+d: Nat, +i: U32, -z: Nat) -> U32:
  match d:
    case 0n:
      work(600000000n, i)
    case 1n+q:
      a b = tree(q, (i * 2 : U32), z) tree(q, (i * 2 + 1 : U32), z)
      (a + b : U32)

def main() -> U32:
  tree(4n, 1, 0n)
