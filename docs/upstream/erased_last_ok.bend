# Control: a balanced fork tree of 2^4 leaves, each running a flat loop. Scales with --threads.
#
#   bend erased_last_ok.bend -o ok && time ./ok --threads 1 && time ./ok --threads 16
import Base

# n steps of a linear congruential generator, seeded per leaf so no two leaves share work
def work(n: Nat, acc: U32) -> U32:
  match n:
    case 0n:
      acc
    case 1n+p:
      work(p, (acc * 1103515245 + 12345 : U32))

def tree(+d: Nat, +i: U32) -> U32:
  match d:
    case 0n:
      work(30000000n, i)
    case 1n+q:
      a b = tree(q, (i * 2 : U32)) tree(q, (i * 2 + 1 : U32))
      (a + b : U32)

def main() -> U32:
  tree(4n, 1)
