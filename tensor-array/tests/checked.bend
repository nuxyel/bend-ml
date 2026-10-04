import Base
import ../main.bend as TA

# 1 if the checked cross-entropy accepted the labels, 0 if it returned None
def ce_flag(-n: Nat, -c: Nat, x: Maybe<&1, TA.MCE<n, c>>) -> Nat:
  match x:
    case Some{v}:
      1n
    case None{}:
      0n

def hits_flag(-n: Nat, -c: Nat, x: Maybe<&1, TA.MHits<n, c>>) -> Nat:
  match x:
    case Some{v}:
      1n
    case None{}:
      0n

def labels(k: Nat) -> List<&2, Nat>:
  match k:
    case 0n:
      Nil{}
    case 1n+p:
      Con{0n, labels(p)}

# digits: ce(2 labels) ce(1) ce(3) hits(2) hits(1) hits(3) for a 2x3 matrix; expected 100100
def main() -> List<&2, Nat>:
  [ce_flag(2n, 3n, TA.Mat.softmax_ce_checked(2n, 3n, TA.Mat.zeros(2n, 3n), labels(2n))),
   ce_flag(2n, 3n, TA.Mat.softmax_ce_checked(2n, 3n, TA.Mat.zeros(2n, 3n), labels(1n))),
   ce_flag(2n, 3n, TA.Mat.softmax_ce_checked(2n, 3n, TA.Mat.zeros(2n, 3n), labels(3n))),
   hits_flag(2n, 3n, TA.Mat.count_correct_checked(2n, 3n, TA.Mat.zeros(2n, 3n), labels(2n))),
   hits_flag(2n, 3n, TA.Mat.count_correct_checked(2n, 3n, TA.Mat.zeros(2n, 3n), labels(1n))),
   hits_flag(2n, 3n, TA.Mat.count_correct_checked(2n, 3n, TA.Mat.zeros(2n, 3n), labels(3n)))]
