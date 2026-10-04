import Base
import ./main.bend as BPE

# ---- reading files ----
def rb_done(m: File & Result<&1, &1, U32 & String, List<&2, U32>>) -> IO(List<&2, U32>):
  (f, r) = m
  do IO<List<&2, U32>>:
    bs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
    File.close(f)
    return bs

def read_bytes(path: String) -> IO(List<&2, U32>):
  do IO<List<&2, U32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_bytes(f, 400000000)
    rb_done(m)

# ---- U32/Nat <-> token conversions ----
def toks_of_bytes(bs: List<&2, U32>) -> List<&2, BPE.Tk>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{+b, t}:
      BPE.B{U32.to_nat(b)} <> toks_of_bytes(t)

def tok_id(+t: BPE.Tk) -> Nat:
  match t:
    case BPE.B{+n}:
      n
    case BPE.M{+k}:
      Nat.add(256n, k)

def id_tok2(small: Bool, +n: Nat) -> BPE.Tk:
  match small:
    case True{}:
      BPE.B{n}
    case False{}:
      BPE.M{Nat.sub(n, 256n)}

def id_tok(+n: Nat) -> BPE.Tk:
  id_tok2(Nat.is_lt(n, 256n), n)

# ---- integers in text: any byte that is not a digit separates ----
def digit2(lo: Bool, hi: Bool, +code: U32) -> Maybe<&2, Nat>:
  match lo hi:
    case True{} True{}:
      Some{U32.to_nat(U32.sub(code, 48))}
    case _ _:
      None{}

def digit(+code: U32) -> Maybe<&2, Nat>:
  digit2(U32.is_le(48, code), U32.is_le(code, 57), code)

def head_digit(xs: List<&2, U32>) -> Maybe<&2, Nat>:
  match xs:
    case Nil{}:
      None{}
    case Con{+c, t}:
      digit(c)

def flush(have: Bool, +cur: Nat, acc: List<&2, Nat>) -> List<&2, Nat>:
  match have:
    case True{}:
      cur <> acc
    case False{}:
      acc

def nums.go(xs: List<&2, U32>, d: Maybe<&2, Nat>, +cur: Nat, have: Bool, acc: List<&2, Nat>) -> List<&2, Nat>:
  match xs d:
    case Nil{} _:
      List.reverse(&2, Nat, flush(have, cur, acc))
    case Con{c, +t} Some{+v}:
      nums.go(t, head_digit(t), Nat.add(Nat.mul(cur, 10n), v), True{}, acc)
    case Con{c, +t} None{}:
      nums.go(t, head_digit(t), 0n, False{}, flush(have, cur, acc))

def nums(+xs: List<&2, U32>) -> List<&2, Nat>:
  nums.go(xs, head_digit(xs), 0n, False{}, Nil{})

# pairs of ids become rules with sequential ids
def rules.go(ns: List<&2, Nat>, +k: Nat) -> List<&2, BPE.Rule>:
  match ns:
    case Nil{}:
      Nil{}
    case Con{+a, t}:
      match t:
        case Nil{}:
          Nil{}
        case Con{+b, u}:
          BPE.Rule{k, id_tok(a), id_tok(b)} <> rules.go(u, 1n+k)

# ---- runtime check of a merges table read from a file ----
# The table must have an even number of ids and rule k may only refer to ids that
# exist before it (bytes < 256, or merges 256..256+k-1); BPE.wf must also hold.
def ids_ok(ns: List<&2, Nat>, +k: Nat) -> Bool:
  match ns:
    case Nil{}:
      True{}
    case Con{+a, t}:
      match t:
        case Nil{}:
          False{}
        case Con{+b, u}:
          Bool.and(Bool.and(Nat.is_lt(a, Nat.add(256n, k)), Nat.is_lt(b, Nat.add(256n, k))), ids_ok(u, 1n+k))

def table_ok(+ns: List<&2, Nat>) -> Bool:
  Bool.and(ids_ok(ns, 0n), BPE.wf(rules.go(ns, 0n)))

def toks_of_ids(ns: List<&2, Nat>) -> List<&2, BPE.Tk>:
  match ns:
    case Nil{}:
      Nil{}
    case Con{n, t}:
      id_tok(n) <> toks_of_ids(t)

# ---- output ----
def show_nats(xs: List<&2, Nat>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      Nat.show(h) ++ " " ++ show_nats(t)

def ids_of(xs: List<&2, BPE.Tk>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      tok_id(h) <> ids_of(t)

def rule_ids(rs: List<&2, BPE.Rule>) -> List<&2, Nat>:
  match rs:
    case Nil{}:
      Nil{}
    case Con{BPE.Rule{i, +a, +b}, t}:
      tok_id(a) <> tok_id(b) <> rule_ids(t)

# ---- modes ----
def do_train(n: Nat, bs: List<&2, U32>) -> IO(Unit):
  IO.print(show_nats(rule_ids(BPE.train(n, 0n, toks_of_bytes(bs)))))

def enc_ok(ok: Bool, +ns: List<&2, Nat>, bs: List<&2, U32>) -> IO(Unit):
  match ok:
    case True{}:
      IO.print(show_nats(ids_of(BPE.encode(rules.go(ns, 0n), toks_of_bytes(bs)))))
    case False{}:
      IO.print("invalid merges table")

def dec_ok(ok: Bool, +ns: List<&2, Nat>, ids: List<&2, U32>) -> IO(Unit):
  match ok:
    case True{}:
      IO.print(show_nats(BPE.decode(rules.go(ns, 0n), toks_of_ids(nums(ids)))))
    case False{}:
      IO.print("invalid merges table")

def do_encode(+ms: List<&2, U32>, bs: List<&2, U32>) -> IO(Unit):
  enc_ok(table_ok(nums(ms)), nums(ms), bs)

def do_decode(+ms: List<&2, U32>, ids: List<&2, U32>) -> IO(Unit):
  dec_ok(table_ok(nums(ms)), nums(ms), ids)

def train_n(mn: Maybe<&2, Nat>, bs: List<&2, U32>) -> IO(Unit):
  match mn:
    case None{}:
      IO.print("invalid N")
    case Some{n}:
      do_train(n, bs)

def mode_train(a: String, b: String) -> IO(Unit):
  do IO<Unit>:
    bs : List<&2, U32> <- read_bytes(b)
    train_n(Nat.read(a), bs)

def mode_encode(a: String, b: String) -> IO(Unit):
  do IO<Unit>:
    ms : List<&2, U32> <- read_bytes(a)
    bs : List<&2, U32> <- read_bytes(b)
    do_encode(ms, bs)

def mode_decode(a: String, b: String) -> IO(Unit):
  do IO<Unit>:
    ms : List<&2, U32> <- read_bytes(a)
    bs : List<&2, U32> <- read_bytes(b)
    do_decode(ms, bs)

def dispatch2(is_train: Bool, is_enc: Bool, is_dec: Bool, a: String, b: String) -> IO(Unit):
  match is_train is_enc is_dec:
    case True{} _ _:
      mode_train(a, b)
    case False{} True{} _:
      mode_encode(a, b)
    case False{} False{} True{}:
      mode_decode(a, b)
    case False{} False{} False{}:
      IO.print("usage: cli train N corpus | encode merges text | decode merges ids")

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{+mode, Con{a, Con{b, Nil{}}}}}:
      dispatch2(String.eq(mode, "train"), String.eq(mode, "encode"), String.eq(mode, "decode"), a, b)
    case _:
      IO.print("usage: cli train N corpus | encode merges text | decode merges ids")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
