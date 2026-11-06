# bend-ml-bpe-tokenizer@0.1.3.0 from BendHub, trained and then used: 256 merges learned from a synthetic text of 4,096
# words, then a second text of 32,768 words encoded with them. Integer and list work only. It prints the number of rules
# (256), the number of tokens (31633) and an FNV-1a hash of the token ids (2633430213), the same on every version.
# The words come from a vocabulary of 64 made by a hash, picked with a skew toward the first ones, separated by spaces.
# bpe_encode.sh builds it with two Bend releases and counts both on one pinned thread.
import Base
import bend-ml-bpe-tokenizer@0.1.3.0/main.bend as BPE

def mix(+x: U32) -> U32:
  +h = ((x * 2654435761 : U32) .^. (x >> 15n) : U32)
  ((h * 2246822519 : U32) .^. (h >> 13n) : U32)

# word w (0..63): 2 to 7 letters from the first 12 of the alphabet
def wlen(+w: U32) -> Nat:
  +h = mix(((w * 7 : U32) + 1 : U32))
  U32.to_nat((2 + (h % 6 : U32) : U32))

def letter(+w: U32, +j: U32) -> Nat:
  +h = mix((((w * 131 : U32) + (j * 17 : U32) : U32) + 5 : U32))
  U32.to_nat((97 + (h % 12 : U32) : U32))

# prepends letters 0..k-1 of word w (the last one first)
def letters(k: Nat, +w: U32, acc: List<&2, Nat>) -> List<&2, Nat>:
  match k:
    case Zero{}: acc
    case Succ{+q}: letters(q, w, Con{letter(w, U32.from_nat(q)), acc})

def put_word(+w: U32, acc: List<&2, Nat>) -> List<&2, Nat>:
  letters(wlen(w), w, Con{32n, acc})

# the i-th word of the text under seed
def word(+seed: U32, +i: U32) -> U32:
  +k = (seed + (i * 2 : U32) : U32)
  +a = mix(k)
  +b = mix((k + 1 : U32))
  U32.min((a % 64 : U32), (b % 64 : U32))

# words 0..left-1 of the text under seed, each followed by a space, prepended to acc (the last word first)
def text_go(left: Nat, +seed: U32, acc: List<&2, Nat>) -> List<&2, Nat>:
  match left:
    case Zero{}: acc
    case Succ{+q}: text_go(q, seed, put_word(word(seed, U32.from_nat(q)), acc))

def text(+words: Nat, +seed: U32) -> List<&2, BPE.Tk>:
  BPE.lift(text_go(words, seed, Nil{}))

def tid(t: BPE.Tk) -> U32:
  match t:
    case BPE.B{n}: U32.from_nat(n)
    case BPE.M{k}: (256 + U32.from_nat(k) : U32)

# the count of tokens and the FNV-1a hash of their ids
def fnv(ts: List<&2, BPE.Tk>, +n: U32, +h: U32) -> U32 & U32:
  match ts:
    case Nil{}: (n, h)
    case Con{t, rest}: fnv(rest, (n + 1 : U32), ((h .^. tid(t)) * 16777619 : U32))

def show(+rules: Nat, r: U32 & U32) -> IO(Unit):
  match r:
    case Tuple{n, h}:
      do IO<Unit>:
        IO.print(U32.show(U32.from_nat(rules)))
        IO.print(U32.show(n))
        IO.print(U32.show(h))

def encode(+table: List<&2, BPE.Rule>) -> IO(Unit):
  show(List.length(&2, BPE.Rule, table), fnv(BPE.encode(table, text(32768n, 2000003)), 0, 2166136261))

def main() -> IO(Unit):
  encode(BPE.train(256n, 0n, text(4096n, 1000003)))
