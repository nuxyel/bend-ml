# The words of the random fills, for reference/test_rng.py's NumPy twin: per line, the Gaussian's and the uniform's
# 32 bits of one cell of counter 77, first cells 0 .. 4095 from fill_gauss and fill_uniform (one bang each), then cells
# 2^32 - 4096 .. 2^32 - 1 from rng_gauss and rng_uniform (the indexes wrap there), then two slices from fill_gauss_at
# and fill_uniform_at: 4,095 cells from the odd index 2^32 - 2047 (they wrap past 2^32 - 1 to 2047) and 777 cells
# from the odd index 12345. The text is built from its last line
# back, each line put in front (a String appended at its end is copied).
import Base
import ../main.bend as TG

def bits(x: F32) -> U32:
  match x:
    case F32{w}: U32{w}

def row(+g: F32, +u: F32) -> String:
  U32.show(bits(g)) ++ " " ++ U32.show(bits(u)) ++ "\n"

def from_arrays(k: Nat, +i: U32, acc: String, rg: Array<F32> & F32, ru: Array<F32> & F32) -> String:
  match k rg ru:
    case 0n Tuple{a, g} Tuple{b, u}: acc
    case 1n+p Tuple{a, +g} Tuple{b, +u}:
      from_arrays(p, (i - 1 : U32), row(g, u) ++ acc, Array.get(F32, a, (i - 1 : U32)), Array.get(F32, b, (i - 1 : U32)))

def from_cells(k: Nat, +i: U32, acc: String) -> String:
  match k:
    case 0n: acc
    case 1n+p: from_cells(p, (i - 1 : U32), row(TG.rng_gauss(77, i), TG.rng_uniform(77, i)) ++ acc)

@unsafe
def words(u: Unit) -> String:
  +sb = from_arrays(777n, 776, "", Array.get(F32, TG.fill_gauss_at(77, 12345, 777n, 10n), 776), Array.get(F32, TG.fill_uniform_at(77, 12345, 777n, 10n), 776))
  +sa = from_arrays(4095n, 4094, sb, Array.get(F32, TG.fill_gauss_at(77, 4294965249, 4095n, 12n), 4094), Array.get(F32, TG.fill_uniform_at(77, 4294965249, 4095n, 12n), 4094))
  from_arrays(4096n, 4095, from_cells(4096n, 4294967295, sa), Array.get(F32, TG.fill_gauss(77, 4096n, 12n), 4095), Array.get(F32, TG.fill_uniform(77, 4096n, 12n), 4095))

# This program takes no arguments (runtime options go between +RTS and -RTS): any argument is an error.
def alone(args: List<String>, go: IO(Unit)) -> IO(Unit):
  match args:
    case Nil{}:
      go
    case Con{p, Nil{}}:
      go
    case Con{p, Con{a, t}}:
      do IO<Unit>:
        IO.print_err("unexpected argument " ++ a ++ ": this program takes none (runtime options go between +RTS and -RTS)")
        IO.exit(2)

@unsafe
def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    alone(args, IO.print(words(Unit{})))
