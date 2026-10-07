# Four IO.fork computations. Each reads the clock INSIDE the fork and then runs a flat loop seeded with it,
# so the work can only happen in the forked computation (not when the action is built, and not in the
# joiner). On Bend 2.0.35 and on main (0ad47fc) this takes the same time with --threads 1 and --threads 16,
# and the same time as io_fork_sequential.bend, which runs the four jobs one after another.
#
#   bend io_fork_parallel.bend -o f && time ./f --threads 1 && time ./f --threads 16
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
    a : Chan(U32) <- IO.fork(U32, job((U32.from_nat(t) + 1 : U32)))
    b : Chan(U32) <- IO.fork(U32, job((U32.from_nat(t) + 2 : U32)))
    c : Chan(U32) <- IO.fork(U32, job((U32.from_nat(t) + 3 : U32)))
    d : Chan(U32) <- IO.fork(U32, job((U32.from_nat(t) + 4 : U32)))
    x : U32 <- IO.join(U32, a)
    y : U32 <- IO.join(U32, b)
    z : U32 <- IO.join(U32, c)
    w : U32 <- IO.join(U32, d)
    IO.print(U32.show((x + y + z + w : U32)))
