import Base

# Matriz r x c: as dimensões estão no TIPO (apagadas em tempo de execução);
# os dados ficam num Array<F32> linha a linha (índice = i*c + j).
type Mat<-r: Nat, -c: Nat> is Type:
  Mat{data: Array<F32>}

def Mat.new(-r: Nat, -c: Nat, depth: Nat, +v: F32) -> Mat<r, c>:
  Mat{Array.new(F32, depth, v)}

# "um match não inspeciona valor calculado": o par devolvido por Array.get
# precisa de uma função própria para ser aberto.
def Mat.get.fin(-r: Nat, -c: Nat, t: Array<F32> & F32) -> Mat<r, c> & F32:
  (d2, x) = t
  (Mat{d2}, x)

def Mat.get(-r: Nat, +c: Nat, m: Mat<r, c>, +i: U32, +j: U32) -> Mat<r, c> & F32:
  match m:
    case Mat{d}:
      Mat.get.fin(r, c, Array.get(F32, d, (i * U32.from_nat(c) + j : U32)))

def Mat.set(-r: Nat, +c: Nat, m: Mat<r, c>, +i: U32, +j: U32, v: F32) -> Mat<r, c>:
  match m:
    case Mat{d}:
      Mat{Array.set(F32, d, (i * U32.from_nat(c) + j : U32), v)}

# acc += A[i][p] * B[p][j] para p = 0..k-1. Os pares (matriz, elemento)
# chegam como parâmetros, que é o que o match aceita abrir.
def Mat.dot(-n: Nat, +k: Nat, +m: Nat, steps: Nat, +i: U32, +j: U32, +p: U32, acc: F32, ax: Mat<n, k> & F32, by: Mat<k, m> & F32) -> Mat<n, k> & Mat<k, m> & F32:
  match steps ax by:
    case 0n Tuple{a, x} Tuple{b, y}:
      (a, b, acc)
    case 1n+s Tuple{a, +x} Tuple{b, +y}:
      Mat.dot(n, k, m, s, i, j, (p + 1 : U32), (acc + (x * y : F32) : F32), Mat.get(n, k, a, i, (p + 1 : U32)), Mat.get(k, m, b, (p + 1 : U32), j))

def show3(t: Mat<1n, 784n> & Mat<784n, 1n> & F32) -> IO(Unit):
  match t:
    case Tuple{a, bf}:
      match bf:
        case Tuple{b, r}:
          IO.print(F32.show(r))

def main() -> IO(Unit):
  show3(Mat.dot(1n, 784n, 1n, 784n, 0, 0, 0, 0.0, Mat.get(1n, 784n, Mat.new(1n, 784n, 10n, 0.5), 0, 0), Mat.get(784n, 1n, Mat.new(784n, 1n, 10n, 2.0), 0, 0)))
