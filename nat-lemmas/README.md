# bend-ml-nat-lemmas

Lemas provados de `Nat` e `List` que a Base do Bend 2 ainda não tem. Servem de alicerce para o resto do bend-ml (shapes de tensores, tokenizer).

- Bend: **2.0.35**
- Licença: MIT
- Sem `@unsafe`, sem `?TODO`: `bend nat-lemmas/main.bend` imprime `ALL PROOFS CHECK`.

## Uso

```python
import Base
import bend-ml-nat-lemmas@0.1.0.0/main.bend as NL

def comm(a: Nat, +b: Nat) -> {Nat.add(a, b) == Nat.add(b, a) : Nat}:
  NL.add_comm(a, b)
```

Um parâmetro leva `+` quando pode ser usado mais de uma vez (regra de variáveis afins do Bend).

## LAWS provadas (em português)

| Lei | O que afirma |
|---|---|
| `add_zero` | `a + 0 = a` |
| `add_succ` | `1 + (a + b) = a + (1 + b)` |
| `add_comm` | `a + b = b + a` (a soma é comutativa) |
| `add_assoc` | `a + (b + c) = (a + b) + c` (a soma é associativa) |
| `mul_zero` | `a * 0 = 0` |
| `mul_succ` | `a * (1 + b) = a + a * b` |
| `mul_comm` | `a * b = b * a` (o produto é comutativo) |
| `mul_dist` | `a*c + b*c = (a + b)*c` (distributividade) |
| `mul_assoc` | `a * (b * c) = (a * b) * c` (o produto é associativo) |
| `mul_one_l`, `mul_one_r` | `1 * a = a` e `a * 1 = a` |
| `append_nil` | `xs ++ [] = xs` |
| `append_assoc` | `(xs ++ ys) ++ zs = xs ++ (ys ++ zs)` |
| `length_append` | `tamanho(xs ++ ys) = tamanho(xs) + tamanho(ys)` |
| `product_append` | `produto(xs ++ ys) = produto(xs) * produto(ys)` |

`product(xs)` multiplica os elementos de uma lista de `Nat` (lista vazia vale 1). É o número de elementos de um tensor cuja shape é `xs`; por isso `product_append` é a base do `reshape` com prova.

## Como ler uma prova

Em Bend, uma prova é uma função cujo tipo é a afirmação. `match` faz análise de casos; a chamada recursiva é a hipótese de indução; `%e : P` reescreve o objetivo usando a igualdade `e`; `{==}` fecha quando os dois lados já são o mesmo termo. Os comentários em `main.bend` explicam cada passo.

## Créditos

As provas de `add_*` e `mul_*` seguem o demo oficial `proof_numerics` do Bend, adaptadas para `Nat.add` e `Nat.mul` da Base.

## Limites conhecidos

- Ainda não rodei `bend --verdict` (reverificação pelo kernel provado em Lean): o Lean 4.34.0 não está instalado na máquina de desenvolvimento.
- Lemas de `reverse`, `take`/`drop` e de ordem (`<=`) ficam para versões futuras.
