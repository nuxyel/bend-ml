# bend-ml — Machine Learning em Bend 2

Contexto do projeto para o Claude Code. Leia este arquivo inteiro antes de qualquer ação.

## Por que este projeto existe

Em 29/09/2026, Victor Taelin (criador do Bend/HVM, empresa HOC) publicou um "info dump" sobre o futuro do Bend:

- HOC está levantando uma rodada de **US$ 20M** e vai montar dois times internos: **BendCore** (teoria, compilador, ecossistema, adoção) e **SupGen** (IA simbólica).
- A wishlist do ecossistema inclui explicitamente: **"AI framework / PyTorch / llama.bend / etc."**, além de "all good parts of mathlib, npm, hackage".
- Lançaram o **BendHub**: bibliotecas nomeadas e versionadas, com um "mini Hacker News" no site para descobrir, votar e discutir pacotes.
- Adoção atual: ~400 IPs não-cloud por dia usando o comando `bend`. Ele mesmo acha baixo.
- Tese do Bend: **"a language evolved by AI, secured by math"**. IA escreve o código, LAWS (provas verificadas pelo kernel) garantem a correção.

Quando perguntaram o que alguém que quer entrar num dos times deveria fazer, ele respondeu:

> build something in Bend so 1. VCs see traction and decide to invest in us 2. send that thing to impress us and be hired

Link: https://x.com/VictorTaelin/status/2104956654542639176

**Objetivo duplo deste projeto:**

1. **Tração mensurável**: pacotes no BendHub que outras pessoas usem (dependentes, upvotes, downloads).
2. **Portfólio**: algo impressionante o bastante para mandar ao Taelin numa candidatura.

**Ângulo escolhido:** o diferencial que só o Bend tem é **tipos dependentes**. A proposta central é um "PyTorch onde erro de shape não compila", usando IA para escrever e LAWS para garantir, ou seja, fazendo dogfooding da própria tese do Bend.

## Sobre o autor

- Renan (nuxyel), desenvolvedor backend. Linguagem principal: Python. Também TypeScript/Node.
- Estudando ML formalmente agora (Stanford/DeepLearning.AI ML Specialization). Experiência prática com NLP (spaCy, stanza).
- **Sem formação em teoria de tipos ou provas formais.** Ao escrever provas, explique o raciocínio em linguagem simples nos comentários e no chat. O projeto também é de aprendizado.
- Máquina: Core Ultra 7 155H, 32 GB RAM, **RTX 4050 (6 GB VRAM)**, Windows/Linux. Considere o limite de VRAM em qualquer demo de GPU.

## Regras de trabalho

1. **Bend 2 é novo e muda rápido.** A sintaxe é diferente da do Bend 1 (que era estilo Python). Não assuma sintaxe de memória. Antes de escrever código, leia a documentação oficial, os exemplos e a biblioteca Base. Na dúvida, leia o código-fonte.
2. **Fixe a versão do Bend** usada no projeto e registre em `NOTES.md`.
3. **Rode o checker o tempo todo.** Ele é rápido; use-o como loop de feedback a cada mudança.
4. **Nunca contorne uma prova.** Se o Bend tiver algum mecanismo equivalente a `sorry`/axioma/assume, não use sem avisar explicitamente e registrar em `NOTES.md` como dívida. Uma LAW falsa ou não provada destrói o valor do projeto.
5. **Floats não são reais.** Não tente provar propriedades numéricas sobre ponto flutuante. Prove o que é estrutural (shapes, tipos, composição) e valide o numérico com testes (gradient checking, comparação com referência em Python).
6. **Referências em Python** ficam em `reference/` (PyTorch, tiktoken etc.) e servem só para gerar valores esperados nos testes.
7. Commits pequenos e frequentes, com mensagens claras.
8. Toda LAW provada deve aparecer no README do pacote correspondente, em linguagem humana.

## Fase 0 — Reconhecimento (fazer primeiro, antes de qualquer código)

Investigar e registrar tudo em `NOTES.md`:

- [ ] Versão do Bend instalada e como atualizar.
- [ ] Onde estão a documentação, os exemplos e a biblioteca Base.
- [ ] O que a Base já oferece: Nat, List, String, Map, arrays. **Quais lemas já existem** (sabe-se que a Base tem poucos ou nenhum lema de Nat).
- [ ] Suporte a números: inteiros (U32/I32/U64?), **floats (F32/F64?)** e operações disponíveis.
- [ ] Backends de compilação disponíveis (JS, C, GPU). **Qual backend de GPU roda numa RTX 4050** (CUDA? só Metal?). Isso decide se a Fase 3 roda em GPU ou CPU.
- [ ] Estado de IO / interop com C: leitura de arquivos é necessária para carregar datasets e pesos.
- [ ] Como publicar no BendHub (namespace, versionamento, formato do pacote).
- [ ] **Se já existe no BendHub** algum tokenizer, lib de tensores ou autograd. Se existir, avaliar se é melhor contribuir ou diferenciar.

Saída esperada: `NOTES.md` com respostas objetivas e uma recomendação sobre ajustes no plano abaixo.

## Fase 1 — `bpe.bend` (meta: poucos dias)

Tokenizer BPE byte-level (estilo GPT-2: vocabulário base de 256 bytes + merges).

**Escopo:**
- `train(corpus, n_merges) -> MergeTable`
- `encode(table, bytes) -> List<Token>`
- `decode(table, tokens) -> bytes`

**LAWS a provar:**
- **Roundtrip:** para qualquer tabela de merges bem formada e qualquer sequência de bytes, `decode(encode(s)) == s`.
- **Limite de vocabulário:** todo token emitido por `encode` é menor que o tamanho do vocabulário da tabela.
- (Se viável) `decode` distribui sobre concatenação de listas de tokens.

**Testes:**
- Comparar ids de tokens contra uma implementação de referência em Python em textos de exemplo (incluindo UTF-8 com acentos).

**Stretch:** carregar a tabela de merges oficial do GPT-2 e bater 100% com tiktoken. Isso é pré-requisito da Fase 3.

**Entrega:** pacote publicado no BendHub + README com as LAWS listadas.

## Fase 2 — `tensor.bend` + autograd (meta: semanas)

**Tensores com shape no tipo.** Ideia geral (pseudocódigo, NÃO sintaxe real do Bend):

```
Tensor(shape: List<Nat>)
matmul : Tensor([n, k]) -> Tensor([k, m]) -> Tensor([n, m])
reshape : Tensor(s1) -> (prova: product(s1) == product(s2)) -> Tensor(s2)
```

**Operações iniciais:** add, mul (elementwise), matmul, transpose, reshape, sum, relu, softmax. Broadcasting fica para depois.

**Autograd:**
1. Primeiro, versão escalar estilo micrograd (grafo de expressões + reverse-mode).
2. Depois, generalizar para tensores.

**LAWS / garantias:**
- Shapes corretos por construção (erro de shape = erro de tipo).
- `reshape` só compila com prova de que o número de elementos se preserva.
- Propriedades estruturais do backward que forem expressáveis (ex.: o backward de uma composição é a composição dos backwards).
- Correção numérica dos gradientes: **testes de gradient checking**, não prova.

**Atenção:** a aritmética de shapes vai exigir lemas de Nat (associatividade, comutatividade, distributividade da multiplicação, propriedades de `product` sobre listas). Se a Base não tiver, criar um pacote separado `nat-lemmas` e publicar também. É mais uma contribuição útil ao ecossistema.

## Fase 3 — Demo de impacto

1. **MLP no MNIST:** treinar, reportar acurácia e tempo. Comparar com um script PyTorch equivalente em `reference/`. Benchmark honesto, com hardware e versões documentados.
2. **Stretch: inferência do GPT-2 small (124M).** Carregar pesos, tokenizar com `bpe.bend`, gerar texto. Atenção ao limite de 6 GB de VRAM. Isso é a semente de um "llama.bend".

## Estrutura sugerida do repositório

```
bend-ml/
├── CLAUDE.md          # este arquivo
├── NOTES.md           # achados da Fase 0, decisões, dívidas
├── bpe/               # pacote bpe.bend
├── nat-lemmas/        # lemas de Nat (se necessário)
├── tensor/            # pacote tensor.bend + autograd
├── demos/
│   └── mnist/
└── reference/         # scripts Python de referência (PyTorch, tiktoken)
```

## Critério de "pronto para mandar ao Taelin"

- Pelo menos um pacote publicado no BendHub, com README claro e LAWS listadas.
- Demo reproduzível com um comando.
- Benchmark honesto (sem esconder onde perde).
- Vídeo curto (~60 s) mostrando um erro de shape sendo pego em tempo de compilação e o treino rodando.
- Thread no X marcando @VictorTaelin, com link para o BendHub e o repositório.

## Riscos conhecidos

- Bend 2 ainda tem bugs e muda rápido; o próprio site avisa isso. Fixar versão e reportar bugs encontrados no GitHub oficial (isso também conta como contribuição).
- O compilador é majoritariamente escrito por IA; só o kernel de provas é auditado por humanos. Se algo se comportar de forma estranha, desconfie do compilador antes de desconfiar da sua prova.
- Suporte a floats, IO e GPU pode estar incompleto. A Fase 0 existe para descobrir isso cedo.
- Base com poucos lemas: provar coisas simples pode custar caro. Estimar com folga.

## Links

- Site oficial: https://bend-lang.com
- Thread do Taelin (info dump + resposta sobre contratação): https://x.com/VictorTaelin/status/2104956654542639176

## Agent skills

### Issue tracker

Issues live in GitHub Issues (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
