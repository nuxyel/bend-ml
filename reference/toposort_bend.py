"""Reordena as definições de um .bend para que cada uma venha depois das que usa."""
import re, sys

def main(path):
    s = open(path).read()
    head_end = s.index("\n\n", s.index("import")) if "import" in s else 0
    # separa cabeçalho (comentários + imports) do resto
    lines = s.split("\n")
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or lines[i].startswith("import") or lines[i].strip() == ""):
        i += 1
    header = "\n".join(lines[:i]).rstrip("\n")
    body = "\n".join(lines[i:])
    blocks = []
    cur = []
    for ln in body.split("\n"):
        if re.match(r"^(def|type|law|@unsafe)\b", ln) and cur and not all(l.startswith("#") or l.strip() == "" for l in cur):
            blocks.append("\n".join(cur).rstrip("\n")); cur = []
        cur.append(ln)
    if cur: blocks.append("\n".join(cur).rstrip("\n"))
    names = {}
    for b in blocks:
        m = re.search(r"^(?:def|type)\s+([^\s(<:]+)", b, re.M)
        if m: names.setdefault(m.group(1), b)
    deps = {}
    for n, b in names.items():
        d = set()
        # corpo sem a primeira linha de declaração
        for other in names:
            if other != n and re.search(r"(?<![\w.])" + re.escape(other) + r"(?![\w]|\.\w)", b):
                d.add(other)
        deps[n] = d
    order, seen, stack = [], set(), set()
    def visit(n):
        if n in seen: return
        if n in stack: return
        stack.add(n)
        for d in sorted(deps[n], key=lambda x: list(names).index(x)):
            visit(d)
        stack.discard(n); seen.add(n); order.append(n)
    for n in names: visit(n)
    out = header + "\n\n" + "\n\n".join(names[n] for n in order) + "\n"
    open(path, "w").write(out)

main(sys.argv[1])
