"""Fails on a Bend runtime option (--threads, --gpu) written outside +RTS ... -RTS.

Since Bend 2.0.38 a compiled program's runtime options go between +RTS and -RTS. A bare `./x --threads 16` is not an
error: the program gets `--threads 16` as arguments and runs on the default thread count, so a check that compares 1
and 16 threads keeps passing while it tests nothing. This reads every file git tracks and reports each line where
`--threads` or `--gpu` is left once the +RTS ... -RTS spans are taken out. The exceptions are in rts-allow.txt at the
repository's root, one per line: `<path glob>` or `<path glob>:<text on the line>`, then `# why`.

    python3 scripts/check_rts.py             # the check (make check runs it)
    python3 scripts/check_rts.py --self-test # a bare flag and a bracketed one, to show the check sees the difference
"""
import fnmatch, os, re, subprocess, sys

FLAG = re.compile(r"--(threads|gpu)\b")
# +RTS up to its -RTS, or, as Bend allows the -RTS to be left off, up to the end of a `code span` or of the line
SPAN = re.compile(r"\+RTS\b(?:(?!-RTS\b)[^`])*(?:-RTS\b)?")
TEXT = (".py", ".sh", ".bend", ".md", ".yml", ".yaml", ".mjs", ".ts", ".js", ".txt", ".toml")


def bare(line):
    return FLAG.search(SPAN.sub("", line)) is not None


def allowed(path, line, rules):
    for glob, text in rules:
        if fnmatch.fnmatch(path, glob) and (text is None or text in line):
            return True
    return False


def load_rules(root):
    rules = []
    p = os.path.join(root, "rts-allow.txt")
    if os.path.exists(p):
        for raw in open(p, encoding="utf-8"):
            entry = raw.split("#", 1)[0].strip()
            if entry:
                glob, _, text = entry.partition(":")
                rules.append((glob.strip(), text.strip() or None))
    return rules


def scan(root, files, rules):
    found = []
    for f in files:
        if not (f.endswith(TEXT) or os.path.basename(f) == "Makefile"):
            continue
        if os.path.basename(f) in ("check_rts.py", "rts_probe.py", "rts-allow.txt"):  # the guards name the bare form
            continue
        try:
            lines = open(os.path.join(root, f), encoding="utf-8").read().splitlines()
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        for n, line in enumerate(lines, 1):
            if bare(line) and not allowed(f, line, rules):
                found.append(f"{f}:{n}: {line.strip()[:160]}")
    return found


def self_test():
    assert bare("./x --threads 16 train 3")
    assert bare('subprocess.run([exe, "--gpu", "off"])')
    assert not bare("./x +RTS --threads 16 -RTS train 3")
    assert not bare('[exe, "+RTS", "--gpu", "off", "--threads", "16", "-RTS", "a"]')
    assert bare("./x +RTS --threads 1 -RTS && ./y --threads 16")
    assert not bare("check `+RTS --threads 1` against `+RTS --threads 16`")
    assert bare("`+RTS --threads 1` against `./y --threads 16`")
    assert allowed("NOTES.md", "x --threads 1", [("NOTES.md", None)])
    assert not allowed("Makefile", "x --threads 1", [("NOTES.md", None)])
    print("check_rts: self-test passed (a bare --threads or --gpu is found, a bracketed one is not)")


def main():
    if "--self-test" in sys.argv:
        self_test()
        return
    here = os.path.dirname(os.path.abspath(__file__))
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, cwd=here, check=True).stdout.strip()
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=root, check=True).stdout.split("\n")
    found = scan(root, [f for f in files if f], load_rules(root))
    if found:
        print("check_rts: runtime options outside +RTS ... -RTS (Bend 2.0.38 passes them to the program, silently):")
        print("\n".join("  " + x for x in found))
        print("Write `+RTS --threads N -RTS`, or add the line to rts-allow.txt with the reason.")
        sys.exit(1)
    print("check_rts: every --threads and --gpu given to a Bend program is inside +RTS ... -RTS")


if __name__ == "__main__":
    main()
