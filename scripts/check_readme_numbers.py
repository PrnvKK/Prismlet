"""Check that every number in README.md also appears in a file under results/."""

import argparse
import pathlib
import re
import sys

FENCED = re.compile(r"```[\s\S]*?```")
INLINE = re.compile(r"`[^`\n]+`")
NUMBER = re.compile(r"(?<![\w.-])-?\d+(?:\.\d+)?%?")


def read_text(path):
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--results", default="results")
    ap.add_argument("--ignore", default="", help="comma-separated numbers to skip")
    a = ap.parse_args()
    text = pathlib.Path(a.readme).read_text(encoding="utf-8")
    text = FENCED.sub(" ", text)
    text = INLINE.sub(" ", text)
    ignore = {x.strip() for x in a.ignore.split(",") if x.strip()}
    wanted = sorted({m.group(0) for m in NUMBER.finditer(text)} - ignore)
    corpus = []
    for p in sorted(pathlib.Path(a.results).rglob("*")):
        if p.is_file() and p.suffix.lower() not in (".npz", ".pt", ".npy"):
            corpus.append(read_text(p))
    corpus = "\n".join(corpus)
    missing = [n for n in wanted if n not in corpus]
    print(f"{len(wanted)} numeric tokens, {len(missing)} not found in results")
    if missing:
        print("missing:", ", ".join(missing))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
