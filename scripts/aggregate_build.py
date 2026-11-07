"""Combine per-variant build reports into one file."""

import argparse
import pathlib

from _common import read_json, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", default="results/pruning")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    root = pathlib.Path(a.reports)
    variants, identity = {}, None
    for p in sorted(root.glob("build_*.json")):
        row = read_json(p)
        name = p.stem[len("build_"):]
        variants[name] = row
        if row.get("identity_check") is not None:
            identity = row["identity_check"]
    out = {"identity_check": identity, "variants": variants}
    write_json(pathlib.Path(a.out) / "pruning" / "build.json", out)
    print(f"{len(variants)} variants")


if __name__ == "__main__":
    main()
