"""Merge per-shard probability files into one JSONL per variant."""

import argparse
import json
import pathlib

from _common import write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True, help="directory with <variant>/test_shard_*.json")
    ap.add_argument("--out", default="results/probabilities")
    ap.add_argument("--split", default="test")
    ap.add_argument("--variants", default=None)
    a = ap.parse_args()
    root = pathlib.Path(a.shards)
    names = [x.strip() for x in a.variants.split(",")] if a.variants else \
        sorted(p.name for p in root.iterdir() if p.is_dir())
    for name in names:
        d = root / name
        files = sorted(d.glob(f"{a.split}_shard_*.json"),
                       key=lambda p: int(p.stem.split("_")[-1]))
        if not files:
            continue
        rows, seconds, tokens, peak = [], 0.0, 0, 0.0
        for f in files:
            j = json.loads(f.read_text(encoding="utf-8"))
            for i, p in zip(j["ids"], j["probs"]):
                rows.append({"id": i, "probs": p})
            seconds += float(j.get("seconds") or 0.0)
            tokens += int(j.get("tokens") or 0)
            peak = max(peak, float(j.get("ram_peak_gb") or j.get("peak_rss_gb") or 0.0))
        out = pathlib.Path(a.out) / name
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{a.split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        write_json(out / f"{a.split}_meta.json",
                   {"variant": name, "split": a.split, "n": len(rows), "shards": len(files),
                    "seconds": seconds, "minutes": seconds / 60.0, "tokens": tokens,
                    "tokens_per_s": tokens / seconds if seconds else 0.0,
                    "peak_rss_gb": peak})
        print(f"{name}: {len(rows)} questions from {len(files)} shards")


if __name__ == "__main__":
    main()
