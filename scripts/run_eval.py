"""Run a variant over an evaluation split and write per-question probabilities.

Each shard runs in a fresh process. When the Level Zero runtime refuses an
allocation, the process exits 75 and the parent retries the shard with a smaller
token budget.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import torch

from _common import pick_device, read_jsonl, write_json

from prismlet.encoder import to_options
from prismlet.engine import PrismEngine

OOM_EXIT = 75


def is_oom(exc):
    s = str(exc).lower()
    return ("out of memory" in s) or ("out_of_resources" in s) or ("ur_error" in s)


def variant_dir(args):
    return pathlib.Path(args.out) / "probabilities" / args.variant


def layer_path(args):
    return pathlib.Path(args.weights) if args.variant == "full" \
        else pathlib.Path(args.pruned) / args.variant


def shard_path(args, k):
    return variant_dir(args) / f"{args.split}_shard_{k}.json"


def peak_rss_gb():
    try:
        import psutil
        mi = psutil.Process().memory_full_info()
        return round((getattr(mi, "peak_wset", None) or mi.rss) / 1e9, 2)
    except Exception:
        return None


def items_of(records):
    items = []
    for r in records:
        kind, keys, opts = to_options({"type": r["type"], "criteria": r["criteria"]})
        items.append((r["state"], kind, opts, r["instructions"]))
    return items


def run_shard(args):
    records = read_jsonl(pathlib.Path(args.data) / "eval" / f"{args.split}.jsonl")
    if args.limit:
        records = records[:args.limit]
    lo, hi = args.shard * args.shard_size, min(len(records), (args.shard + 1) * args.shard_size)
    recs = records[lo:hi]
    items = items_of(recs)
    eng = PrismEngine(weights_dir=args.weights, layers_dir=layer_path(args),
                      device=pick_device(args.device), mode="disk", dtype=torch.bfloat16,
                      max_tokens=args.max_tokens, nthreads=args.threads, prefetch=True)
    t0 = time.time()
    try:
        probs = eng.forward_items(items)
    except RuntimeError as e:
        if is_oom(e):
            print(f"out of memory in shard {args.shard}: {str(e)[:160]}", flush=True)
            return OOM_EXIT
        raise
    dt = time.time() - t0
    payload = {"ids": [r["id"] for r in recs], "probs": [p.tolist() for p in probs],
               "seconds": dt, "tokens": int(sum(len(eng.enc(s, ins, opts).ids)
                                                 for (s, k, opts, ins) in items)),
               "max_tokens": args.max_tokens, "peak_rss_gb": peak_rss_gb(),
               "timings": getattr(eng, "last_timings_total", {})}
    payload["tokens_per_s"] = payload["tokens"] / dt if dt else 0.0
    for p in probs:
        assert torch.isfinite(p).all()
        assert abs(float(p.sum()) - 1.0) <= 1e-3
    d = variant_dir(args)
    d.mkdir(parents=True, exist_ok=True)
    write_json(shard_path(args, args.shard), payload)
    print(json.dumps({"shard": args.shard, "n": len(recs), "seconds": round(dt, 1),
                      "tokens_per_s": round(payload["tokens_per_s"], 1)}), flush=True)
    return 0


def merge(args):
    d = variant_dir(args)
    files = sorted(d.glob(f"{args.split}_shard_*.json"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise SystemExit(f"no shards in {d}")
    rows, seconds, tokens, peak = [], 0.0, 0, 0.0
    for f in files:
        j = json.loads(f.read_text(encoding="utf-8"))
        for i, p in zip(j["ids"], j["probs"]):
            rows.append({"id": i, "probs": p})
        seconds += float(j.get("seconds") or 0.0)
        tokens += int(j.get("tokens") or 0)
        peak = max(peak, float(j.get("peak_rss_gb") or 0.0))
    with open(d / f"{args.split}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    meta = {"variant": args.variant, "split": args.split, "n": len(rows),
            "shards": len(files), "shard_size": args.shard_size,
            "seconds": seconds, "minutes": seconds / 60.0, "tokens": tokens,
            "tokens_per_s": tokens / seconds if seconds else 0.0,
            "peak_rss_gb": peak, "max_tokens": args.max_tokens}
    write_json(d / f"{args.split}_meta.json", meta)
    print(json.dumps({k: meta[k] for k in ("variant", "split", "n", "shards",
                                           "seconds", "tokens_per_s")}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--pruned", default="pruned")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--split", default="test", choices=["test", "calib"])
    ap.add_argument("--shard-size", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--shard", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.merge_only:
        merge(args)
        return
    if args.worker:
        raise SystemExit(run_shard(args))

    records = read_jsonl(pathlib.Path(args.data) / "eval" / f"{args.split}.jsonl")
    n = min(len(records), args.limit) if args.limit else len(records)
    n_shards = (n + args.shard_size - 1) // args.shard_size
    variant_dir(args).mkdir(parents=True, exist_ok=True)
    todo = [k for k in range(n_shards) if not shard_path(args, k).exists()]
    print(json.dumps({"variant": args.variant, "split": args.split, "n": n,
                      "shards": n_shards, "todo": todo}), flush=True)
    for k in todo:
        budget, ok = args.max_tokens, False
        for attempt in range(args.restarts + 1):
            cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), *sys.argv[1:]]
            cmd = [x for x in cmd if x != "--merge-only"]
            if "--worker" not in cmd:
                cmd += ["--worker"]
            cmd = [x for x in cmd if x not in ("--shard", str(args.shard))]
            cmd += ["--shard", str(k), "--max-tokens", str(budget)]
            rc = subprocess.call(cmd)
            if rc == 0:
                ok = True
                break
            if rc == OOM_EXIT and attempt < args.restarts:
                budget = max(1024, budget // 2)
                continue
            raise SystemExit(f"shard {k} exited with {rc}")
        if not ok:
            raise SystemExit(f"shard {k} did not complete")
    merge(args)


if __name__ == "__main__":
    main()
