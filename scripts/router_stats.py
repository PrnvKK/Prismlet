"""Collect router and saliency statistics over a calibration set."""

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from _common import pick_device, prompt_items, read_jsonl, write_json

from prismlet.encoder import to_options
from prismlet.engine import PrismEngine
from prismlet.stats import Stats


def records_to_items(records):
    items = []
    for r in records:
        kind, keys, opts = to_options({"type": r["type"], "criteria": r["criteria"]})
        items.append((r["state"], kind, opts, r["instructions"]))
    return items


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def summary(stats, n_prompts, topk):
    cnt = stats.cnt.numpy()
    ntok = stats.ntok.numpy()
    per_layer = {}
    for l in range(1, 32):
        c = cnt[l]
        order = np.argsort(-c)
        per_layer[str(l)] = {
            "tokens": float(ntok[l]),
            "expert_tokens": float(c.sum()),
            "nonzero_experts": int((c > 0).sum()),
            "zero_fraction": float((c == 0).mean()),
            "top8": [int(x) for x in order[:8]],
            "top8_share": float(c[order[:8]].sum() / max(1.0, c.sum())),
        }
    kept = {}
    for ranking in ("total", "reap"):
        kept[ranking] = {str(n): {str(l): stats.keep(ranking, n)[l] for l in range(1, 32)}
                         for n in (32, 16, 8)}
    return {
        "n_prompts": n_prompts,
        "topk": topk,
        "tokens": float(ntok[1]) if len(ntok) > 1 else float(ntok.max()),
        "tokens_per_layer": float(ntok[1]) if len(ntok) > 1 else float(ntok.max()),
        "count_matches_tokens_times_topk": all(
            int(cnt[l].sum()) == int(ntok[l]) * topk for l in range(1, 32)),
        "zero_fraction_mean": float(np.mean([(cnt[l] == 0).mean() for l in range(1, 32)])),
        "zero_fraction_max": float(np.max([(cnt[l] == 0).mean() for l in range(1, 32)])),
        "per_layer": per_layer,
        "kept": kept,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--data", default=None, help="calibration JSONL from build_eval_set.py")
    ap.add_argument("--prompts", default=None, help="geometry prompt JSONL instead of --data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="router_stats")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--mode", default="disk", choices=["disk", "ram"])
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--compare", default=None, help="second .npz for kept-set overlap")
    a = ap.parse_args()
    if not a.data and not a.prompts:
        ap.error("pass --data or --prompts")
    if a.prompts:
        records = read_jsonl(a.prompts)
        records = [r for r in records if r.get("label") is not None]
        items, _ = prompt_items(records)
    else:
        records = read_jsonl(a.data)
        items = records_to_items(records)
    eng = PrismEngine(weights_dir=a.weights, device=pick_device(a.device), mode=a.mode,
                      dtype=torch.bfloat16, topk=a.topk,
                      max_tokens=a.max_tokens, nthreads=8)
    stats = Stats()
    t0 = time.time()
    eng.forward_items(items, stats=stats)
    dt = time.time() - t0
    out_dir = pathlib.Path(a.out) / "pruning"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats.save(out_dir / f"{a.name}.npz")
    out = summary(stats, len(records), a.topk)
    out["seconds"] = dt
    out["tokens_per_s"] = out["tokens"] / dt if dt else 0.0
    if a.compare:
        other = Stats.load(a.compare)
        overlap = {}
        for n in (32, 16):
            a1 = stats.keep("total", n)
            b1 = other.keep("total", n)
            per_layer = {str(l): round(jaccard(a1[l], b1[l]), 4) for l in range(1, 32)}
            per_layer["mean"] = round(float(np.mean([per_layer[str(l)] for l in range(1, 32)])), 4)
            overlap[str(n)] = per_layer
        out["kept_overlap_total"] = overlap
    write_json(out_dir / f"{a.name}.json", out)
    print(json.dumps({k: out[k] for k in ("n_prompts", "topk", "tokens", "seconds",
                                          "tokens_per_s", "zero_fraction_mean",
                                          "count_matches_tokens_times_topk")}, indent=1))


if __name__ == "__main__":
    main()
