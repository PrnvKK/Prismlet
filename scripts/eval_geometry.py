"""Evaluate a variant on the held-out synthetic geometry prompts."""

import argparse
import pathlib
import time

import numpy as np
import torch

from _common import pick_device, prompt_items, read_json, read_jsonl, write_json

from prismlet.engine import PrismEngine


def metrics(ref, probs, labels, n_pixels):
    rows = []
    for i in range(n_pixels):
        f = np.clip(np.asarray(ref[i], dtype=np.float64), 1e-12, None)
        v = np.clip(np.asarray(probs[i], dtype=np.float64), 1e-12, None)
        f, v = f / f.sum(), v / v.sum()
        rows.append({"agree": int(f.argmax() == v.argmax()),
                     "kl": float(np.sum(f * np.log(f / v))),
                     "tv": float(0.5 * np.abs(f - v).sum()),
                     "correct_full": int(f.argmax() == labels[i]),
                     "correct_var": int(v.argmax() == labels[i])})
    if not rows:
        return {"n_pixels": 0}
    af = float(np.mean([r["correct_full"] for r in rows]))
    av = float(np.mean([r["correct_var"] for r in rows]))
    return {"n_pixels": len(rows),
            "argmax_agreement": float(np.mean([r["agree"] for r in rows])),
            "mean_kl": float(np.mean([r["kl"] for r in rows])),
            "mean_tv": float(np.mean([r["tv"] for r in rows])),
            "pixel_acc": av, "full_pixel_acc": af, "pixel_acc_drop": (af - av) * 100.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--pruned", default="pruned")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--prompts", default="calibration/geometry_heldout.jsonl")
    ap.add_argument("--reference", default="results/pruning/geometry_reference.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--mode", default="disk", choices=["disk", "ram"])
    ap.add_argument("--skip", default="", help="comma-separated layer ids to omit")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--write-reference", action="store_true")
    a = ap.parse_args()
    records = read_jsonl(a.prompts)
    if a.limit:
        records = records[:a.limit]
    items, labels = prompt_items(records)
    n_pixels = sum(1 for x in labels if x is not None)
    layers = pathlib.Path(a.weights) if a.variant == "full" \
        else pathlib.Path(a.pruned) / a.variant
    eng = PrismEngine(weights_dir=a.weights, layers_dir=layers, device=pick_device(a.device),
                      mode=a.mode, dtype=torch.bfloat16, topk=a.topk,
                      skip_layers=tuple(int(x) for x in a.skip.split(",") if x.strip()),
                      max_tokens=8192, nthreads=8, prefetch=True)
    t0 = time.time()
    probs = [p.tolist() for p in eng.forward_items(items)]
    dt = time.time() - t0
    tokens = int(sum(len(eng.enc(s, ins, opts).ids) for (s, k, opts, ins) in items))

    if a.write_reference:
        out = {"probs": probs, "labels": labels,
               "pixels": [r.get("pixel") for r in records], "n_pixels": n_pixels,
               "seconds": dt, "tokens": tokens}
        write_json(a.reference, out)
        print(f"wrote reference for {len(records)} prompts ({n_pixels} pixels)")
        return

    ref = read_json(a.reference)
    m = metrics(ref["probs"], probs, labels, n_pixels)
    m.update({"tag": a.tag or f"geometry_{a.variant}", "variant": a.variant,
              "skip": [int(x) for x in a.skip.split(",") if x.strip()],
              "topk": a.topk, "mode": a.mode, "device": str(eng.device),
              "seconds": dt, "tokens": tokens, "tokens_per_s": tokens / dt if dt else 0.0})
    out = pathlib.Path(a.out) / "pruning" / "geometry_runs" / f"{m['tag']}.json"
    write_json(out, m)
    print(f"{m['tag']}: {m}")


if __name__ == "__main__":
    main()
