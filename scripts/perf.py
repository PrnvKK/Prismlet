"""Measure one streaming configuration and write throughput results."""

import argparse
import json
import pathlib
import time

import psutil
import torch

from _common import pick_device, read_jsonl, write_json

from prismlet.engine import PrismEngine
from prismlet.examples import QUESTIONS, STATE
from prismlet.io import LayerStream, STFile

FLOP_PER_TOKEN = 20.534e9


def layer_bytes(path):
    return sum(p.stat().st_size for p in pathlib.Path(path).glob("model-layer-*.safetensors"))


def peak_rss_gb():
    try:
        mi = psutil.Process().memory_full_info()
        return round((getattr(mi, "peak_wset", None) or mi.rss) / 1e9, 2)
    except Exception:
        return round(psutil.Process().memory_info().rss / 1e9, 2)


def load_items(prompts, n):
    records = read_jsonl(prompts)
    items = []
    while len(items) < n:
        for r in records:
            items.append((r["state"], r["type"], list(r["options"]), r["instructions"]))
            if len(items) >= n:
                break
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--pruned", default="pruned")
    ap.add_argument("--variant", default="full", help="full or a directory name under --pruned")
    ap.add_argument("--prompts", default="calibration/geometry_calib.jsonl")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--mode", default="disk", choices=["disk", "ram"])
    ap.add_argument("--batches", default="3,16,64")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--skip-golden", action="store_true")
    a = ap.parse_args()

    weights = pathlib.Path(a.weights)
    layers = weights if a.variant == "full" else pathlib.Path(a.pruned) / a.variant
    device = pick_device(a.device)
    tag = a.tag or f"{a.variant}_{a.mode}_{device}"
    files = [STFile(layers / f"model-layer-{i:02d}.safetensors") for i in range(32)]
    t0 = time.time()
    n = 0
    for li, v in LayerStream(files, list(range(32)), nbuf=2, nthreads=a.threads,
                             pin=device == "xpu"):
        n += 1
    floor = time.time() - t0
    lb = layer_bytes(layers)

    eng = PrismEngine(weights_dir=weights, layers_dir=layers, device=device, mode=a.mode,
                      dtype=torch.bfloat16, max_tokens=a.max_tokens, nthreads=a.threads,
                      prefetch=True)
    golden = None
    if not a.skip_golden:
        t0 = time.time()
        ans = eng.decide(STATE, QUESTIONS)
        golden = {"seconds": time.time() - t0, "probabilities": ans,
                  "timings": getattr(eng, "last_timings_total", {})}

    runs = []
    for b in [int(x) for x in a.batches.split(",") if x.strip()]:
        items = load_items(a.prompts, b)
        tokens = sum(len(eng.enc(s, ins, opts).ids) for (s, k, opts, ins) in items)
        passes = []
        npass = 2 if a.mode == "ram" else 1
        for pi in range(npass):
            if device == "xpu":
                torch.xpu.reset_peak_memory_stats()
            t0 = time.time()
            eng.forward_items(items)
            dt = time.time() - t0
            waits = sum(t.get("wait", 0.0) for t in getattr(eng, "last_timings", []))
            comp = max(1e-9, dt - waits)
            tim = getattr(eng, "last_timings_total", {})
            xmem = round(torch.xpu.max_memory_allocated() / 1e9, 2) if device == "xpu" else 0.0
            passes.append({"pass": pi + 1, "total_s": dt, "compute_s": comp,
                           "layers_s": float(tim.get("layers", dt)),
                           "head_s": float(tim.get("head", 0.0)),
                           "tokens_per_s": tokens / dt if dt else 0.0,
                           "eff_tflops": tokens * FLOP_PER_TOKEN / comp / 1e12,
                           "peak_rss_gb": peak_rss_gb(), "xpu_peak_gb": xmem})
            if device == "xpu":
                torch.xpu.empty_cache()
        runs.append({"batch": b, "tokens": tokens, "passes": passes, "reported": passes[-1]})

    out = {"tag": tag, "variant": a.variant, "mode": a.mode, "device": device,
           "layer_bytes": lb, "disk_floor_s": floor, "disk_gbps": lb / floor / 1e9,
           "n_layers": n, "torch": torch.__version__, "threads": torch.get_num_threads(),
           "golden": golden, "runs": runs}
    if a.variant == "full":
        out["best_config"] = {"device": device, "mode": a.mode, "dtype": "bfloat16",
                              "max_tokens": a.max_tokens, "nthreads": a.threads,
                              "prefetch": True, "dequant_backend": "lut"}
        out["by_batch"] = [{"n": r["batch"], "tokens": r["tokens"],
                            "total_s": r["reported"]["total_s"],
                            "compute_s": r["reported"]["compute_s"],
                            "tokens_per_s": r["reported"]["tokens_per_s"],
                            "eff_tflops": r["reported"]["eff_tflops"]} for r in runs]
        out["golden_seconds"] = (golden or {}).get("seconds")
        out["disk_gbps"] = lb / floor / 1e9
        best = max(runs, key=lambda r: r["reported"]["tokens_per_s"])
        out["best_tokens_per_s"] = best["reported"]["tokens_per_s"]
        out["best_n"] = best["batch"]
        out["best_eff_tflops"] = best["reported"]["eff_tflops"]
        write_json(pathlib.Path(a.out) / "engine" / "perf.json", out)
    else:
        write_json(pathlib.Path(a.out) / "pruning" / "perf_runs" / f"{tag}.json", out)
    print(json.dumps({"tag": tag, "layer_bytes": lb, "disk_floor_s": floor,
                      "runs": [{"batch": r["batch"], "tokens": r["tokens"],
                                "tokens_per_s": r["reported"]["tokens_per_s"]} for r in runs]},
                     indent=1))


if __name__ == "__main__":
    main()
