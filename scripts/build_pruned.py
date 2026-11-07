"""Build pruned layer files from router statistics and verify the writer."""

import argparse
import json
import pathlib
import shutil
import time

import torch

from _common import pick_device, write_json

from prismlet.config import SIZE_LAYER0, layer_bytes
from prismlet.engine import PrismEngine
from prismlet.examples import QUESTIONS, STATE_TEXT
from prismlet.encoder import to_options
from prismlet.io import STFile
from prismlet.prune import build_layer, copy_file, slice_layer
from prismlet.stats import Stats


def size_of(path):
    return sum(p.stat().st_size for p in pathlib.Path(path).glob("model-layer-*.safetensors"))


def verify_slab(weights, out_dir, keep, n):
    sf = STFile(pathlib.Path(weights) / "model-layer-10.safetensors")
    sb = torch.empty(sf.nbytes, dtype=torch.uint8)
    sf.read_all(sb, 8)
    sv = sf.views(sb)
    of = STFile(pathlib.Path(out_dir) / "model-layer-10.safetensors")
    ob = torch.empty(of.nbytes, dtype=torch.uint8)
    of.read_all(ob, 8)
    ov = of.views(ob)
    ids = keep[10]
    e = len(ids) // 2
    key = "model.layers.10.mlp.experts.gate_up_proj"
    ok = bool(torch.equal(sv[key][ids[e]], ov[key][e]))
    del sb, ob, sv, ov
    return ok


def identity_check(weights, pruned_root, device):
    tmp = pathlib.Path(pruned_root) / ".identity"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        slice_layer(pathlib.Path(weights) / "model-layer-00.safetensors",
                    tmp / "model-layer-00.safetensors", [])
        slice_layer(pathlib.Path(weights) / "model-layer-01.safetensors",
                    tmp / "model-layer-01.safetensors", list(range(128)))
        items = []
        for name in ("team", "urgent", "anger"):
            q = QUESTIONS[name]
            kind, keys, opts = to_options(q)
            items.append((STATE_TEXT, kind, opts, q["instructions"]))
        e0 = PrismEngine(weights_dir=weights, device=device, mode="disk",
                         dtype=torch.bfloat16, topk=8, max_tokens=8192, nthreads=8)
        h0 = e0.forward_items(items, stop_after=1)
        e1 = PrismEngine(weights_dir=weights, layers_dir=tmp, device=device, mode="disk",
                         dtype=torch.bfloat16, topk=8, max_tokens=8192, nthreads=8)
        h1 = e1.forward_items(items, stop_after=1)
        ok = all(bool(torch.equal(x, y)) for x, y in zip(h0, h1))
        del e0, e1, h0, h1
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--pruned", default="pruned")
    ap.add_argument("--stats", required=True, help="router statistics .npz")
    ap.add_argument("--ranking", default="total", choices=["total", "reap", "count"])
    ap.add_argument("--n", default="32", help="comma-separated expert counts")
    ap.add_argument("--suffix", default="general", help="directory suffix: N32_<suffix>")
    ap.add_argument("--calibration-questions", type=int, default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--no-identity-check", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    a = ap.parse_args()
    ns = [int(x) for x in a.n.split(",") if x.strip()]
    stats = Stats.load(a.stats)
    keep = {n: stats.keep(a.ranking, n) for n in ns}
    for n in ns:
        for l in range(32):
            k = keep[n][l]
            assert k == sorted(set(k)) and len(k) == n and all(0 <= x < 128 for x in k), (n, l)
    for n1, n2 in zip(ns, ns[1:]):
        if n2 < n1:
            for l in range(32):
                assert set(keep[n2][l]) <= set(keep[n1][l]), (n2, n1, l)

    need = sum(31 * layer_bytes(n) + SIZE_LAYER0 for n in ns)
    free = shutil.disk_usage(pathlib.Path(a.pruned).anchor or ".").free
    if not a.verify_only and free - need < a.min_free_gb * 1e9:
        raise SystemExit(f"need {need / 1e9:.1f} GB, free {free / 1e9:.1f} GB; "
                         f"refusing to leave less than {a.min_free_gb:.0f} GB free")

    dirs = {}
    for n in ns:
        name = f"N{n}" + (f"_{a.suffix}" if a.suffix else "")
        d = pathlib.Path(a.pruned) / name
        dirs[n] = d
        if a.verify_only:
            continue
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.tmp*"):
            f.unlink()
        copy_file(pathlib.Path(a.weights) / "model-layer-00.safetensors",
                  d / "model-layer-00.safetensors")

    if not a.verify_only:
        t0 = time.time()
        for li in range(1, 32):
            src = pathlib.Path(a.weights) / f"model-layer-{li:02d}.safetensors"
            build_layer(src, {n: dirs[n] / f"model-layer-{li:02d}.safetensors" for n in ns},
                        {n: keep[n][li] for n in ns})
        build_seconds = time.time() - t0
    else:
        build_seconds = 0.0

    tokens = float(stats.ntok[1]) if stats.ntok.shape[0] > 1 else float(stats.ntok.sum())
    reports = {}
    for n in ns:
        name = f"N{n}" + (f"_{a.suffix}" if a.suffix else "")
        d = dirs[n]
        if not a.verify_only:
            manifest = {
                "name": f"Prismlet-{name}",
                "base_model": "sarvamai/prism",
                "ranking": a.ranking,
                "N": n,
                "calibration": {"questions": a.calibration_questions, "tokens": tokens,
                                "stats": pathlib.Path(a.stats).name},
                "kept": {str(l): keep[n][l] for l in range(1, 32)},
            }
            write_json(d / "manifest.json", manifest)
        actual = size_of(d)
        expected = 31 * layer_bytes(n) + SIZE_LAYER0
        rel = abs(actual - expected) / expected
        slab = verify_slab(a.weights, d, keep[n], n)
        reports[name] = {"bytes": actual, "expected_bytes": expected, "relative_error": rel,
                         "expert_slab_ok": slab, "n": n, "ranking": a.ranking}
    id_ok = None if a.no_identity_check else identity_check(a.weights, a.pruned,
                                                            pick_device(a.device))
    report = {"build_seconds": build_seconds, "identity_check": id_ok,
              "variants": reports, "free_gb_after": shutil.disk_usage(
                  pathlib.Path(a.pruned).anchor or ".").free / 1e9}
    for name, r in reports.items():
        write_json(pathlib.Path(a.out) / "pruning" / f"build_{name}.json", {**r, **{
            "identity_check": id_ok, "free_gb_after": report["free_gb_after"]}})
    print(json.dumps({"variants": reports, "identity_check": id_ok,
                      "build_seconds": round(build_seconds, 1)}, indent=1))


if __name__ == "__main__":
    main()
