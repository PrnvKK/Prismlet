"""Aggregate per-configuration performance runs into one table."""

import argparse
import json
import pathlib

from _common import read_json, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="results/pruning/perf_runs")
    ap.add_argument("--full-perf", default="results/engine/perf.json")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rows, cpu_run = [], None
    for f in sorted(pathlib.Path(a.runs).glob("*.json")):
        j = read_json(f)
        if "runs" not in j:
            continue
        if j["tag"].startswith("cpu"):
            cpu_run = j
        for r in j["runs"]:
            rep = r["reported"]
            first = None
            if len(r["passes"]) > 1:
                first = r["passes"][0]["total_s"] - r["passes"][-1]["total_s"]
            rows.append({"tag": j["tag"], "variant": j["variant"], "mode": j["mode"],
                         "device": j["device"], "batch": r["batch"], "tokens": r["tokens"],
                         "total_s": rep["total_s"], "compute_s": rep["compute_s"],
                         "tokens_per_s": rep["tokens_per_s"], "eff_tflops": rep["eff_tflops"],
                         "peak_rss_gb": rep["peak_rss_gb"],
                         "xpu_peak_gb": rep.get("xpu_peak_gb"),
                         "disk_floor_s": j["disk_floor_s"],
                         "first_pass_load_s": first})
    full = None
    fp = pathlib.Path(a.full_perf)
    if fp.exists():
        f = read_json(fp)
        full = {"tag": "full", "variant": "full", "mode": f.get("mode", "disk"),
                "device": f.get("device"), "batch": f.get("best_n"),
                "tokens_per_s": f.get("best_tokens_per_s"),
                "eff_tflops": f.get("best_eff_tflops"),
                "disk_floor_s": f.get("disk_floor_s"),
                "by_batch": f.get("runs") or f.get("by_batch")}
    out = {"measured": rows, "full_reused": full,
           "cpu_run": cpu_run}
    out_dir = pathlib.Path(a.out) / "pruning"
    write_json(out_dir / "perf.json", out)

    lines = ["# Pruned-variant throughput (Intel Arc 140V, 32 GB RAM)", "",
             "Selected set: geometry prompts of mixed token length; for batch b the first b prompts.",
             "Effective TFLOPS = tokens * 20.534 GFLOP / compute seconds.",
             "",
             "| config | variant | mode | device | batch | tokens | total s | tok/s | "
             "eff TFLOPS | peak RSS GB | XPU peak GB |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %s | %s | %d | %d | %.1f | %.1f | %.3f | %.2f | %s |"
                     % (r["tag"], r["variant"], r["mode"], r["device"], r["batch"], r["tokens"],
                        r["total_s"], r["tokens_per_s"], r["eff_tflops"], r["peak_rss_gb"],
                        r["xpu_peak_gb"] if r["xpu_peak_gb"] is not None else "-"))
    if full:
        lines += ["", "Full model: %.1f tok/s at batch %s, layer-stream floor %.1f s."
                  % (full["tokens_per_s"], full["batch"], full["disk_floor_s"]), ""]
    (out_dir / "perf_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "full": bool(full), "cpu": bool(cpu_run)}, indent=1))


if __name__ == "__main__":
    main()
