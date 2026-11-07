"""Collect layer-skip, ranking and calibration-domain ablations into one table."""

import argparse
import json
import pathlib

from _common import read_json, write_json


def load(path):
    p = pathlib.Path(path)
    return read_json(p) if p.exists() else None


def compact(m):
    if not m:
        return None
    keys = ("n_pixels", "argmax_agreement", "mean_kl", "mean_tv",
            "pixel_acc", "tokens_per_s", "seconds")
    return {k: m[k] for k in keys if k in m}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="results/pruning/geometry_runs")
    ap.add_argument("--fidelity", default="results/pruning/fidelity.json")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    runs = pathlib.Path(a.runs)
    base = load(runs / "N32_total.json")
    skip2 = load(runs / "skip2.json")
    skip4 = load(runs / "skip4.json")
    gen_n32 = load(runs / "N32_general.json")
    gen_n16 = load(runs / "N16_general.json")
    reap_quick = load(runs / "N32_reap_quick.json")
    fid = read_json(a.fidelity) if pathlib.Path(a.fidelity).exists() else {"variants": {}}

    out = {"geometry": {"base": compact(base), "layer_skip": {}}}
    if base:
        for name, m in (("skip2", skip2), ("skip4", skip4)):
            if not m:
                continue
            row = compact(m)
            row["layers"] = m.get("skip")
            row["delta_agree_vs_unskipped"] = m["argmax_agreement"] - base["argmax_agreement"]
            row["delta_kl_vs_unskipped"] = m["mean_kl"] - base["mean_kl"]
            out["geometry"]["layer_skip"][name] = row

    variant = fid.get("variants", {})

    def overall(name):
        m = variant.get(name, {}).get("overall")
        if not m:
            return None
        return {"n": m["n"], "argmax_agreement": m["argmax_agreement"],
                "mean_kl": m["mean_kl"], "mean_tv": m["mean_tv"],
                "acc": m["acc_var"], "acc_full": m["acc_full"],
                "acc_drop_pp": m["acc_drop_pp"]}

    out["ranking"] = {
        "general_n16": {"TOTAL": overall("N16_general"), "REAP": overall("N16_reap")},
        "geometry_n32_quick": {
            "TOTAL": compact(base) if base else None,
            "REAP": compact(reap_quick) if reap_quick else None,
        },
    }
    if reap_quick and base:
        out["ranking"]["geometry_n32_quick"]["note"] = (
            "TOTAL and REAP measured on the same 63 held-out geometry pixels; "
            "the REAP entry is the summary recorded during the ranking comparison."
        )

    out["calibration_domain"] = {
        "general_test": {"general_calibrated": {"N32": overall("N32_general"),
                                               "N16": overall("N16_general")},
                         "geometry_calibrated": {"N32": overall("N32_total"),
                                                 "N16": overall("N16_total")}},
        "geometry_test": {"general_calibrated": {"N32": compact(gen_n32),
                                                 "N16": compact(gen_n16)},
                          "geometry_calibrated": {"N32": compact(base)}},
    }
    out_dir = pathlib.Path(a.out) / "pruning"
    write_json(out_dir / "ablations.json", out)

    lines = ["# Ablations", ""]
    if base:
        lines += ["## Layer skip (N32_total, 63 held-out geometry pixels)", "",
                  "| config | layers skipped | agree | KL | TV | pixel acc | tok/s |",
                  "|---|---|---|---|---|---|---|"]
        b = out["geometry"]["base"]
        lines.append("| unskipped | none | %.4f | %.4f | %.4f | %.4f | %.1f |"
                     % (b["argmax_agreement"], b["mean_kl"], b["mean_tv"],
                        b["pixel_acc"], b.get("tokens_per_s") or 0))
        for name, m in out["geometry"]["layer_skip"].items():
            lines.append("| %s | %s | %.4f | %.4f | %.4f | %.4f | %.1f |"
                         % (name, m.get("layers"), m["argmax_agreement"], m["mean_kl"],
                            m["mean_tv"], m["pixel_acc"], m.get("tokens_per_s") or 0))
        lines.append("")
    lines += ["## Ranking at N16 (1200 general test questions)", "",
              "| ranking | n | agree | KL | TV | accuracy | drop (pp) |",
              "|---|---|---|---|---|---|---|"]
    for name in ("TOTAL", "REAP"):
        m = out["ranking"]["general_n16"].get(name)
        if m:
            lines.append("| %s | %d | %.4f | %.4f | %.4f | %.4f | %+.2f |"
                         % (name, m["n"], m["argmax_agreement"], m["mean_kl"],
                            m["mean_tv"], m["acc"], m["acc_drop_pp"]))
    lines += ["", "## Calibration domain, same 1200 general questions", "",
              "| calibration | N | agree | KL | accuracy | drop (pp) |",
              "|---|---|---|---|---|---|"]
    for cal, block in (("general", out["calibration_domain"]["general_test"]["general_calibrated"]),
                       ("geometry", out["calibration_domain"]["general_test"]["geometry_calibrated"])):
        for n, m in block.items():
            if m:
                lines.append("| %s | %s | %.4f | %.4f | %.4f | %+.2f |"
                             % (cal, n, m["argmax_agreement"], m["mean_kl"],
                                m["acc"], m["acc_drop_pp"]))
    lines.append("")
    (out_dir / "ablations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"layer_skip": list(out["geometry"]["layer_skip"]),
                      "ranking": list(out["ranking"]),
                      "calibration": list(out["calibration_domain"])}, indent=1))


if __name__ == "__main__":
    main()
