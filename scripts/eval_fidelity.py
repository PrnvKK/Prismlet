"""Compare saved variant probabilities with the full model and write fidelity tables."""

import argparse
import csv
import json
import math
import pathlib

import numpy as np

from _common import read_json, read_jsonl, write_json

MAX_OPTIONS = 20


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def boot_ci(x, n_boot=1000, seed=1):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = [float(x[rng.integers(0, len(x), len(x))].mean()) for _ in range(n_boot)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def load_probs(path):
    return {r["id"]: r["probs"] for r in read_jsonl(path)}


def rows_for(labels, full, var):
    rows = []
    for rec in labels:
        i = rec["id"]
        if i not in full or i not in var:
            continue
        f = np.clip(np.asarray(full[i], dtype=np.float64), 1e-12, None)
        v = np.clip(np.asarray(var[i], dtype=np.float64), 1e-12, None)
        f, v = f / f.sum(), v / v.sum()
        a_f, a_v = int(f.argmax()), int(v.argmax())
        lab = int(rec["label"])
        rows.append({
            "id": i, "dataset": rec["dataset"], "domain": rec["domain"], "lang": rec["lang"],
            "type": rec["type"], "label": lab, "split": rec["split"],
            "a_full": a_f, "a_var": a_v,
            "p_full_top": float(f.max()), "p_var_top": float(v.max()),
            "p_full": f, "p_var": v,
            "kl": float(np.sum(f * np.log(f / v))),
            "tv": float(0.5 * np.abs(f - v).sum()),
            "max_abs": float(np.max(np.abs(f - v))),
            "correct_full": int(a_f == lab), "correct_var": int(a_v == lab),
            "score_full": float(np.dot(np.arange(len(f)), f)) if rec["type"] == "score" else None,
            "score_var": float(np.dot(np.arange(len(v)), v)) if rec["type"] == "score" else None,
        })
    return rows


def agg(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    agree = float(np.mean([r["a_full"] == r["a_var"] for r in rows]))
    kl = [r["kl"] for r in rows]
    tv = [r["tv"] for r in rows]
    af = float(np.mean([r["correct_full"] for r in rows]))
    av = float(np.mean([r["correct_var"] for r in rows]))
    top = [r for r in rows if r["p_full_top"] >= 0.9]
    sc = [r for r in rows if r["type"] == "score"]
    out = {
        "n": n,
        "argmax_agreement": agree,
        "agreement_ci95": wilson(int(round(agree * n)), n),
        "mean_kl": float(np.mean(kl)), "mean_kl_ci95": boot_ci(kl),
        "mean_tv": float(np.mean(tv)),
        "acc_full": af, "acc_full_ci95": wilson(int(sum(r["correct_full"] for r in rows)), n),
        "acc_var": av, "acc_var_ci95": wilson(int(sum(r["correct_var"] for r in rows)), n),
        "acc_drop_pp": (af - av) * 100.0,
        "agree_at_p090": float(np.mean([r["a_full"] == r["a_var"] for r in top])) if top else None,
        "coverage_p090": len(top) / n,
        "mean_max_abs": float(np.mean([r["max_abs"] for r in rows])),
        "max_max_abs": float(np.max([r["max_abs"] for r in rows])),
    }
    if sc:
        out["mae_expected_score"] = float(np.mean(
            [abs(r["score_full"] - r["score_var"]) for r in sc]))
        out["n_score"] = len(sc)
    return out


def by_slice(rows, key):
    return {v: agg([r for r in rows if r[key] == v]) for v in sorted({r[key] for r in rows})}


def fmt_row(name, m):
    return ("| %s | %d | %.4f | [%.3f, %.3f] | %.4f | [%.4f, %.4f] | %.4f | %.4f | %.4f | %+.2f | %s | %.3f |"
            % (name, m["n"], m["argmax_agreement"], m["agreement_ci95"][0], m["agreement_ci95"][1],
               m["mean_kl"], m["mean_kl_ci95"][0], m["mean_kl_ci95"][1], m["mean_tv"],
               m["acc_full"], m["acc_var"], m["acc_drop_pp"],
               ("%.3f" % m["agree_at_p090"]) if m.get("agree_at_p090") is not None else "-",
               m.get("coverage_p090") or 0))


def write_csv(path, rows, full_key, var_key):
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ["id", "dataset", "split", "label", "pred_full", "pred_variant"]
    head += [f"p_full_{i}" for i in range(MAX_OPTIONS)]
    head += [f"p_variant_{i}" for i in range(MAX_OPTIONS)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        for r in rows:
            pf = list(r[full_key])
            pv = list(r[var_key])
            w.writerow([r["id"], r["dataset"], r["split"], r["label"], r["a_full"], r["a_var"]]
                       + [f"{x:.6f}" for x in pf] + [""] * (MAX_OPTIONS - len(pf))
                       + [f"{x:.6f}" for x in pv] + [""] * (MAX_OPTIONS - len(pv)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probabilities", default="results/probabilities")
    ap.add_argument("--labels", default="results/eval/labels.json")
    ap.add_argument("--reference", default="full")
    ap.add_argument("--variants", default=None, help="comma-separated variant names")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-csv", action="store_true")
    a = ap.parse_args()
    np.random.seed(a.seed)
    root = pathlib.Path(a.probabilities)
    labels = [r for r in read_json(a.labels) if r["split"] == "test"]
    full = load_probs(root / a.reference / "test.jsonl")
    labels = [r for r in labels if r["id"] in full]
    if a.variants:
        names = [x for x in a.variants.split(",") if x.strip()]
    else:
        names = sorted(p.parent.name for p in root.glob("*/test.jsonl"))
    names = [n for n in names if n != a.reference]
    out = {"reference": a.reference, "n_questions": len(labels), "variants": {},
           "runtime": {}}
    table = ["# Fidelity against the full model", "",
             "Reference = full model, identical batching. n = questions present in both runs.",
             "",
             "| variant | n | argmax agree | 95% CI | mean KL(full||var) | 95% CI | mean TV | "
             "acc full | acc var | acc drop (pp) | agree@p>=0.9 | coverage |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    slices_out = {}
    for name in names:
        p = root / name / "test.jsonl"
        if not p.exists():
            continue
        probs = load_probs(p)
        rows = rows_for(labels, full, probs)
        if not rows:
            continue
        overall = agg(rows)
        slices = {"type": by_slice(rows, "type"), "domain": by_slice(rows, "domain"),
                  "lang": by_slice(rows, "lang"), "dataset": by_slice(rows, "dataset")}
        meta_p = root / name / "test_meta.json"
        out["variants"][name] = {"overall": overall, "slices": slices}
        out["runtime"][name] = read_json(meta_p) if meta_p.exists() else {"n": len(rows)}
        table.append(fmt_row(name, overall))
        slices_out[name] = slices
        if not a.no_csv:
            write_csv(pathlib.Path(a.out) / "per_question" / f"per_question_{name}.csv",
                      rows, "p_full", "p_var")
    table.append("")
    for name, slices in slices_out.items():
        table += [f"## {name}", "",
                  "| slice | group | n | agree | mean KL | mean TV | acc full | acc var | acc drop (pp) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for kind in ("type", "domain", "lang", "dataset"):
            for g, m in slices[kind].items():
                if m["n"] == 0:
                    continue
                table.append("| %s | %s | %d | %.4f | %.4f | %.4f | %.4f | %.4f | %+.2f |"
                             % (kind, g, m["n"], m["argmax_agreement"], m["mean_kl"], m["mean_tv"],
                                m["acc_full"], m["acc_var"], m["acc_drop_pp"]))
        table.append("")
    out_dir = pathlib.Path(a.out) / "pruning"
    write_json(out_dir / "fidelity.json", out)
    (out_dir / "fidelity_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps({"n_questions": len(labels),
                      "variants": {k: v["overall"]["argmax_agreement"]
                                   for k, v in out["variants"].items()}}, indent=1))


if __name__ == "__main__":
    main()
