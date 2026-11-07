"""Compare saved engine probabilities with the hosted Sarvam Prism API.

The API key is read from ``SARVAM_API_KEY`` only and is never written to disk.
Use ``--fetch`` to collect responses; without it the script reads the recorded
responses and rebuilds the parity tables.
"""

import argparse
import json
import os
import pathlib
import random
import threading
import time

import numpy as np

from _common import read_json, read_jsonl, write_json

URL = "https://api.sarvam.ai/v2/systemone"


def redact(text, key):
    return str(text).replace(key, "[REDACTED]") if key else str(text)


def payload_for(state, questions):
    return {"model": "sarvam-prism", "state": state, "questions": questions}


def q_for(rec):
    if rec["type"] == "choice":
        return {"type": "choice", "instructions": rec["instructions"],
                "criteria": rec["criteria"]}
    if rec["type"] == "yes_no":
        return {"type": "noul", "instructions": rec["instructions"]}
    if rec["type"] == "score":
        return {"type": "score", "instructions": rec["instructions"],
                "criteria": rec["criteria"]}
    raise ValueError(rec["type"])


def parse_answer(rec, ans):
    if rec["type"] == "choice":
        pr = ans.get("probabilities")
        if not isinstance(pr, dict) or set(pr) != set(rec["keys"]):
            raise ValueError("choice probability keys do not match")
        return [float(pr[k]) for k in rec["keys"]]
    if rec["type"] == "yes_no":
        p = ans.get("noul", ans.get("yes"))
        if p is None:
            raise ValueError("yes/no answer without a value")
        p = float(p)
        return [1.0 - p, p]
    pr = ans.get("probabilities")
    if not isinstance(pr, dict) or set(pr) != set(rec["keys"]):
        raise ValueError("score probability keys do not match")
    return [float(pr[k]) for k in rec["keys"]]


def fetch(args):
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not key:
        raise SystemExit("SARVAM_API_KEY is not set")
    records = read_jsonl(pathlib.Path(args.data) / "eval" / "test.jsonl")
    if args.limit:
        records = records[:args.limit]
    rng = random.Random(args.seed)
    noise = [records[i] for i in rng.sample(range(len(records)),
                                           min(args.noise, len(records)))] if args.noise else []
    jobs = [(r["id"], r) for r in records] + [(r["id"] + "_noise", r) for r in noise]
    out = pathlib.Path(args.responses)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for row in read_jsonl(out):
            done.add(row["id"])
    todo = [(name, r) for name, r in jobs if name not in done]
    if args.max_requests and len(done) + len(todo) > args.max_requests:
        todo = todo[:max(0, args.max_requests - len(done))]
    lock = threading.Lock()
    state = {"done": 0, "stop": False, "errors": 0, "tokens": 0}

    def one(name, rec):
        if state["stop"]:
            return
        payload = payload_for(rec["state"], {"q": q_for(rec)})
        attempts = 0
        while True:
            attempts += 1
            try:
                r = requests.post(URL, json=payload, timeout=70,
                                  headers={"api-subscription-key": key,
                                           "Content-Type": "application/json"})
                if r.status_code in (402, 403):
                    with lock:
                        state["stop"] = True
                    return
                if r.status_code in (429, 502, 503, 504) and attempts <= 5:
                    time.sleep(min(32, 2 ** attempts))
                    continue
                body = r.json()
                ans = body["answers"]["q"]
                vec = parse_answer(rec, ans)
                row = {"id": name, "probs": vec, "status": r.status_code,
                       "latency_s": r.elapsed.total_seconds(), "attempts": attempts,
                       "usage_input_tokens": int((body.get("usage") or {}).get("input_tokens") or 0)}
                with lock:
                    with open(out, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, separators=(",", ":")) + "\n")
                    state["done"] += 1
                    state["tokens"] += row["usage_input_tokens"]
                    if state["done"] % 100 == 0:
                        print(f"{state['done']} responses, {state['tokens']} input tokens",
                              flush=True)
                return
            except Exception as e:
                if attempts <= 3:
                    time.sleep(min(16, 2 ** attempts))
                    continue
                with lock:
                    state["errors"] += 1
                    with open(out, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"id": name, "error": redact(e, key)[:300],
                                            "status": -1}) + "\n")
                return

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(as_completed([ex.submit(one, name, rec) for name, rec in todo]))
    total = len(done) + state["done"]
    write_json(out.parent / "run_summary.json",
               {"jobs": len(jobs), "done_total": total, "errors": state["errors"],
                "tokens": state["tokens"]})
    print(json.dumps({"collected": state["done"], "errors": state["errors"],
                      "input_tokens": state["tokens"]}, indent=1))


def probe(args):
    import requests
    from prismlet.encoder import to_options
    from prismlet.examples import QUESTIONS, STATE_TEXT

    key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not key:
        raise SystemExit("SARVAM_API_KEY is not set")
    questions = {}
    for name, q in QUESTIONS.items():
        kind, keys, opts = to_options(q)
        if kind == "yes_no":
            questions[name] = {"type": "noul", "instructions": q["instructions"]}
        elif kind == "choice":
            questions[name] = {"type": "choice", "instructions": q["instructions"],
                               "criteria": q["criteria"]}
        else:
            questions[name] = {"type": "score", "instructions": q["instructions"],
                               "criteria": q["criteria"]}
    payload = payload_for(STATE_TEXT, questions)
    r = requests.post(URL, json=payload, timeout=70,
                      headers={"api-subscription-key": key, "Content-Type": "application/json"})
    body = r.json()
    answers = body["answers"]
    out = pathlib.Path(args.out)
    write_json(out / "api" / "probe.json",
               {"requests": [{"status": r.status_code, "latency_s": r.elapsed.total_seconds(),
                              "usage": body.get("usage")}],
                "all_shapes_parsed": all(k in answers for k in ("team", "urgent", "anger"))})
    golden_path = out / "engine" / "golden.json"
    published = read_json(golden_path)["expected"] if golden_path.exists() else None
    delta = None
    if published:
        vals = []
        for k, v in published["team"].items():
            vals.append(abs(float(answers["team"]["probabilities"][k]) - v))
        vals.append(abs(float(answers["urgent"]["noul"]) - published["urgent"]["yes"]))
        for k, v in published["anger"].items():
            vals.append(abs(float(answers["anger"]["probabilities"][k]) - v))
        delta = max(vals)
    write_json(out / "api" / "golden.json",
               {"answers": {k: answers[k] for k in ("team", "urgent", "anger")},
                "max_abs_delta_vs_published": delta})
    print(json.dumps({"status": r.status_code, "max_abs_delta_vs_published": delta}, indent=1))


def kl(a, b):
    a = np.clip(np.asarray(a, dtype=np.float64), 1e-12, None)
    b = np.clip(np.asarray(b, dtype=np.float64), 1e-12, None)
    a, b = a / a.sum(), b / b.sum()
    return float(np.sum(a * np.log(a / b)))


def tv(a, b):
    return float(0.5 * np.abs(np.asarray(a, dtype=np.float64) -
                              np.asarray(b, dtype=np.float64)).sum())


def load_api(path):
    out, meta = {}, {"n_files": 0, "n_ok": 0, "input_tokens": 0, "errors": 0}
    if not pathlib.Path(path).exists():
        return out, meta
    for row in read_jsonl(path):
        meta["n_files"] += 1
        if row.get("probs"):
            out[row["id"]] = row["probs"]
            meta["n_ok"] += 1
            meta["input_tokens"] += int(row.get("usage_input_tokens") or 0)
        else:
            meta["errors"] += 1
    return out, meta


def load_probs(path):
    return {r["id"]: r["probs"] for r in read_jsonl(path)}


def aggregate(rows):
    if not rows:
        return {"n": 0}
    return {"n": len(rows),
            "argmax_agreement": float(np.mean([r["agree"] for r in rows])),
            "mean_kl": float(np.mean([r["kl"] for r in rows])),
            "mean_tv": float(np.mean([r["tv"] for r in rows])),
            "mean_abs": float(np.mean([r["max_abs"] for r in rows])),
            "max_abs": float(np.max([r["max_abs"] for r in rows])),
            "api_acc": float(np.mean([r["api_correct"] for r in rows])),
            "sys_acc": float(np.mean([r["sys_correct"] for r in rows])),
            "acc_delta_pp": float(np.mean([r["sys_correct"] for r in rows]) -
                                  np.mean([r["api_correct"] for r in rows])) * 100.0}


def by(rows, key):
    return {v: aggregate([r for r in rows if r[key] == v])
            for v in sorted({r[key] for r in rows})}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true", help="collect responses from the API")
    ap.add_argument("--probe", action="store_true", help="send the three model-card questions")
    ap.add_argument("--data", default="data")
    ap.add_argument("--responses", default="results/api/responses.jsonl")
    ap.add_argument("--probabilities", default="results/probabilities")
    ap.add_argument("--labels", default="results/eval/labels.json")
    ap.add_argument("--variants", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--noise", type=int, default=50)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-requests", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20261107)
    a = ap.parse_args()
    if a.probe:
        probe(a)
        return
    if a.fetch:
        fetch(a)
        return
    labels = [r for r in read_json(a.labels) if r["split"] == "test"]
    api, api_meta = load_api(a.responses)
    if not api:
        raise SystemExit("no API responses; run with --fetch and SARVAM_API_KEY set")
    root = pathlib.Path(a.probabilities)
    if a.variants:
        systems = [x for x in a.variants.split(",") if x.strip()]
    else:
        systems = sorted(p.parent.name for p in root.glob("*/test.jsonl"))
    probs = {s: load_probs(root / s / "test.jsonl") for s in systems}
    common = set(api)
    for p in probs.values():
        common &= set(p)
    recs = [r for r in labels if r["id"] in common]
    out = {"n_questions": len(recs), "api": api_meta, "systems": {}}
    for s, p in probs.items():
        rows = []
        for r in recs:
            av = np.asarray(api[r["id"]], dtype=np.float64)
            sv = np.asarray(p[r["id"]], dtype=np.float64)
            rows.append({"id": r["id"], "dataset": r["dataset"], "lang": r["lang"],
                         "type": r["type"], "label": int(r["label"]),
                         "agree": int(av.argmax() == sv.argmax()),
                         "kl": kl(av, sv), "tv": tv(av, sv),
                         "max_abs": float(np.max(np.abs(av - sv))),
                         "api_correct": int(av.argmax() == r["label"]),
                         "sys_correct": int(sv.argmax() == r["label"])})
        out["systems"][s] = {"overall": aggregate(rows), "by_type": by(rows, "type"),
                             "by_lang": by(rows, "lang"), "by_dataset": by(rows, "dataset")}
    noise = []
    for r in recs:
        a1, b1 = api.get(r["id"]), api.get(r["id"] + "_noise")
        if a1 is not None and b1 is not None:
            noise.append({"agree": int(np.argmax(a1) == np.argmax(b1)),
                          "mean_abs": float(np.mean(np.abs(np.asarray(a1) - np.asarray(b1)))),
                          "max_abs": float(np.max(np.abs(np.asarray(a1) - np.asarray(b1)))),
                          "kl": kl(a1, b1), "tv": tv(a1, b1)})
    if noise:
        out["noise_floor"] = {"n": len(noise),
                              "argmax_agreement": float(np.mean([x["agree"] for x in noise])),
                              "mean_abs": float(np.mean([x["mean_abs"] for x in noise])),
                              "max_abs": float(np.max([x["max_abs"] for x in noise])),
                              "mean_kl": float(np.mean([x["kl"] for x in noise])),
                              "mean_tv": float(np.mean([x["tv"] for x in noise]))}
    golden = pathlib.Path(a.out) / "engine" / "golden.json"
    api_golden = pathlib.Path(a.out) / "api" / "golden.json"
    if golden.exists() and api_golden.exists():
        g, ag = read_json(golden), read_json(api_golden)
        out["golden"] = {
            "engine": g.get("probabilities"),
            "api": ag.get("answers"),
            "max_abs_delta": ag.get("max_abs_delta_vs_published"),
        }
    out["cost"] = {"responses": api_meta["n_ok"], "input_tokens": api_meta["input_tokens"]}
    out_dir = pathlib.Path(a.out)
    write_json(out_dir / "pruning" / "api_parity.json", out)

    lines = ["# Parity against the hosted Sarvam Prism API", "",
             "Reference = API. n = questions present in every run.", "",
             "| system | n | argmax agree | mean KL(api||system) | mean TV | mean abs | "
             "api acc | system acc | delta (pp) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s, block in out["systems"].items():
        m = block["overall"]
        lines.append("| %s | %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %+.2f |"
                     % (s, m["n"], m["argmax_agreement"], m["mean_kl"], m["mean_tv"],
                        m["mean_abs"], m["api_acc"], m["sys_acc"], m["acc_delta_pp"]))
    if noise:
        n = out["noise_floor"]
        lines += ["", "Noise floor from %d repeated identical calls: argmax agreement %.4f, "
                  "mean abs %.6f, mean KL %.6f, mean TV %.6f."
                  % (n["n"], n["argmax_agreement"], n["mean_abs"], n["mean_kl"], n["mean_tv"])]
    lines.append("")
    (out_dir / "pruning" / "api_parity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"n_questions": len(recs), "systems": list(out["systems"]),
                      "noise": out.get("noise_floor")}, indent=1))


if __name__ == "__main__":
    main()
