"""Build the 1200-question test set and 500-question calibration set from public datasets."""

import argparse
import csv
import gzip
import json
import pathlib
import time

import numpy as np
import pyarrow.parquet as pq

from _common import write_json

from prismlet.encoder import Encoder, to_options
from prismlet.examples import QUESTIONS, STATE_TEXT

SEED = 20261107
CAP_TOKENS = 900

DBPEDIA_NAMES = ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder",
                 "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal",
                 "Plant", "Album", "Film", "WrittenWork"]
SIB_LABELS = ["science/technology", "travel", "politics", "sports", "health",
              "entertainment", "geography"]

INSTR = {
    "banking77": "Which banking customer-support intent does this message express?",
    "massive": "Which intent does this assistant request express?",
    "amazon_polarity": "Is this product review positive?",
    "civil_comments": "How toxic is this comment?",
    "snli": "If the premise is true, must the hypothesis also be true?",
    "boolq": "Answer yes or no.",
    "mmlu": "Choose the correct answer.",
    "global_mmlu": "Choose the correct answer.",
    "sib200": "What is the topic of this passage?",
    "dbpedia14": "What kind of entity is this Wikipedia article about?",
    "when2call": "What should the assistant do next?",
}

CAPS = {"banking77": 1000, "massive": 800, "amazon_polarity": 1000, "civil_comments": 800,
        "snli": 1200, "boolq": 1400, "mmlu": 1200, "global_mmlu": 1200, "sib200": 1000,
        "dbpedia14": 800, "when2call": 1200}


def clean(s, cap):
    s = " ".join(str(s).replace("\x00", " ").split())
    return s[:cap]


def normkey(state, instr):
    return " ".join((state + "\n" + instr).lower().split())


def rng_for(idx):
    return np.random.default_rng(SEED + idx)


def shuffled(rows, rng):
    idx = rng.permutation(len(rows))
    return [rows[int(i)] for i in idx]


def per_stratum(rows, keyfn, rng):
    groups = {}
    for r in rows:
        groups.setdefault(keyfn(r), []).append(r)
    return {k: shuffled(v, rng) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def interleave(groups, order):
    pos = {k: 0 for k in order}
    out = []
    alive = [k for k in order if groups.get(k)]
    while alive:
        nxt = []
        for k in alive:
            g = groups[k]
            if pos[k] < len(g):
                out.append(g[pos[k]])
                pos[k] += 1
            if pos[k] < len(g):
                nxt.append(k)
        alive = nxt
    return out


def proportional_interleave(groups, order):
    items = []
    for k in order:
        g = groups.get(k) or []
        L = len(g)
        for j, r in enumerate(g):
            items.append((j / L if L else 0.0, order.index(k), r))
    items.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in items]


def mk_choice(state, criteria, gold_key, instr, meta):
    cfg = {"type": "choice", "criteria": criteria}
    kind, keys, opts = to_options(cfg)
    lab = keys.index(str(gold_key))
    return dict(kind=kind, keys=keys, opts=opts, label=lab, state=state, instructions=instr, meta=meta)


def mk_yes(state, yes, instr, meta):
    return dict(kind="yes_no", keys=["no", "yes"], opts=["no", "yes"],
                label=1 if yes else 0, state=state, instructions=instr, meta=meta)


def mk_score(state, level, levels, instr, meta):
    cfg = {"type": "score", "criteria": levels}
    kind, keys, opts = to_options(cfg)
    return dict(kind=kind, keys=keys, opts=opts, label=int(level), state=state,
                instructions=instr, meta=meta)


class Builder:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.seen = set()
        self.dups = 0
        self.over999 = 0
        self.enc = None

    def cap(self, key):
        return int(CAPS[key] * self.scale)

    def make(self, ds, lang, domain, typ, raw, split, src_row):
        k = normkey(raw["state"], raw["instructions"])
        if k in self.seen:
            self.dups += 1
            return None
        ids = self.enc(raw["state"], raw["instructions"], raw["opts"]).ids
        if len(ids) > CAP_TOKENS:
            self.over999 += 1
            return None
        self.seen.add(k)
        return dict(id=None, split=split, dataset=ds, domain=domain, lang=lang, type=typ,
                    state=raw["state"], instructions=raw["instructions"],
                    criteria=None, keys=raw["keys"], label=raw["label"], src_row=src_row,
                    meta=raw.get("meta"))

    def fill(self, cands, n, ds, lang, domain, typ, split):
        out = []
        for r in cands:
            if len(out) >= n:
                break
            rec = self.make(ds, lang, domain, typ, r["raw"], split, r["src"])
            if rec is not None:
                rec["criteria"] = r["criteria"]
                rec["meta"] = r.get("meta")
                out.append(rec)
        return out


def read_parquet(p):
    return pq.read_table(str(p)).to_pylist()


def label_names_from_parquet(path, field="label"):
    t = pq.read_table(str(path))
    md = t.schema.metadata or {}
    raw = md.get(b"huggingface")
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))["info"]["features"][field]["names"]
    except Exception:
        return None


def b_banking77(B, idx, D):
    key = "banking77"
    test = read_parquet(D / "raw" / key / "data" / "test-00000-of-00001.parquet")
    train = read_parquet(D / "raw" / key / "data" / "train-00000-of-00001.parquet")
    names = label_names_from_parquet(D / "raw" / key / "data" / "test-00000-of-00001.parquet")
    if not names:
        raise RuntimeError("banking77 label names not in parquet metadata")
    rng = rng_for(idx)

    def cands(rows, split):
        g = per_stratum(rows, lambda r: r["label"], rng)
        groups = {}
        for lab, lst in g.items():
            out = []
            for r in lst:
                gold = names[int(r["label"])]
                others = [x for x in names if x != gold]
                picks = rng.choice(len(others), size=19, replace=False)
                crit = {gold: None}
                for p in picks:
                    crit[others[int(p)]] = None
                out.append(dict(raw=mk_choice(clean(r["text"], B.cap(key)), crit, gold,
                                              INSTR[key], {"label": gold}),
                                criteria=crit, src=f"{split}/{r.get('row_idx')}",
                                meta={"label": gold}))
            groups[lab] = out
        return interleave(groups, sorted(groups))

    for i, r in enumerate(test):
        r["row_idx"] = i
    for i, r in enumerate(train):
        r["row_idx"] = i
    return (B.fill(cands(test, "test"), 120, key, "en", "intent routing", "choice", "test"),
            B.fill(cands(train, "train"), 50, key, "en", "intent routing", "choice", "calib"))


def b_massive(B, idx, D):
    key = "massive"
    langs = ["en", "hi", "bn", "ta", "te"]
    rng = rng_for(idx)
    tests, calibs = [], []
    for lang in langs:
        for split, fname, want, target in (("test", "test", 32, tests),
                                           ("calib", "validation", 16, calibs)):
            p = D / "raw" / key / fname / f"{lang}.json.gz"
            rows = []
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rows.append(json.loads(line))
            labels = sorted({r["label_text"] for r in rows})
            g = per_stratum(rows, lambda r: r["label_text"], rng)
            groups = {}
            for lab, lst in g.items():
                out = []
                for r in lst:
                    gold = r["label_text"]
                    others = [x for x in labels if x != gold]
                    picks = rng.choice(len(others), size=19, replace=False)
                    crit = {gold: None}
                    for q in picks:
                        crit[others[int(q)]] = None
                    out.append(dict(raw=mk_choice(clean(r["text"], B.cap(key)), crit, gold,
                                                  INSTR[key], {"label": gold}),
                                    criteria=crit, src=f"{fname}/{lang}/{r.get('id')}",
                                    meta={"label": gold}))
                groups[lab] = out
            cands = interleave(groups, sorted(groups))
            taken = 0
            for r in cands:
                if taken >= want:
                    break
                rec = B.make(key, lang, "intent routing", "choice", r["raw"], split, r["src"])
                if rec is None:
                    continue
                rec["criteria"] = r["criteria"]
                rec["meta"] = r["meta"]
                target.append(rec)
                taken += 1
    return tests, calibs


def b_amazon_polarity(B, idx, D):
    key = "amazon_polarity"
    rows = read_parquet(D / "raw" / key / "data" / "test-00000-of-00001.parquet")
    rng = rng_for(idx)
    for i, r in enumerate(rows):
        r["row_idx"] = i
    g = per_stratum(rows, lambda r: int(r["label"]), rng)
    out_t, out_c = [], []
    for split, per_label in (("test", 40), ("calib", 20)):
        for lab in sorted(g):
            lst = [r for r in g[lab]
                   if not (normkey(clean(r["text"], B.cap(key)), INSTR[key]) in B.seen)]
            taken = 0
            for r in lst:
                if taken >= per_label:
                    break
                raw = mk_yes(clean(r["text"], B.cap(key)), int(r["label"]) == 1,
                             INSTR[key], {"label": int(r["label"])})
                rec = B.make(key, "en", "sentiment", "yes_no", raw, split, f"test/{r['row_idx']}")
                if rec is None:
                    continue
                rec["criteria"] = None
                (out_t if split == "test" else out_c).append(rec)
                taken += 1
    return out_t, out_c


def b_civil(B, idx, D):
    key = "civil_comments"
    rng = rng_for(idx)
    levels = ["not toxic", "mildly rude", "toxic", "severely toxic"]
    bins = [0.0, 0.1, 0.4, 0.7, 1.0001]

    def binof(x):
        for i in range(4):
            if bins[i] <= x < bins[i + 1]:
                return i
        return 3

    out = {}
    for split, fname, counts in (("test", "test-00000-of-00001.parquet", [38, 38, 37, 37]),
                                 ("calib", "validation-00000-of-00001.parquet", [15, 15, 15, 15])):
        rows = read_parquet(D / "raw" / key / "data" / fname)
        for i, r in enumerate(rows):
            r["row_idx"] = i
        g = per_stratum([r for r in rows if clean(r["text"], B.cap(key))],
                        lambda r: binof(float(r["toxicity"])), rng)
        recs = []
        for bi in range(4):
            taken = 0
            for r in g.get(bi, []):
                if taken >= counts[bi]:
                    break
                raw = mk_score(clean(r["text"], B.cap(key)), bi, levels, INSTR[key],
                               {"toxicity": float(r["toxicity"])})
                rec = B.make(key, "en", "toxicity / moderation", "score", raw, split,
                             f"{fname}/{r['row_idx']}")
                if rec is None:
                    continue
                rec["criteria"] = levels
                recs.append(rec)
                taken += 1
        out[split] = recs
    return out["test"], out["calib"]


def b_snli(B, idx, D):
    key = "snli"
    rng = rng_for(idx)
    out = {}
    for split, fname, counts in (("test", "test-00000-of-00001.parquet", {0: 50, 1: 25, 2: 25}),
                                 ("calib", "validation-00000-of-00001.parquet", {0: 20, 1: 10, 2: 10})):
        rows = [r for r in read_parquet(D / "raw" / key / "plain_text" / fname)
                if int(r["label"]) in (0, 1, 2)]
        for i, r in enumerate(rows):
            r["row_idx"] = i
        g = per_stratum(rows, lambda r: int(r["label"]), rng)
        recs = []
        for lab, want in counts.items():
            taken = 0
            for r in g.get(lab, []):
                if taken >= want:
                    break
                st = clean("Premise: " + r["premise"] + "\nHypothesis: " + r["hypothesis"],
                           B.cap(key))
                raw = mk_yes(st, lab == 0, INSTR[key], {"nli": int(r["label"])})
                rec = B.make(key, "en", "NLI", "yes_no", raw, split, f"{fname}/{r['row_idx']}")
                if rec is None:
                    continue
                rec["criteria"] = None
                recs.append(rec)
                taken += 1
        out[split] = recs
    return out["test"], out["calib"]


def b_boolq(B, idx, D):
    key = "boolq"
    rng = rng_for(idx)
    out = {}
    for split, fname, per in (("test", "validation-00000-of-00001.parquet", 40),
                              ("calib", "train-00000-of-00001.parquet", 20)):
        rows = read_parquet(D / "raw" / key / "data" / fname)
        for i, r in enumerate(rows):
            r["row_idx"] = i
        g = per_stratum(rows, lambda r: bool(r["answer"]), rng)
        recs = []
        for ans in (False, True):
            taken = 0
            for r in g.get(ans, []):
                if taken >= per:
                    break
                q = clean(r["question"], 400)
                q = q[0].upper() + q[1:] if q else q
                if not q.endswith("?"):
                    q = q + "?"
                ins = q + " Answer yes or no."
                raw = mk_yes(clean(r["passage"], B.cap(key)), bool(r["answer"]), ins,
                             {"answer": bool(r["answer"])})
                rec = B.make(key, "en", "reading-comprehension QA", "yes_no", raw, split,
                             f"{fname}/{r['row_idx']}")
                if rec is None:
                    continue
                rec["criteria"] = None
                recs.append(rec)
                taken += 1
        out[split] = recs
    return out["test"], out["calib"]


def b_mmlu(B, idx, D):
    key = "mmlu"
    rng = rng_for(idx)
    out = {}
    for split, fname, n in (("test", "test-00000-of-00001.parquet", 130),
                            ("calib", "dev-00000-of-00001.parquet", 52)):
        rows = read_parquet(D / "raw" / key / "all" / fname)
        for i, r in enumerate(rows):
            r["row_idx"] = i
        g = per_stratum(rows, lambda r: r["subject"], rng)
        cands = interleave(g, list(g.keys()))
        recs = []
        for r in cands:
            if len(recs) >= n:
                break
            ch = list(r["choices"])
            if len(ch) != 4:
                continue
            crit = {"A": clean(ch[0], 300), "B": clean(ch[1], 300),
                    "C": clean(ch[2], 300), "D": clean(ch[3], 300)}
            gold = "ABCD"[int(r["answer"])]
            raw = mk_choice(clean(r["question"], B.cap(key)), crit, gold, INSTR[key],
                            {"subject": r["subject"]})
            rec = B.make(key, "en", "knowledge MCQ", "choice", raw, split,
                         f"{fname}/{r['row_idx']}")
            if rec is None:
                continue
            rec["criteria"] = crit
            rec["meta"] = {"subject": r["subject"]}
            recs.append(rec)
        out[split] = recs
    return out["test"], out["calib"]


def b_global_mmlu(B, idx, D):
    key = "global_mmlu"
    rng = rng_for(idx)
    out = {}
    for split, fname, per in (("test", "test-00000-of-00001.parquet", 40),
                              ("calib", "dev-00000-of-00001.parquet", 15)):
        recs = []
        for lang in ("hi", "bn", "te"):
            rows = read_parquet(D / "raw" / key / lang / fname)
            for i, r in enumerate(rows):
                r["row_idx"] = i
            g = per_stratum(rows, lambda r: r["subject"], rng)
            cands = interleave(g, list(g.keys()))
            taken = 0
            for r in cands:
                if taken >= per:
                    break
                crit = {"A": clean(r["option_a"], 300), "B": clean(r["option_b"], 300),
                        "C": clean(r["option_c"], 300), "D": clean(r["option_d"], 300)}
                gold = str(r["answer"]).strip().upper()[:1]
                if gold not in "ABCD":
                    continue
                raw = mk_choice(clean(r["question"], B.cap(key)), crit, gold, INSTR[key],
                                {"subject": r["subject"],
                                 "cult": r.get("cultural_sensitivity_label")})
                rec = B.make(key, lang, "knowledge MCQ", "choice", raw, split,
                             f"{lang}/{fname}/{r['row_idx']}")
                if rec is None:
                    continue
                rec["criteria"] = crit
                rec["meta"] = {"subject": r["subject"],
                               "cult": r.get("cultural_sensitivity_label")}
                recs.append(rec)
                taken += 1
        out[split] = recs
    return out["test"], out["calib"]


def b_sib200(B, idx, D):
    key = "sib200"
    rng = rng_for(idx)
    files = {"en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml"}
    out = {}
    for split, src, per in (("test", "test", 30), ("calib", "train", 12)):
        recs = []
        for lang, code in files.items():
            p = D / "raw" / key / "data" / code / f"{src}.tsv"
            rows = []
            with open(p, encoding="utf-8") as fh:
                rd = csv.DictReader(fh, delimiter="\t")
                for i, r in enumerate(rd):
                    r["row_idx"] = i
                    rows.append(r)
            g = per_stratum(rows, lambda r: r["category"], rng)
            cands = interleave(g, sorted(g.keys()))
            taken = 0
            for r in cands:
                if taken >= per:
                    break
                crit = {name: None for name in SIB_LABELS}
                raw = mk_choice(clean(r["text"], B.cap(key)), crit, r["category"],
                                INSTR[key], {"category": r["category"]})
                rec = B.make(key, lang, "topic classification", "choice", raw, split,
                             f"{lang}/{src}/{r['row_idx']}")
                if rec is None:
                    continue
                rec["criteria"] = crit
                rec["meta"] = {"category": r["category"]}
                recs.append(rec)
                taken += 1
        out[split] = recs
    return out["test"], out["calib"]


def b_dbpedia(B, idx, D):
    key = "dbpedia14"
    rng = rng_for(idx)
    rows = read_parquet(D / "raw" / key / "dbpedia_14" / "test-00000-of-00001.parquet")
    for i, r in enumerate(rows):
        r["row_idx"] = i
    g = per_stratum(rows, lambda r: int(r["label"]), rng)
    names = DBPEDIA_NAMES
    out = {}
    for split, n in (("test", 70), ("calib", 25)):
        cands = []
        for lab, lst in g.items():
            for r in lst:
                gold = names[int(r["label"])]
                crit = {x: None for x in names}
                cands.append(dict(raw=mk_choice(clean(r["title"] + ": " + r["content"], B.cap(key)),
                                                crit, gold, INSTR[key], {"label": gold}),
                                  criteria=crit, src=f"test/{r['row_idx']}", meta={"label": gold}))
        by = {}
        for c in cands:
            by.setdefault(c["raw"]["meta"]["label"], []).append(c)
        order = sorted(by.keys())
        seq = []
        pos = {k: 0 for k in order}
        while len(seq) < len(cands):
            for k in order:
                if pos[k] < len(by[k]):
                    seq.append(by[k][pos[k]])
                    pos[k] += 1
        recs = []
        for c in seq:
            if len(recs) >= n:
                break
            rec = B.make(key, "en", "topic / entity type", "choice", c["raw"], split, c["src"])
            if rec is None:
                continue
            rec["criteria"] = c["criteria"]
            rec["meta"] = c["meta"]
            recs.append(rec)
        out[split] = recs
    return out["test"], out["calib"]


def b_when2call(B, idx, D):
    key = "when2call"
    rng = rng_for(idx)
    rows = []
    with open(D / "raw" / key / "test" / "when2call_test_mcq.jsonl", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if line.strip():
                r = json.loads(line)
                r["row_idx"] = i
                rows.append(r)
    g = per_stratum(rows, lambda r: r.get("correct_answer"), rng)
    seq = interleave(g, sorted(g.keys()))
    amap = [("A", "direct"), ("B", "tool_call"), ("C", "request_for_info"), ("D", "cannot_answer")]
    out = {}
    for split, n in (("test", 70), ("calib", 20)):
        recs = []
        for r in seq:
            if len(recs) >= n:
                break
            ans = r.get("answers") or {}
            names = []
            for t in (r.get("tools") or []):
                try:
                    j = json.loads(t)
                    if isinstance(j, dict) and j.get("name"):
                        names.append(str(j["name"]))
                except Exception:
                    pass
            st = clean(r.get("question", ""), 700)
            if names:
                st = (st + "\nAvailable tools: " + ", ".join(names[:20]))[:B.cap(key)]
            crit = {}
            for k, src in amap:
                txt = str(ans.get(src, ""))[:240]
                crit[k] = txt if txt else None
            gold = [k for k, src in amap if src == r.get("correct_answer")]
            if not gold:
                continue
            raw = mk_choice(clean(st, B.cap(key)), crit, gold[0], INSTR[key],
                            {"answer": r.get("correct_answer")})
            rec = B.make(key, "en", "tool-use decision", "choice", raw, split,
                         f"test/{r['row_idx']}")
            if rec is None:
                continue
            rec["criteria"] = crit
            rec["meta"] = {"answer": r.get("correct_answer")}
            recs.append(rec)
        out[split] = recs
    return out["test"], out["calib"]


BUILDERS = [b_banking77, b_massive, b_amazon_polarity, b_civil, b_snli, b_boolq, b_mmlu,
            b_global_mmlu, b_sib200, b_dbpedia, b_when2call]


def build_all(weights, data, scale=1.0):
    B = Builder(scale)
    B.enc = Encoder(weights / "tokenizer.json")
    test, calib = [], []
    for idx, fn in enumerate(BUILDERS):
        t, c = fn(B, idx, data)
        test += t
        calib += c
    return B, test, calib


def tokenise(records, weights):
    enc = Encoder(weights / "tokenizer.json")
    toks = []
    for r in records:
        kind, keys, opts = to_options({"type": r["type"], "criteria": r["criteria"]})
        toks.append(len(enc(r["state"], r["instructions"], opts).ids))
    return toks


def chance_of(r):
    if r["type"] == "yes_no":
        return 0.5
    return 1.0 / len(r["keys"])


def add_ids(records, split):
    pre = "t" if split == "test" else "c"
    for i, r in enumerate(records):
        r["id"] = f"{r['dataset']}-{pre}{i:04d}"


def order_recs(records):
    groups, order = {}, []
    for r in records:
        k = (r["dataset"], r["lang"])
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(r)
    order.sort()
    return proportional_interleave(groups, order)


def main():
    global SEED, CAP_TOKENS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="weights", help="weights directory with tokenizer.json")
    ap.add_argument("--data", default="data", help="dataset directory with raw/")
    ap.add_argument("--out", default="results", help="results directory")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--cap-tokens", type=int, default=CAP_TOKENS)
    a = ap.parse_args()
    SEED, CAP_TOKENS = a.seed, a.cap_tokens
    weights, data, out = pathlib.Path(a.weights), pathlib.Path(a.data), pathlib.Path(a.out)
    t_start = time.time()
    scale = 1.0
    for attempt in (1, 2):
        B, test, calib = build_all(weights, data, scale)
        add_ids(test, "test")
        add_ids(calib, "calib")
        params = [normkey(r["state"], r["instructions"]) for r in test + calib]
        dup_all = len(params) - len(set(params))
        toks_t, toks_c = tokenise(test, weights), tokenise(calib, weights)
        if (sum(toks_t) <= 330_000 and sum(toks_c) <= 140_000) or attempt == 2:
            break
        for k in ("boolq", "mmlu", "global_mmlu", "dbpedia14"):
            CAPS[k] = int(CAPS[k] * 0.75)

    toks_t, toks_c = tokenise(test, weights), tokenise(calib, weights)
    for r, n in zip(test, toks_t):
        r["n_tokens"] = n
    for r, n in zip(calib, toks_c):
        r["n_tokens"] = n

    key_t = [(r["dataset"], r["src_row"]) for r in test]
    key_c = [(r["dataset"], r["src_row"]) for r in calib]
    overlap_src = sorted(set(key_t) & set(key_c))
    overlap_text = sorted(set(normkey(r["state"], r["instructions"]) for r in test) &
                          set(normkey(r["state"], r["instructions"]) for r in calib))
    test, calib = order_recs(test), order_recs(calib)

    (data / "eval").mkdir(parents=True, exist_ok=True)
    with open(data / "eval" / "test.jsonl", "w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(data / "eval" / "calib.jsonl", "w", encoding="utf-8") as f:
        for r in calib:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    gold = []
    for name, q in QUESTIONS.items():
        kind, keys, opts = to_options(q)
        gold.append(dict(id="golden-" + name, split="golden", dataset="golden", domain="golden",
                         lang="en", type=kind, state=STATE_TEXT, instructions=q["instructions"],
                         criteria=q.get("criteria"), keys=keys, label=None, src_row="model card",
                         n_tokens=len(Encoder(weights / "tokenizer.json")(
                             STATE_TEXT, q["instructions"], opts).ids)))
    write_json(data / "eval" / "golden.json", {"items": gold})

    def counts(recs, field):
        d = {}
        for r in recs:
            d[str(r[field])] = d.get(str(r[field]), 0) + 1
        return dict(sorted(d.items()))

    per_ds = {}
    for r in test + calib:
        k = r["dataset"]
        d = per_ds.setdefault(k, {"test": 0, "calib": 0, "labels": {}, "type": r["type"]})
        d[r["split"]] += 1
        d["labels"][str(r["label"])] = d["labels"].get(str(r["label"]), 0) + 1

    chance = {}
    for k in per_ds:
        rr = next(r for r in test + calib if r["dataset"] == k)
        chance[k] = round(chance_of(rr), 4)

    report = {
        "seed": SEED,
        "n_test": len(test), "n_calib": len(calib),
        "tokens_test": int(sum(toks_t)), "tokens_calib": int(sum(toks_c)),
        "max_tokens_per_question_test": int(max(toks_t)),
        "max_tokens_per_question_calib": int(max(toks_c)),
        "overlap_src_rows": len(overlap_src),
        "overlap_normalised_text": len(overlap_text),
        "dup_normalised_all": dup_all,
        "missing_label": sum(1 for r in test + calib if r["label"] is None),
        "counts_test": {"dataset": counts(test, "dataset"), "domain": counts(test, "domain"),
                        "lang": counts(test, "lang"), "type": counts(test, "type")},
        "counts_calib": {"dataset": counts(calib, "dataset"), "domain": counts(calib, "domain"),
                         "lang": counts(calib, "lang"), "type": counts(calib, "type")},
        "per_dataset": per_ds,
        "random_baseline": chance,
        "caps_chars": CAPS,
        "budget": {"tokens_test_max": 330000, "tokens_calib_max": 140000,
                   "max_tokens_per_question": CAP_TOKENS},
        "build_seconds": round(time.time() - t_start, 1),
    }
    checks = {
        "test_count": len(test) == 1200,
        "calib_count": len(calib) == 500,
        "no_source_overlap": len(overlap_src) == 0,
        "no_text_overlap": len(overlap_text) == 0,
        "all_labelled": report["missing_label"] == 0,
        "tokens_test": sum(toks_t) <= 330_000,
        "tokens_calib": sum(toks_c) <= 140_000,
        "max_tokens": max(toks_t + toks_c) <= CAP_TOKENS,
    }
    report["checks"] = checks
    report["all_checks"] = all(checks.values())
    write_json(out / "eval" / "eval_set.json", report)

    labels = []
    for r in test + calib:
        labels.append({"id": r["id"], "split": r["split"], "dataset": r["dataset"],
                       "domain": r["domain"], "lang": r["lang"], "type": r["type"],
                       "label": r["label"], "n_options": len(r["keys"])})
    write_json(out / "eval" / "labels.json", labels)
    print(json.dumps({k: report[k] for k in ("n_test", "n_calib", "tokens_test",
                                             "tokens_calib", "all_checks")}, indent=1))
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
