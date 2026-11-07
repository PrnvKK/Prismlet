"""Compare decoder layers 0 and 1 against the fp32 reference in three precision modes."""

import argparse
import json
import pathlib
import time

import torch
import torch.nn.functional as F

from _common import pick_device, write_json

from prismlet.encoder import Encoder, to_options
from prismlet.engine import PrismEngine
from prismlet.examples import QUESTIONS, STATE_TEXT
from reference.oracle import load_oracle
from reference.reference import ref_layers

SKIP = {"ids", "pos", "role", "seg", "topk_ids"}


def compare(got, want):
    if got.shape != want.shape:
        got = got.reshape(want.shape)
    gf, wf = got.float(), want.float()
    if gf.dim() == 1:
        gf, wf = gf[None], wf[None]
    g2, w2 = gf.reshape(gf.shape[0], -1), wf.reshape(wf.shape[0], -1)
    cos = F.cosine_similarity(g2, w2, dim=-1)
    rel = float((gf - wf).norm() / wf.norm())
    return dict(cos_min=float(cos.min()), cos_mean=float(cos.mean()), rel_l2=rel)


def check_mode(name, mode_results, limits):
    failures = []
    for lk, res in mode_results.items():
        h = res["h_out"]
        if h["cos_min"] < limits["h_cos_min"] or h["rel_l2"] > limits["h_rel"]:
            failures.append(f"{name} {lk} h_out {h}")
        for k, v in res["traces"].items():
            if k == "topk_ids_set_eq":
                if v < limits["topk"]:
                    failures.append(f"{name} {lk} topk {v}")
            elif v["rel_l2"] > limits["trace_rel"]:
                failures.append(f"{name} {lk} {k} {v}")
            elif v["cos_min"] < limits.get("trace_cos", -1.0):
                failures.append(f"{name} {lk} {k} {v}")
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="weights", help="directory with the official weights")
    ap.add_argument("--out", default="results", help="results directory")
    ap.add_argument("--device", default="auto", help="auto, xpu or cpu")
    ap.add_argument("--modes", default="fp32,bf16,bf16_emu")
    ap.add_argument("--layers", default="0,1")
    a = ap.parse_args()
    weights = pathlib.Path(a.weights)
    device = pick_device(a.device)
    layers = [int(x) for x in a.layers.split(",") if x.strip()]

    kind, keys, opts = to_options(QUESTIONS["team"])
    enc = Encoder(weights / "tokenizer.json")(STATE_TEXT, QUESTIONS["team"]["instructions"], opts)

    out = {"n_tokens": len(enc.ids), "device": device, "modes": {}}

    prism_py = weights / "prism.py"
    if prism_py.exists():
        o = load_oracle(prism_py)
        oh = o.DecisionHead(4096, dh=512, layers=2)
        from prismlet.head import DecisionHead
        dh = DecisionHead(weights / "decision_head.safetensors")
        oh.load_state_dict({k[len("head."):]: v for k, v in dh.p.items() if k.startswith("head.")})
        g = torch.Generator().manual_seed(3)
        h_ans = torch.randn(4096, generator=g)
        h_opts = torch.randn(3, 4096, generator=g)
        head_delta = {}
        for qt in ("choice", "yes_no", "score"):
            a1 = dh(h_ans, h_opts, qt)
            b1 = torch.softmax(oh(h_ans, h_opts, qt), -1)
            head_delta[qt] = float((a1 - b1).abs().max().detach())
        out["head_oracle_max_abs"] = head_delta
        out["head_oracle"] = max(head_delta.values()) < 1e-5
    else:
        out["head_oracle"] = None
        out["head_oracle_note"] = "weights/prism.py not present"

    refs = {}
    for upto in layers:
        t0 = time.time()
        trace = {}
        h, _ = ref_layers(enc, upto, weights, trace=trace)
        refs[upto] = (h, trace)
        out.setdefault("reference_seconds", {})[str(upto)] = time.time() - t0

    mode_kwargs = {
        "fp32": dict(dtype=torch.float32, emulate_act=False),
        "bf16": dict(dtype=torch.bfloat16, emulate_act=False),
        "bf16_emu": dict(dtype=torch.bfloat16, emulate_act=True),
    }
    limits = {
        "fp32": dict(h_cos_min=0.9999, h_rel=2e-3, trace_cos=0.99999, trace_rel=1e-3, topk=0.995),
        "bf16": dict(h_cos_min=0.995, h_rel=3e-2, trace_rel=5e-2, topk=0.90),
        "bf16_emu": dict(h_cos_min=-1.0, h_rel=8e-2, trace_rel=1.5e-1, topk=0.75),
    }
    failures = []
    for name in [m for m in a.modes.split(",") if m.strip()]:
        eng = PrismEngine(weights_dir=weights, device=device, mode="disk",
                          max_tokens=8192, prefetch=False, pin_memory=False, **mode_kwargs[name])
        res = {}
        for upto in layers:
            trace = {}
            t0 = time.time()
            eng.forward_items([(STATE_TEXT, kind, opts, QUESTIONS["team"]["instructions"])],
                              trace=trace, stop_after=upto)
            dt = time.time() - t0
            comp = {}
            for k, v in refs[upto][1].items():
                if k in SKIP:
                    continue
                comp[k] = compare(trace[k], v)
            if "topk_ids" in refs[upto][1]:
                ids_r = refs[upto][1]["topk_ids"]
                ids_g = trace["topk_ids"]
                comp["topk_ids_set_eq"] = float(
                    (ids_r.sort(dim=-1).values == ids_g.sort(dim=-1).values).all(dim=-1).float().mean())
            res[f"L{upto}"] = dict(seconds=dt, traces=comp,
                                   h_out=compare(trace["h_out"], refs[upto][0]))
            del trace
        out["modes"][name] = res
        failures += check_mode(name, res, limits[name])
        del eng

    out["failures"] = failures
    out["passed"] = not failures and out["head_oracle"] is not False
    write_json(pathlib.Path(a.out) / "engine" / "layer.json", out)
    print(json.dumps({"device": device, "n_tokens": len(enc.ids),
                      "head_oracle": out["head_oracle"],
                      "modes": {m: {k: {"h_out": v[k]["h_out"]} for k in v}
                                for m, v in out["modes"].items()},
                      "failures": failures[:6], "passed": out["passed"]}, indent=1))


if __name__ == "__main__":
    main()
