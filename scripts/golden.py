"""Run the three-question example and compare it with the published probabilities."""

import argparse
import json
import pathlib
import time

import torch

from _common import pick_device, write_json

from prismlet.engine import PrismEngine
from prismlet.examples import QUESTIONS, STATE

PUBLISHED = {
    "team": {"billing": 0.986, "tech": 0.007, "delivery": 0.006},
    "urgent": {"yes": 0.45},
    "anger": {"0": 0.032, "1": 0.267, "2": 0.510, "3": 0.191},
}


def flat8(ans):
    return [ans["team"]["probabilities"][k] for k in ("billing", "tech", "delivery")] + \
           [ans["urgent"]["yes"]] + [ans["anger"]["probabilities"][k] for k in ("0", "1", "2", "3")]


def expected8():
    return [PUBLISHED["team"]["billing"], PUBLISHED["team"]["tech"], PUBLISHED["team"]["delivery"],
            PUBLISHED["urgent"]["yes"]] + [PUBLISHED["anger"][k] for k in ("0", "1", "2", "3")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="weights", help="directory with the official weights")
    ap.add_argument("--out", default="results", help="results directory")
    ap.add_argument("--device", default="auto", help="auto, xpu or cpu")
    ap.add_argument("--mode", default="fp8emu", choices=["bf16", "fp8emu"],
                    help="bf16 with or without FP8 activation emulation")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--dequant-backend", default="lut", choices=["lut", "native", "bits"])
    a = ap.parse_args()
    weights = pathlib.Path(a.weights)
    t0 = time.time()
    eng = PrismEngine(weights_dir=weights, device=pick_device(a.device), mode="disk",
                      dtype=torch.bfloat16, emulate_act=(a.mode == "fp8emu"),
                      dequant_backend=a.dequant_backend,
                      max_tokens=a.max_tokens, nthreads=8, prefetch=True)
    ans = eng.decide(STATE, QUESTIONS)
    dt = time.time() - t0
    ours = flat8(ans)
    exp = expected8()
    argmax = (ans["team"]["choice"], ans["urgent"]["yes"] < 0.5, ans["anger"]["level"])
    out = {
        "mode": a.mode,
        "probabilities": {
            "team": ans["team"]["probabilities"],
            "urgent": ans["urgent"]["yes"],
            "anger": ans["anger"]["probabilities"],
        },
        "expected": PUBLISHED,
        "max_abs_delta": max(abs(x - y) for x, y in zip(ours, exp)),
        "argmax": list(argmax),
        "argmax_matches_published": list(argmax) == ["billing", True, "2"],
        "seconds": dt,
        "timings": getattr(eng, "last_timings_total", {}),
    }
    write_json(pathlib.Path(a.out) / "engine" / "golden.json", out)
    print(json.dumps({k: out[k] for k in ("mode", "max_abs_delta", "argmax",
                                          "argmax_matches_published", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
