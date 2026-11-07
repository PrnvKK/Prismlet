"""Check FP8 dequantisation and the input encoder against the official implementation."""

import argparse
import json
import pathlib
import random
import time

import torch

from _common import pick_device, write_json

from prismlet.encoder import ANSWER_ROLE, CONTEXT, PAD, Encoder, build_allow, to_options
from prismlet.fp8 import build_lut, count_nan_codes, dequant, linear_fp8, quant_act, round_e4m3
from prismlet.io import STFile
from reference.oracle import TokShim, load_oracle

STATE = {"ticket": "My card was charged twice for the same order and I still haven't got a refund."}
QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
             "criteria": {"billing": "payments, charges and refunds",
                          "tech": "app or website problems",
                          "delivery": "shipping and order status"}},
    "urgent": {"type": "yes_no", "instructions": "Does this need a reply within the hour?"},
    "anger": {"type": "score", "instructions": "How upset is the customer?",
              "criteria": ["calm", "annoyed", "angry", "furious"]},
}


def rel_l2(a, b):
    return float((a - b).norm() / b.norm())


def check_fp8(weights, device):
    res = {}
    lut = build_lut()
    finite = torch.isfinite(lut)
    res.update({"lut_ok": True, "lut_finite_codes": int(finite.sum()),
                "lut_max": float(lut[finite].max()),
                "lut_min_positive": float(lut[finite & (lut > 0)].min()),
                "lut_nan_codes": (~finite).nonzero().flatten().tolist()})
    codes = (torch.arange(1 << 20, dtype=torch.int64) % 256).to(torch.uint8).to(device)
    keep = (codes != 127) & (codes != 255)
    ref = lut[codes.cpu().long()].float()
    big = (torch.arange(25165824, dtype=torch.int64) % 254).to(torch.uint8).to(device)
    backends = {}
    for backend in ("native", "bits", "lut"):
        try:
            y = dequant(codes, backend, torch.float32)
            ok = torch.equal(y[keep].cpu(), ref[keep.cpu()]) and \
                torch.equal(torch.signbit(y.cpu()), torch.signbit(ref))
            t0 = time.time()
            for _ in range(2):
                z = dequant(big, backend, torch.bfloat16)
            if device.type == "xpu":
                torch.xpu.synchronize()
            dt = (time.time() - t0) / 2
            backends[backend] = {"ok": bool(ok), "gbps": big.numel() / dt / 1e9}
        except Exception as e:
            backends[backend] = {"ok": False, "error": repr(e)}
    ok_backends = [k for k, v in backends.items() if v.get("ok")]
    chosen = max(ok_backends, key=lambda k: backends[k].get("gbps", 0))
    res["backend_results"] = backends
    res["backend_chosen"] = chosen
    res["native_cast_xpu_ok"] = bool(backends.get("native", {}).get("ok"))

    f = STFile(pathlib.Path(weights) / "model-layer-00.safetensors")
    buf = torch.empty(f.nbytes, dtype=torch.uint8)
    f.read_all(buf, 4)
    v = f.views(buf)
    w8 = v["model.layers.0.self_attn.q_proj.weight"][:256]
    sc = v["model.layers.0.self_attn.q_proj.weight_scale"][:256].float()
    wref = w8.view(torch.float8_e4m3fn).float() * sc[:, None]
    ours = dequant(w8.to(device), chosen, torch.bfloat16).float().cpu() * sc[:, None]
    assert torch.equal(ours, wref)
    res["nan_codes_qproj"] = count_nan_codes(v["model.layers.0.self_attn.q_proj.weight"])
    res["real_tensor_equal"] = True

    g = torch.Generator().manual_seed(7)
    tests = [torch.clamp(torch.randn(2_000_000, generator=g) * 150, -448, 448),
             lut[finite], (lut[finite][:-1] + lut[finite][1:]) / 2,
             torch.cat([torch.linspace(0, 2 ** -6, 100000),
                        torch.linspace(-2 ** -6, 0, 100000)]),
             torch.tensor([0.0, -0.0, 1e-30, -1e-30, 448.0, -448.0, 464.0, -464.0])]
    for x in tests:
        assert torch.equal(round_e4m3(x.float()), x.float().to(torch.float8_e4m3fn).float())
    res["round_e4m3_equal"] = True

    x = torch.randn(257, 4096, generator=g)
    x[:, 0] *= 30
    xb = x.to(torch.bfloat16).to(device)
    xq, s = quant_act(xb)
    xf = xb.float().cpu()
    sref = (xf.abs().amax(-1, keepdim=True) / 448).clamp(min=1e-12)
    assert torch.equal(xq.cpu().float(), round_e4m3(xf / sref))
    assert torch.equal(s.cpu(), sref)
    res["quant_act_equal"] = True

    W = wref
    xb = torch.randn(32, 4096, generator=g).to(torch.bfloat16).to(device)
    ref32 = xb.cpu().float() @ W.T
    out_bf16 = linear_fp8(xb, dequant(w8.to(device), chosen, torch.bfloat16), sc.to(device), False)
    r_bf = rel_l2(out_bf16.cpu().float(), ref32)
    x32 = xb.float()
    out32 = linear_fp8(x32, dequant(w8.to(device), chosen, torch.float32), sc.to(device), False)
    r32 = rel_l2(out32.cpu(), x32.cpu() @ W.T)
    xs = torch.clamp(xb.cpu().float().abs().amax(-1, keepdim=True) / 448, min=1e-12)
    xq_cpu = (xb.cpu().float() / xs).to(torch.float8_e4m3fn).float()
    ref_emu = (xq_cpu @ W.T) * xs
    out_emu = linear_fp8(xb, dequant(w8.to(device), chosen, torch.bfloat16), sc.to(device), True)
    res.update(linear_bf16_rel_l2=r_bf, linear_fp32_rel_l2=r32,
               linear_emu_rel_l2=rel_l2(out_emu.cpu().float(), ref_emu))
    return res


def check_encoder(weights):
    res = {}
    enc = Encoder(pathlib.Path(weights) / "tokenizer.json")
    res["tags"] = enc.tags
    assert enc.tags["state"] == [3112, 236787]
    assert enc.tags["question"] == [107, 14977, 236787]
    assert enc.tags["answer"] == [107, 7925, 236787]

    prism = pathlib.Path(weights) / "prism.py"
    if not prism.exists():
        res["oracle"] = "weights/prism.py not present"
        return res
    oracle = load_oracle(prism)
    oenc = oracle.Encoder(TokShim(enc.tk))

    def rand_text(rng, n=None):
        n = n or rng.randint(0, 40)
        alphabet = "abc XYZ0123\n\t<option></option>‹éहिन्दीதமிழ்🙂"
        return "".join(rng.choice(alphabet) for _ in range(n))

    rng = random.Random(1234)
    cases = []
    for q in QUESTIONS.values():
        kind, keys, opts = to_options(q)
        cases.append((json.dumps(STATE, ensure_ascii=False, separators=(",", ":")),
                      q["instructions"], opts))
    for _ in range(200):
        k = rng.randint(1, 8)
        opts = [rand_text(rng, rng.randint(0, 12)) for _ in range(k)]
        cases.append((rand_text(rng, rng.randint(0, 300)),
                      rand_text(rng, rng.randint(0, 60)), opts))
    for st, ins, opts in cases:
        a, b = enc(st, ins, opts), oenc(st, ins, opts)
        assert (a.ids, a.pos, a.role, a.seg, a.close_idx, a.answer_idx, a.n_context) == \
               (b.ids, b.pos, b.role, b.seg, b.close_idx, b.answer_idx, b.n_context)
    res["encoder_oracle_cases"] = len(cases)

    kind, keys, opts = to_options(QUESTIONS["team"])
    e = enc(json.dumps(STATE, ensure_ascii=False, separators=(",", ":")),
            QUESTIONS["team"]["instructions"], opts)
    for j in range(len(opts)):
        first = next(i for i, r in enumerate(e.role) if r == j)
        assert e.pos[first] == e.n_context
        assert e.close_idx[j] == max(i for i, r in enumerate(e.role) if r == j)
    res["structural_ok"] = True

    L = len(e.ids)

    def rule_mask(roles):
        allow = torch.zeros(L, L, dtype=torch.bool)
        for i in range(L):
            for j in range(L):
                if i == j:
                    allow[i, j] = True
                elif roles[j] == PAD:
                    continue
                elif roles[i] == CONTEXT:
                    allow[i, j] = roles[j] == CONTEXT
                elif roles[i] >= 0:
                    allow[i, j] = roles[j] == CONTEXT or roles[j] == roles[i]
                elif roles[i] == ANSWER_ROLE:
                    allow[i, j] = j <= i
        return allow

    assert torch.equal(rule_mask(e.role)[None], build_allow(torch.tensor([e.role])))
    assert torch.equal(build_allow(torch.tensor([e.role])),
                       oracle.attention_mask([e.role], L, "cpu", torch.float32)[:, 0] == 0)
    for _ in range(300):
        ctx = rng.randint(1, 20)
        nopt = rng.randint(1, 6)
        roles = [CONTEXT] * ctx
        for j in range(nopt):
            roles += [j] * rng.randint(1, 8)
        roles += [ANSWER_ROLE] * rng.randint(1, 4)
        roles += [PAD] * rng.randint(0, 5)
        L2 = len(roles)
        a = torch.zeros(L2, L2, dtype=torch.bool)
        for i in range(L2):
            for j in range(L2):
                if i == j:
                    a[i, j] = True
                elif roles[j] == PAD:
                    continue
                elif roles[i] == CONTEXT:
                    a[i, j] = roles[j] == CONTEXT
                elif roles[i] >= 0:
                    a[i, j] = roles[j] == CONTEXT or roles[j] == roles[i]
                elif roles[i] == ANSWER_ROLE:
                    a[i, j] = j <= i
        b = build_allow(torch.tensor([roles]))
        c = oracle.attention_mask([roles], L2, "cpu", torch.float32)[:, 0] == 0
        nonpad = torch.tensor([x != PAD for x in roles])
        assert torch.equal(a[None][:, nonpad][:, :, nonpad], b[:, nonpad][:, :, nonpad])
        assert torch.equal(b[:, nonpad][:, :, nonpad], c[:, nonpad][:, :, nonpad])
    res["mask_random_cases"] = 300

    for _ in range(200):
        kind = rng.choice(["choice", "yes_no", "score"])
        if kind == "choice":
            ks = rng.sample(["a", "b", "10", "9", "z", 3, 2], rng.randint(1, 5))
            q = {"type": kind, "criteria": {k: rng.choice([None, "", "text", 0, 3])
                                            for k in ks}}
        elif kind == "score":
            q = {"type": kind, "criteria": [rand_text(rng, 5)
                                            for _ in range(rng.randint(1, 5))]}
        else:
            q = {"type": kind}
        assert to_options(q) == oracle.to_options(q)
    res["to_options_random_cases"] = 200
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    device = torch.device(pick_device(a.device))
    fp8 = check_fp8(a.weights, device)
    write_json(pathlib.Path(a.out) / "engine" / "fp8.json", fp8)
    enc = check_encoder(a.weights)
    write_json(pathlib.Path(a.out) / "engine" / "encoder.json", enc)
    print(json.dumps({"fp8_backend": fp8["backend_chosen"],
                      "linear_bf16_rel_l2": fp8["linear_bf16_rel_l2"],
                      "encoder": enc.get("oracle", "ok")}, indent=1))


if __name__ == "__main__":
    main()
