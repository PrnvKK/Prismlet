import torch
import torch.nn.functional as F

from reference.oracle import load_oracle
from prismlet.encoder import Encoder, to_options
from prismlet.engine import PrismEngine
from prismlet.examples import QUESTIONS, STATE_TEXT
from reference.reference import ref_layers


def compare(got, want):
    if got.shape != want.shape:
        got = got.reshape(want.shape)
    gf, wf = got.float(), want.float()
    if gf.dim() == 1:
        gf, wf = gf[None], wf[None]
    cos = F.cosine_similarity(gf.reshape(gf.shape[0], -1), wf.reshape(wf.shape[0], -1), dim=-1)
    return float(cos.min()), float((gf - wf).norm() / wf.norm())


def test_head_matches_official(weights_dir):
    prism = weights_dir / "prism.py"
    if not prism.exists():
        import pytest
        pytest.skip("weights/prism.py not present")
    o = load_oracle(prism)
    oh = o.DecisionHead(4096, dh=512, layers=2)
    from prismlet.head import DecisionHead
    dh = DecisionHead(weights_dir / "decision_head.safetensors")
    oh.load_state_dict({k[5:]: v for k, v in dh.p.items() if k.startswith("head.")})
    g = torch.Generator().manual_seed(3)
    h_ans = torch.randn(4096, generator=g)
    h_opts = torch.randn(3, 4096, generator=g)
    for qt in ("choice", "yes_no", "score"):
        a = dh(h_ans, h_opts, qt)
        b = torch.softmax(oh(h_ans, h_opts, qt), -1)
        assert float((a - b).abs().max().detach()) < 1e-5


def test_layers_match_reference(weights_dir):
    kind, keys, opts = to_options(QUESTIONS["team"])
    enc = Encoder(weights_dir / "tokenizer.json")(
        STATE_TEXT, QUESTIONS["team"]["instructions"], opts)
    refs = {}
    for upto in (0, 1):
        trace = {}
        h, _ = ref_layers(enc, upto, weights_dir, trace=trace)
        refs[upto] = (h, trace)

    cases = [
        ("fp32", dict(dtype=torch.float32, emulate_act=False), (0, 1),
         dict(cos_min=0.9999, rel=2e-3, trace_rel=1e-3, topk=0.995)),
        ("bf16", dict(dtype=torch.bfloat16, emulate_act=False), (0,),
         dict(cos_min=0.995, rel=3e-2, trace_rel=5e-2, topk=0.90)),
        ("bf16_emu", dict(dtype=torch.bfloat16, emulate_act=True), (0,),
         dict(cos_min=-1.0, rel=8e-2, trace_rel=1.5e-1, topk=0.75)),
    ]
    for name, kwargs, layers, lim in cases:
        device = "xpu" if torch.xpu.is_available() else "cpu"
        eng = PrismEngine(weights_dir=weights_dir, device=device, mode="disk",
                          max_tokens=8192, prefetch=False, pin_memory=False, **kwargs)
        for upto in layers:
            trace = {}
            eng.forward_items([(STATE_TEXT, kind, opts, QUESTIONS["team"]["instructions"])],
                              trace=trace, stop_after=upto)
            cos_min, rel = compare(trace["h_out"], refs[upto][0])
            assert cos_min >= lim["cos_min"], (name, upto, cos_min)
            assert rel <= lim["rel"], (name, upto, rel)
            for key, want in refs[upto][1].items():
                if key in ("ids", "pos", "role", "seg", "topk_ids"):
                    continue
                _, rel_k = compare(trace[key], want)
                assert rel_k <= lim["trace_rel"], (name, upto, key, rel_k)
            if "topk_ids" in refs[upto][1]:
                a = refs[upto][1]["topk_ids"].sort(dim=-1).values
                b = trace["topk_ids"].sort(dim=-1).values
                assert float((a == b).all(dim=-1).float().mean()) >= lim["topk"]
