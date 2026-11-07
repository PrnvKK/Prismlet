import json
import random

import torch

from reference.oracle import TokShim, load_oracle
from prismlet.encoder import ANSWER_ROLE, CONTEXT, PAD, Encoder, build_allow, to_options


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


def rule_mask(roles, L):
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


def rand_text(rng, n=None):
    n = n or rng.randint(0, 40)
    alphabet = "abc XYZ0123\n\t<option></option>‹éहिन्दीதமிழ்🙂"
    return "".join(rng.choice(alphabet) for _ in range(n))


def test_encoder_matches_official(weights_dir):
    prism = weights_dir / "prism.py"
    if not prism.exists():
        import pytest
        pytest.skip("weights/prism.py not present")
    ours = Encoder(weights_dir / "tokenizer.json")
    o = load_oracle(prism)
    oenc = o.Encoder(TokShim(ours.tk))
    assert ours.tags["state"] == [3112, 236787]
    assert ours.tags["question"] == [107, 14977, 236787]
    assert ours.tags["answer"] == [107, 7925, 236787]

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
        a, b = ours(st, ins, opts), oenc(st, ins, opts)
        assert (a.ids, a.pos, a.role, a.seg, a.close_idx, a.answer_idx, a.n_context) == \
               (b.ids, b.pos, b.role, b.seg, b.close_idx, b.answer_idx, b.n_context)


def test_attention_mask(weights_dir):
    prism = weights_dir / "prism.py"
    if not prism.exists():
        import pytest
        pytest.skip("weights/prism.py not present")
    ours = Encoder(weights_dir / "tokenizer.json")
    o = load_oracle(prism)
    kind, keys, opts = to_options(QUESTIONS["team"])
    e = ours(json.dumps(STATE, ensure_ascii=False, separators=(",", ":")),
             QUESTIONS["team"]["instructions"], opts)
    m_rule = rule_mask(e.role, len(e.ids))[None]
    m_ours = build_allow(torch.tensor([e.role]))
    m_orc = o.attention_mask([e.role], len(e.ids), "cpu", torch.float32)[:, 0] == 0
    assert torch.equal(m_rule, m_ours)
    assert torch.equal(m_ours, m_orc)


def test_to_options_matches_official(weights_dir):
    prism = weights_dir / "prism.py"
    if not prism.exists():
        import pytest
        pytest.skip("weights/prism.py not present")
    o = load_oracle(prism)
    rng = random.Random(99)
    for _ in range(100):
        kind = rng.choice(["choice", "yes_no", "score"])
        if kind == "choice":
            ks = rng.sample(["a", "b", "10", "9", "z", 3, 2], rng.randint(1, 5))
            q = {"type": kind, "criteria": {k: rng.choice([None, "", "text", 0, 3])
                                            for k in ks}}
        elif kind == "score":
            q = {"type": kind, "criteria": [rand_text(rng, 5) for _ in range(rng.randint(1, 5))]}
        else:
            q = {"type": kind}
        assert to_options(q) == o.to_options(q)
