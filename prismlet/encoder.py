"""Input encoding for Prism, a transcription of the ``prism.py`` encoder."""

import json
from dataclasses import dataclass

import torch
from tokenizers import Tokenizer

TAGS = {"state": "State:", "question": "\nQuestion:", "open": "\n<option>", "close": "</option>",
        "answer": "\nAnswer:"}
CONTEXT, ANSWER_ROLE, PAD = -1, -2, -3
SEG_STATE, SEG_QUESTION, SEG_OPTION, SEG_ANSWER = range(4)
TYPES = {"choice": 0, "yes_no": 1, "score": 2}


def _text(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


@dataclass
class Encoded:
    ids: list
    pos: list
    role: list
    seg: list
    close_idx: list
    answer_idx: int
    n_context: int


class Encoder:
    def __init__(self, tok_json):
        self.tk = Tokenizer.from_file(str(tok_json))
        self.tags = {k: self._raw(v) for k, v in TAGS.items()}

    def _raw(self, s):
        return self.tk.encode(s, add_special_tokens=False).ids

    def _ids(self, s):
        for t in ("</option>", "<option>"):
            s = s.replace(t, t.replace("<", "\u2039"))
        return self._raw(s)

    def __call__(self, state, instructions, options):
        t = self.tags
        ids = t["state"] + self._ids(" " + state)
        seg = [SEG_STATE] * len(ids)
        q = t["question"] + self._ids(" " + instructions)
        ids += q
        seg += [SEG_QUESTION] * len(q)
        role, pos = [CONTEXT] * len(ids), list(range(len(ids)))
        p0, close, longest = len(ids), [], 0
        for j, o in enumerate(options):
            span = t["open"] + self._ids(o) + t["close"]
            ids += span
            role += [j] * len(span)
            seg += [SEG_OPTION] * len(span)
            pos += range(p0, p0 + len(span))
            close.append(len(ids) - 1)
            longest = max(longest, len(span))
        a, p1 = t["answer"], p0 + longest
        ids += a
        role += [ANSWER_ROLE] * len(a)
        seg += [SEG_ANSWER] * len(a)
        pos += range(p1, p1 + len(a))
        return Encoded(ids, pos, role, seg, close, len(ids) - 1, p0)


def to_options(q):
    kind = q["type"]
    if kind == "choice":
        items = sorted(q["criteria"].items(), key=lambda kv: str(kv[0]))
        keys, opts = [k for k, _ in items], []
        for k, v in items:
            v = None if v is None else _text(v)
            opts.append(f"{k}: {v}" if v and v != k else str(k))
        return kind, keys, opts
    if kind == "yes_no":
        return kind, ["no", "yes"], ["no", "yes"]
    if kind == "score":
        c = q["criteria"]
        keys = [str(i) for i in range(len(c))] if isinstance(c, list) else list(c)
        levels = [_text(v) for v in (c if isinstance(c, list) else c.values())]
        return kind, keys, [f"{i}: {lvl}" for i, lvl in enumerate(levels)]
    raise ValueError(f"unknown question type {q['type']!r} (choice, yes_no or score)")


def build_allow(roles):
    L = roles.shape[1]
    rq, rk = roles[:, :, None], roles[:, None, :]
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool))[None]
    allow = causal & (rk != PAD) & ((rk < 0) | (rq == ANSWER_ROLE) | (rq == rk))
    allow |= (rq == rk) & ((rq == CONTEXT) | (rq >= 0))
    allow |= torch.eye(L, dtype=torch.bool)[None]
    return allow


def collate(encs, pad_id=0):
    B, L = len(encs), max(len(e.ids) for e in encs)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    pos = torch.zeros(B, L, dtype=torch.long)
    seg = torch.zeros(B, L, dtype=torch.long)
    role = torch.full((B, L), PAD, dtype=torch.long)
    for b, e in enumerate(encs):
        n = len(e.ids)
        ids[b, :n] = torch.tensor(e.ids)
        pos[b, :n] = torch.tensor(e.pos)
        seg[b, :n] = torch.tensor(e.seg)
        role[b, :n] = torch.tensor(e.role)
    valid = role != PAD
    return dict(ids=ids, pos=pos, seg=seg, role=role, valid=valid, allow=build_allow(role),
                answer_idx=[e.answer_idx for e in encs], close_idx=[e.close_idx for e in encs])
