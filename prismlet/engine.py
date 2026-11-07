"""Streaming Prism engine."""

import pathlib
import time

import numpy as np
import torch

from .config import (DEFAULT_WEIGHTS, HID, MAX_POS, NLAYERS, TOPK)
from .encoder import Encoder, collate, to_options, _text
from .head import DecisionHead
from .io import LayerStream, STFile
from .layers import LayerDev, decoder_layer, rmsnorm
from .rope import yarn_tables


def _sync(dev):
    if dev.type == "xpu":
        torch.xpu.synchronize()


class PrismEngine:
    """Runs a Prism model one layer at a time from disk, RAM, or device memory."""

    def __init__(self, weights_dir=None, layers_dir=None, device="xpu", dtype=torch.bfloat16,
                 emulate_act=False, topk=None, skip_layers=(), mode="disk", dequant_backend="lut",
                 max_tokens=8192, nthreads=4, prefetch=True, pin_memory=True, profile=False):
        self.weights_dir = pathlib.Path(weights_dir or DEFAULT_WEIGHTS)
        self.layers_dir = pathlib.Path(layers_dir or self.weights_dir)
        self.device = torch.device(device)
        self.dtype = dtype
        self.emulate_act = emulate_act
        self.topk = TOPK if topk is None else topk
        self.skip_layers = tuple(skip_layers)
        self.mode = mode
        self.backend = dequant_backend
        self.max_tokens = int(max_tokens)
        self.nthreads = nthreads
        self.prefetch = prefetch
        self.pin_memory = pin_memory
        self.nbuf = 1
        self.profile = profile
        self.cache_experts = "auto"
        self.enc = Encoder(self.weights_dir / "tokenizer.json")
        self.head = DecisionHead(self.weights_dir / "decision_head.safetensors")
        self.files = {}
        for li in range(NLAYERS):
            p = self.layers_dir / f"model-layer-{li:02d}.safetensors"
            if p.exists():
                self.files[li] = STFile(p)
        ef = STFile(self.weights_dir / "model-embed.safetensors")
        dt, shp, a, b = ef.t["model.embed_tokens.weight"]
        self.emb = np.memmap(str(ef.path), dtype=np.int16, mode="r", offset=ef.data_start + a,
                             shape=tuple(shp))
        ndt, nshp, na, nb = ef.t["model.norm.weight"]
        nt = torch.empty(nb - na, dtype=torch.uint8)
        with open(ef.path, "rb") as f:
            f.seek(ef.data_start + na)
            f.readinto(memoryview(nt.numpy()))
        self.norm_w = nt.view(torch.bfloat16).reshape(tuple(nshp)).to(dtype).to(self.device)
        hf = STFile(self.weights_dir / "decision_head.safetensors")
        hbuf = torch.empty(hf.nbytes, dtype=torch.uint8)
        hf.read_all(hbuf, 2)
        self.seg_tbl = hf.views(hbuf)["segment_embedding.weight"].float().to(dtype).to(self.device)
        self.cos, self.sin = yarn_tables(MAX_POS, dtype=dtype)
        self.cos, self.sin = self.cos.to(self.device), self.sin.to(self.device)
        self._ram = {}
        self._dev = {}
        self.last_timings = []
        self.last_timings_total = {}

    # ---------------- embedding ----------------
    def _embed(self, b):
        ids = b["ids"]
        B, L = ids.shape
        flat = ids.reshape(-1).numpy()
        rows = np.ascontiguousarray(self.emb[flat])
        x = torch.from_numpy(rows).view(torch.bfloat16)
        if self.pin_memory:
            x = x.pin_memory()
        x = x.to(self.device)
        seg = self.seg_tbl[b["seg"].reshape(-1).to(self.device)]
        return (x + seg).view(B, L, HID)

    # ---------------- layer sources ----------------
    def _iter_layers(self, order):
        if self.mode == "disk":
            yield from LayerStream([self.files[i] for i in range(max(self.files) + 1)], order,
                                   nbuf=self.nbuf, nthreads=self.nthreads,
                                   pin=self.pin_memory)
        elif self.mode == "ram":
            for li in order:
                if li not in self._ram:
                    f = self.files[li]
                    buf = torch.empty(f.nbytes, dtype=torch.uint8)
                    f.read_all(buf, self.nthreads)
                    self._ram[li] = buf
                yield li, self.files[li].views(self._ram[li])
        elif self.mode == "device":
            for li in order:
                if li not in self._dev:
                    f = self.files[li]
                    buf = torch.empty(f.nbytes, dtype=torch.uint8)
                    f.read_all(buf, self.nthreads)
                    d = buf.to(self.device)
                    self._dev[li] = d
                    del buf
                yield li, self.files[li].views(self._dev[li])
        else:
            raise ValueError(self.mode)

    # ---------------- main ----------------
    def _prepare(self, items):
        encs = [self.enc(s, ins, opts) for (s, qt, opts, ins) in items]
        order = sorted(range(len(items)), key=lambda i: len(encs[i].ids))
        mbs, cur = [], []
        for i in order:
            if cur and (len(cur) + 1) * len(encs[i].ids) > self.max_tokens:
                mbs.append(cur)
                cur = []
            cur.append(i)
        if cur:
            mbs.append(cur)
        batches = []
        for mb in mbs:
            b = collate([encs[i] for i in mb])
            b["pos"] = b["pos"].to(self.device)
            b["seg"] = b["seg"].to(self.device)
            b["role"] = b["role"].to(self.device)
            b["valid"] = b["valid"].to(self.device)
            b["allow"] = b["allow"].to(self.device)
            batches.append((mb, b))
        return encs, batches

    def forward_items(self, items, stats=None, trace=None, stop_after=None):
        self.last_timings = []
        t_start = time.time()
        encs, batches = self._prepare(items)
        hs = [self._embed(b) for _, b in batches]
        order = [li for li in range(NLAYERS) if li not in self.skip_layers and li in self.files
                 and (stop_after is None or li <= stop_after)]
        opts = dict(emulate_act=self.emulate_act, topk=self.topk)
        for li, views in self._iter_layers(order):
            tw = time.time()
            use_ec = self.cache_experts if isinstance(self.cache_experts, bool) else (len(batches) > 1)
            Wl = LayerDev(views, li, self.device, self.dtype, self.backend, cache_experts=use_ec)
            tl = time.time()
            for j, (mb, b) in enumerate(batches):
                tr = trace if (trace is not None and len(batches) == 1 and len(mb) == 1) else None
                hs[j] = decoder_layer(hs[j], b["pos"], b["allow"], b["valid"], Wl, self.cos, self.sin,
                                      opts, stats=stats, trace=tr)
            if self.device.type == "xpu" and self.profile:
                _sync(self.device)
            tc = time.time()
            self.last_timings.append(dict(
                layer=li,
                wait=tw - (self.last_timings[-1]["end"] if self.last_timings else t_start),
                load=tl - tw, compute=tc - tl, end=tc))
            del Wl
        t_layers = time.time()
        # Final norm and decision head run in fp32 on the CPU.
        probs = [None] * len(items)
        stop_h = [None] * len(items) if stop_after is not None else None
        for j, (mb, b) in enumerate(batches):
            h = rmsnorm(hs[j], self.norm_w)
            hf = h.float().cpu()
            hs[j] = None
            for bi, oi in enumerate(mb):
                if stop_after is not None:
                    stop_h[oi] = hf[bi][:int(b["valid"][bi].sum())]
                    continue
                h_ans = hf[bi, b["answer_idx"][bi]]
                h_opts = hf[bi, torch.tensor(b["close_idx"][bi])]
                probs[oi] = self.head(h_ans, h_opts, items[oi][1])
        self.last_timings_total = dict(total=time.time() - t_start, layers=t_layers - t_start,
                                       head=time.time() - t_layers)
        if self.device.type == "xpu":
            torch.xpu.empty_cache()
        if stop_after is not None:
            return stop_h
        return probs

    def decide(self, state, questions):
        s = _text(state) if state not in (None, "", {}, []) else ""
        items, meta = [], []
        for name, q in questions.items():
            kind, keys, opts = to_options(q)
            ins = _text(q.get("instructions", ""))
            items.append((s, kind, opts, ins))
            meta.append((name, kind, keys))
        P = self.forward_items(items)
        out = {}
        for (name, kind, keys), p in zip(meta, P):
            pl = p.tolist()
            probs = dict(zip(keys, pl))
            if kind == "yes_no":
                out[name] = {"type": "yes_no", "yes": pl[1]}
            elif kind == "score":
                out[name] = {"type": "score", "score": sum(i * v for i, v in enumerate(pl)),
                             "level": max(keys, key=probs.get), "probabilities": probs}
            else:
                out[name] = {"type": "choice", "choice": max(keys, key=probs.get),
                             "probabilities": probs}
        return out

    def decide_many(self, requests):
        flat, groups = [], []
        for state, questions in requests:
            s = _text(state) if state not in (None, "", {}, []) else ""
            idx = []
            for name, q in questions.items():
                kind, keys, opts = to_options(q)
                ins = _text(q.get("instructions", ""))
                idx.append((len(flat), name, kind, keys))
                flat.append((s, kind, opts, ins))
            groups.append(idx)
        P = self.forward_items(flat)
        out = []
        for idx in groups:
            d = {}
            for i, name, kind, keys in idx:
                pl = P[i].tolist()
                probs = dict(zip(keys, pl))
                if kind == "yes_no":
                    d[name] = {"type": "yes_no", "yes": pl[1]}
                elif kind == "score":
                    d[name] = {"type": "score", "score": sum(j * v for j, v in enumerate(pl)),
                               "level": max(keys, key=probs.get), "probabilities": probs}
                else:
                    d[name] = {"type": "choice", "choice": max(keys, key=probs.get),
                               "probabilities": probs}
            out.append(d)
        return out
