"""Device implementation of one Prism decoder layer (MLA attention plus dense or MoE MLP)."""

import torch
import torch.nn.functional as F

from .config import SOFTMAX_SCALE
from .fp8 import dequant, linear_fp8

SCALE = SOFTMAX_SCALE


def rmsnorm(x, w, eps=1e-6):
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def _trace_put(trace, name, t):
    if trace is not None:
        trace[name] = t.detach().float().cpu()


class WMat:
    """Dequantised weight (exact e4m3 values) plus a per-output-row scale."""

    __slots__ = ("wq", "ws")

    def __init__(self, wq, ws):
        self.wq, self.ws = wq, ws

    def __call__(self, x, emu):
        return linear_fp8(x, self.wq, self.ws, emu)


class LayerDev:
    """One layer's device tensors. Routed experts stay in FP8 and are dequantised per expert."""

    def __init__(self, views, li, device, dtype, backend="lut", chunk=16, cache_experts=False):
        self.device, self.dtype, self.backend = device, dtype, backend
        self.li = li
        p = f"model.layers.{li}."
        self.is_moe = f"{p}mlp.experts.gate_up_proj" in views

        def mv(name):
            return views[p + name].to(device, non_blocking=False)

        def mat(name):
            wq = dequant(mv(name + ".weight"), backend, dtype)
            ws = mv(name + ".weight_scale").float()
            return WMat(wq, ws)

        self.ln1 = mv("input_layernorm.weight").to(dtype)
        self.ln2 = mv("post_attention_layernorm.weight").to(dtype)
        self.kv_ln = mv("self_attn.kv_a_layernorm.weight").to(dtype)
        self.q_proj = mat("self_attn.q_proj")
        self.kv_a = mat("self_attn.kv_a_proj_with_mqa")
        self.kv_b = mat("self_attn.kv_b_proj")
        self.o_proj = mat("self_attn.o_proj")
        if self.is_moe:
            self.shared = [mat("mlp.shared_experts.gate_proj"), mat("mlp.shared_experts.up_proj"),
                           mat("mlp.shared_experts.down_proj")]
            self.gate_w = mv("mlp.gate.weight").float()
            self.gate_b = mv("mlp.gate.e_score_correction_bias").float()
            E = self.gate_w.shape[0]
            self.E = E
            gu = mv("mlp.experts.gate_up_proj")
            dn = mv("mlp.experts.down_proj")
            self.gus = mv("mlp.experts.gate_up_proj_scale").float()
            self.ds = mv("mlp.experts.down_proj_scale").float()
            self.chunk = chunk
            self.gu_chunks = [gu[c:c + chunk].contiguous() for c in range(0, E, chunk)]
            self.d_chunks = [dn[c:c + chunk].contiguous() for c in range(0, E, chunk)]
            del gu, dn
            self._ecache = None
            if cache_experts:
                if self.device.type == "xpu":
                    torch.xpu.empty_cache()
                cache = []
                for e in range(E):
                    c, j = divmod(e, chunk)
                    cache.append((dequant(self.gu_chunks[c][j], backend, dtype), self.gus[e],
                                  dequant(self.d_chunks[c][j], backend, dtype), self.ds[e]))
                self._ecache = cache
                self.gu_chunks = None
                self.d_chunks = None
        else:
            self.gate_p = mat("mlp.gate_proj")
            self.up_p = mat("mlp.up_proj")
            self.down_p = mat("mlp.down_proj")

    def expert(self, e):
        if self._ecache is not None:
            return self._ecache[e]
        c, j = divmod(e, self.chunk)
        Wgu = dequant(self.gu_chunks[c][j], self.backend, self.dtype)
        Wd = dequant(self.d_chunks[c][j], self.backend, self.dtype)
        return Wgu, self.gus[e], Wd, self.ds[e]


def _rope(z, c, s):
    b, h, l, d = z.shape
    z = z.reshape(b, h, l, d // 2, 2).transpose(4, 3).contiguous().reshape(b, h, l, d)
    return z * c + torch.cat([-z[..., d // 2:], z[..., : d // 2]], -1) * s


def mla_attention(x, pos, allow, A, cos, sin, emu, chunk_bytes=1 << 30, trace=None):
    B, L, _ = x.shape
    dt = x.dtype
    x2 = x.reshape(B * L, 4096)
    q = A.q_proj(x2, emu).view(B, L, 64, 192).transpose(1, 2)
    t = A.kv_a(x2, emu).view(B, L, 576)
    ckv, k_pe = t[..., :512], t[..., 512:].reshape(B, L, 1, 64).transpose(1, 2)
    kvn = rmsnorm(ckv, A.kv_ln)
    kv = A.kv_b(kvn.reshape(B * L, 512), emu).view(B, L, 64, 256).transpose(1, 2)
    q_nope, q_pe = q[..., :128], q[..., 128:]
    k_nope, v = kv[..., :128], kv[..., 128:]
    c = cos[pos].unsqueeze(1).to(dt)
    s = sin[pos].unsqueeze(1).to(dt)
    q_pe, k_pe = _rope(q_pe, c, s), _rope(k_pe, c, s)
    if trace is not None:
        _trace_put(trace, "q", q.transpose(1, 2).contiguous().reshape(B, L, -1))
        _trace_put(trace, "ckv", ckv)
        _trace_put(trace, "k_pe_raw", t[..., 512:])
        _trace_put(trace, "kv_norm", kvn)
        _trace_put(trace, "kv", kv.transpose(1, 2).contiguous().reshape(B, L, -1))
        _trace_put(trace, "q_pe_rot", q_pe.transpose(1, 2).contiguous().reshape(B, L, 64, 64))
        _trace_put(trace, "k_pe_rot", k_pe.transpose(1, 2).contiguous().reshape(B, L, 64))
    Q = torch.cat([q_nope, q_pe], -1)
    K = torch.cat([k_nope, k_pe.expand(-1, 64, -1, -1)], -1)
    ctx = torch.empty(B, L, 64 * 128, dtype=dt, device=x.device)
    step = max(1, chunk_bytes // max(1, 64 * L * L * 4))
    for i in range(0, B, step):
        S = (Q[i:i + step] @ K[i:i + step].transpose(-1, -2)).float() * SCALE
        S = S.masked_fill(~allow[i:i + step, None], float("-inf"))
        P = torch.softmax(S, -1)
        if trace is not None:
            _trace_put(trace, "attn_probs_h0", P[:, 0])
            _trace_put(trace, "attn_probs_h63", P[:, 63])
        ctx[i:i + step] = (P.to(dt) @ v[i:i + step]).transpose(1, 2).contiguous().reshape(-1, L, 64 * 128)
    _trace_put(trace, "attn_ctx", ctx.reshape(B, L, -1))
    out = A.o_proj(ctx.reshape(B * L, 8192), emu).view(B, L, 4096)
    _trace_put(trace, "attn_out", out)
    return out


def swiglu(g, u, dt):
    return (F.silu(g.float()).to(dt).float() * u.float()).to(dt)


def mlp_swiglu(x2, Ws, emu):
    g = Ws[0](x2, emu)
    u = Ws[1](x2, emu)
    return Ws[2](swiglu(g, u, x2.dtype), emu)


def moe_forward(x2, M, k, emu, stats=None, layer=None, trace=None):
    dt = x2.dtype
    T = x2.shape[0]
    E = M.gate_w.shape[0]
    k = min(k, E)
    scores = (x2.float() @ M.gate_w.t()).sigmoid()
    sel = scores + M.gate_b[None]
    ids = torch.topk(sel, k, dim=-1).indices
    w = scores.gather(1, ids)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * 2.5
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    tok = order // k
    counts = torch.bincount(flat, minlength=E).tolist()
    xs = x2[tok]
    out = torch.empty_like(xs)
    norms = torch.empty(T * k, dtype=torch.float32, device=x2.device) if stats is not None else None
    off = 0
    for e, n in enumerate(counts):
        if n == 0:
            continue
        Wgu, sgu, Wd, sd = M.expert(e)
        gu = linear_fp8(xs[off:off + n], Wgu, sgu, emu)          # [n,4096] = [gate|up]
        o = linear_fp8(swiglu(gu[:, :2048], gu[:, 2048:], dt), Wd, sd, emu)
        out[off:off + n] = o
        if norms is not None:
            norms[off:off + n] = o.float().norm(dim=-1)
        off += n
        del Wgu, Wd
    y = torch.empty_like(out)
    y[order] = out
    yv = y.view(T, k, 4096)
    routed = torch.zeros(T, 4096, dtype=torch.float32, device=x2.device)
    for j in range(k):
        routed += yv[:, j].float() * w[:, j:j + 1]
    routed = routed.to(dt)
    shared = mlp_swiglu(x2, M.shared, emu)
    if stats is not None:
        nt = torch.empty_like(norms)
        nt[order] = norms
        stats.update(layer, ids, w, nt.view(T, k))
    if trace is not None:
        _trace_put(trace, "router_scores", scores)
        sids, sidx = torch.sort(ids, dim=-1)
        _trace_put(trace, "topk_ids", sids)
        _trace_put(trace, "topk_w", w.gather(1, sidx))
        _trace_put(trace, "routed_out", routed)
        _trace_put(trace, "shared_out", shared)
        _trace_put(trace, "mlp_out", routed + shared)
    return routed + shared


def decoder_layer(h, pos, allow, valid, Wl, cos, sin, opts, stats=None, trace=None):
    emu = opts.get("emulate_act", False)
    x = rmsnorm(h, Wl.ln1)
    _trace_put(trace, "ln1", x)
    attn = mla_attention(x, pos, allow, Wl, cos, sin, emu, trace=trace)
    h = h + attn
    _trace_put(trace, "h_mid", h)
    x2 = rmsnorm(h, Wl.ln2)
    _trace_put(trace, "ln2", x2)
    B, L, _ = h.shape
    if Wl.is_moe:
        xv = x2.reshape(-1, 4096)[valid.reshape(-1)]
        m = moe_forward(xv, Wl, opts.get("topk", 8), emu, stats=stats, layer=Wl.li, trace=trace)
        mf = torch.zeros(B * L, 4096, dtype=h.dtype, device=h.device)
        mf[valid.reshape(-1)] = m
        m = mf.view(B, L, 4096)
    else:
        m = mlp_swiglu(x2.reshape(-1, 4096), [Wl.gate_p, Wl.up_p, Wl.down_p], emu).view(B, L, 4096)
        _trace_put(trace, "mlp_out", m)
    h = h + m
    _trace_put(trace, "h_out", h)
    return h
