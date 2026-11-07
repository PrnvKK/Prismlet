"""Independent, deliberately naive fp32 CPU reference for one or more decoder layers."""

import collections

import torch
import torch.nn.functional as F

from prismlet.config import EPS, SOFTMAX_SCALE
from prismlet.io import STFile
from prismlet.rope import yarn_inv_freq


def ref_rmsnorm(x, w, eps=EPS):
    xf = x.float()
    return w.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps))


def ref_rope(z, pos):
    """z [T,H,64] interleaved pairs -> [T,H,64] half-split [new_evens | new_odds]."""
    inv = yarn_inv_freq()                                     # float64 [32]
    theta = pos.double()[:, None] * inv[None, :]               # [T,32]
    c, s = torch.cos(theta).float(), torch.sin(theta).float()
    c, s = c[:, None, :], s[:, None, :]
    e, o = z[..., 0::2], z[..., 1::2]
    return torch.cat([e * c - o * s, e * s + o * c], -1)


def rule_mask(roles):
    """Double-loop implementation of the attention role rules."""
    L = len(roles)
    allow = torch.zeros(L, L, dtype=torch.bool)
    for i in range(L):
        for j in range(L):
            if i == j:
                allow[i, j] = True
                continue
            if roles[j] == -3:
                continue
            if roles[i] == -1:
                allow[i, j] = roles[j] == -1
            elif roles[i] >= 0:
                allow[i, j] = roles[j] == -1 or roles[j] == roles[i]
            elif roles[i] == -2:
                allow[i, j] = j <= i
    return allow


class RefWeights:
    def __init__(self, path, li):
        f = STFile(path)
        self.buf = torch.empty(f.nbytes, dtype=torch.uint8)
        f.read_all(self.buf, nthreads=4)
        self.v = f.views(self.buf)
        self.li = li
        self.cache = {}
        self.expert_cache = collections.OrderedDict()

    def raw(self, name):
        return self.v[f"model.layers.{self.li}.{name}"]

    def norm(self, name):
        return self.raw(name).float()

    def mat(self, name):
        if name not in self.cache:
            w8 = self.raw(name + ".weight")
            sc = self.raw(name + ".weight_scale").float()
            self.cache[name] = w8.view(torch.float8_e4m3fn).float() * sc[:, None]
        return self.cache[name]

    def expert(self, e):
        if e in self.expert_cache:
            self.expert_cache.move_to_end(e)
            return self.expert_cache[e]
        gu = self.raw("mlp.experts.gate_up_proj")[e]
        gus = self.raw("mlp.experts.gate_up_proj_scale")[e].float()
        dn = self.raw("mlp.experts.down_proj")[e]
        ds = self.raw("mlp.experts.down_proj_scale")[e].float()
        val = (gu.view(torch.float8_e4m3fn).float() * gus[:, None],
               dn.view(torch.float8_e4m3fn).float() * ds[:, None])
        self.expert_cache[e] = val
        if len(self.expert_cache) > 40:
            self.expert_cache.popitem(last=False)
        return val


def _put(trace, name, t):
    if trace is not None:
        trace[name] = t.detach().float().cpu()


def ref_layer(h, pos, allow, rw, trace=None):
    x = ref_rmsnorm(h, rw.norm("input_layernorm.weight"))
    _put(trace, "ln1", x)
    T = x.shape[0]
    q = x @ rw.mat("self_attn.q_proj").T
    _put(trace, "q", q)
    t = x @ rw.mat("self_attn.kv_a_proj_with_mqa").T
    ckv, k_pe = t[:, :512], t[:, 512:]
    _put(trace, "ckv", ckv)
    _put(trace, "k_pe_raw", k_pe)
    kvn = ref_rmsnorm(ckv, rw.norm("self_attn.kv_a_layernorm.weight"))
    _put(trace, "kv_norm", kvn)
    kv = kvn @ rw.mat("self_attn.kv_b_proj").T
    _put(trace, "kv", kv)
    q = q.view(T, 64, 192)
    kv = kv.view(T, 64, 256)
    q_nope, q_pe = q[..., :128], q[..., 128:]
    k_nope, vv = kv[..., :128], kv[..., 128:]
    q_pe = ref_rope(q_pe, pos)
    k_pe = ref_rope(k_pe[:, None, :], pos)[:, 0, :]
    _put(trace, "q_pe_rot", q_pe)
    _put(trace, "k_pe_rot", k_pe)
    Q = torch.cat([q_nope, q_pe], -1).transpose(0, 1)          # [H,T,192]
    K = torch.cat([k_nope, k_pe[:, None, :].expand(T, 64, 64)], -1).transpose(0, 1)
    V = vv.transpose(0, 1)
    S = torch.einsum("hld,hkd->hlk", Q, K).float() * SOFTMAX_SCALE
    S = S.masked_fill(~allow[None], float("-inf"))
    P = torch.softmax(S, -1)
    _put(trace, "attn_probs_h0", P[0])
    _put(trace, "attn_probs_h63", P[63])
    ctx = torch.einsum("hlk,hkd->hld", P, V).transpose(0, 1).reshape(T, 8192)
    _put(trace, "attn_ctx", ctx)
    attn_out = ctx @ rw.mat("self_attn.o_proj").T
    _put(trace, "attn_out", attn_out)
    h = h + attn_out
    _put(trace, "h_mid", h)
    x2 = ref_rmsnorm(h, rw.norm("post_attention_layernorm.weight"))
    _put(trace, "ln2", x2)
    if f"model.layers.{rw.li}.mlp.experts.gate_up_proj" in rw.v:
        gw = rw.raw("mlp.gate.weight").float()
        gb = rw.raw("mlp.gate.e_score_correction_bias").float()
        scores = (x2.float() @ gw.T).sigmoid()
        ids = torch.topk(scores + gb[None], 8, dim=-1).indices
        w = scores.gather(1, ids)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * 2.5
        _put(trace, "router_scores", scores)
        sids, sidx = torch.sort(ids, dim=-1)
        _put(trace, "topk_ids", sids)
        _put(trace, "topk_w", w.gather(1, sidx))
        routed = torch.zeros_like(x2)
        for e in range(128):
            sel = (ids == e)
            if not bool(sel.any()):
                continue
            tok_idx, slot = sel.nonzero(as_tuple=True)
            Wgu, Wd = rw.expert(e)
            xe = x2[tok_idx]
            gu = xe @ Wgu.T
            o = (F.silu(gu[:, :2048]) * gu[:, 2048:]) @ Wd.T
            routed.index_add_(0, tok_idx, o * w[tok_idx, slot][:, None])
            rw.expert_cache.clear()
        _put(trace, "routed_out", routed)
        sh = rw.mat("mlp.shared_experts.down_proj") @ (
            F.silu(x2 @ rw.mat("mlp.shared_experts.gate_proj").T)
            * (x2 @ rw.mat("mlp.shared_experts.up_proj").T)).T
        shared = sh.T
        _put(trace, "shared_out", shared)
        m = routed + shared
    else:
        g = x2 @ rw.mat("mlp.gate_proj").T
        u = x2 @ rw.mat("mlp.up_proj").T
        m = (F.silu(g) * u) @ rw.mat("mlp.down_proj").T
        _put(trace, "mlp_out", m)
    h = h + m
    _put(trace, "h_out", h)
    return h


_INPUTS = {}


def load_inputs(enc, weights_dir):
    """fp32 x0 from the bf16 embedding rows plus the fp32 segment embedding."""
    key = str(weights_dir)
    if key not in _INPUTS:
        f = STFile(weights_dir / "model-embed.safetensors")
        buf = torch.empty(f.nbytes, dtype=torch.uint8)
        f.read_all(buf, nthreads=4)
        v = f.views(buf)
        hf = STFile(weights_dir / "decision_head.safetensors")
        hb = torch.empty(hf.nbytes, dtype=torch.uint8)
        hf.read_all(hb, nthreads=4)
        hv = hf.views(hb)
        _INPUTS[key] = dict(emb=v["model.embed_tokens.weight"], norm=v["model.norm.weight"],
                            headv=hv, seg=hv["segment_embedding.weight"])
    d = _INPUTS[key]
    x = d["emb"][torch.tensor(enc.ids)].float()
    seg = d["seg"][torch.tensor(enc.seg)]
    return x + seg, d["norm"].float(), d["headv"]


def ref_layers(enc, upto, weights_dir, trace=None):
    x0, norm_w, headv = load_inputs(enc, weights_dir)
    allow = rule_mask(enc.role)
    pos = torch.tensor(enc.pos, dtype=torch.long)
    h = x0
    for li in range(upto + 1):
        rw = RefWeights(weights_dir / f"model-layer-{li:02d}.safetensors", li)
        h = ref_layer(h, pos, allow, rw, trace=trace if li == upto else None)
    return h, headv
