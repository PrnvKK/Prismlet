"""Check the published weight layout and write the inventory and size table."""

import argparse
import json
import os
import pathlib

from _common import write_json

import prismlet.config as C
from prismlet.config import file_for_layer


def bytes_of(dt, shape):
    n = 1
    for s in shape:
        n *= s
    return n * {"F8_E4M3": 1, "BF16": 2, "F32": 4}[dt]


def expected():
    exp = {}
    for li in range(32):
        p = f"model.layers.{li}."
        d = {
            "input_layernorm.weight": ("BF16", (4096,)),
            "post_attention_layernorm.weight": ("BF16", (4096,)),
            "self_attn.kv_a_layernorm.weight": ("BF16", (512,)),
            "self_attn.q_proj.weight": ("F8_E4M3", (12288, 4096)),
            "self_attn.q_proj.weight_scale": ("F32", (12288,)),
            "self_attn.kv_a_proj_with_mqa.weight": ("F8_E4M3", (576, 4096)),
            "self_attn.kv_a_proj_with_mqa.weight_scale": ("F32", (576,)),
            "self_attn.kv_b_proj.weight": ("F8_E4M3", (16384, 512)),
            "self_attn.kv_b_proj.weight_scale": ("F32", (16384,)),
            "self_attn.o_proj.weight": ("F8_E4M3", (4096, 8192)),
            "self_attn.o_proj.weight_scale": ("F32", (4096,)),
        }
        if li == 0:
            d.update({
                "mlp.gate_proj.weight": ("F8_E4M3", (16384, 4096)),
                "mlp.gate_proj.weight_scale": ("F32", (16384,)),
                "mlp.up_proj.weight": ("F8_E4M3", (16384, 4096)),
                "mlp.up_proj.weight_scale": ("F32", (16384,)),
                "mlp.down_proj.weight": ("F8_E4M3", (4096, 16384)),
                "mlp.down_proj.weight_scale": ("F32", (4096,)),
            })
        else:
            d.update({
                "mlp.experts.gate_up_proj": ("F8_E4M3", (128, 4096, 4096)),
                "mlp.experts.gate_up_proj_scale": ("F32", (128, 4096)),
                "mlp.experts.down_proj": ("F8_E4M3", (128, 4096, 2048)),
                "mlp.experts.down_proj_scale": ("F32", (128, 4096)),
                "mlp.gate.weight": ("F32", (128, 4096)),
                "mlp.gate.e_score_correction_bias": ("F32", (128,)),
                "mlp.shared_experts.gate_proj.weight": ("F8_E4M3", (2048, 4096)),
                "mlp.shared_experts.gate_proj.weight_scale": ("F32", (2048,)),
                "mlp.shared_experts.up_proj.weight": ("F8_E4M3", (2048, 4096)),
                "mlp.shared_experts.up_proj.weight_scale": ("F32", (2048,)),
                "mlp.shared_experts.down_proj.weight": ("F8_E4M3", (4096, 2048)),
                "mlp.shared_experts.down_proj.weight_scale": ("F32", (4096,)),
            })
        for k, v in d.items():
            exp[p + k] = v
    exp["model.embed_tokens.weight"] = ("BF16", (262144, 4096))
    exp["model.norm.weight"] = ("BF16", (4096,))
    return exp


def head_expected():
    d = {}
    for b in range(2):
        p = f"head.blocks.{b}."
        d[p + "q.weight"] = (512, 4096); d[p + "q.bias"] = (512,)
        d[p + "k.weight"] = (512, 4096); d[p + "k.bias"] = (512,)
        d[p + "v.weight"] = (512, 4096); d[p + "v.bias"] = (512,)
        d[p + "o.weight"] = (4096, 512); d[p + "o.bias"] = (4096,)
        d[p + "mlp.0.weight"] = (512, 4096); d[p + "mlp.0.bias"] = (512,)
        d[p + "mlp.2.weight"] = (4096, 512); d[p + "mlp.2.bias"] = (4096,)
        d[p + "n1.weight"] = (4096,); d[p + "n1.bias"] = (4096,)
        d[p + "n2.weight"] = (4096,); d[p + "n2.bias"] = (4096,)
    d["head.norm_in.weight"] = (4096,); d["head.norm_in.bias"] = (4096,)
    d["head.norm_out.weight"] = (4096,); d["head.norm_out.bias"] = (4096,)
    d["head.pd.weight"] = (512, 4096); d["head.pd.bias"] = (512,)
    d["head.po.weight"] = (512, 4096); d["head.po.bias"] = (512,)
    d["head.task.weight"] = (3, 4096)
    d["segment_embedding.weight"] = (4, 4096)
    return d


def category(key):
    if key.startswith("model.embed"):
        return "embed"
    if key.startswith("model.norm"):
        return "final_norm"
    if key.startswith("head.") or key.startswith("segment_"):
        return "head"
    if ".self_attn." in key:
        return "attention"
    if ".mlp.experts." in key:
        return "experts"
    if ".mlp.shared_experts." in key:
        return "shared"
    if ".mlp.gate." in key:
        return "router"
    if ".mlp." in key:
        return "dense_mlp"
    return "norm"


def size_table():
    rows = []
    for n in (128, 64, 32, 16, 8, 4):
        lb = C.layer_bytes(n)
        tb = C.total_backbone(n)
        rows.append(dict(N=n, layer_bytes=lb, moe_layers_bytes=31 * lb, total_bytes=tb,
                         GB=tb / 1e9, GiB=tb / 2 ** 30,
                         with_embed_GB=(tb + C.SIZE_EMBED) / 1e9,
                         reduction=C.total_backbone(128) / tb))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Check the tensor layout and sizes.")
    ap.add_argument("--weights", default="weights", help="directory with the official weight files")
    ap.add_argument("--out", default="results", help="results directory")
    a = ap.parse_args()
    w = pathlib.Path(a.weights)
    idx = json.loads((w / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wm = idx["weight_map"]
    files = sorted(set(wm.values()))
    exp = expected()
    mism = []
    cats = {}
    for f in files:
        from prismlet.io import STFile
        sf = STFile(w / f)
        keys = [k for k, v in wm.items() if v == f]
        if set(keys) != set(sf.t):
            mism.append({"file": f, "missing": sorted(set(keys) - set(sf.t))[:5],
                         "extra": sorted(set(sf.t) - set(keys))[:5]})
        offs = sorted(v[2:] for v in sf.t.values())
        pos = 0
        for lo, hi in offs:
            if lo != pos:
                mism.append({"file": f, "gap_at": lo, "expected": pos})
            pos = hi
        size = os.path.getsize(sf.path)
        if size != sf.data_start + pos:
            mism.append({"file": f, "file_bytes": size, "header_plus_data": sf.data_start + pos})
        for k in keys:
            dt, shape, lo, hi = sf.t[k]
            e = exp.get(k)
            if e is None or e[0] != dt or tuple(e[1]) != tuple(shape):
                mism.append({"file": f, "key": k, "dtype": dt, "shape": shape})
            c = category(k)
            cats.setdefault(c, [0, 0])
            cats[c][0] += 1
            cats[c][1] += hi - lo

    per_layer = {}
    from prismlet.io import STFile
    for li in range(32):
        sf = STFile(w / file_for_layer(li))
        per_layer[li] = sum(hi - lo for _, _, lo, hi in sf.t.values())

    hf = STFile(w / "decision_head.safetensors")
    he = head_expected()
    head_mism = [k for k, (dt, shape, lo, hi) in hf.t.items()
                 if dt != "F32" or tuple(he[k]) != tuple(shape)]
    unusual = [k for k in wm if any(s in k for s in
               ("q_norm", "k_norm", "lm_head", "q_a_proj", "q_b_proj"))]

    out = {
        "keys": len(wm),
        "files": len(files),
        "total_bytes": idx["metadata"].get("total_size"),
        "total_gb": idx["metadata"].get("total_size", 0) / 1e9,
        "layers": 32,
        "per_layer_bytes": {str(k): v for k, v in per_layer.items()},
        "categories": {k: {"tensors": v[0], "bytes": v[1]} for k, v in cats.items()},
        "head_tensors": len(hf.t),
        "head_bytes": hf.nbytes,
        "mismatches": mism,
        "head_mismatches": head_mism,
        "unexpected_tensor_names": unusual,
    }
    write_json(pathlib.Path(a.out) / "engine" / "inventory.json", out)
    write_json(pathlib.Path(a.out) / "engine" / "size_table.json", size_table())
    print(json.dumps({k: out[k] for k in ("keys", "files", "total_bytes", "head_tensors",
                                          "head_bytes", "mismatches", "head_mismatches")}, indent=1))


if __name__ == "__main__":
    main()
