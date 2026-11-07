import numpy as np
import torch

from prismlet.io import STFile, write_safetensors
from prismlet.prune import build_layer, slice_layer


def raw_bytes(t):
    return t.contiguous().view(torch.uint8).numpy()


def make_source(path):
    g = torch.Generator().manual_seed(5)
    tensors = [
        ("model.layers.0.input_layernorm.weight", "BF16", (4,),
         raw_bytes(torch.randn(4, generator=g).to(torch.bfloat16))),
        ("model.layers.0.mlp.experts.gate_up_proj", "BF16", (4, 2, 2),
         raw_bytes(torch.randn(4, 2, 2, generator=g).to(torch.bfloat16))),
        ("model.layers.0.mlp.experts.down_proj", "BF16", (4, 2, 2),
         raw_bytes(torch.randn(4, 2, 2, generator=g).to(torch.bfloat16))),
        ("model.layers.0.mlp.experts.gate_up_proj_scale", "F32", (4, 2),
         raw_bytes(torch.rand(4, 2, generator=g).float())),
        ("model.layers.0.mlp.experts.down_proj_scale", "F32", (4, 2),
         raw_bytes(torch.rand(4, 2, generator=g).float())),
        ("model.layers.0.mlp.gate.weight", "F32", (4, 2),
         raw_bytes(torch.randn(4, 2, generator=g).float())),
        ("model.layers.0.mlp.gate.e_score_correction_bias", "F32", (4,),
         raw_bytes(torch.randn(4, generator=g).float())),
    ]
    write_safetensors(str(path) + ".tmp", tensors)
    return tensors


def test_write_read_roundtrip(tmp_path):
    tensors = make_source(tmp_path / "src.safetensors")
    f = STFile(tmp_path / "src.safetensors")
    assert len(f.t) == len(tensors)
    buf = torch.empty(f.nbytes, dtype=torch.uint8)
    f.read_all(buf, nthreads=2)
    v = f.views(buf)
    gu = v["model.layers.0.mlp.experts.gate_up_proj"]
    assert tuple(gu.shape) == (4, 2, 2)


def test_slice_layer_keeps_rows(tmp_path):
    tensors = make_source(tmp_path / "src.safetensors")
    out = tmp_path / "sliced.safetensors"
    slice_layer(tmp_path / "src.safetensors", out, [1, 3])
    f = STFile(out)
    buf = torch.empty(f.nbytes, dtype=torch.uint8)
    f.read_all(buf, 2)
    v = f.views(buf)
    assert tuple(v["model.layers.0.mlp.experts.gate_up_proj"].shape) == (2, 2, 2)
    assert tuple(v["model.layers.0.mlp.gate.weight"].shape) == (2, 2)
    assert tuple(v["model.layers.0.input_layernorm.weight"].shape) == (4,)


def test_build_layer_multiple_counts(tmp_path):
    make_source(tmp_path / "src.safetensors")
    outs = {2: tmp_path / "n2.safetensors", 3: tmp_path / "n3.safetensors"}
    build_layer(tmp_path / "src.safetensors", outs, {2: [0, 2], 3: [0, 1, 3]})
    for n, p in outs.items():
        f = STFile(p)
        buf = torch.empty(f.nbytes, dtype=torch.uint8)
        f.read_all(buf, 2)
        v = f.views(buf)
        assert tuple(v["model.layers.0.mlp.experts.gate_up_proj"].shape) == (n, 2, 2)
        assert tuple(v["model.layers.0.mlp.gate.e_score_correction_bias"].shape) == (n,)
