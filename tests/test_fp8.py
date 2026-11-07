import torch

from prismlet.fp8 import (build_lut, count_nan_codes, dequant, linear_fp8, quant_act,
                          round_e4m3)


def test_lut_grid():
    lut = build_lut()
    finite = torch.isfinite(lut)
    assert int(finite.sum()) == 254
    assert float(lut[finite].max()) == 448.0
    assert float(lut[finite & (lut > 0)].min()) == 2 ** -9
    assert (~finite).nonzero().flatten().tolist() == [127, 255]


def test_dequant_matches_native():
    codes = (torch.arange(1 << 16, dtype=torch.int64) % 256).to(torch.uint8)
    keep = (codes != 127) & (codes != 255)
    ref = codes.view(torch.float8_e4m3fn).float()
    for backend in ("lut", "bits", "native"):
        got = dequant(codes, backend, torch.float32)
        assert torch.equal(got[keep], ref[keep])


def test_round_e4m3_matches_torch_cast():
    g = torch.Generator().manual_seed(7)
    x = torch.clamp(torch.randn(200_000, generator=g) * 120, -448, 448)
    assert torch.equal(round_e4m3(x), x.to(torch.float8_e4m3fn).float())
    edges = torch.tensor([0.0, -0.0, 1e-30, 448.0, -448.0, 464.0, -464.0])
    assert torch.equal(round_e4m3(edges), edges.to(torch.float8_e4m3fn).float())


def test_quant_act_matches_reference():
    g = torch.Generator().manual_seed(11)
    x = torch.randn(257, 4096, generator=g)
    x[:, 0] *= 30
    xq, s = quant_act(x)
    sref = (x.abs().amax(-1, keepdim=True) / 448).clamp(min=1e-12)
    qref = round_e4m3(x / sref)
    assert torch.equal(xq, qref)
    assert torch.equal(s, sref)


def test_linear_fp8():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(16, 64, generator=g)
    w = torch.randn(8, 64, generator=g)
    scale = torch.rand(8, generator=g) + 0.5
    out = linear_fp8(x, w, scale, False)
    ref = (x @ w.t()) * scale[None, :]
    assert torch.allclose(out, ref, atol=1e-5)


def test_count_nan_codes():
    u = torch.tensor([0, 127, 255, 1], dtype=torch.uint8)
    assert count_nan_codes(u) == 2
