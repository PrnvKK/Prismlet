"""e4m3fn dequantisation and activation quantisation."""

import torch

_LUTS = {}
_NAN_CODES = (127, 255)


def build_lut():
    return torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)


def _lut(device, dtype):
    k = (str(device), dtype)
    if k not in _LUTS:
        lut = build_lut()
        _LUTS[k] = lut.to(dtype).to(device)
    return _LUTS[k]


def dequant(u8, backend="lut", dtype=torch.bfloat16):
    """Map uint8 e4m3 codes to ``dtype``; exact for finite codes."""
    if backend == "native":
        return u8.view(torch.float8_e4m3fn).to(dtype)
    if backend == "bits":
        x = u8.to(torch.int32)
        bits = ((x & 0x7F) << 4) | ((x & 0x80) << 8)
        bits = torch.where(bits >= 32768, bits - 65536, bits).to(torch.int16)
        return (bits.view(torch.bfloat16).to(torch.float32) * (2.0 ** 120)).to(dtype)
    lut = _lut(u8.device, dtype)
    flat = u8.reshape(-1)
    out = torch.empty(flat.numel(), dtype=dtype, device=u8.device)
    CH = 1 << 23
    for i in range(0, flat.numel(), CH):
        out[i:i + CH] = lut.index_select(0, flat[i:i + CH].to(torch.int64))
    return out.view(u8.shape)


def round_e4m3(v):
    """Round fp32 to the e4m3fn grid (round-to-nearest-even, saturating at +-448)."""
    a = v.abs().clamp(max=448.0).contiguous()
    e = ((a.view(torch.int32) >> 23) & 0xFF) - 127
    e = e.clamp(min=-6)
    step = torch.exp2((e - 3).to(torch.float32))
    q = (torch.round(a / step) * step).clamp(max=448.0)
    return torch.copysign(q, v)


def quant_act(x):
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True)
    if x.device.type == "cpu":
        s = (amax / 448.0).clamp(min=1e-12)
    else:
        # Intel XPU fp32 division is not correctly rounded (1 ulp). Compute the
        # per-row scale on the CPU so activation quantisation matches the reference.
        s = (amax.cpu() / 448.0).clamp(min=1e-12).to(x.device)
    return round_e4m3(xf / s).to(x.dtype), s


def linear_fp8(x, wq, ws, emulate_act):
    dt = x.dtype
    if emulate_act:
        xq, s = quant_act(x)
        return ((xq @ wq.t()).float() * s * ws[None, :]).to(dt)
    return ((x @ wq.t()).float() * ws[None, :]).to(dt)


def count_nan_codes(u8):
    flat = u8.reshape(-1)
    return int(((flat == 127) | (flat == 255)).sum().item())
