"""YaRN rotary tables, following modeling_sarvam_moe.py."""

import math

import torch


def yarn_tables(max_pos, dtype=torch.bfloat16, dim=64, base=10000.0, factor=40.0, orig=4096,
                beta_fast=32, beta_slow=1, mscale=1.0, mscale_all=1.0):
    ar = torch.arange(0, dim, 2, dtype=torch.float32)
    f_extra = 1.0 / (base ** (ar / dim))
    f_inter = 1.0 / (factor * base ** (ar / dim))

    def cd(n):
        return dim * math.log(orig / (n * 2 * math.pi)) / (2 * math.log(base))

    low, high = max(math.floor(cd(beta_fast)), 0), min(math.ceil(cd(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    mask = 1.0 - ramp
    inv = f_inter * (1 - mask) + f_extra * mask

    def ms(s, m):
        return 1.0 if s <= 1 else 0.1 * m * math.log(s) + 1.0

    k = ms(factor, mscale) / ms(factor, mscale_all)
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv)
    emb = torch.cat([freqs, freqs], -1)
    return (emb.cos() * k).to(dtype), (emb.sin() * k).to(dtype)


def yarn_inv_freq(dim=64, base=10000.0, factor=40.0, orig=4096, beta_fast=32, beta_slow=1):
    ar = torch.arange(0, dim, 2, dtype=torch.float64)
    f_extra = 1.0 / (base ** (ar / dim))
    f_inter = 1.0 / (factor * base ** (ar / dim))

    def cd(n):
        return dim * math.log(orig / (n * 2 * math.pi)) / (2 * math.log(base))

    low, high = max(math.floor(cd(beta_fast)), 0), min(math.ceil(cd(beta_slow)), dim - 1)
    ramp = ((torch.arange(dim // 2, dtype=torch.float64) - low) / (high - low)).clamp(0, 1)
    mask = 1.0 - ramp
    return f_inter * (1 - mask) + f_extra * mask
