"""Measure the hardware and disk benchmark used by the engine."""

import argparse
import os
import pathlib
import platform
import time

# The Level Zero allocator keeps a smaller per-process budget unless this is set.
os.environ.setdefault("UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS", "1")

import psutil  # noqa: E402
import torch  # noqa: E402

from _common import pick_device, write_json  # noqa: E402

from prismlet.io import STFile  # noqa: E402


def bench_mm(M, N, K, dtype, device, iters=10, warm=3):
    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)
    c = None
    for _ in range(warm):
        c = a @ b
    if device.type == "xpu":
        torch.xpu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        c = a @ b
    if device.type == "xpu":
        torch.xpu.synchronize()
    dt = (time.time() - t0) / iters
    del a, b, c
    if device.type == "xpu":
        torch.xpu.empty_cache()
    return 2 * M * N * K / dt / 1e12


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    device = torch.device(pick_device(a.device))
    out = {"python": platform.python_version(), "torch_version": torch.__version__,
           "xpu_available": bool(torch.xpu.is_available()),
           "ram_bytes": psutil.virtual_memory().total,
           "ram_gb": psutil.virtual_memory().total / 1e9}
    if device.type == "xpu":
        out["device_name"] = torch.xpu.get_device_name(0)
        props = torch.xpu.get_device_properties(0)
        out["total_memory_mb"] = props.total_memory // (1 << 20)
        out["max_compute_units"] = props.max_compute_units
        try:
            out["driver_version"] = props.driver_version
        except AttributeError:
            pass
        try:
            out["mem_get_info"] = list(torch.xpu.mem_get_info())
        except Exception as e:
            out["mem_get_info"] = repr(e)
    out["float8_e4m3fn"] = str(torch.float8_e4m3fn)
    try:
        codes = torch.arange(256, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
        lut = codes.to(torch.bfloat16).float().cpu()
        ref = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        ok = torch.equal(lut[torch.isfinite(ref)], ref[torch.isfinite(ref)])
        out["fp8_cast_xpu_ok"] = bool(ok)
    except Exception as e:
        out["fp8_cast_xpu_ok"] = False
        out["fp8_cast_error"] = repr(e)

    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device=device)
    y = torch.randn(1024, 1024, dtype=torch.bfloat16, device=device)
    got = (x @ y).float().cpu()
    ref = x.float().cpu() @ y.float().cpu()
    out["bf16_1024_ok_rel"] = float((got - ref).abs().max() / ref.abs().max())
    del x, y, got, ref

    out["bf16_tflops"] = {}
    for M, N, K in [(4096, 4096, 4096), (256, 4096, 2048), (256, 2048, 4096),
                    (1024, 4096, 4096), (64, 4096, 4096), (8192, 4096, 4096)]:
        out["bf16_tflops"][f"{M}x{N}x{K}"] = round(
            bench_mm(M, N, K, torch.bfloat16, device), 3)
    out["fp32_tflops"] = {}
    for M, N, K in [(4096, 4096, 4096), (256, 4096, 2048)]:
        torch.set_float32_matmul_precision("high")
        out["fp32_tflops"][f"{M}x{N}x{K}"] = round(
            bench_mm(M, N, K, torch.float32, device), 3)

    n = 1 << 30
    buf = torch.empty(n, dtype=torch.uint8, device=device)
    if device.type == "xpu":
        torch.xpu.synchronize()
    t0 = time.time()
    clone = buf.clone()
    if device.type == "xpu":
        torch.xpu.synchronize()
    out["memcpy_gbps"] = n * 2 / (time.time() - t0) / 1e9
    del buf, clone

    host = torch.empty(n, dtype=torch.uint8)
    if device.type == "xpu":
        torch.xpu.synchronize()
    t0 = time.time()
    moved = host.to(device)
    if device.type == "xpu":
        torch.xpu.synchronize()
    out["h2d_gbps"] = n / (time.time() - t0) / 1e9
    del host, moved

    out["largest_alloc_gib"] = 0
    for gib in (1, 2, 3, 4, 6):
        try:
            z = torch.empty(gib << 30, dtype=torch.uint8, device=device)
            del z
            out["largest_alloc_gib"] = gib
        except Exception as e:
            out["largest_alloc_error"] = f"{gib} GiB: {e!r}"
            break
    if device.type == "xpu":
        torch.xpu.empty_cache()

    weights = pathlib.Path(a.weights)
    files = [weights / f"model-layer-{i}.safetensors" for i in range(20, 28)]
    files = [p for p in files if p.exists()]
    if files:
        sts = [STFile(p) for p in files]
        size = max(s.nbytes for s in sts)
        buf = torch.empty(size, dtype=torch.uint8)
        best = None
        out["disk"] = []
        for ti, th in enumerate((1, 2, 4, 8)):
            for bi, blk in enumerate((16 << 20, 64 << 20)):
                st = sts[(ti * 2 + bi) % len(sts)]
                t0 = time.time()
                st.read_all(buf, nthreads=th, block=blk)
                gbps = st.nbytes / (time.time() - t0) / 1e9
                out["disk"].append({"threads": th, "block_mb": blk // (1 << 20),
                                    "gbps": gbps})
                if best is None or gbps > best["gbps"]:
                    best = {"threads": th, "block_mb": blk // (1 << 20), "gbps": gbps}
        st = sts[0]
        t0 = time.time()
        st.read_all(buf, nthreads=best["threads"], block=best["block_mb"] << 20)
        out["disk_cached_gbps"] = st.nbytes / (time.time() - t0) / 1e9
        out["disk_best"] = best
        out["disk_gbps"] = best["gbps"]
        del buf

    write_json(pathlib.Path(a.out) / "engine" / "environment.json", out)
    print({k: out[k] for k in ("device_name", "total_memory_mb", "ram_gb",
                               "disk_gbps", "fp8_cast_xpu_ok") if k in out})


if __name__ == "__main__":
    main()
