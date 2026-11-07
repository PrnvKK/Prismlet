"""Writers for pruned layer files."""

import shutil

import torch

from .io import STFile, write_safetensors

DT_ORDER = {"F32": 0, "BF16": 1, "F8_E4M3": 2}


def slice_layer(src_path, out_path, keep_ids):
    f = STFile(src_path)
    buf = torch.empty(f.nbytes, dtype=torch.uint8)
    f.read_all(buf, nthreads=8)
    v = f.views(buf)
    keep = torch.tensor(keep_ids, dtype=torch.long)
    tensors = []
    names = sorted(f.t, key=lambda k: (DT_ORDER[f.t[k][0]], k))
    for k in names:
        dt, shape, a, b = f.t[k]
        t = v[k]
        if k.endswith("mlp.experts.gate_up_proj") or k.endswith("mlp.experts.down_proj") \
                or k.endswith("mlp.experts.gate_up_proj_scale") or k.endswith("mlp.experts.down_proj_scale"):
            t = t[keep]
        elif k.endswith("mlp.gate.weight") or k.endswith("mlp.gate.e_score_correction_bias"):
            t = t[keep]
        arr = t.contiguous().view(torch.uint8).numpy()
        tensors.append((k, dt, tuple(t.shape), arr))
    tmp = str(out_path) + ".tmp"
    write_safetensors(tmp, tensors)
    return [t[3].shape for t in tensors]


def build_layer(src_path, out_paths, keep_map):
    """Write one pruned file per N from a single read of ``src_path``.

    ``keep_map`` maps N to the kept expert ids for this layer.
    """
    f = STFile(src_path)
    buf = torch.empty(f.nbytes, dtype=torch.uint8)
    f.read_all(buf, nthreads=8)
    v = f.views(buf)
    names = sorted(f.t, key=lambda k: (DT_ORDER[f.t[k][0]], k))
    for n, ids in keep_map.items():
        keep = torch.tensor(ids, dtype=torch.long)
        tensors = []
        for k in names:
            dt = f.t[k][0]
            t = v[k]
            if k.endswith("mlp.experts.gate_up_proj") or k.endswith("mlp.experts.down_proj") \
                    or k.endswith("mlp.experts.gate_up_proj_scale") or k.endswith("mlp.experts.down_proj_scale"):
                t = t[keep]
            elif k.endswith("mlp.gate.weight") or k.endswith("mlp.gate.e_score_correction_bias"):
                t = t[keep]
            arr = t.contiguous().view(torch.uint8).numpy()
            tensors.append((k, dt, tuple(t.shape), arr))
        tmp = str(out_paths[n]) + ".tmp"
        write_safetensors(tmp, tensors)
        del tensors
    del v, buf


def copy_file(src, dst):
    shutil.copyfile(src, dst)
