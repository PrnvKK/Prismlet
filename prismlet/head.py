"""Decision head (fp32, CPU)."""

import torch
import torch.nn.functional as F

from .config import TYPES
from .io import STFile


class DecisionHead:
    def __init__(self, path):
        f = STFile(path)
        buf = torch.empty(f.nbytes, dtype=torch.uint8)
        f.read_all(buf, 2)
        self.p = {k: v.clone().float() for k, v in f.views(buf).items()}

    def __call__(self, h_ans, h_opts, qtype):
        p = self.p

        def ln(x, n):
            return F.layer_norm(x, (4096,), p[n + ".weight"], p[n + ".bias"], 1e-5)

        def lin(x, n):
            return F.linear(x, p[n + ".weight"], p[n + ".bias"])

        x = ln(torch.cat([p["head.task.weight"][TYPES[qtype]][None], h_opts, h_ans[None]]), "head.norm_in")
        for b in range(2):
            pre = f"head.blocks.{b}."
            y = ln(x, pre + "n1")
            att = torch.softmax(lin(y, pre + "q") @ lin(y, pre + "k").T / 512 ** 0.5, -1) @ lin(y, pre + "v")
            x = x + lin(att, pre + "o")
            x = x + lin(F.gelu(lin(ln(x, pre + "n2"), pre + "mlp.0")), pre + "mlp.2")
        x = ln(x, "head.norm_out")
        pd = lin(x[-1], "head.pd")
        return torch.softmax(lin(x[1:-1], "head.po") @ pd / 512 ** 0.5, -1)
