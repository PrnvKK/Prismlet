"""Router and saliency statistics for expert pruning."""

import numpy as np
import torch


class Stats:
    def __init__(self, nl=32, ne=128):
        self.cnt = torch.zeros(nl, ne, dtype=torch.float64)
        self.wsum = torch.zeros(nl, ne, dtype=torch.float64)
        self.sal = torch.zeros(nl, ne, dtype=torch.float64)
        self.ntok = torch.zeros(nl, dtype=torch.float64)

    def update(self, layer, ids, w, norms):
        E = self.cnt.shape[1]
        f = ids.reshape(-1).detach().cpu()
        wc = w.reshape(-1).detach().cpu().double()
        nc = norms.reshape(-1).detach().cpu().double()
        idx = f.long()
        self.cnt[layer] += torch.bincount(idx, minlength=E).double()
        self.wsum[layer] += torch.zeros(E, dtype=torch.float64).index_add_(0, idx, wc)
        self.sal[layer] += torch.zeros(E, dtype=torch.float64).index_add_(0, idx, wc * nc)
        self.ntok[layer] += ids.shape[0]

    def merge(self, other):
        self.cnt += other.cnt
        self.wsum += other.wsum
        self.sal += other.sal
        self.ntok += other.ntok

    def save(self, path):
        np.savez(path, cnt=self.cnt.numpy(), wsum=self.wsum.numpy(), sal_sum=self.sal.numpy(),
                 ntok=self.ntok.numpy())

    @classmethod
    def load(cls, path):
        z = np.load(path)
        s = cls(z["cnt"].shape[0], z["cnt"].shape[1])
        s.cnt = torch.from_numpy(z["cnt"])
        s.wsum = torch.from_numpy(z["wsum"])
        s.sal = torch.from_numpy(z["sal_sum"])
        s.ntok = torch.from_numpy(z["ntok"])
        return s

    def ranking(self, name):
        if name == "reap":
            # Mean routing weight times routed-output norm per activation.
            return self.sal / self.cnt.clamp(min=1)
        if name == "total":
            # Total saliency of an expert divided by the number of tokens in the layer.
            return self.sal / self.ntok[:, None].clamp(min=1)
        if name == "count":
            return self.cnt
        raise ValueError(name)

    def keep(self, name, n):
        r = self.ranking(name)
        out = []
        for l in range(r.shape[0]):
            ids = torch.topk(r[l], min(n, r.shape[1])).indices.sort().values
            out.append(ids.long().tolist())
        return out
