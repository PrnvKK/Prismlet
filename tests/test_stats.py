import torch

from prismlet.stats import Stats


def test_update_and_rankings():
    s = Stats(nl=2, ne=4)
    ids = torch.tensor([[0, 1], [0, 2]])
    w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    norms = torch.tensor([[1.0, 1.0], [1.0, 10.0]])
    s.update(1, ids, w, norms)
    s.sal[1] = torch.tensor([4.0, 2.0, 40.0, 0.0], dtype=torch.float64)
    assert s.cnt[1].tolist() == [2.0, 1.0, 1.0, 0.0]
    assert s.ntok[1].item() == 2
    total = s.ranking("total")[1]
    reap = s.ranking("reap")[1]
    assert total[0] > total[1]
    assert total[2] > total[0]
    assert reap[2] > reap[1]


def test_keep_is_sorted_and_nested():
    s = Stats(nl=1, ne=8)
    s.cnt[0] = torch.arange(8, dtype=torch.float64)
    s.ntok[0] = 100
    s.sal[0] = torch.tensor([8, 7, 6, 5, 4, 3, 2, 1], dtype=torch.float64)
    kmax = s.keep("total", 6)[0]
    kmin = s.keep("total", 2)[0]
    assert kmax == sorted(kmax) and kmin == sorted(kmin)
    assert set(kmin) <= set(kmax)


def test_save_load(tmp_path):
    s = Stats(nl=1, ne=4)
    s.cnt[0] = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    s.ntok[0] = 10
    p = tmp_path / "stats.npz"
    s.save(p)
    t = Stats.load(p)
    assert torch.equal(t.cnt, s.cnt)
    assert torch.equal(t.ntok, s.ntok)
