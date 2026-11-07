import pathlib

from prismlet import config as C


def test_softmax_scale():
    assert abs(C.SOFTMAX_SCALE - 0.1352337788608801) < 1e-15


def test_layer_sizes():
    assert C.layer_bytes(128) == 3347500800
    assert C.layer_bytes(32) == 926862720
    assert C.layer_bytes(16) == 523423040
    assert C.layer_bytes(8) == 321703200
    assert C.total_backbone(128) == 104068783616
    assert C.total_backbone(128) + C.SIZE_EMBED == 106216275456


def test_size_reduction():
    full = C.total_backbone(128)
    assert abs(full / C.total_backbone(32) - 3.5849933643412037) < 1e-9
    assert abs(full / C.total_backbone(16) - 6.298658386617657) < 1e-9
    assert abs(full / C.total_backbone(8) - 10.134209335837099) < 1e-9


def test_rope_parameters():
    assert C.ROPE["dim"] == 64
    assert C.ROPE["factor"] == 40.0
    assert C.ROPE["orig"] == 4096
