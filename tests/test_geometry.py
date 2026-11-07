import pathlib

from calibration.geometry_questions import (PALETTE_SORTED, default_questions, grid_labels,
                                            random_layout, verify_recorded)


def test_recorded_file_has_all_prompts():
    path = pathlib.Path(__file__).resolve().parent.parent / "calibration" / "geometry_calib.jsonl"
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 99
    assert sum(1 for x in lines if '"label":null' not in x) == 96


def test_generator_matches_recorded_prefix():
    path = pathlib.Path(__file__).resolve().parent.parent / "calibration" / "geometry_calib.jsonl"
    ok, info = verify_recorded(path)
    assert ok, info
    assert info["n"] == 48


def test_layout_labels_are_palette_indices():
    layout = random_layout(101)
    labels = grid_labels(layout)
    assert labels.shape == (32, 32)
    assert int(labels.min()) >= 0
    assert int(labels.max()) < len(PALETTE_SORTED)


def test_generated_prompts_are_list_shaped():
    qs = default_questions(per_layout=2)
    assert len(qs) == 6
    for q in qs:
        assert q["type"] == "choice"
        assert q["options"] == PALETTE_SORTED
        assert q["instructions"].startswith("Pixel x=")
        assert isinstance(q["label"], int)
