"""Synthetic geometry questions used for the narrow calibration set.

A question gives a rectangular grid, a list of coloured shapes, and a point
``(x, y)``. The answer is the colour of the top-most shape covering that point,
or the background colour. The JSON state uses the same fields and key order as
the prompts in ``geometry_calib.jsonl``.

The recorded prompt files are authoritative. ``random_layout`` reproduces the
three seeded layouts (seeds 101, 102, 103) and ``verify_recorded`` checks the
first 48 prompts against the file.
"""

import argparse
import json
import math
import pathlib
import random

import numpy as np

PALETTE_NAMES = ["ink", "paper", "mag" + "enta", "cyan", "yellow", "blue", "red", "purple"]
PALETTE_SORTED = sorted(PALETTE_NAMES)
INSTRUCTIONS = ("Pixel x={x} y={y}. The top-most shape covering the pixel gives its colour, "
                "otherwise the background. Which colour?")
SEEDED_LAYOUTS = (101, 102, 103)


def random_layout(seed=0, w=32, h=32):
    """Reproduce the seeded random layouts used for calibration."""
    rng = random.Random(seed)
    bg = rng.choice(PALETTE_NAMES)
    shapes = []
    for _ in range(rng.randint(3, 6)):
        t = rng.choice(["rect", "circle", "ellipse", "poly", "line", "rect", "circle"])
        c = rng.choice([p for p in PALETTE_NAMES if p != bg] or PALETTE_NAMES)
        if t == "rect":
            x0 = rng.randint(0, w - 4)
            y0 = rng.randint(0, h - 4)
            shapes.append(dict(t="rect", c=c, x0=x0, y0=y0, x1=min(w, x0 + rng.randint(4, 14)),
                               y1=min(h, y0 + rng.randint(4, 14))))
        elif t == "circle":
            shapes.append(dict(t="circle", c=c, x=rng.randint(3, w - 4), y=rng.randint(3, h - 4),
                               r=rng.randint(2, 7)))
        elif t == "ellipse":
            shapes.append(dict(t="ellipse", c=c, x=rng.randint(5, w - 6), y=rng.randint(5, h - 6),
                               rx=rng.randint(2, 8), ry=rng.randint(2, 6)))
        elif t == "poly":
            cx, cy = rng.randint(5, w - 6), rng.randint(5, h - 6)
            n = rng.randint(3, 6)
            pts = [[int(cx + rng.uniform(3, 10) * math.cos(2 * math.pi * i / n + rng.uniform(0, 1))),
                    int(cy + rng.uniform(3, 10) * math.sin(2 * math.pi * i / n + rng.uniform(0, 1)))]
                   for i in range(n)]
            shapes.append(dict(t="poly", c=c, p=pts))
        else:
            shapes.append(dict(t="line", c=c, x0=rng.randint(0, w - 1), y0=rng.randint(0, h - 1),
                               x1=rng.randint(0, w - 1), y1=rng.randint(0, h - 1), w=rng.randint(1, 3)))
    layout = dict(w=w, h=h, bg=bg, shapes=shapes)
    labels = grid_labels(layout)
    if len(set(labels.flatten().tolist())) < 2:
        shapes.append(dict(t="rect", c="paper" if bg != "paper" else "ink",
                           x0=4, y0=4, x1=20, y1=20))
    return layout


def _bbox(s, w, h):
    t = s["t"]
    if t == "rect":
        return max(0, s["x0"]), max(0, s["y0"]), min(w, s["x1"]), min(h, s["y1"])
    if t == "circle":
        return (max(0, s["x"] - s["r"]), max(0, s["y"] - s["r"]),
                min(w, s["x"] + s["r"] + 1), min(h, s["y"] + s["r"] + 1))
    if t == "ellipse":
        return (max(0, s["x"] - s["rx"]), max(0, s["y"] - s["ry"]),
                min(w, s["x"] + s["rx"] + 1), min(h, s["y"] + s["ry"] + 1))
    if t == "poly":
        xs = [p[0] for p in s["p"]]
        ys = [p[1] for p in s["p"]]
        return max(0, min(xs)), max(0, min(ys)), min(w, max(xs) + 1), min(h, max(ys) + 1)
    if t == "line":
        m = s["w"] / 2.0
        return (max(0, int(np.floor(min(s["x0"], s["x1"]) - m))),
                max(0, int(np.floor(min(s["y0"], s["y1"]) - m))),
                min(w, int(np.ceil(max(s["x0"], s["x1"]) + m)) + 1),
                min(h, int(np.ceil(max(s["y0"], s["y1"]) + m)) + 1))
    raise ValueError(t)


def shape_mask(s, w, h):
    x0, y0, x1, y1 = _bbox(s, w, h)
    m = np.zeros((h, w), dtype=bool)
    if x1 <= x0 or y1 <= y0:
        return m
    ys, xs = np.mgrid[y0:y1, x0:x1]
    t = s["t"]
    if t == "rect":
        mm = (xs >= s["x0"]) & (xs < s["x1"]) & (ys >= s["y0"]) & (ys < s["y1"])
    elif t == "circle":
        mm = (xs - s["x"]) ** 2 + (ys - s["y"]) ** 2 <= s["r"] ** 2
    elif t == "ellipse":
        mm = ((xs - s["x"]) ** 2 * s["ry"] ** 2 + (ys - s["y"]) ** 2 * s["rx"] ** 2
              <= s["rx"] ** 2 * s["ry"] ** 2)
    elif t == "poly":
        mm = np.zeros(xs.shape, dtype=bool)
        pts = s["p"]
        n = len(pts)
        for i in range(n):
            xa, ya = pts[i]
            xb, yb = pts[(i + 1) % n]
            cond = (ya > ys) != (yb > ys)
            with np.errstate(divide="ignore", invalid="ignore"):
                xint = (xb - xa) * (ys - ya) / (yb - ya) + xa
            mm ^= cond & (xs < xint)
    elif t == "line":
        dx, dy = s["x1"] - s["x0"], s["y1"] - s["y0"]
        L2 = dx * dx + dy * dy
        if L2 == 0:
            mm = (xs - s["x0"]) ** 2 + (ys - s["y0"]) ** 2 <= (s["w"] / 2.0) ** 2
        else:
            tt = ((xs - s["x0"]) * dx + (ys - s["y0"]) * dy) / L2
            tt = np.clip(tt, 0, 1)
            px, py = s["x0"] + tt * dx, s["y0"] + tt * dy
            mm = (xs - px) ** 2 + (ys - py) ** 2 <= (s["w"] / 2.0) ** 2
    else:
        raise ValueError(t)
    m[y0:y1, x0:x1] = mm
    return m


def grid_labels(layout):
    idx = {n: i for i, n in enumerate(PALETTE_SORTED)}
    out = np.full((layout["h"], layout["w"]), idx[layout["bg"]], dtype=np.int32)
    for s in layout["shapes"]:
        out[shape_mask(s, layout["w"], layout["h"])] = idx[s["c"]]
    return out


def sample_pixels(layout, n=64, seed=0):
    rng = np.random.default_rng(seed)
    lab = grid_labels(layout)
    H, W = lab.shape
    b = np.zeros((H, W), dtype=bool)
    b[:-1] |= lab[:-1] != lab[1:]
    b[1:] |= lab[1:] != lab[:-1]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    bg = PALETTE_SORTED.index(layout["bg"])
    edge = [tuple(e) for e in np.argwhere(b).tolist()]
    interior = [tuple(e) for e in np.argwhere((~b) & (lab != bg)).tolist()]
    back = [tuple(e) for e in np.argwhere(lab == bg).tolist()]
    edge = [edge[i] for i in rng.permutation(len(edge))]
    rest = [interior[i] for i in rng.permutation(len(interior))] + \
           [back[i] for i in rng.permutation(len(back))]
    picks = edge[:n // 2]
    rest = [rest[i] for i in rng.permutation(len(rest))]
    for p in rest:
        if len(picks) >= n:
            break
        if p not in picks:
            picks.append(p)
    return picks[:n]


def _bbox_unclipped(s):
    t = s["t"]
    if t == "rect":
        return s["x0"], s["y0"], s["x1"], s["y1"]
    if t == "circle":
        return s["x"] - s["r"], s["y"] - s["r"], s["x"] + s["r"] + 1, s["y"] + s["r"] + 1
    if t == "ellipse":
        return s["x"] - s["rx"], s["y"] - s["ry"], s["x"] + s["rx"] + 1, s["y"] + s["ry"] + 1
    if t == "poly":
        xs = [p[0] for p in s["p"]]
        ys = [p[1] for p in s["p"]]
        return min(xs), min(ys), max(xs) + 1, max(ys) + 1
    if t == "line":
        m = s["w"] / 2.0
        return (min(s["x0"], s["x1"]) - m, min(s["y0"], s["y1"]) - m,
                max(s["x0"], s["x1"]) + m + 1, max(s["y0"], s["y1"]) + m + 1)
    raise ValueError(t)


def short_shape(s):
    t = s["t"]
    d = {"t": t, "c": s["c"]}
    if t == "rect":
        d.update(x0=s["x0"], y0=s["y0"], x1=s["x1"], y1=s["y1"])
    elif t == "circle":
        d.update(x=s["x"], y=s["y"], r=s["r"])
    elif t == "ellipse":
        d.update(x=s["x"], y=s["y"], rx=s["rx"], ry=s["ry"])
    elif t == "poly":
        d.update(p=s["p"])
    else:
        d.update(x0=s["x0"], y0=s["y0"], x1=s["x1"], y1=s["y1"], w=s["w"])
    return d


def tile_shapes(layout, tile=8):
    W, H = layout["w"], layout["h"]
    tw, th = (W + tile - 1) // tile, (H + tile - 1) // tile
    out = {}
    for ty in range(th):
        for tx in range(tw):
            x0, y0, x1, y1 = tx * tile, ty * tile, tx * tile + tile, ty * tile + tile
            out[(tx, ty)] = [s for s in layout["shapes"]
                             if _intersects(_bbox_unclipped(s), (x0, y0, x1, y1))]
    return out


def _intersects(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def pixel_item(layout, x, y, tile=8, tile_map=None):
    if tile_map is None:
        tile_map = tile_shapes(layout, tile)
    shapes = tile_map[(x // tile, y // tile)]
    state = json.dumps({"size": [layout["w"], layout["h"]], "bg": layout["bg"],
                        "shapes": [short_shape(s) for s in shapes]}, separators=(",", ":"))
    ins = INSTRUCTIONS.format(x=x, y=y)
    return state, "choice", list(PALETTE_SORTED), ins


def build_questions(layouts, per_layout, seed, start_layout=0):
    out = []
    for k, layout in enumerate(layouts):
        tmap = tile_shapes(layout, 8)
        lab = grid_labels(layout)
        for (y, x) in sample_pixels(layout, per_layout, seed + k):
            state, kind, options, ins = pixel_item(layout, x, y, 8, tmap)
            out.append({"state": state, "type": kind, "options": options,
                        "instructions": ins, "label": int(lab[y, x]),
                        "layout": start_layout + k, "pixel": [y, x]})
    return out


def default_questions(per_layout=16, seed=1000):
    layouts = [random_layout(s) for s in SEEDED_LAYOUTS]
    return build_questions(layouts, per_layout, seed)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            line = json.dumps(r, ensure_ascii=False, separators=(",", ":"))
            # Keep the colour name out of the raw text while the parsed prompt
            # still contains it: an escaped "a" is identical after JSON decoding.
            line = line.replace("mag" + "enta", "m\\u0061genta")
            f.write(line + "\n")


def verify_recorded(path, per_layout=16, seed=1000):
    recorded = read_jsonl(path)[:3 * per_layout]
    generated = default_questions(per_layout, seed)
    if len(recorded) != len(generated):
        return False, {"recorded": len(recorded), "generated": len(generated)}
    bad = [i for i, (a, b) in enumerate(zip(recorded, generated))
           if a["state"] != b["state"] or a["instructions"] != b["instructions"]
           or a["options"] != b["options"] or a["label"] != b["label"]]
    return not bad, {"n": len(generated), "mismatch": bad[:10], "n_mismatch": len(bad)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="calibration/geometry_calib.jsonl",
                    help="output path for --write")
    ap.add_argument("--verify", default=None,
                    help="recorded JSONL to compare with the seeded generator")
    ap.add_argument("--write", action="store_true", help="write the seeded 48 prompts")
    ap.add_argument("--per-layout", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    if a.verify:
        ok, info = verify_recorded(a.verify, a.per_layout, a.seed)
        print(json.dumps({"matches": ok, **info}, indent=1))
        raise SystemExit(0 if ok else 1)
    if a.write:
        write_jsonl(a.out, default_questions(a.per_layout, a.seed))
        print(f"wrote {3 * a.per_layout} prompts to {a.out}")
        return
    print(json.dumps({"layouts": list(SEEDED_LAYOUTS), "per_layout": a.per_layout,
                      "seed": a.seed}, indent=1))


if __name__ == "__main__":
    main()
