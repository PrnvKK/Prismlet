"""Small helpers shared by the experiment scripts."""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def pick_device(name="auto"):
    import torch
    if name != "auto":
        return name
    return "xpu" if torch.xpu.is_available() else "cpu"


def prompt_items(records):
    """Turn prompt records into engine items plus labels."""
    items, labels = [], []
    for r in records:
        items.append((r["state"], r["type"], list(r["options"]), r["instructions"]))
        labels.append(r.get("label"))
    return items, labels
