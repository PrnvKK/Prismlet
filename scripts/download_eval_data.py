"""Download the public evaluation datasets and record their licences."""

import argparse
import json
import pathlib
import time

from _common import write_json

MB = 1_000_000
HARD_FILE = 400 * MB
HARD_TOTAL = 2_000 * MB

SPEC = [
    dict(key="banking77", repo="PolyAI/banking77",
         files=["data/test-00000-of-00001.parquet", "data/train-00000-of-00001.parquet"],
         license="cc-by-4.0", type="choice", domain="intent routing", langs=["en"],
         n_test=120, n_calib=50),
    dict(key="massive", repo="mteb/amazon_massive_intent",
         files=[f"{split}/{lang}.json.gz" for split in ("test", "validation")
                for lang in ("en", "hi", "bn", "ta", "te")],
         license="apache-2.0", license_note="card apache-2.0; origin AmazonScience/massive CC-BY-4.0",
         type="choice", domain="intent routing", langs=["en", "hi", "bn", "ta", "te"],
         n_test=160, n_calib=80),
    dict(key="amazon_polarity", repo="mteb/amazon_polarity",
         files=["data/test-00000-of-00001.parquet"],
         license="apache-2.0", license_note="card apache-2.0; derived from Amazon product reviews",
         type="yes_no", domain="sentiment", langs=["en"], n_test=80, n_calib=40),
    dict(key="civil_comments", repo="google/civil_comments",
         files=["data/test-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet"],
         license="cc0-1.0", type="score", domain="toxicity / moderation", langs=["en"],
         n_test=150, n_calib=60),
    dict(key="snli", repo="stanfordnlp/snli",
         files=["plain_text/test-00000-of-00001.parquet", "plain_text/validation-00000-of-00001.parquet"],
         license="cc-by-sa-4.0", type="yes_no", domain="NLI", langs=["en"], n_test=100, n_calib=40),
    dict(key="boolq", repo="google/boolq",
         files=["data/validation-00000-of-00001.parquet", "data/train-00000-of-00001.parquet"],
         license="cc-by-sa-3.0", type="yes_no", domain="reading-comprehension QA", langs=["en"],
         n_test=80, n_calib=40),
    dict(key="mmlu", repo="cais/mmlu",
         files=["all/test-00000-of-00001.parquet", "all/dev-00000-of-00001.parquet"],
         license="mit", type="choice", domain="knowledge MCQ", langs=["en"], n_test=130, n_calib=52),
    dict(key="global_mmlu", repo="CohereLabs/Global-MMLU",
         files=[f"{lang}/{split}-00000-of-00001.parquet" for lang in ("hi", "bn", "te")
                for split in ("test", "dev")],
         license="apache-2.0", type="choice", domain="knowledge MCQ", langs=["hi", "bn", "te"],
         n_test=120, n_calib=45),
    dict(key="sib200", repo="Davlan/sib200",
         files=[f"data/{lang}/{split}.tsv" for lang in ("eng_Latn", "hin_Deva", "ben_Beng", "tam_Taml")
                for split in ("test", "train")] + ["data/eng_Latn/labels.txt"],
         license="cc-by-sa-4.0", type="choice", domain="topic classification",
         langs=["en", "hi", "bn", "ta"], n_test=120, n_calib=48),
    dict(key="dbpedia14", repo="fancyzhx/dbpedia_14",
         files=["dbpedia_14/test-00000-of-00001.parquet"],
         license="cc-by-sa-3.0", type="choice", domain="topic / entity type", langs=["en"],
         n_test=70, n_calib=25),
    dict(key="when2call", repo="nvidia/When2Call",
         files=["test/when2call_test_mcq.jsonl"],
         license="cc-by-4.0", type="choice", domain="tool-use decision", langs=["en"],
         n_test=70, n_calib=20),
]

REJECTED = [
    {"repo_id": "stanfordnlp/sst2", "reason": "card licence 'unknown'"},
    {"repo_id": "fancyzhx/ag_news", "reason": "card licence 'unknown'"},
    {"repo_id": "stanfordnlp/imdb", "reason": "card licence 'other'"},
    {"repo_id": "yelp_polarity", "reason": "no licence"},
    {"repo_id": "cardiffnlp/tweet_eval", "reason": "licence unknown"},
    {"repo_id": "ai4bharat/IndicXNLI", "reason": "card says cc0 but text says CC BY-NC 4.0 (conflict)"},
    {"repo_id": "ai4bharat/IndicSentiment", "reason": "no licence stated"},
    {"repo_id": "facebook/xnli", "reason": "CC BY-NC"},
    {"repo_id": "allenai/wildguardmix", "reason": "gated (auto); HF login required"},
    {"repo_id": "cais/hle", "reason": "gated; images ~1 GB"},
]


def norm_license(v):
    if isinstance(v, list):
        return ",".join(str(x) for x in v)
    return "" if v is None else str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data", help="destination directory")
    ap.add_argument("--out", default="results", help="results directory")
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    root = pathlib.Path(a.data)
    api = HfApi()
    out, dropped, total = [], [], 0
    t0 = time.time()
    for spec in SPEC:
        key, repo = spec["key"], spec["repo"]
        try:
            info = api.dataset_info(repo, files_metadata=True)
        except Exception as e:
            dropped.append({"key": key, "reason": "dataset_info failed: " + type(e).__name__})
            continue
        lic = norm_license((info.cardData or {}).get("license"))
        sizes = {s.rfilename: s.size for s in (info.siblings or [])}
        files, ok, why = [], True, ""
        for f in spec["files"]:
            b = sizes.get(f)
            if b is None:
                ok, why = False, "missing file " + f
                break
            if b > HARD_FILE:
                ok, why = False, f"file {f} {b / 1e6:.0f} MB > 400 MB"
                break
            files.append({"path": f, "bytes": b})
        need = sum(x["bytes"] for x in files)
        if total + need > HARD_TOTAL:
            ok, why = False, "total would exceed 2 GB"
        if ok and lic and lic != spec["license"]:
            ok, why = False, f"licence card {lic!r} != expected {spec['license']!r}"
        if ok and not lic:
            ok, why = False, "no licence on card"
        if not ok:
            dropped.append({"key": key, "reason": why})
            continue
        local = root / "raw" / key
        local.mkdir(parents=True, exist_ok=True)
        got = []
        for x in files:
            p = hf_hub_download(repo_id=repo, filename=x["path"], repo_type="dataset",
                                local_dir=str(local))
            got.append({"path": x["path"], "bytes": pathlib.Path(p).stat().st_size})
        dl = sum(g["bytes"] for g in got)
        total += dl
        out.append({
            "key": key, "repo_id": repo, "revision": info.sha,
            "url": f"https://huggingface.co/datasets/{repo}",
            "files": got, "license_card": lic,
            "license_note": spec.get("license_note", {
                "cc-by-4.0": "CC-BY-4.0 attribution required",
                "cc-by-sa-4.0": "CC-BY-SA-4.0 attribution and share-alike for derived files",
                "cc-by-sa-3.0": "CC-BY-SA-3.0 attribution and share-alike for derived files",
                "cc0-1.0": "CC0-1.0 public domain dedication",
                "mit": "MIT attribution",
                "apache-2.0": "Apache-2.0 attribution",
            }.get(spec["license"], spec["license"])),
            "type": spec["type"], "domain": spec["domain"], "langs": spec["langs"],
            "n_test": spec["n_test"], "n_calib": spec["n_calib"], "downloaded_bytes": dl,
        })
    doc = {"datasets": out, "rejected": REJECTED + dropped, "total_downloaded_bytes": total,
           "notes": "licences read from the HF dataset card at download time"}
    write_json(root / "datasets.json", doc)
    write_json(pathlib.Path(a.out) / "eval" / "datasets.json", doc)
    print(json.dumps({"datasets": [d["key"] for d in out], "bytes": total,
                      "skipped": [d["key"] for d in dropped],
                      "seconds": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()
