"""Join teacher labels, Laya scores and the manual gold set into train/val/test splits."""
import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def split_for(uri: str) -> str:
    h = int(hashlib.sha1(uri.encode()).hexdigest()[:8], 16)
    if h % 10 < 8:
        return "train"
    if h % 10 == 8:
        return "val"
    return "test"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="data/seed_pool.jsonl")
    parser.add_argument("--teacher", default="data/teacher_labels.jsonl")
    parser.add_argument("--laya", default="data/laya_labels.jsonl")
    parser.add_argument("--gold", default="data/gold.jsonl")
    parser.add_argument("--out", default="data/dataset.jsonl")
    args = parser.parse_args()

    teacher = {r["uri"]: r for r in load_jsonl(args.teacher)}
    laya = {r["uri"]: r for r in load_jsonl(args.laya)}
    gold = {r["uri"]: r for r in load_jsonl(args.gold)}

    rows = []
    missing = 0
    for post in load_jsonl(args.pool):
        t = teacher.get(post["uri"])
        laya_row = laya.get(post["uri"], {})
        # choice + score + simple was the best Laya ensemble on the 120-post
        # gold set (AUC 0.958); the long definition and 'names' prompts hurt.
        scores = [
            laya_row.get("laya_simple"),
            laya_row.get("laya_choice"),
            laya_row.get("laya_score"),
        ]
        scores = [s for s in scores if isinstance(s, (int, float))]
        laya_mean = sum(scores) / len(scores) if scores else None
        if t is not None:
            label = t["label"]
            label_source = "teacher"
            teacher_name = t.get("teacher")
            agrees = laya_mean is not None and ((label == 1 and laya_mean >= 0.6) or (label == 0 and laya_mean <= 0.4))
            confidence = 1.0 if agrees else 0.5
        elif laya_mean is not None:
            label = 1 if laya_mean >= 0.5 else 0
            label_source = "laya"
            teacher_name = None
            agrees = None
            confidence = 1.0 if abs(laya_mean - 0.5) >= 0.2 else 0.5
        else:
            missing += 1
            continue
        rows.append({
            "uri": post["uri"],
            "text": post["text"],
            "alt": post.get("alt"),
            "tags": post.get("tags"),
            "external": post.get("external"),
            "createdAt": post.get("createdAt"),
            "langs": post.get("langs"),
            "source": post.get("source"),
            "keywords": post.get("keywords"),
            "label": label,
            "label_source": label_source,
            "teacher": teacher_name,
            "laya_mean": laya_mean,
            "laya_agrees": agrees,
            "confidence": confidence,
            "split": split_for(post["uri"]),
        })

    for uri, g in gold.items():
        rows.append({
            "uri": uri,
            "text": g["text"],
            "alt": g.get("alt"),
            "tags": g.get("tags"),
            "external": g.get("external"),
            "source": "gold",
            "label": g["label"],
            "label_source": "manual",
            "laya_mean": None,
            "laya_agrees": None,
            "confidence": 1.0,
            "split": "gold",
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {}
    for row in rows:
        counts[(row["split"], row["label"])] = counts.get((row["split"], row["label"]), 0) + 1
    print(json.dumps({
        "rows": len(rows),
        "missing": missing,
        "counts": {f"{k[0]}/{k[1]}": v for k, v in sorted(counts.items())},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
