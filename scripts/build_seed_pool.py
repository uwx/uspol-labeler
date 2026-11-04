"""Build a stratified seed pool from raw firehose dumps.

Rare 2026-cycle names are sampled exhaustively first; the rest is filled
deterministically from candidates, negatives and the unbiased random sample.
"""
import argparse
import json
import random
from pathlib import Path

RARE = [
    "talarico", "paxton", "hinojosa", "peltola", "hickenlooper", "lujan", "mizusawa",
    "hageman", "el-sayed", "tafoya", "hyde-smith", "husted", "merkley", "fetty anderson",
]


def read(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", nargs="+", default=["data/raw24", "data/raw"])
    parser.add_argument("--out", default="data/seed_pool.jsonl")
    parser.add_argument("--candidates", type=int, default=4000)
    parser.add_argument("--negatives", type=int, default=2500)
    parser.add_argument("--random", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    gold = {json.loads(line)["uri"] for line in open("data/gold.jsonl", encoding="utf-8")}
    rng = random.Random(args.seed)
    seen = set(gold)
    candidates, negatives, randoms = [], [], []

    for raw in args.raw:
        root = Path(raw)
        for post in read(root / "candidates.jsonl"):
            if post["uri"] in seen:
                continue
            seen.add(post["uri"])
            candidates.append(post)
        for post in read(root / "negatives.jsonl"):
            if post["uri"] in seen:
                continue
            seen.add(post["uri"])
            negatives.append(post)
        for post in read(root / "random.jsonl"):
            if post["uri"] in seen:
                continue
            seen.add(post["uri"])
            randoms.append(post)

    def surface(post):
        return "\n".join(filter(None, [
            post.get("text"), post.get("alt"), " ".join(post.get("tags") or []),
            (post.get("external") or {}).get("title"),
            (post.get("external") or {}).get("description"),
        ])).lower()

    rare = [p for p in candidates if any(name in surface(p) for name in RARE)]
    rng.shuffle(rare)
    rare = rare[:1500]
    rare_uris = {p["uri"] for p in rare}

    rest = [p for p in candidates if p["uri"] not in rare_uris]
    rng.shuffle(rest)
    picked_candidates = rare + rest[: max(0, args.candidates - len(rare))]
    picked_negatives = rng.sample(negatives, min(args.negatives, len(negatives)))
    picked_random = rng.sample(randoms, min(args.random, len(randoms)))

    rows = []
    for source, group in (("candidate", picked_candidates), ("negative", picked_negatives), ("random", picked_random)):
        for post in group:
            post["source"] = source
            post["keywords"] = [name for name in RARE if name in surface(post)]
            rows.append(post)
    rng.shuffle(rows)

    with open(args.out, "w", encoding="utf-8") as out:
        for post in rows:
            out.write(json.dumps(post, ensure_ascii=False) + "\n")
    print(json.dumps({
        "candidates": len(picked_candidates),
        "negatives": len(picked_negatives),
        "random": len(picked_random),
        "rare": len(rare),
        "total": len(rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
