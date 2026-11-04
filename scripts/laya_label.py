import json, os, sys, time
os.environ.setdefault("USE_TF", "0")
from laya import Router

QUESTIONS = {
    "simple": {"type": "noul", "instructions": "Is this post about United States politics?"},
    "names": {
        "type": "noul",
        "instructions": "Does this post mention or discuss American politicians, elections, government policy, political parties, or political institutions?",
    },
    "def": {
        "type": "noul",
        "instructions": "Does this post contain or reference United States politics? US politics means American political figures, politicians, parties, elections, campaigns, legislation, government policy, political institutions, or political events. Foreign politics, sports, entertainment, and ordinary personal life do not count.",
    },
    "choice": {
        "type": "choice",
        "instructions": "What is the main topic of this post?",
        "criteria": {
            "us_politics": "United States politics: American politicians, political parties, elections, voting, government policy, legislation, political institutions or political events.",
            "other": "Anything else: personal life, entertainment, sports, foreign politics, business, or general news.",
        },
    },
    "score": {
        "type": "score",
        "instructions": "How strongly does this post reference United States politics?",
        "criteria": ["not at all", "slightly", "clearly", "strongly"],
    },
}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="laya")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--questions", default="all", help="all, or a comma-separated subset: simple,names,def,choice,score")
    args = parser.parse_args()
    selected = {k: v for k, v in QUESTIONS.items() if args.questions == "all" or k in args.questions.split(",")}

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    done = set()
    try:
        for line in open(args.output, encoding="utf-8"):
            done.add(json.loads(line)["uri"])
    except FileNotFoundError:
        pass
    rows = [r for r in rows if r["uri"] not in done]
    print(f"{len(rows)} to label with {args.checkpoint}", file=sys.stderr)

    router = Router(device="cuda", max_loaded=1)
    router.preload([args.checkpoint])
    start = time.time()
    out = open(args.output, "a", encoding="utf-8")
    for offset in range(0, len(rows), args.batch_size):
        chunk = rows[offset:offset + args.batch_size]
        reqs = [{"state": r["text"], "questions": selected, "model": args.checkpoint} for r in chunk]
        results = router.predict_batch(reqs, batch_size=args.batch_size, sort_by_length=True)
        for row, result in zip(chunk, results):
            a = result["answers"]
            record = {"uri": row["uri"]}
            if "simple" in a: record["laya_simple"] = a["simple"]["noul"]
            if "names" in a: record["laya_names"] = a["names"]["noul"]
            if "def" in a: record["laya_def"] = a["def"]["noul"]
            if "choice" in a:
                probs = a["choice"].get("probabilities") or a["choice"].get("probability") or {}
                record["laya_choice"] = float(probs.get("us_politics", 0.0)) if isinstance(probs, dict) else float(a["choice"].get("choice") == "us_politics")
            if "score" in a: record["laya_score"] = a["score"].get("score", 0) / 3.0
            out.write(json.dumps(record) + "\n")
        out.flush()
        done_now = min(offset + len(chunk), len(rows))
        print(f"{done_now}/{len(rows)} ({done_now / (time.time() - start):.0f}/s)", file=sys.stderr)
    out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
