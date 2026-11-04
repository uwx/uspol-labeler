import json, os, re, sys, time, statistics
os.environ.setdefault("USE_TF", "0")
from laya import Router

rows = [json.loads(line) for line in open("data/gold.jsonl", encoding="utf-8")]
states = [r["text"] for r in rows]
labels = [r["label"] for r in rows]

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

def scores(ckpt, batch_size=64):
    router = Router(device="cuda", max_loaded=1)
    router.preload([ckpt])
    reqs = [{"state": s, "questions": QUESTIONS, "model": ckpt} for s in states]
    t0 = time.time()
    res = router.predict_batch(reqs, batch_size=batch_size)
    dt = time.time() - t0
    out = {k: [] for k in QUESTIONS}
    for r in res:
        a = r["answers"]
        out["simple"].append(a["simple"]["noul"])
        out["names"].append(a["names"]["noul"])
        out["def"].append(a["def"]["noul"])
        c = a["choice"]
        probs = c.get("probabilities") or c.get("probability") or {}
        out["choice"].append(float(probs.get("us_politics", 0.0)) if isinstance(probs, dict) else float(c.get("choice") == "us_politics"))
        out["score"].append(a["score"].get("score", 0) / 3.0)
    print(f"{ckpt}: {len(states)} in {dt:.2f}s = {len(states)/dt:.0f}/s", file=sys.stderr)
    return out

def auc(vals, ys):
    pos = [v for v, y in zip(vals, ys) if y]
    neg = [v for v, y in zip(vals, ys) if not y]
    wins = sum(1 for p in pos for n in neg if p > n) + 0.5 * sum(1 for p in pos for n in neg if p == n)
    return wins / (len(pos) * len(neg))

all_scores = {}
for ckpt in ["laya", "typed-decisions"]:
    all_scores[ckpt] = scores(ckpt)

for ckpt, sc in all_scores.items():
    print(ckpt)
    for k, vals in sc.items():
        print(f"  {k:8s} auc={auc(vals, labels):.3f} pos_mean={statistics.mean(v for v,y in zip(vals,labels) if y):.3f} neg_mean={statistics.mean(v for v,y in zip(vals,labels) if not y):.3f}")

# simple rank-average ensemble
def rank(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    for rank_i, idx in enumerate(order): r[idx] = rank_i / (len(vals) - 1)
    return r
ensemble = [sum(rank(all_scores[c][k])[i] for c in all_scores for k in ["simple","names","def","choice","score"]) / 10 for i in range(len(labels))]
print("ensemble auc", round(auc(ensemble, labels),3))
print("threshold 0.5 acc", round(sum((v>=0.5)==bool(y) for v,y in zip(ensemble,labels))/len(labels),3))

json.dump({c: {k: v for k, v in sc.items()} for c, sc in all_scores.items()}, open("data/gold_laya_scores.json","w"))
