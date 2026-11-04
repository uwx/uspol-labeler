#!/usr/bin/env python
"""Label a post pool with a local Ollama teacher.

The teacher is intentionally a stronger general model than the tiny classifier we
distill; Laya's zero-shot decisions were too weak on this novel task (see README).
Laya scores are still recorded for every post by `laya_label.py`.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

FEWSHOT = """Examples:
Post: "The Senate voted 51-49 to advance the budget bill." -> {"label": 1}
Post: "The UK parliament is debating its budget today." -> {"label": 0}
Post: "I voted today! Get out and vote." -> {"label": 1}
Post: "My cat knocked over the trash can again lol." -> {"label": 0}
Post: "Trudeau resigned as party leader. Canadian politics is wild." -> {"label": 0}
Post: "Looks fabulous!" -> {"label": 0}
Post: "Trump just announced new tariffs on Canada." -> {"label": 1}
Post: "Our city council is voting on a bike lane ordinance." -> {"label": 1}
Post: "Happy Election Day!" -> {"label": 1}
Post: "Find your polling place at vote.org." -> {"label": 1}
Post: "I got stuck in a voting booth for ten minutes." -> {"label": 1}
Post: "Good luck to all the voters today." -> {"label": 1}
Post: "You've met Obama and Biden?" -> {"label": 1}
Post: "The mayor vetoed the ordinance at a press conference." -> {"label": 1}
Post: "The Spanish opposition party is campaigning on the economy." -> {"label": 0}"""

SYSTEM = """You are labeling Bluesky posts for United States politics.
Label 1 if the post is about: American politicians or political figures; US political parties; US elections, campaigns or voting; US legislation, government policy or regulation; US political institutions (Congress, courts, agencies); or US political events and controversies.
Label 0 for everything else.
Important: if the post's main subject is another country, label 0 even when the US is mentioned. Sports, entertainment, personal life, ads, and generic news are 0. A post that is just a link or a few words with no political content is 0.
Return only JSON: {"label": 1} or {"label": 0}.

""" + FEWSHOT


def post_text(post: dict) -> str:
    parts = [post.get("text") or ""]
    if post.get("alt"):
        parts.append("ALT: " + post["alt"])
    if post.get("tags"):
        parts.append("TAGS: " + " ".join(post["tags"]))
    ext = post.get("external") or {}
    if ext.get("title"):
        parts.append("LINK TITLE: " + ext["title"])
    if ext.get("description"):
        parts.append("LINK DESCRIPTION: " + ext["description"])
    return "\n".join(parts)[:2000]


def ask(model: str, text: str) -> tuple[int | None, str, float]:
    prompt = f"Post:\n{text}\n\nLabel this post. Return JSON."
    body = json.dumps(
        {
            "model": model,
            "system": SYSTEM,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            "options": {"temperature": 0, "seed": 42, "num_predict": 16, "num_ctx": 2048},
        }
    ).encode()
    req = urllib.request.Request(
        os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=300) as response:
        payload = json.loads(response.read())
    raw = payload.get("response", "")
    label = None
    try:
        parsed = json.loads(raw)
        value = parsed.get("label")
        if isinstance(value, bool):
            label = int(value)
        elif isinstance(value, (int, float)) and value in (0, 1):
            label = int(value)
        elif isinstance(value, str) and value.strip() in {"0", "1"}:
            label = int(value.strip())
    except json.JSONDecodeError:
        pass
    return label, raw, time.time() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="qwen2.5-coder:14b")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["uri"])
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"resuming with {len(seen)} already labeled", file=sys.stderr)

    total = 0
    done = 0
    failed = 0
    start = time.time()
    with out_path.open("a", encoding="utf-8") as out, open(args.input, encoding="utf-8") as inp:
        for line in inp:
            post = json.loads(line)
            if post["uri"] in seen:
                continue
            if args.limit and done >= args.limit:
                break
            label, raw, elapsed = ask(args.model, post_text(post))
            if label is None:
                failed += 1
                print(f"unparsable response for {post['uri']}: {raw!r}", file=sys.stderr)
                continue
            out.write(json.dumps({
                "uri": post["uri"],
                "label": label,
                "teacher": args.model,
                "ms": round(elapsed * 1000),
                "raw": raw,
            }, ensure_ascii=False) + "\n")
            out.flush()
            done += 1
            total += 1
            if done % 100 == 0:
                rate = done / (time.time() - start)
                print(f"{done} labeled ({rate:.2f}/s, {failed} failed)", file=sys.stderr, flush=True)
    print(f"done: {total} labeled, {failed} failed, {time.time() - start:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
