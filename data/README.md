# US politics post dataset

Posts collected from the Bluesky Jetstream firehose and labeled for the `us-politics` label.

## Definition

A post is **positive** when it contains or references United States politics:

- American politicians or political figures, current or former;
- US political parties, campaigns, elections, voting, or ballots;
- US legislation, government policy, regulation, or political debate;
- US political institutions (Congress, courts, agencies, state and local government);
- US political events and controversies.

Negative: foreign politics without a US subject, sports, entertainment, personal life, ads, and generic news. A post that only shares a link with no political text is negative unless the link title/description is political.

## Provenance

- Source: `wss://jetstream2.us-east.bsky.network`, `app.bsky.feed.post` creates.
- Window: approximately 2026-11-03T18:48Z to 2026-11-04T18:47Z (24h replay from a cursor).
- Collected: 4,457,499 post creates; 656,774 matched the keyword net; a deterministic 1/4 of candidates (164,193), 1/16 of the rest (238,410), and 1/500 of all posts (8,925) were written.
- Seed pool: 8,000 posts — 4,000 candidates (rare 2026 names sampled first), 2,500 negatives, 1,500 uniform random.
- Teacher: `qwen2.5:7b` via Ollama with a few-shot prompt (325 rows by `qwen2.5-coder:14b` before switching); 2,279 rows use the teacher label.
- Laya (`laya` checkpoint): zero-shot `choice`, `score` and `simple` answers for all 8,000 rows; 5,721 rows use Laya as the label source when no teacher label exists. Ensemble AUC 0.958 on the manual gold set.
- Gold: 120 posts manually labeled by the author from the collected candidates; held out of training and evaluation splits.

## Fields

| field | meaning |
| --- | --- |
| `uri` | AT URI of the post |
| `text` | post text |
| `alt`, `tags`, `external` | image alt text, rich-text tags, link card title/description, when present |
| `label` | 1 = US politics, 0 = not |
| `label_source` | `teacher`, `laya`, or `manual` |
| `teacher` | teacher model for teacher labels |
| `laya_mean` | mean of Laya's `simple`, `choice` and `score` answers (0–1) |
| `laya_agrees` | whether Laya's answer is on the same side of 0.5 as the teacher |
| `confidence` | 1.0 when teacher and Laya agree at the 0.6/0.4 margins, otherwise 0.5 |
| `source` | `candidate`, `negative`, `random`, or `gold` |
| `split` | `train`, `val`, `test`, or `gold` |

Splits are deterministic by SHA-1 of the URI (80/10/10). The gold set is never in train/val/test. Current counts: 6,423 train (4,044 negative / 2,379 positive), 829 validation, 748 test, 120 gold.

## Caveats

- Teacher labels are noisy; the 120-post gold set is the evaluation set, not a claim of perfection.
- The keyword net is deliberately broad (for example `policy`, `voting`, `senator`, 2026 candidate names). The teacher makes the final decision.
- The sample is English-heavy because MiniLM-L6 is English-first.
- Counts in `README.md` and in the dataset file are generated from the local run; rerunning the collection at a different time changes them.
