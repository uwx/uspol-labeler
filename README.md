# uspol-labeler

A tiny US-politics classifier distilled from a local teacher, exported to ONNX, and run as a Bluesky labeler over the Jetstream firehose with ONNX Runtime (DirectML on Windows).

> For step-by-step account setup, training your own model, and Docker Compose deployment, see [`GUIDE.md`](GUIDE.md).

The label is `us-politics`: the post contains or references United States politics — American political figures, parties, elections and voting, legislation or government policy, political institutions, or political events.

## Results

| | |
| --- | --- |
| Firehose replay | 4,457,499 post creates over ~24h |
| Keyword candidates | 656,774 (14.7%); 164,193 written to disk after sampling |
| Seed pool | 8,000 posts (4,000 candidates, 2,500 negatives, 1,500 uniform random) |
| Dataset | 8,120 rows: 6,423 train / 829 val / 748 test / 120 gold |
| Label sources | 2,279 teacher-labeled, 5,721 Laya-labeled |
| Manual gold set | 120 posts, 67 positive / 53 negative |
| Laya ensemble (zero-shot) | AUC 0.958 on the gold set |
| Teacher agreement | 94.2% (113/120) on the gold set |
| Final classifier | MiniLM-L6 (22M parameters), ONNX, 128 tokens |
| Gold metrics (manual) | AUC 0.956; P 0.940 / R 0.940 / F1 0.940 at 0.5; P 0.931 / R 0.806 at 0.85 |
| Distilled test metrics | AUC 0.927; P 0.794 / R 0.828 / F1 0.811 at 0.6 |
| ONNX parity | max \|Δ\| 6.9e-7 vs PyTorch; 0.933 gold accuracy |
| Labeler provider | DirectML (`dml`), RTX 4070 SUPER; CUDA 13.0 (`cuda`), RTX 5060 Ti |
| Classifier throughput | 1,613 posts/s DirectML; 2,253 posts/s CUDA, batch 128 |
| CPU int8 fallback | 90 posts/s at batch 64 (1.45x live post traffic) |
| Live firehose (observed) | 62 post creates/s, 664 commit events/s |
| Headroom | ~26x DirectML; ~36x CUDA over post traffic |

`data/gold.jsonl` contains the manual labels. `data/dataset.jsonl` is the assembled dataset (see `data/README.md`).

## Pipeline

```mermaid
flowchart LR
    A[Jetstream replay<br/>24h cursor] --> B[keyword net]
    B --> C[seed pool<br/>8k posts]
    J[Laya zero-shot<br/>AUC 0.958] --> D
    T[Local teacher<br/>qwen2.5:7b] --> D[dataset builder]
    D --> E[train MiniLM-L6<br/>binary classifier]
    E --> F[export ONNX<br/>fp32 + int8]
    F --> G[labeler<br/>DirectML batching]
    G --> H[SQLite + NDJSON]
    G --> I[@atcute labeler service<br/>queryLabels / subscribeLabels]
```

1. **Collect.** `scripts/collect.ts` replays Jetstream from a cursor one day old, keeps post creates, and writes keyword candidates plus a uniform sample of negatives. The 24h replay needed nine reconnects because the public instance caps a single replay connection.
2. **Seed.** `scripts/build_seed_pool.py` samples candidates, negatives, and the uniform pool; rare 2026-cycle names are sampled exhaustively first.
3. **Label.** `scripts/laya_label.py` records zero-shot Laya decisions and `scripts/teacher_label.py` records a local Ollama teacher. `data/gold.jsonl` is a manually labeled holdout. Laya's `choice`, `score` and `simple` answers form the ensemble used for second-opinion confidence.
4. **Train.** `scripts/train.py` fine-tunes `sentence-transformers/all-MiniLM-L6-v2` for binary classification and reports test and gold metrics.
5. **Export.** `scripts/export_onnx.py` exports fixed-length ONNX and an int8 CPU fallback, and checks PyTorch/ONNX parity.
6. **Label.** `labeler/src/index.ts` consumes Jetstream through `@atcute/jetstream`, batches 64 posts, classifies on DirectML, and stores labels. With `DID` and `SIGNING_KEY` set, it signs labels through `@atcute/labeler` and serves `com.atproto.label.queryLabels` and `subscribeLabels`.
7. **Bench.** `labeler/src/bench.ts` measures classifier throughput and a live-rate simulation.

## Setup

Prerequisites: Python 3.11, Node.js 22+, pnpm, Bun (collector), and Ollama for the teacher.

```bash
pnpm install
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install laya onnx onnxruntime
ollama pull qwen2.5:7b
```

On Windows, `onnxruntime-node` uses the DirectML execution provider. `PROVIDER=cpu` forces CPU.

### Rebuild the dataset

```bash
bun scripts/collect.ts --hours=24 --out=data/raw24 --cand-every=4 --neg-every=16 --rand-every=500
py -3.11 scripts/build_seed_pool.py
USE_TF=0 .venv/Scripts/python.exe scripts/laya_label.py --input data/seed_pool.jsonl --output data/laya_labels.jsonl --checkpoint laya
.venv/Scripts/python.exe scripts/teacher_label.py --input data/seed_pool.jsonl --output data/teacher_labels.jsonl --model qwen2.5:7b
py -3.11 scripts/build_dataset.py
```

The teacher and Laya steps are resumable; both append to their output JSONL and skip URIs already present.

### Train and export

```bash
.venv/Scripts/python.exe scripts/train.py
.venv/Scripts/python.exe scripts/export_onnx.py
```

`model/onnx/model.onnx` is fp32 with dynamic batch (87 MB). `model/onnx/model.int8.onnx` is the CPU fallback (22 MB). On the gold set the int8 model reaches 0.908 accuracy versus 0.933 for fp32, so DirectML is the default.

### Run the labeler

```bash
cp .env.example .env
pnpm labeler
```

Defaults: live Jetstream, threshold 0.85, batch 64, SQLite at `data/labels.db`, NDJSON at `data/labels.ndjson`, metrics on `http://127.0.0.1:4103/metrics`.

To serve signed labels, set `DID` and `SIGNING_KEY` (a 64-character hex secp256k1 private key). The service listens on `PORT` (default 4102) and exposes the standard endpoints under `/xrpc`. The label definition is in `labeler/label-definition.json`; the labeler account must publish it in its `app.bsky.labeler.service` record and point its DID document at the public endpoint.

For a local service test:

```bash
pnpm serve:labels
```

This generates `data/test-signing-key.txt`, starts on port 4199 with a test DID, and serves one signed label.

### Expose for testing with cloudflared

```bash
./tools/cloudflared.exe tunnel --config tools/empty-cloudflared.yml --url http://127.0.0.1:4199 --no-autoupdate
```

The `--config` flag is required on this machine: the default `~/.cloudflared/config.yml` has a catch-all `http_status:501` rule that otherwise wins over the quick tunnel.

Query a label through the tunnel:

```bash
curl "https://<tunnel>.trycloudflare.com/xrpc/com.atproto.label.queryLabels?uriPatterns=at%3A%2F%2Fdid%3Aplc%3Aexample%2Fapp.bsky.feed.post%2F_&limit=10"
```

`subscribeLabels` frames are CBOR, not JSON; `scripts/test_subscribe.ts` decodes them.

### Benchmark

```bash
pnpm bench --input data/bench_posts.jsonl --batch 128
pnpm bench --input data/bench_posts.jsonl --batch 64 --live-rate 62
```

The first mode reports sustained throughput. The second feeds posts at the observed live rate and reports the maximum queue lag; `kept_up` is true when the drain finishes within a second of the input.

Observed on an RTX 4070 SUPER:

```
provider=dml  posts=22000  batch=128
posts_per_second=1613  p50=78.9ms  p95=82.5ms  p99=89.8ms
```

Live-rate simulation (4,000 posts at 62/s, batch 64):

```
effective_feed_rate=62  sim_seconds=64.57  max_queue_lag=6
drain_tail_ms=40.6  kept_up=true
```

The live-rate mode (4,000 posts fed at 62/s) reports `kept_up`, the effective feed rate, and the maximum queue lag. The queue stays bounded by one batch because inference runs ~26x faster than the feed.

### End-to-end verification

A four-minute live run with `THRESHOLD=0.85`, `BATCH_SIZE=64`, and signed labels enabled:

```
posts_total=12204  classified_total=12204  labels_total=1559
queue=0  dropped=0  p50_batch=22.3ms  p99_batch=28.0ms  lag=0.8s
```

After a restart the consumer resumed from the saved cursor (`1793820509064148`), replayed the gap, and returned to `queue=0` without dropping posts.

The first positive post was queried back through the public cloudflared endpoint and returned the signed label:

```
{"src":"did:web:localhost","uri":"at://did:plc:.../app.bsky.feed.post/...","val":"us-politics","ver":1,"sig":{"$bytes":"..."}}
```

Other checks:

```bash
pnpm typecheck
pnpm rate 15                    # live post/commit rate
tsx scripts/verify_onnx_gold.ts # ONNX Runtime vs PyTorch, 120 gold posts
tsx scripts/test_service.ts     # queryLabels + signature
tsx scripts/test_subscribe.ts   # subscribeLabels CBOR frames over a tunnel
```

## Labeler service

The live labeler account is `uspol.puppykitty.racing` (`did:plc:xwq5acimyyzd5pv4iklkez5a`), with endpoint `https://uspol.puppykitty.racing`. It runs in Docker on `littleguy` with CUDA 13.0 (`provider=cuda`). Its DID document has an `#atproto_labeler` service and an `#atproto_label` signing key; `labeler/label-definition.json` is published as `app.bsky.labeler.service/self`. The endpoint serves `/.well-known/atproto-did` for handle verification.

The service uses `@atcute/labeler` for signing and sequence management and `@atcute/xrpc-server` + `@atcute/xrpc-server-node` for the XRPC routes. `labeler/src/store.ts` is a SQLite `LabelStore`; `labeler/src/service.ts` wires the router and the WebSocket adapter.

Verified locally and through the public tunnel:

- `queryLabels` returns signed labels with `src`, `cts`, `ver` and `sig`.
- `subscribeLabels` streams CBOR label frames over WebSocket.
- Labels carry `src=did:plc:xwq5acimyyzd5pv4iklkez5a` and verify against the `#atproto_label` key in the DID document.
- The firehose consumer's cursor advances with the last event and survives restarts.

## Limitations

- The classifier is English-first (MiniLM-L6). Non-English posts can still be classified, but recall is not measured on them.
- The teacher is a 7B local model, not Jev. The TypeSafe API needs an account key; Laya was evaluated instead and its zero-shot ensemble is stored in the dataset. Teacher labels carry noise; the 120-post gold set is the honest evaluation.
- The labeler subscribes to `app.bsky.feed.post` creates only. Likes, follows and profile updates are ignored. At the observed 664 commits/s a full-firehose variant would still fit under the measured classifier throughput, but it is not implemented.
- Publishing to the appview requires a labeler account, PLC setup, a public HTTPS endpoint, and a published label definition. This run published all of them for `uspol.puppykitty.racing`.
- The public AppView still reports `handle.invalid` for the account while its cached actor index is stale. The PDS handle, DID document `alsoKnownAs`, DNS TXT, and HTTPS `/.well-known/atproto-did` all resolve to the new handle; the AppView revalidates handles on its hourly schedule or on an identity event.
- The threshold (0.85) trades recall for precision. `data/gold_predictions.jsonl` and `data/gold.jsonl` make that tradeoff measurable.
