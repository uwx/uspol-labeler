# Labeler setup, training, and deployment guide

This guide covers the full operational path:

1. Turn a Bluesky account into a labeler and publish its service record.
2. Collect data and train your own US-politics classifier.
3. Deploy the labeler with Docker Compose.

The short version of the results and pipeline is in [`README.md`](README.md).

---

## 0. Prerequisites

| Tool | Use |
| --- | --- |
| Node.js 22+ | labeler, classifier, scripts |
| pnpm 10 | dependency management |
| Python 3.11 | training, Laya, ONNX export |
| Bun | Jetstream collector |
| Docker + Compose | deployment |
| A PDS account | labeler identity and repo records |
| An HTTPS endpoint | labeler XRPC, reachable by clients |

A GPU is optional. The Docker deployment uses CUDA on Linux; native Windows can use DirectML. The int8 CPU model remains available as a fallback.

```bash
git clone <repository>
cd uspol-labeler
pnpm install
cp .env.example .env
```

---

## 1. Set up the labeler account

### 1.1 Create the account and note its DID

Create the account on your PDS (the Bluesky app is fine), then resolve its DID:

```bash
curl "https://public.api.bsky.app/xrpc/com.atproto.identity.resolveHandle?handle=YOUR_HANDLE"
```

Save the returned `did:plc:...`. You will need:

- `DID` — the labeler account DID.
- `PDS` — the PDS base URL, e.g. `https://bsky.social` or your provider.
- Labeler handle and endpoint, e.g. `labels.example.com`.
- A signing key (`SIGNING_KEY`, 32-byte secp256k1 hex).

### 1.2 Prove control of the handle

The handle must resolve to the DID by either:

**DNS TXT**

```text
_atproto.labels.example.com.  TXT  "did=did:plc:..."
```

**HTTPS well-known**

```text
https://labels.example.com/.well-known/atproto-did
```

The response body is the DID, e.g. `did:plc:...`.

This repository serves that route when `DID` and `SIGNING_KEY` are configured:

```bash
curl https://labels.example.com/.well-known/atproto-did
```

If the endpoint is not live yet, use the DNS TXT method first.

### 1.3 Add the labeler service and signing key

The account's DID document needs:

- service `#atproto_labeler`, type `AtprotoLabeler`, endpoint `https://labels.example.com`;
- verification method `#atproto_label`, containing the public key for label signatures;
- `alsoKnownAs` set to the labeler handle.

The setup helper does these steps and publishes `app.bsky.labeler.service/self`:

```bash
export PDS=https://your-pds.example
export BSKY_IDENTIFIER=did:plc:...       # DID is safest; the handle also works
export BSKY_PASSWORD='...'
export LABELER_HANDLE=labels.example.com
export LABELER_ENDPOINT=https://labels.example.com

node scripts/setup-labeler-account.mjs status
node scripts/setup-labeler-account.mjs keygen
node scripts/setup-labeler-account.mjs request-token
```

The PLC token request sends an email to the address on the account. Then:

```bash
PLC_TOKEN='XXXXX-XXXXX' node scripts/setup-labeler-account.mjs configure
```

`configure`:

- updates the handle if it differs;
- submits a PLC operation adding `#atproto_labeler` and `#atproto_label`;
- publishes or updates `app.bsky.labeler.service/self` with `labeler/label-definition.json`;
- writes the private key to `data/uspolly-signing-key.txt`.

If you already have a key, put it in `SIGNING_KEY` before running `configure`. If the key file exists, the helper reuses it; it does not silently rotate keys.

Put the result in `.env`:

```dotenv
DID=did:plc:...
WELL_KNOWN_DID=did:plc:...
SIGNING_KEY=<64 hex characters>
PORT=4199
METRICS_PORT=4113
```

Never commit `.env` or the signing key. Both are ignored by `.gitignore`.

### 1.4 Start and test the labeler

```bash
# native (DirectML on Windows when available)
pnpm labeler
```

```bash
# query a label by URI
curl -G \
  --data-urlencode 'uriPatterns=at://did:plc:.../app.bsky.feed.post/...' \
  https://labels.example.com/xrpc/com.atproto.label.queryLabels
```

Expected: `labels` contains `src=did:plc:...`, `val=us-politics`, and `sig`.

For local-only testing, `pnpm serve:labels` starts a local test service on port 4199 using a generated test DID. It is not a production labeler.

### 1.5 AppView indexing

The AppView keeps its own handle index. After a handle change it can keep showing `handle.invalid` until its hourly handle revalidation or the next identity event, even when DNS TXT, `/.well-known/atproto-did`, the PDS, and the PLC document are correct. The labeler itself works by DID while that cache is stale.

---

## 2. Train your own model

The default model is a fine-tuned `sentence-transformers/all-MiniLM-L6-v2` binary classifier. You can replace the label definition, data, and model.

### 2.1 Collect firehose posts

The collector replays Jetstream from a cursor and keeps keyword candidates plus negative/random samples.

```bash
bun scripts/collect.ts \
  --hours=24 \
  --out=data/raw24 \
  --cand-every=4 \
  --neg-every=16 \
  --rand-every=500
```

Jetstream replay connections are capped; the collector reconnects with the last cursor. Then build the seed pool:

```bash
py -3.11 scripts/build_seed_pool.py
```

### 2.2 Label the seed pool

Use Laya's zero-shot decisions and/or a local teacher. Both labelers append to JSONL and skip URIs already present, so they are resumable.

```bash
USE_TF=0 .venv/Scripts/python.exe scripts/laya_label.py \
  --input data/seed_pool.jsonl \
  --output data/laya_labels.jsonl \
  --checkpoint laya

.venv/Scripts/python.exe scripts/teacher_label.py \
  --input data/seed_pool.jsonl \
  --output data/teacher_labels.jsonl \
  --model qwen2.5:7b
```

Manually label a small gold set and edit `data/gold.jsonl` with the same fields as the dataset (`uri`, `text`, `label`, `split: "gold"`). The gold set is the honest evaluation set; do not train on it.

Build the splits:

```bash
py -3.11 scripts/build_dataset.py
```

Output: `data/dataset.jsonl` with `train`, `val`, `test`, and `gold` splits.

### 2.3 Train and export

```bash
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install laya onnx onnxruntime
.venv/Scripts/python.exe scripts/train.py
.venv/Scripts/python.exe scripts/export_onnx.py
```

`scripts/export_onnx.py` writes:

```text
model/onnx/model.onnx         # fp32, dynamic batch
model/onnx/model.int8.onnx    # int8 CPU fallback
model/onnx/tokenizer.json
model/onnx/tokenizer_config.json
model/onnx/export.json
```

Evaluate the exported model against PyTorch gold predictions:

```bash
tsx scripts/verify_onnx_gold.ts
PROVIDER=cpu MODEL_FILE=model.int8.onnx tsx scripts/verify_onnx_gold.ts
```

### 2.4 Tune threshold and batch size

The labeler defaults to threshold `0.85`, which favors precision. At 0.85 the current gold metrics are P 0.931 / R 0.806; at 0.5 they are P 0.940 / R 0.940.

```bash
pnpm bench --input data/bench_posts.jsonl --batch 128
pnpm bench --input data/bench_posts.jsonl --batch 64 --live-rate 62
pnpm bench --input data/bench_posts.jsonl --batch 64 --provider cpu --model-file model.int8.onnx
```

Use `MODEL_DIR` and `MODEL_FILE` to point the labeler at another model:

```dotenv
MODEL_DIR=model/onnx
MODEL_FILE=model.onnx
```

For CPU-only deployments use `model.int8.onnx`. The measured CPU int8 throughput is ~90 posts/s, above the observed ~62 post creates/s.

---

## 3. Deploy with Docker Compose

The Compose deployment runs the labeler in a CUDA 13.0 container on Linux. It installs the CUDA execution-provider binaries for `onnxruntime-node` with `ONNXRUNTIME_NODE_INSTALL=cuda12` and runs the fp32 ONNX model on the GPU. The package still uses the historical `cuda12` flag name, but its current Linux binaries link CUDA 13, hence the CUDA 13.0.1 base image. The int8 CPU model is still available as a fallback.

### 3.1 Prepare files

Required host files:

```text
.env
data/
model/onnx/model.onnx
model/onnx/tokenizer.json
model/onnx/tokenizer_config.json
```

If you have not exported the model yet, train it with `scripts/train.py` and run `scripts/export_onnx.py`. The model files are generated artifacts and are not committed to the repository.

`.env` should contain at least:

```dotenv
DID=did:plc:...
WELL_KNOWN_DID=did:plc:...
SIGNING_KEY=<64 hex characters>
FIREHOSE_URL=wss://jetstream2.us-east.bsky.network
MODEL_DIR=model/onnx
MODEL_FILE=model.onnx
CURSOR_FILE=data/uspolly-cursor.txt
LABEL_DB=data/uspolly-labels.db
LABELS_NDJSON=data/uspolly-labels.ndjson
LABEL_VALUE=us-politics
THRESHOLD=0.85
BATCH_SIZE=64
PORT=4199
METRICS_PORT=4113
```

`docker-compose.yml` defaults to `PROVIDER=cuda` and `MODEL_FILE=model.onnx`. The host `./data` and `./model` are mounted into the container.

On Linux, make the bind-mounted data directory writable by the container's `ubuntu` user (UID 1000):

```bash
mkdir -p data
sudo chown -R 1000:1000 data
```

Verify GPU access before building:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04 nvidia-smi
```

### 3.2 Start

```bash
docker compose up -d --build
docker compose logs -f labeler
```

The startup log should show `"provider":"cuda"`.

Check health and metrics:

```bash
curl http://127.0.0.1:4113/health
curl http://127.0.0.1:4113/metrics
```

The labeler listens on `PORT` (default 4199) and serves:

- `/xrpc/com.atproto.label.queryLabels`
- `/xrpc/com.atproto.label.subscribeLabels`
- `/.well-known/atproto-did`

### 3.3 Public endpoint

Point your tunnel or reverse proxy at the published port:

```bash
cloudflared tunnel --url http://127.0.0.1:4199
```

Then set `LABELER_ENDPOINT` to that public HTTPS URL and update the PLC record:

```bash
LABELER_ENDPOINT=https://labels.example.com \
PLC_TOKEN='XXXXX-XXXXX' \
node scripts/setup-labeler-account.mjs configure
```

The Compose service does not create the tunnel. Keep the tunnel process running on the host, or terminate TLS at your own reverse proxy.

### 3.4 Restart, update, and back up

```bash
docker compose restart labeler
docker compose up -d --build labeler   # after source/model changes
docker compose down                    # stops the service, keeps bind mounts
```

Persist and back up:

- `data/uspolly-labels.db` — signed labels;
- `data/uspolly-cursor.txt` — resume position;
- `data/uspolly-labels.ndjson` — human-readable label log.

The consumer saves its cursor every five seconds and on `SIGTERM`; Compose uses a 15-second stop grace period.

### 3.5 Troubleshooting

| Symptom | Fix |
| --- | --- |
| `model.int8.onnx` not found | Run `scripts/export_onnx.py`, or set `MODEL_FILE=model.onnx` |
| permission denied on `data/` | `sudo chown -R 1000:1000 data` |
| container unhealthy | Check `docker compose logs labeler`; confirm `METRICS_PORT` matches the published port |
| `handle.invalid` in the AppView | PDS/DNS/PLC are correct; wait for AppView handle revalidation or trigger an identity event |
| labels fail to verify | The `#atproto_label` key in PLC must match `SIGNING_KEY`; do not rotate the key silently |
| CUDA not selected | Check `docker compose logs labeler`; verify `--gpus all`, the NVIDIA container toolkit, and the `onnxruntime-node` CUDA binaries |
| CPU cannot keep up | Use `LABELER_PROVIDER=cpu LABELER_MODEL_FILE=model.int8.onnx docker compose up -d` |

### 3.6 GPU note

The Compose image contains CUDA 13.0 and cuDNN from `nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04`. Override `CUDA_VERSION` at build time if your driver requires a different 13.x runtime. The `onnxruntime-node` CUDA execution-provider binaries are installed at image build time. Measured on the RTX 5060 Ti: 2,253 posts/s at batch 128.

Native Windows DirectML remains the alternative fast path when Docker is not used. For CPU-only operation in this image, override `LABELER_PROVIDER=cpu` and `LABELER_MODEL_FILE=model.int8.onnx`.
