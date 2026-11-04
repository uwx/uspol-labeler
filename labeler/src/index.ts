import { createWriteStream } from 'node:fs';
import { readFile, rename, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { loadEnvFile } from 'node:process';
import type { GenericUri } from '@atcute/lexicons';
import { Classifier } from './classifier.js';
import { Jetstream, type Post } from './jetstream.js';
import { createLabelService } from './service.js';
import { SqliteLabelStore } from './store.js';

try {
	loadEnvFile();
} catch {
	// .env is optional; explicit environment variables still work.
}

const env = process.env;
const MODEL_DIR = env.MODEL_DIR ?? 'model/onnx';
const FIREHOSE_URL = env.FIREHOSE_URL ?? 'wss://jetstream2.us-east.bsky.network/subscribe';
const CURSOR_FILE = env.CURSOR_FILE ?? 'data/cursor.txt';
const LABEL_DB = env.LABEL_DB ?? 'data/labels.db';
const LABELS_NDJSON = env.LABELS_NDJSON ?? 'data/labels.ndjson';
const LABEL_VALUE = env.LABEL_VALUE ?? 'us-politics';
const THRESHOLD = Number(env.THRESHOLD ?? 0.85);
const BATCH_SIZE = Math.max(1, Number(env.BATCH_SIZE ?? 64));
const FLUSH_MS = Math.max(0, Number(env.FLUSH_MS ?? 25));
const MAX_QUEUE = Math.max(BATCH_SIZE, Number(env.MAX_QUEUE ?? 20_000));
const METRICS_PORT = Number(env.METRICS_PORT ?? 4103);
const PROVIDER = (env.PROVIDER as 'auto' | 'cpu' | 'dml' | undefined) ?? 'auto';
const DID = env.DID;
const SIGNING_KEY = env.SIGNING_KEY;
const WELL_KNOWN_DID = env.WELL_KNOWN_DID ?? DID;
const PORT = Number(env.PORT ?? 4102);

async function loadCursor(): Promise<number> {
	try {
		const raw = (await readFile(CURSOR_FILE, 'utf8')).trim();
		const value = Number(raw);
		return Number.isFinite(value) && value > 0 ? value : 0;
	} catch {
		return 0;
	}
}

async function saveCursor(cursor: number): Promise<void> {
	const tmp = `${CURSOR_FILE}.tmp`;
	await writeFile(tmp, String(cursor));
	await rename(tmp, CURSOR_FILE);
}

function delay(ms: number): Promise<void> {
	const { promise, resolve } = Promise.withResolvers<void>();
	setTimeout(resolve, ms);
	return promise;
}

const classifier = await Classifier.create({ modelDir: MODEL_DIR, provider: PROVIDER });
const store = new SqliteLabelStore(LABEL_DB);
const labelService = DID && SIGNING_KEY
	? await createLabelService({ did: DID, signingKeyHex: SIGNING_KEY, store, port: PORT, wellKnownDid: WELL_KNOWN_DID, onError: (error) => console.error(JSON.stringify({ serviceError: String(error) })) })
	: null;
const labelsOut = createWriteStream(LABELS_NDJSON, { flags: 'a' });

console.error(JSON.stringify({
	provider: classifier.stats.provider,
	loadMs: classifier.stats.loadMs,
	maxLength: classifier.stats.maxLength,
	threshold: THRESHOLD,
	batchSize: BATCH_SIZE,
}));

const queue: Post[] = [];
let head = 0;
let dropped = 0;
let labelled = 0;
let classified = 0;
const counters = { events: 0, posts: 0, started: Date.now() };
let lastEventUs = 0;
let lastReport = Date.now();
let lastRateSample = Date.now();
let rateEvents = 0;
let ratePosts = 0;
let rateClassified = 0;
let batchLatencies: number[] = [];
let flushTimer: NodeJS.Timeout | undefined;
let wakeWorker: (() => void) | null = null;
let stopped = false;

function notify(): void {
	if (wakeWorker) {
		wakeWorker();
		wakeWorker = null;
	}
}

const jetstream = new Jetstream({
	url: FIREHOSE_URL,
	cursor: await loadCursor(),
	onStatus: (status) => console.error(JSON.stringify({ status, cursor: jetstream.cursor })),
	onEvent: (timeUs) => {
		counters.events++;
		rateEvents++;
		lastEventUs = Math.max(lastEventUs, timeUs);
	},
	onPost: (post) => {
		counters.posts++;
		ratePosts++;
		queue.push(post);
		const overflow = queue.length - head - MAX_QUEUE;
		if (overflow > 0) {
			head += overflow;
			dropped += overflow;
		}
		clearTimeout(flushTimer);
		flushTimer = setTimeout(notify, FLUSH_MS);
		notify();
	},
});
void jetstream.start().catch((error) => {
	console.error(JSON.stringify({ jetstreamFatal: String(error) }));
	process.exitCode = 1;
});

function takeBatch(): Post[] {
	const end = Math.min(head + BATCH_SIZE, queue.length);
	const batch = queue.slice(head, end);
	head = end;
	if (head > 8192 && head * 2 > queue.length) {
		queue.splice(0, head);
		head = 0;
	}
	return batch;
}

async function worker(): Promise<void> {
	while (!stopped) {
		if (queue.length - head === 0) {
			const { promise, resolve } = Promise.withResolvers<void>();
			wakeWorker = resolve;
			setTimeout(resolve, 25);
			await promise;
			continue;
		}
		// wait a beat so a live stream fills a batch instead of running batch-of-one
		if (queue.length - head < BATCH_SIZE) await delay(Math.min(FLUSH_MS, 5));
		const batch = takeBatch();
		const started = performance.now();
		let probs: Float32Array;
		try {
			probs = await classifier.classify(batch.map((post) => post.text));
		} catch (error) {
			console.error(JSON.stringify({ error: String(error), batch: batch.length }));
			continue;
		}
		const elapsed = performance.now() - started;
		batchLatencies.push(elapsed);
		if (batchLatencies.length > 512) batchLatencies = batchLatencies.slice(-256);
		classified += batch.length;
		rateClassified += batch.length;
		const cts = new Date().toISOString();
		const positives: { post: Post; prob: number }[] = [];
		for (let i = 0; i < batch.length; i++) {
			const prob = probs[i] ?? 0;
			if (prob < THRESHOLD) continue;
			const post = batch[i]!;
			if (store.hasLabel(post.uri, LABEL_VALUE)) continue;
			positives.push({ post, prob });
		}
		if (labelService) {
			await labelService.labeler.applyLabels(
				positives.map(({ post }) => ({ uri: post.uri as GenericUri, cid: post.cid, value: LABEL_VALUE })),
				{ issuedAt: cts },
			);
		} else {
			for (const { post, prob } of positives) {
				store.putLocal({ uri: post.uri, cid: post.cid, val: LABEL_VALUE, prob, cts });
			}
		}
		for (const { post, prob } of positives) {
			labelsOut.write(JSON.stringify({
				uri: post.uri,
				cid: post.cid,
				did: post.did,
				text: post.text,
				val: LABEL_VALUE,
				prob,
				cts,
			}) + '\n');
			labelled++;
		}
	}
}

function percentile(values: number[], p: number): number {
	if (values.length === 0) return 0;
	const sorted = [...values].sort((a, b) => a - b);
	return sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))] ?? 0;
}

function report(): void {
	const now = Date.now();
	const seconds = Math.max(0.001, (now - lastRateSample) / 1000);
	const backlog = queue.length - head;
	const stats = {
		uptime_s: Math.round((now - counters.started) / 1000),
		events_per_s: Math.round(rateEvents / seconds),
		posts_per_s: Math.round(ratePosts / seconds),
		classified_per_s: Math.round(rateClassified / seconds),
		queue: backlog,
		dropped,
		labelled,
		labels: store.count(),
		threshold: THRESHOLD,
		p50_ms: Number(percentile(batchLatencies, 50).toFixed(2)),
		p99_ms: Number(percentile(batchLatencies, 99).toFixed(2)),
		lag_s: lastEventUs ? Number(((now * 1000 - lastEventUs) / 1e6).toFixed(1)) : null,
		provider: classifier.stats.provider,
	};
	console.error(JSON.stringify(stats));
	rateEvents = 0;
	ratePosts = 0;
	rateClassified = 0;
	lastRateSample = now;
}

setInterval(() => {
	if (Date.now() - lastReport >= 5000) {
		lastReport = Date.now();
		report();
	}
}, 1000).unref();

setInterval(() => {
	if (lastEventUs > 0) void saveCursor(lastEventUs);
}, 5000).unref();

const metrics = createServer((req, res) => {
	if (req.url === '/health') {
		res.writeHead(200, { 'content-type': 'application/json' });
		res.end(JSON.stringify({ ok: true, provider: classifier.stats.provider }));
		return;
	}
	const backlog = queue.length - head;
	res.writeHead(200, { 'content-type': 'text/plain; version=0.0.4' });
	res.end([
		'# TYPE uspol_events_total counter',
		`uspol_events_total ${counters.events}`,
		'# TYPE uspol_posts_total counter',
		`uspol_posts_total ${counters.posts}`,
		'# TYPE uspol_classified_total counter',
		`uspol_classified_total ${classified}`,
		'# TYPE uspol_labels_total counter',
		`uspol_labels_total ${labelled}`,
		'# TYPE uspol_queue gauge',
		`uspol_queue ${backlog}`,
		'# TYPE uspol_dropped_total counter',
		`uspol_dropped_total ${dropped}`,
		'# TYPE uspol_batch_p50_ms gauge',
		`uspol_batch_p50_ms ${percentile(batchLatencies, 50).toFixed(2)}`,
		'# TYPE uspol_batch_p99_ms gauge',
		`uspol_batch_p99_ms ${percentile(batchLatencies, 99).toFixed(2)}`,
		'',
	].join('\n'));
}).listen(METRICS_PORT, () => console.error(JSON.stringify({ metrics: `http://127.0.0.1:${METRICS_PORT}/metrics` })));

void worker().catch((error) => {
	console.error(JSON.stringify({ fatal: String(error) }));
	process.exitCode = 1;
});

let shuttingDown = false;
async function shutdown(): Promise<void> {
	if (shuttingDown) return;
	shuttingDown = true;
	stopped = true;
	notify();
	jetstream.stop();
	const deadline = Date.now() + 3000;
	while (queue.length - head > 0 && Date.now() < deadline) {
		await delay(50);
	}
	await saveCursor(jetstream.cursor);
	labelsOut.end();
	await labelService?.close();
	store.close();
	metrics.close();
	process.exit(process.exitCode ?? 0);
}

process.on('SIGINT', () => void shutdown());
process.on('SIGTERM', () => void shutdown());
