import { readFile } from 'node:fs/promises';
import { Classifier } from './classifier.js';

interface Row { text: string }

const args = new Map<string, string>();
for (let i = 2; i < process.argv.length; i++) {
	const arg = process.argv[i]!;
	if (!arg.startsWith('--')) continue;
	const [key = '', ...rest] = arg.slice(2).split('=');
	if (rest.length > 0) {
		args.set(key, rest.join('='));
	} else if (process.argv[i + 1] && !process.argv[i + 1]!.startsWith('--')) {
		args.set(key, process.argv[++i]!);
	} else {
		args.set(key, 'true');
	}
}

const MODEL_DIR = args.get('model') ?? process.env.MODEL_DIR ?? 'model/onnx';
const INPUT = args.get('input') ?? 'data/bench_posts.jsonl';
const BATCH_SIZE = Math.max(1, Number(args.get('batch') ?? 64));
const PROVIDER = (args.get('provider') as 'auto' | 'cpu' | 'dml' | undefined) ?? 'auto';
const LIVE_RATE = Number(args.get('live-rate') ?? 0);

const raw = (await readFile(INPUT, 'utf8')).trim().split('\n');
const texts = raw.map((line) => (JSON.parse(line) as Row).text).filter((text) => text.length > 0);
const classifier = await Classifier.create({ modelDir: MODEL_DIR, provider: PROVIDER });
console.log(JSON.stringify({ provider: classifier.stats.provider, loadMs: classifier.stats.loadMs, posts: texts.length, batch: BATCH_SIZE }));

// warmup: first call pays kernel compilation / DML graph capture
await classifier.classify(texts.slice(0, BATCH_SIZE));

function percentile(values: number[], p: number): number {
	const sorted = [...values].sort((a, b) => a - b);
	return sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))] ?? 0;
}

const latencies: number[] = [];
const start = performance.now();
let positive = 0;
for (let offset = 0; offset < texts.length; offset += BATCH_SIZE) {
	const batch = texts.slice(offset, offset + BATCH_SIZE);
	const t0 = performance.now();
	const probs = await classifier.classify(batch);
	latencies.push(performance.now() - t0);
	for (const prob of probs) if (prob >= 0.5) positive++;
}
const seconds = (performance.now() - start) / 1000;
const perSecond = texts.length / seconds;

const result = {
	mode: 'throughput',
	provider: classifier.stats.provider,
	posts: texts.length,
	seconds: Number(seconds.toFixed(2)),
	posts_per_second: Math.round(perSecond),
	batches: latencies.length,
	p50_ms: Number(percentile(latencies, 50).toFixed(2)),
	p95_ms: Number(percentile(latencies, 95).toFixed(2)),
	p99_ms: Number(percentile(latencies, 99).toFixed(2)),
	positive_at_0_5: positive,
};

if (LIVE_RATE > 0) {
	// Simulate a live firehose. Enqueue against a wall-clock target so Windows
	// timer resolution cannot make a fast consumer look slow; the meaningful
	// signal is the queue lag and the drain tail after the last enqueue.
	const queue = [...texts];
	let head = 0;
	let enqueued = 0;
	let drained = 0;
	let maxLag = 0;
	let lastEnqueueAt = performance.now();
	const simStart = performance.now();
	const targetInterval = 1000 / LIVE_RATE;

	const enqueue = async () => {
		while (enqueued < texts.length) {
			const target = Math.min(texts.length, Math.floor(((performance.now() - simStart) / 1000) * LIVE_RATE));
			while (enqueued < target) {
				enqueued++;
				lastEnqueueAt = performance.now();
				maxLag = Math.max(maxLag, enqueued - drained);
			}
			await new Promise((r) => setTimeout(r, Math.max(1, targetInterval / 4)));
		}
	};
	const drain = async () => {
		while (drained < texts.length) {
			const end = Math.min(head + BATCH_SIZE, enqueued);
			if (end <= head) {
				await new Promise((r) => setTimeout(r, 1));
				continue;
			}
			const batch = queue.slice(head, end);
			head = end;
			await classifier.classify(batch);
			drained += batch.length;
			maxLag = Math.max(maxLag, enqueued - drained);
		}
	};
	await Promise.all([enqueue(), drain()]);
	const simSeconds = (performance.now() - simStart) / 1000;
	const feedSeconds = (lastEnqueueAt - simStart) / 1000;
	result.mode = 'live-rate';
	Object.assign(result, {
		live_rate: LIVE_RATE,
		sim_seconds: Number(simSeconds.toFixed(2)),
		feed_seconds: Number(feedSeconds.toFixed(2)),
		effective_feed_rate: Number((texts.length / Math.max(feedSeconds, 0.001)).toFixed(1)),
		drain_tail_ms: Number((simSeconds - feedSeconds) * 1000).toFixed(1),
		max_queue_lag: maxLag,
		kept_up: maxLag <= BATCH_SIZE * 2,
	});
}

console.log(JSON.stringify(result, null, 2));
