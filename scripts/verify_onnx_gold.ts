import { readFileSync } from 'node:fs';
import { Classifier } from '../labeler/src/classifier.js';

const classifier = await Classifier.create({ modelDir: process.env.MODEL_DIR ?? 'model/onnx', provider: (process.env.PROVIDER as 'dml' | 'cpu' | 'auto' | undefined) ?? 'dml' });
const gold = readFileSync('data/gold.jsonl', 'utf8').trim().split('\n').map((line) => JSON.parse(line) as { uri: string; text: string; label: number });
const predictions = new Map(
	readFileSync('data/gold_predictions.jsonl', 'utf8').trim().split('\n').map((line) => {
		const row = JSON.parse(line) as { uri: string; prob: number };
		return [row.uri, row.prob] as const;
	}),
);
const probs = await classifier.classify(gold.map((row) => row.text));
let maxDiff = 0;
let correct = 0;
for (let i = 0; i < gold.length; i++) {
	const prob = probs[i]!;
	maxDiff = Math.max(maxDiff, Math.abs(prob - (predictions.get(gold[i]!.uri) ?? 0)));
	if ((prob >= 0.5 ? 1 : 0) === gold[i]!.label) correct++;
}
console.log(JSON.stringify({
	provider: classifier.stats.provider,
	rows: gold.length,
	max_abs_diff_vs_pytorch: maxDiff,
	accuracy_at_0_5: correct / gold.length,
}, null, 2));
