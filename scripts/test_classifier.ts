import { Classifier } from '../labeler/src/classifier.js';

const classifier = await Classifier.create({ modelDir: process.env.MODEL_DIR ?? 'model/onnx', provider: (process.env.PROVIDER as 'dml' | 'cpu' | 'auto' | undefined) ?? 'dml' });
console.log(classifier.stats);

for (const n of [1, 8, 32, 64, 128, 256, 512]) {
	const texts = Array(n).fill('The Senate voted to advance the budget bill after Trump threatened a veto.');
	const start = performance.now();
	const probs = await classifier.classify(texts);
	const ms = performance.now() - start;
	console.log(JSON.stringify({ n, ms: Number(ms.toFixed(2)), per_s: Math.round((n / ms) * 1000), p: Number((probs[0] ?? 0).toFixed(4)) }));
}
