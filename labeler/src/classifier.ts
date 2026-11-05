import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { Tokenizer } from '@huggingface/tokenizers';
import * as ort from 'onnxruntime-node';

export interface ClassifierOptions {
	modelDir: string;
	/** ONNX file inside modelDir. Defaults to model.onnx. */
	modelFile?: string;
	maxLength?: number;
	provider?: 'cuda' | 'dml' | 'cpu' | 'auto';
	threads?: number;
}

export interface ClassifierStats {
	provider: string;
	maxLength: number;
	loadMs: number;
}

const SEP = '[SEP]';

/**
 * MiniLM classifier over ONNX Runtime.
 *
 * Batches are tokenized to a fixed sequence length so DirectML never has to
 * re-specialize the graph for a new shape mid-stream.
 */
export class Classifier {
	readonly stats: ClassifierStats;
	#tokenizer: Tokenizer;
	#session: ort.InferenceSession;
	#maxLength: number;
	#inputNames: Set<string>;
	#sepId: number;
	#padId = 0;

	private constructor(tokenizer: Tokenizer, session: ort.InferenceSession, maxLength: number, stats: ClassifierStats) {
		this.#tokenizer = tokenizer;
		this.#session = session;
		this.#maxLength = maxLength;
		this.#inputNames = new Set(session.inputNames);
		this.#sepId = tokenizer.token_to_id(SEP) ?? 102;
		this.#padId = tokenizer.token_to_id('[PAD]') ?? 0;
		this.stats = stats;
	}

	static async create(options: ClassifierOptions): Promise<Classifier> {
		const started = Date.now();
		const maxLength = options.maxLength ?? 128;
		const tokenizerJson = JSON.parse(await readFile(join(options.modelDir, 'tokenizer.json'), 'utf8'));
		const tokenizerConfig = JSON.parse(await readFile(join(options.modelDir, 'tokenizer_config.json'), 'utf8'));
		const tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);

		const requested = options.provider ?? 'auto';
		const modelPath = join(options.modelDir, options.modelFile ?? 'model.onnx');
		const providers: string[] =
			requested === 'cpu' ? ['cpu']
			: requested === 'dml' ? ['dml']
			: requested === 'cuda' ? ['cuda']
			: process.platform === 'win32' ? ['dml', 'cpu']
			: ['cuda', 'cpu'];
		let session: ort.InferenceSession | null = null;
		let used = '';
		let lastError: unknown = null;
		for (const provider of providers) {
			try {
				session = await ort.InferenceSession.create(modelPath, {
					executionProviders: [provider],
					intraOpNumThreads: options.threads ?? 1,
					interOpNumThreads: 1,
					logSeverityLevel: 3,
				});
				used = provider;
				break;
			} catch (error) {
				lastError = error;
			}
		}
		if (!session) throw lastError ?? new Error('could not create ONNX session');
		if ((requested === 'dml' || requested === 'cuda') && used !== requested) {
			throw lastError ?? new Error(`${requested} requested but unavailable`);
		}

		return new Classifier(tokenizer, session, maxLength, {
			provider: used,
			maxLength,
			loadMs: Date.now() - started,
		});
	}

	/** Returns P(us_politics) for each text, preserving order. */
	async classify(texts: string[]): Promise<Float32Array> {
		if (texts.length === 0) return new Float32Array(0);
		const batch = texts.length;
		const seq = this.#maxLength;
		const inputIds = new BigInt64Array(batch * seq);
		const attentionMask = new BigInt64Array(batch * seq);
		const tokenTypes = new BigInt64Array(batch * seq);

		for (let row = 0; row < batch; row++) {
			const encoded = this.#tokenizer.encode(texts[row]!, { add_special_tokens: true, return_token_type_ids: true });
			let ids = encoded.ids;
			if (ids.length > seq) {
				// keep [CLS] and [SEP]; drop from the middle
				ids = [ids[0]!, ...ids.slice(1, seq - 1), this.#sepId];
			}
			const offset = row * seq;
			for (let i = 0; i < seq; i++) {
				inputIds[offset + i] = BigInt(i < ids.length ? ids[i]! : this.#padId);
				attentionMask[offset + i] = BigInt(i < ids.length ? 1 : 0);
				tokenTypes[offset + i] = BigInt(encoded.token_type_ids?.[i] ?? 0);
			}
		}

		const feeds: Record<string, ort.Tensor> = {};
		if (this.#inputNames.has('input_ids')) feeds.input_ids = new ort.Tensor('int64', inputIds, [batch, seq]);
		if (this.#inputNames.has('attention_mask')) feeds.attention_mask = new ort.Tensor('int64', attentionMask, [batch, seq]);
		if (this.#inputNames.has('token_type_ids')) feeds.token_type_ids = new ort.Tensor('int64', tokenTypes, [batch, seq]);

		const output = await this.#session.run(feeds);
		const logits = output.logits ?? output[Object.keys(output)[0]!]!;
		const data = logits.data as Float32Array;
		const dims = logits.dims;
		const classes = dims[dims.length - 1] ?? 2;
		const probs = new Float32Array(batch);
		for (let row = 0; row < batch; row++) {
			const base = row * classes;
			const negative = data[base] ?? 0;
			const positive = data[base + 1] ?? 0;
			const max = Math.max(negative, positive);
			const eNeg = Math.exp(negative - max);
			const ePos = Math.exp(positive - max);
			probs[row] = ePos / (eNeg + ePos);
		}
		return probs;
	}
}
