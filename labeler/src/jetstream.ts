import { JetstreamSubscription } from '@atcute/jetstream';

export interface Post {
	uri: string;
	cid: string;
	did: string;
	rkey: string;
	text: string;
	createdAt: string;
	langs: string[];
	time_us: number;
}

export interface JetstreamOptions {
	/** base URL, e.g. wss://jetstream2.us-east.bsky.network (without /subscribe) */
	url: string;
	cursor?: number;
	onPost: (post: Post) => void;
	onEvent?: (timeUs: number) => void;
	onStatus?: (status: string) => void;
}

function isObject(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null;
}

export function parsePost(event: unknown): Post | null {
	if (!isObject(event) || event.kind !== 'commit') return null;
	const commit = event.commit;
	if (!isObject(commit) || commit.operation !== 'create' || commit.collection !== 'app.bsky.feed.post') return null;
	const record = commit.record;
	if (!isObject(record) || typeof record.text !== 'string') return null;
	const text = record.text.trim();
	if (text.length < 4) return null;
	const did = typeof event.did === 'string' ? event.did : null;
	const rkey = typeof commit.rkey === 'string' ? commit.rkey : null;
	const cid = typeof commit.cid === 'string' ? commit.cid : null;
	if (!did || !rkey || !cid) return null;
	return {
		uri: `at://${did}/${commit.collection}/${rkey}`,
		cid,
		did,
		rkey,
		text,
		createdAt: typeof record.createdAt === 'string' ? record.createdAt : '',
		langs: Array.isArray(record.langs) ? record.langs.filter((v): v is string => typeof v === 'string') : [],
		time_us: typeof event.time_us === 'number' ? event.time_us : 0,
	};
}

/**
 * Jetstream post consumer backed by @atcute/jetstream: reconnects, validates
 * events, and advances the cursor without bespoke socket management.
 */
export class Jetstream {
	#subscription: JetstreamSubscription;
	#stopped = false;
	#options: JetstreamOptions;
	lastEventUs = 0;
	events = 0;

	constructor(options: JetstreamOptions) {
		this.lastEventUs = options.cursor ?? 0;
		const url = options.url.replace(/\/subscribe\/?$/, '');
		this.#subscription = new JetstreamSubscription({
			url,
			cursor: options.cursor || undefined,
			wantedCollections: ['app.bsky.feed.post'],
			validateEvents: true,
			onConnectionOpen: () => options.onStatus?.('open'),
			onConnectionClose: () => options.onStatus?.('closed'),
			onConnectionError: (error) => options.onStatus?.(`connection error: ${String(error)}`),
			onError: (error) => options.onStatus?.(`event error: ${String(error)}`),
		});
		this.#options = options;
	}

	async start(): Promise<void> {
		for await (const event of this.#subscription) {
			if (this.#stopped) break;
			this.events++;
			if (typeof event.time_us === 'number') {
				this.lastEventUs = Math.max(this.lastEventUs, event.time_us);
				this.#options.onEvent?.(event.time_us);
			}
			const post = parsePost(event);
			if (post) this.#options.onPost(post);
		}
	}

	stop(): void {
		this.#stopped = true;
	}

	get cursor(): number {
		return this.lastEventUs;
	}
}
