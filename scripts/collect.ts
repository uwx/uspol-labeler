#!/usr/bin/env bun
/**
 * Collect app.bsky.feed.post creates from a Jetstream replay cursor.
 *
 * Writes three JSONL files:
 *   candidates.jsonl  posts matching the US-politics keyword net (enriched positives)
 *   negatives.jsonl   deterministic 1/8 sample of everything else
 *   random.jsonl      deterministic 1/200 sample of every post (unbiased eval pool)
 *
 * The keyword net is intentionally generous: Laya makes the actual decision.
 */
import { createWriteStream } from 'node:fs';
import { mkdir } from 'node:fs/promises';

const args = new Map<string, string>();
for (const arg of process.argv.slice(2)) {
	const [k = '', ...rest] = arg.replace(/^--/, '').split('=');
	args.set(k, rest.join('=') || 'true');
}

const HOST = args.get('host') ?? 'jetstream2.us-east.bsky.network';
const HOURS = Number(args.get('hours') ?? 24);
const OUT = args.get('out') ?? 'data/raw';
const NEG_EVERY = Number(args.get('neg-every') ?? 8);
const CAND_EVERY = Number(args.get('cand-every') ?? 1);
const RAND_EVERY = Number(args.get('rand-every') ?? 200);
const MAX_POSTS = Number(args.get('max-posts') ?? 0);
const START_CURSOR = args.get('cursor') ? Number(args.get('cursor')) : Date.now() * 1000 - HOURS * 3600_000_000;

if (!args.get('cursor')) {
	// stay just inside the public instance's retention window
	console.error(`starting ${HOURS}h back at ${new Date(START_CURSOR / 1000).toISOString()}`);
}

await mkdir(OUT, { recursive: true });

const files = {
	candidates: createWriteStream(`${OUT}/candidates.jsonl`, { flags: 'w' }),
	negatives: createWriteStream(`${OUT}/negatives.jsonl`, { flags: 'w' }),
	random: createWriteStream(`${OUT}/random.jsonl`, { flags: 'w' }),
};

// US politics keyword net. Broad on purpose: false negatives here are lost forever,
// false positives are cheap because Laya re-checks each candidate.
const KEYWORD_ALTERNATIVES = [
	// people (unambiguous US political figures)
	'trump', 'biden', 'obama', 'harris', 'walz', 'vance', 'pelosi', 'schumer', 'mcconnell',
	'desantis', 'newsom', 'buttigieg', 'whitmer', 'mamdani', 'rfk', 'aoc', 'tlaib',
	// 2026 cycle: winners, losers and statewide candidates
	'talarico', 'paxton', 'hinojosa', 'allred', 'cornyn', 'ossoff', 'warnock', 'kemp',
	'fetterm[ae]n', 'mccormick', 'slotkin', 'jeffries', 'thune', 'stefanik', 'schiff',
	'porter', 'swalwell', 'khanna', 'kounalakis', 'villaraigosa', 'bonta', 'kotek',
	'ricketts', 'kreiss-tomkins', 'pritzker', 'bailey', 'schimel', 'crawford', 'kaul',
	'peltola', 'hickenlooper', 'coons', 'risch', 'stratton', 'turek', 'hinson', 'letlow',
	'markey', 'deaton', 'el-sayed', 'flanagan', 'tafoya', 'colom', 'hyde-smith', 'bankhead',
	'osborn', 'pappas', 'sununu', 'lujan', 'husted', 'merkley', 'mckay', 'bradshaw', 'hagerty',
	'mizusawa', 'hageman', 'fetty anderson', 'moody', 'sullivan', 'moore capito',
	'senator-elect', 'governor-elect', 'president-elect', 'election results', 'called the race',
	'blue tsunami', 'red tsunami', 'wave election', 'ballotpedia',
	// executive branch / agencies
	'hegseth', 'noem', 'bondi', 'patel', 'gabbard', 'bessent', 'lutnick', 'mcmahon',
	'burgum', 'wright', 'rollins', 'turner', 'zeldin', 'rubio', 'wiles', 'miller',

	// institutions / offices
	'white house', 'congress', 'senate', 'house of representatives', 'supreme court',
	'justice department', 'pentagon', 'state department', 'federal reserve',
	// politics
	'politics', 'political', 'politician', 'election', 'midterms?', 'ballot', 'voters?',
	'vote', 'votes', 'voted', 'voting', 'electoral', 'electors?', 'caucus', 'president',
	'administration', 'oval office', 'west wing', 'government', 'inaugurat', 'impeach', 'filibuster',
	'gerrymand', 'legislat', 'statute', 'executive order', 'veto', 'shutdown', 'debt ceiling',
	'appropriat', 'federal budget', 'gop', 'maga', 'republicans?', 'democrats?', 'liberals?',
	'conservatives?', 'progressives?', 'libertarians?', 'socialists?', 'far right', 'far left',
	'alt-right', 'antifa', 'presidency', 'presidential', 'senators?', 'representatives?',
	'governors?', 'congressm[ae]n', 'congresswom[ae]n', 'attorney general', 'lawmakers?',
	'capitol', 'washington dc',
	// policy areas commonly framed politically
	'abortion', 'pro-life', 'pro-choice', 'roe v', 'dobbs', 'immigration', 'border crisis',
	'tariffs?', 'trade war', 'medicare', 'medicaid', 'social security', 'obamacare', 'aca',
	'gun control', 'second amendment', 'civil rights', 'voting rights', 'gerrymander',
	'climate change', 'green new deal', 'dei', 'trans rights', 'gay marriage',
	'policy', 'policies', 'regulat', 'campaign', 'protest', 'polling', 'town hall',
	'ballot measure', 'voter fraud', 'electoral fraud', 'dark money', 'super pac', 'lobbying',
	'federal agency', 'federal government', 'government shutdown', 'state legislature',
	'school choice', 'book ban', 'crt', 'affirmative action', 'title ix',
	'ukraine aid', 'israel aid', 'nato', 'foreign policy', 'war powers',
	// political events / media
	'january 6', 'jan 6', 'insurrection', 'coup', 'tea party', 'red wave', 'blue wave',
	'swing state', 'battleground state', 'electoral college', 'popular vote', 'exit poll',
	'approval rating', 'debate night', 'fox news', 'msnbc', 'cnn', 'newsmax', 'oann',
	'breitbart', 'daily wire', 'huffpost', 'politico', 'the hill', 'axios', 'associated press',
	// US-specific
	'american politics', 'us politics', 'u\\.s\\. politics', 'united states', 'america first',
	'declaration of independence', 'first amendment', 'fourth amendment',
	'fourteenth amendment', 'electoral count', 'federal court', 'supreme court justice',
	'speaker of the house', 'majority leader', 'minority leader',
];

// Word boundaries matter: "cia" used to match "social", "ice" matched "nice", and
// "irs" matched "first". Those tokens are gone entirely.
const KEYWORDS = new RegExp(
	`(?<![\\p{L}\\p{N}])(?:${KEYWORD_ALTERNATIVES.join('|')})(?![\\p{L}\\p{N}])`,
	'iu',
);

interface Post {
	uri: string;
	cid: string;
	did: string;
	rkey: string;
	createdAt: string;
	langs: string[];
	text: string;
	alt?: string;
	tags?: string[];
	external?: { uri: string; title: string; description: string };
	quote?: { uri: string; cid: string };
	reply?: { parent: string; root: string };
	time_us: number;
}

interface RawFacetFeature { $type?: string; tag?: string }
interface RawFacet { features?: RawFacetFeature[] }
interface RawImage { alt?: string }
interface RawExternal { uri?: string; title?: string; description?: string }
interface RawEmbed {
	external?: RawExternal;
	record?: { uri?: string; cid?: string };
	images?: RawImage[];
	video?: { alt?: string };
}
interface RawRecord {
	$type?: string;
	text?: string;
	createdAt?: string;
	langs?: string[];
	facets?: RawFacet[];
	embed?: RawEmbed;
	reply?: { parent?: { uri?: string }; root?: { uri?: string } };
}

function isObject(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null;
}

function asString(value: unknown): string | undefined {
	return typeof value === 'string' ? value : undefined;
}

function postFromEvent(event: unknown): Post | null {
	if (!isObject(event) || event.kind !== 'commit') return null;
	const commit = event.commit;
	if (!isObject(commit) || commit.operation !== 'create' || commit.collection !== 'app.bsky.feed.post') return null;
	const record = commit.record;
	if (!isObject(record) || typeof record.text !== 'string') return null;
	const text = record.text.trim();
	if (text.length < 4) return null;

	const did = asString(event.did);
	const rkey = asString(commit.rkey);
	const cid = asString(commit.cid);
	if (!did || !rkey || !cid) return null;

	const raw = record as RawRecord;
	const embed = raw.embed ?? {};
	const external = embed.external
		? { uri: embed.external.uri ?? '', title: embed.external.title ?? '', description: embed.external.description ?? '' }
		: undefined;
	const alt = [
		...(embed.images ?? []).map((image) => image?.alt ?? ''),
		...(embed.video?.alt ? [embed.video.alt] : []),
	].filter(Boolean).join(' ');
	const tags = (raw.facets ?? [])
		.flatMap((facet) => facet?.features ?? [])
		.filter((feature) => feature?.$type === 'app.bsky.richtext.facet#tag')
		.map((feature) => feature.tag)
		.filter((tag): tag is string => Boolean(tag));

	return {
		uri: `at://${did}/${commit.collection}/${rkey}`,
		cid,
		did,
		rkey,
		createdAt: raw.createdAt ?? '',
		langs: raw.langs ?? [],
		text,
		alt: alt || undefined,
		tags: tags.length ? tags : undefined,
		external,
		quote: embed.record?.uri ? { uri: embed.record.uri, cid: embed.record.cid ?? '' } : undefined,
		reply: raw.reply?.parent?.uri
			? { parent: raw.reply.parent.uri, root: raw.reply.root?.uri ?? '' }
			: undefined,
		time_us: typeof event.time_us === 'number' ? event.time_us : 0,
	};
}

function surfaceOf(post: Post): string {
	return [
		post.text,
		post.alt,
		post.tags?.join(' '),
		post.external?.title,
		post.external?.description,
	].filter(Boolean).join('\n');
}

function fnv1a(s: string): number {
	let h = 0x811c9dc5;
	for (let i = 0; i < s.length; i++) {
		h ^= s.charCodeAt(i);
		h = Math.imul(h, 0x01000193);
	}
	return h >>> 0;
}

const counts = { posts: 0, candidates: 0, negatives: 0, random: 0, skipped: 0, reconnects: 0 };
const start = Date.now();
const targetUs = HOURS > 0 ? Date.now() * 1000 - 60_000_000 : Number.POSITIVE_INFINITY;
let closed = false;
let lastEventUs = START_CURSOR;
let ws: WebSocket | null = null;

function write(stream: NodeJS.WritableStream, obj: unknown): void {
	stream.write(JSON.stringify(obj) + '\n');
}

function connect(cursor: number): void {
	ws = new WebSocket(
		`wss://${HOST}/subscribe?wantedCollections=app.bsky.feed.post&cursor=${cursor}`,
	);
	ws.addEventListener('message', (ev) => {
		let event: unknown;
		try { event = JSON.parse(typeof ev.data === 'string' ? ev.data : ''); } catch { counts.skipped++; return; }
		if (isObject(event) && typeof event.time_us === 'number') lastEventUs = Math.max(lastEventUs, event.time_us);
		const post = postFromEvent(event);
		if (!post) return;
		counts.posts++;
		const h = fnv1a(post.uri);

		if (KEYWORDS.test(surfaceOf(post))) {
			counts.candidates++;
			if (h % CAND_EVERY === 0) write(files.candidates, post);
		} else if (h % NEG_EVERY === 0) {
			counts.negatives++;
			write(files.negatives, post);
		}
		if (h % RAND_EVERY === 0) {
			counts.random++;
			write(files.random, post);
		}

		if (counts.posts % 25_000 === 0) {
			const secs = (Date.now() - start) / 1000;
			console.error(JSON.stringify({
				...counts,
				rate: Math.round(counts.posts / secs),
				cursorISO: new Date(lastEventUs / 1000).toISOString(),
			}));
		}
		if (MAX_POSTS && counts.posts >= MAX_POSTS) void finish();
	});
	ws.addEventListener('error', (e) => { console.error('ws error', e); process.exitCode = 1; });
	ws.addEventListener('close', (e) => {
		console.error('ws closed', e.code, e.reason, 'at', new Date(lastEventUs / 1000).toISOString());
		void reconnectOrFinish();
	});
}

async function reconnectOrFinish(): Promise<void> {
	if (closed) return;
	// The public instances cap a replay connection at roughly half a million
	// events; resume from the last event until we reach the target time.
	if (lastEventUs + 1_000_000 < targetUs && counts.reconnects < 40) {
		counts.reconnects++;
		setTimeout(() => connect(lastEventUs + 1), 250);
		return;
	}
	await finish();
}

async function finish(): Promise<void> {
	if (closed) return;
	closed = true;
	try { ws?.close(); } catch {}
	for (const f of Object.values(files)) {
		await new Promise<void>((resolve) => f.end(resolve));
	}
	console.log(JSON.stringify({ ...counts, seconds: (Date.now() - start) / 1000 }, null, 2));
	process.exit(process.exitCode ?? 0);
}

connect(START_CURSOR);

// replay drains and closes; a live cursor never does, so cap wall time too
if (HOURS > 0) setTimeout(() => void finish(), 20 * 60_000);
