import { JetstreamSubscription } from '@atcute/jetstream';

const seconds = Number(process.argv[2] ?? 15);
const url = process.env.FIREHOSE_URL?.replace(/\/subscribe\/?$/, '') ?? 'wss://jetstream2.us-east.bsky.network';

async function probe(label: string, wantedCollections?: string[]) {
	const subscription = new JetstreamSubscription({ url, wantedCollections, validateEvents: false });
	const start = Date.now();
	let events = 0;
	for await (const event of subscription) {
		if (event.kind !== 'commit') continue;
		if (wantedCollections?.length && event.commit.collection !== 'app.bsky.feed.post') continue;
		if (wantedCollections?.length && event.commit.operation !== 'create') continue;
		events++;
		if (Date.now() - start >= seconds * 1000) break;
	}
	const elapsed = (Date.now() - start) / 1000;
	return { label, events, seconds: elapsed, per_second: Number((events / elapsed).toFixed(1)) };
}

console.log(JSON.stringify({
	posts: await probe('post creates', ['app.bsky.feed.post']),
	all: await probe('all commits'),
}, null, 2));
