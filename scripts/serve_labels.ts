/**
 * Standalone labeler-service harness for testing queryLabels/subscribeLabels
 * and tunneling with cloudflared before a real DID is configured.
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Secp256k1PrivateKeyExportable } from '@atcute/crypto';
import type { GenericUri } from '@atcute/lexicons';
import { createLabelService } from '../labeler/src/service.js';
import { SqliteLabelStore } from '../labeler/src/store.js';

const KEY_FILE = process.env.TEST_SIGNING_KEY_FILE ?? 'data/test-signing-key.txt';
const DB_FILE = process.env.LABEL_DB ?? 'data/test-label-service.db';
const PORT = Number(process.env.PORT ?? 4199);
const DID = process.env.DID ?? 'did:web:localhost';

let hex = process.env.SIGNING_KEY;
if (!hex) {
	if (existsSync(KEY_FILE)) hex = readFileSync(KEY_FILE, 'utf8').trim();
	else {
		const key = await Secp256k1PrivateKeyExportable.createKeypair();
		hex = await key.exportPrivateKey('rawHex');
		writeFileSync(KEY_FILE, hex);
	}
}

const store = new SqliteLabelStore(DB_FILE);
const service = await createLabelService({
	did: DID,
	signingKeyHex: hex,
	store,
	port: PORT,
	onError: (error) => console.error(JSON.stringify({ serviceError: String(error) })),
});

// one signed label so queryLabels/subscribeLabels have something to return
await service.labeler.applyLabels([{
	uri: 'at://did:plc:example/app.bsky.feed.post/test' as GenericUri,
	cid: 'bafyreitest',
	value: process.env.LABEL_VALUE ?? 'us-politics',
}]);

console.error(JSON.stringify({
	did: DID,
	port: PORT,
	db: DB_FILE,
	query: `http://127.0.0.1:${PORT}/xrpc/com.atproto.label.queryLabels?uriPatterns=${encodeURIComponent('at://did:plc:example/app.bsky.feed.post/_')}`,
}));

process.on('SIGINT', () => {
	void service.close().then(() => {
		store.close();
		process.exit(0);
	});
});
