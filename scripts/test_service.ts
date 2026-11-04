import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Secp256k1PrivateKeyExportable } from '@atcute/crypto';
import type { GenericUri } from '@atcute/lexicons';
import { createLabelService } from '../labeler/src/service.js';
import { SqliteLabelStore } from '../labeler/src/store.js';

const dir = await mkdtemp(join(tmpdir(), 'uspol-label-'));
const store = new SqliteLabelStore(join(dir, 'labels.db'));
const key = await Secp256k1PrivateKeyExportable.createKeypair();
const hex = await key.exportPrivateKey('rawHex');
const service = await createLabelService({
	did: 'did:example:uspol-test',
	signingKeyHex: hex,
	store,
	port: 4199,
	onError: (error) => console.error('service error', error),
});

const uri = 'at://did:plc:test/app.bsky.feed.post/abc';
await service.labeler.applyLabels([{ uri: uri as GenericUri, cid: 'bafyreitest', value: 'us-politics' }]);

const response = await fetch(`http://127.0.0.1:4199/xrpc/com.atproto.label.queryLabels?uriPatterns=${encodeURIComponent(uri)}&limit=10`);
const body = await response.json();
console.log(JSON.stringify({ status: response.status, labels: body.labels?.length, first: body.labels?.[0], cursor: body.cursor }, null, 2));

const bad = await fetch(`http://127.0.0.1:4199/xrpc/com.atproto.label.queryLabels?uriPatterns=${encodeURIComponent('at://did:plc:test/app.bsky.feed.post/_')}&limit=10`).then((r) => r.json());
console.log('prefix query', bad.labels.length);

await service.close();
store.close();
await rm(dir, { recursive: true, force: true });
