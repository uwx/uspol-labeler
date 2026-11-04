import { readFileSync } from 'node:fs';
import WebSocket from 'ws';
import { decodeFirst } from '@atcute/cbor';
import { Secp256k1PrivateKey } from '@atcute/crypto';
import { Labeler } from '@atcute/labeler';
import type { GenericUri } from '@atcute/lexicons';
import { SqliteLabelStore } from '../labeler/src/store.js';

const base = process.env.TUNNEL_WS;
if (!base) throw new Error('set TUNNEL_WS to the public wss://.../xrpc URL');
const store = new SqliteLabelStore('data/test-label-service.db');
const key = await Secp256k1PrivateKey.importRaw(new Uint8Array(Buffer.from(readFileSync('data/test-signing-key.txt', 'utf8').trim(), 'hex')));
const labeler = new Labeler({ serviceDid: 'did:web:localhost', signingKey: key, store });

const messages: unknown[] = [];
const socket = new WebSocket(`${base.replace(/\/$/, '')}/com.atproto.label.subscribeLabels?cursor=0`);
socket.on('message', (data) => {
	let buf: Uint8Array = Buffer.isBuffer(data) ? data : new Uint8Array(data as ArrayBuffer);
	while (buf.length > 0) {
		const [frame, rest] = decodeFirst(buf);
		messages.push(frame);
		buf = rest;
	}
});
await new Promise<void>((resolve, reject) => {
	socket.once('open', () => resolve());
	socket.once('error', reject);
});
const uri = `at://did:plc:example/app.bsky.feed.post/ws-${Date.now()}` as GenericUri;
await labeler.applyLabels([{ uri: uri as GenericUri, value: 'us-politics' }]);
await new Promise((resolve) => setTimeout(resolve, 3000));
console.log(JSON.stringify({ messages: messages.length, last: messages[messages.length - 1] }, null, 2));
socket.close();
store.close();
