import { createServer, type IncomingMessage, type Server } from 'node:http';
import { Readable } from 'node:stream';
import { ComAtprotoLabelQueryLabels, ComAtprotoLabelSubscribeLabels } from '@atcute/atproto';
import { Secp256k1PrivateKey } from '@atcute/crypto';
import { FutureCursorError, Labeler } from '@atcute/labeler';
import type { ComAtprotoLabelDefs } from '@atcute/atproto';
import { XRPCRouter, XRPCSubscriptionError, json } from '@atcute/xrpc-server';
import { createNodeWebSocket } from '@atcute/xrpc-server-node';
import type { SqliteLabelStore } from './store.js';

export interface LabelServiceOptions {
	did: string;
	signingKeyHex: string;
	store: SqliteLabelStore;
	port: number;
	/** DID returned at /.well-known/atproto-did for handle verification. Defaults to `did`. */
	wellKnownDid?: string;
	onError?: (error: unknown) => void;
}

export interface LabelService {
	labeler: Labeler;
	server: Server;
	close: () => Promise<void>;
}

function requestFromNode(req: IncomingMessage): Request {
	const host = req.headers.host ?? '127.0.0.1';
	const url = new URL(req.url ?? '/', `http://${host}`);
	const method = req.method ?? 'GET';
	const hasBody = method !== 'GET' && method !== 'HEAD';
	return new Request(url, {
		method,
		headers: req.headers as Record<string, string>,
		body: hasBody ? (Readable.toWeb(req) as ReadableStream<Uint8Array>) : null,
		duplex: hasBody ? 'half' : undefined,
	} as RequestInit);
}

export async function createLabelService(options: LabelServiceOptions): Promise<LabelService> {
	const key = await Secp256k1PrivateKey.importRaw(new Uint8Array(Buffer.from(options.signingKeyHex, 'hex')));
	const labeler = new Labeler({
		serviceDid: options.did as ComAtprotoLabelDefs.Label['src'],
		signingKey: key,
		store: options.store,
	});
	const { adapter, injectWebSocket } = createNodeWebSocket();
	const router = new XRPCRouter({
		websocket: adapter,
		onError: ({ error }) => options.onError?.(error),
		onSocketError: ({ error }) => options.onError?.(error),
	});

	router.addQuery(ComAtprotoLabelQueryLabels, {
		handler: ({ params }) =>
			json(options.store.queryLabels({
				uriPatterns: params.uriPatterns,
				sources: params.sources,
				cursor: params.cursor ? Number(params.cursor) : undefined,
				limit: params.limit ?? 50,
			})),
	});

	router.addSubscription(ComAtprotoLabelSubscribeLabels, {
		async *handler({ params, signal }) {
			try {
				for await (const event of labeler.subscribeLabels({ cursor: params.cursor, signal })) {
					yield {
						$type: 'com.atproto.label.subscribeLabels#labels' as const,
						seq: event.seq,
						labels: event.labels,
					};
				}
			} catch (error) {
				if (error instanceof FutureCursorError) throw new XRPCSubscriptionError({ error: 'FutureCursor' });
				throw error;
			}
		},
	});

	const server = createServer((req, res) => {
		if (req.method === 'GET' && req.url === '/.well-known/atproto-did') {
			res.writeHead(200, { 'content-type': 'text/plain; charset=utf-8' });
			res.end(options.wellKnownDid ?? options.did);
			return;
		}
		if (!req.url?.startsWith('/xrpc/')) {
			res.writeHead(404, { 'content-type': 'application/json' });
			res.end(JSON.stringify({ error: 'NotFound' }));
			return;
		}
		void router
			.fetch(requestFromNode(req))
			.then(async (response) => {
				res.writeHead(response.status, Object.fromEntries(response.headers.entries()));
				if (response.body) {
					const reader = response.body.getReader();
					for (;;) {
						const { done, value } = await reader.read();
						if (done) break;
						res.write(value);
					}
				}
				res.end();
			})
			.catch((error) => {
				options.onError?.(error);
				if (!res.headersSent) res.writeHead(500, { 'content-type': 'application/json' });
				res.end(JSON.stringify({ error: 'InternalServerError' }));
			});
	});
	injectWebSocket(server, router);

	await new Promise<void>((resolve, reject) => {
		server.once('error', reject);
		server.listen(options.port, () => resolve());
	});
	console.error(JSON.stringify({ labelService: `http://127.0.0.1:${options.port}/xrpc`, did: options.did }));

	return {
		labeler,
		server,
		close: () =>
			new Promise<void>((resolve) => {
				server.close(() => resolve());
			}),
	};
}
