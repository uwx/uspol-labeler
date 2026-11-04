#!/usr/bin/env node
/**
 * One-off setup for the uspol labeler account.
 *
 * status        - show session, handle, DID, recommended DID credentials
 * request-token - request the PLC operation signature email
 * handle        - set the account handle via com.atproto.identity.updateHandle
 * record        - publish/update app.bsky.labeler.service
 * configure     - set handle, PLC labeler service + signing key, labeler record
 */
import { readFile, writeFile } from 'node:fs/promises';
import { Secp256k1PrivateKeyExportable } from '@atcute/crypto';

const PDS = process.env.PDS ?? 'https://eurosky.social';
const IDENTIFIER = process.env.BSKY_IDENTIFIER ?? 'did:plc:xwq5acimyyzd5pv4iklkez5a';
const PASSWORD = process.env.BSKY_PASSWORD;
const HANDLE = process.env.LABELER_HANDLE ?? 'uspol.puppykitty.racing';
const ENDPOINT = process.env.LABELER_ENDPOINT ?? 'https://uspol.puppykitty.racing';
const PLC_TOKEN = process.env.PLC_TOKEN;
const SIGNING_KEY = process.env.SIGNING_KEY;
const KEY_FILE = process.env.SIGNING_KEY_FILE ?? 'data/uspolly-signing-key.txt';
const action = process.argv[2] ?? 'status';

if (!PASSWORD) throw new Error('BSKY_PASSWORD is required');

async function xrpc(path, { method = 'GET', token, body } = {}) {
	const headers = {};
	if (token) headers.authorization = `Bearer ${token}`;
	if (body !== undefined) headers['content-type'] = 'application/json';
	const response = await fetch(`${PDS}/xrpc/${path}`, {
		method,
		headers,
		body: body === undefined ? undefined : JSON.stringify(body),
	});
	const text = await response.text();
	let data;
	try {
		data = text ? JSON.parse(text) : {};
	} catch {
		data = { raw: text };
	}
	if (!response.ok) {
		throw new Error(`${path} ${response.status}: ${text}`);
	}
	return data;
}

async function login() {
	return xrpc('com.atproto.server.createSession', {
		method: 'POST',
		body: { identifier: IDENTIFIER, password: PASSWORD },
	});
}

async function labelerRecord() {
	const definition = JSON.parse(await readFile('labeler/label-definition.json', 'utf8'));
	return {
		$type: 'app.bsky.labeler.service',
		policies: {
			labelValues: [definition.identifier],
			labelValueDefinitions: [definition],
		},
		createdAt: new Date().toISOString(),
	};
}

async function publishRecord(session) {
	const record = await labelerRecord();
	await xrpc('com.atproto.repo.putRecord', {
		method: 'POST',
		token: session.accessJwt,
		body: {
			repo: session.did,
			collection: 'app.bsky.labeler.service',
			rkey: 'self',
			record,
			validate: true,
		},
	});
	return record;
}

async function updateHandle(session) {
	await xrpc('com.atproto.identity.updateHandle', {
		method: 'POST',
		token: session.accessJwt,
		body: { handle: HANDLE },
	});
}

async function main() {
	const session = await login();
	if (action === 'status') {
		const full = await xrpc('com.atproto.server.getSession', { token: session.accessJwt });
		const recommended = await xrpc('com.atproto.identity.getRecommendedDidCredentials', {
			token: session.accessJwt,
		});
		let declaration = null;
		try {
			const existing = await xrpc(
				`com.atproto.repo.getRecord?repo=${encodeURIComponent(session.did)}&collection=app.bsky.labeler.service&rkey=self`,
				{ token: session.accessJwt },
			);
			declaration = existing.value ?? null;
		} catch {
			declaration = null;
		}
		console.log(JSON.stringify({
			did: session.did,
			handle: full.handle,
			email: full.email,
			emailConfirmed: full.emailConfirmed,
			recommended,
			hasDeclaration: Boolean(declaration),
			declaration,
		}, null, 2));
		return;
	}
	if (action === 'request-token') {
		await xrpc('com.atproto.identity.requestPlcOperationSignature', {
			method: 'POST',
			token: session.accessJwt,
		});
		const full = await xrpc('com.atproto.server.getSession', { token: session.accessJwt });
		console.log(JSON.stringify({ requested: true, email: full.email }));
		return;
	}
	if (action === 'handle') {
		await updateHandle(session);
		const full = await xrpc('com.atproto.server.getSession', { token: session.accessJwt });
		console.log(JSON.stringify({ handle: full.handle, did: full.did }));
		return;
	}
	if (action === 'record') {
		const record = await publishRecord(session);
		console.log(JSON.stringify({ published: true, policies: record.policies.labelValues }, null, 2));
		return;
	}
	if (action === 'confirm-email') {
		await xrpc('com.atproto.server.requestEmailConfirmation', {
			method: 'POST',
			token: session.accessJwt,
		});
		console.log(JSON.stringify({ requested: true }));
		return;
	}
	if (action === 'keygen') {
		const existing = await readFile(KEY_FILE, 'utf8').catch(() => null);
		const key = existing
			? await Secp256k1PrivateKeyExportable.importRaw(Buffer.from(existing.trim(), 'hex'))
			: await Secp256k1PrivateKeyExportable.createKeypair();
		if (!existing) {
			await writeFile(KEY_FILE, `${await key.exportPrivateKey('rawHex')}\n`, { mode: 0o600 });
		}
		console.log(JSON.stringify({
			publicKeyDid: await key.exportPublicKey('did'),
			signingKeyFile: KEY_FILE,
			created: !existing,
		}, null, 2));
		return;
	}
	if (action === 'configure') {
		if (!PLC_TOKEN) throw new Error('PLC_TOKEN is required');
		const current = await xrpc('com.atproto.server.getSession', { token: session.accessJwt });
		if (current.handle !== HANDLE) await updateHandle(session);

		const key = SIGNING_KEY
			? await Secp256k1PrivateKeyExportable.importRaw(Buffer.from(SIGNING_KEY, 'hex'))
			: await Secp256k1PrivateKeyExportable.createKeypair();
		const privateKeyHex = await key.exportPrivateKey('rawHex');
		const publicKeyDid = await key.exportPublicKey('did');
		await writeFile(KEY_FILE, `${privateKeyHex}\n`, { mode: 0o600 });

		const recommended = await xrpc('com.atproto.identity.getRecommendedDidCredentials', {
			token: session.accessJwt,
		});
		const operation = {
			verificationMethods: {
				...(recommended.verificationMethods ?? {}),
				atproto_label: publicKeyDid,
			},
			services: {
				...(recommended.services ?? {}),
				atproto_labeler: { type: 'AtprotoLabeler', endpoint: ENDPOINT },
			},
		};
		const signed = await xrpc('com.atproto.identity.signPlcOperation', {
			method: 'POST',
			token: session.accessJwt,
			body: { token: PLC_TOKEN, ...operation },
		});
		await xrpc('com.atproto.identity.submitPlcOperation', {
			method: 'POST',
			token: session.accessJwt,
			body: { operation: signed.operation },
		});
		const record = await publishRecord(session);
		console.log(JSON.stringify({
			did: session.did,
			handle: HANDLE,
			endpoint: ENDPOINT,
			publicKeyDid,
			signingKeyFile: KEY_FILE,
			labelValues: record.policies.labelValues,
			plcOperationSubmitted: true,
		}, null, 2));
		return;
	}
	throw new Error(`unknown action: ${action}`);
}

main().catch((error) => {
	console.error(String(error));
	process.exit(1);
});
