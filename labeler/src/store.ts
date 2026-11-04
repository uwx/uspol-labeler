import Database from 'better-sqlite3';
import type { LabelEvent, LabelStore, SignedLabel } from '@atcute/labeler';
import type { Bytes, GenericUri } from '@atcute/lexicons';

interface LabelRow {
	seq: number;
	src: string;
	uri: string;
	cid: string | null;
	val: string;
	neg: number;
	cts: string;
	exp: string | null;
	ver: number | null;
	sig: string | null;
}

export interface LocalLabel {
	uri: string;
	cid: string;
	val: string;
	prob: number;
	cts: string;
}

/**
 * SQLite-backed LabelStore for @atcute/labeler, plus an unsigned path for
 * dry-runs that have no service DID or signing key yet.
 */
export class SqliteLabelStore implements LabelStore {
	#db: Database.Database;
	#insert: Database.Statement;
	#latest: Database.Statement;
	#has: Database.Statement;

	constructor(path: string) {
		this.#db = new Database(path);
		this.#db.pragma('journal_mode = WAL');
		this.#db.pragma('synchronous = NORMAL');
		this.#db.exec(`
			create table if not exists labels (
				seq integer primary key autoincrement,
				src text not null,
				uri text not null,
				cid text,
				val text not null,
				neg integer not null default 0,
				cts text not null,
				exp text,
				ver integer,
				sig text,
				prob real
			);
			create index if not exists labels_uri on labels(uri);
			create index if not exists labels_val on labels(val);
		`);
		this.#insert = this.#db.prepare(`
			insert into labels (src, uri, cid, val, neg, cts, exp, ver, sig, prob)
			values (@src, @uri, @cid, @val, @neg, @cts, @exp, @ver, @sig, @prob)
		`);
		this.#latest = this.#db.prepare('select coalesce(max(seq), 0) as latest from labels');
		this.#has = this.#db.prepare('select 1 from labels where uri = ? and val = ? limit 1');
	}

	async appendLabels(labels: SignedLabel[]): Promise<LabelEvent[]> {
		const insert = this.#db.transaction((batch: SignedLabel[]) =>
			batch.map((label) => {
				const result = this.#insert.run({
					src: label.src as string,
					uri: label.uri as string,
					cid: label.cid ?? null,
					val: label.val,
					neg: label.neg ? 1 : 0,
					cts: label.cts as string,
					exp: label.exp ?? null,
					ver: label.ver ?? 1,
					sig: label.sig.$bytes,
					prob: null,
				});
				return { seq: Number(result.lastInsertRowid), label };
			}),
		);
		return insert(labels).map(({ seq, label }) => ({ seq, labels: [label] }));
	}

	async getLatestSeq(): Promise<number | null> {
		const row = this.#latest.get() as { latest: number };
		return row.latest > 0 ? row.latest : null;
	}

	async listLabelEvents(options: { after?: number; limit?: number }): Promise<LabelEvent[]> {
		const rows = this.#db
			.prepare('select * from labels where seq > ? order by seq asc limit ?')
			.all(options.after ?? 0, options.limit ?? 1000) as LabelRow[];
		return rows.map((row) => ({ seq: row.seq, labels: [this.#toSigned(row)] }));
	}

	putLocal(label: LocalLabel, src = 'local'): void {
		this.#insert.run({
			src,
			uri: label.uri,
			cid: label.cid,
			val: label.val,
			neg: 0,
			cts: label.cts,
			exp: null,
			ver: 1,
			sig: null,
			prob: label.prob,
		});
	}

	hasLabel(uri: string, val: string): boolean {
		return this.#has.get(uri, val) !== undefined;
	}

	count(): number {
		const row = this.#db.prepare('select count(*) as n from labels').get() as { n: number };
		return row.n;
	}

	queryLabels(options: {
		uriPatterns: string[];
		sources?: string[];
		cursor?: number;
		limit: number;
	}): { labels: SignedLabel[]; cursor?: string } {
		const clauses: string[] = ['sig is not null'];
		const params: (string | number)[] = [];
		if (options.cursor) {
			clauses.push('seq > ?');
			params.push(options.cursor);
		}
		if (options.sources?.length) {
			clauses.push(`src in (${options.sources.map(() => '?').join(',')})`);
			params.push(...options.sources);
		}
		const patterns = options.uriPatterns.map((pattern) => {
			if (pattern.endsWith('_')) {
				return { sql: 'uri like ?', value: `${pattern.slice(0, -1)}%` };
			}
			return { sql: 'uri = ?', value: pattern };
		});
		if (patterns.length > 0) {
			clauses.push(`(${patterns.map((p) => p.sql).join(' or ')})`);
			params.push(...patterns.map((p) => p.value));
		}
		const rows = this.#db
			.prepare(`select * from labels where ${clauses.join(' and ')} order by seq asc limit ?`)
			.all(...params, options.limit) as LabelRow[];
		return {
			labels: rows.map((row) => this.#toSigned(row)),
			cursor: rows.length ? String(rows[rows.length - 1]!.seq) : options.cursor ? String(options.cursor) : undefined,
		};
	}

	#toSigned(row: LabelRow): SignedLabel {
		return {
			src: row.src as SignedLabel['src'],
			uri: row.uri as GenericUri,
			cid: row.cid ?? undefined,
			val: row.val,
			neg: row.neg ? true : undefined,
			cts: row.cts as SignedLabel['cts'],
			exp: row.exp ?? undefined,
			ver: row.ver ?? 1,
			sig: { $bytes: row.sig ?? '' } satisfies Bytes,
		};
	}

	close(): void {
		this.#db.close();
	}
}
