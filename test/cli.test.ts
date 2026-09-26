import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DiskStore } from '../src/disk-store.ts';
import { embedKey, keyToSegments } from '../src/key.ts';
import { gcCorrupt, listEntries, parseDuration, pruneOlderThan, statsFor } from '../src/cli.ts';

async function withTempDir(t: import('node:test').TestContext): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'embed-cache-cli-test-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  return dir;
}

test('parseDuration accepts a bare number as milliseconds', () => {
  assert.strictEqual(parseDuration('500'), 500);
});

test('parseDuration accepts suffixed units', () => {
  assert.strictEqual(parseDuration('500ms'), 500);
  assert.strictEqual(parseDuration('30s'), 30_000);
  assert.strictEqual(parseDuration('10m'), 600_000);
  assert.strictEqual(parseDuration('2h'), 7_200_000);
  assert.strictEqual(parseDuration('7d'), 604_800_000);
});

test('parseDuration rejects garbage', () => {
  assert.throws(() => parseDuration('soon'));
  assert.throws(() => parseDuration('7 days'));
});

test('listEntries and statsFor report every file with its size', async (t) => {
  const dir = await withTempDir(t);
  const store = new DiskStore(dir);
  await store.set(embedKey('ns', 'model', 'one'), Float32Array.from([1, 2]));
  await store.set(embedKey('ns', 'model', 'two'), Float32Array.from([1, 2, 3, 4]));

  const entries = await listEntries(dir);
  assert.strictEqual(entries.length, 2);
  assert.ok(entries.every((e) => e.size > 0));

  const stats = await statsFor(dir);
  assert.strictEqual(stats.count, 2);
  assert.strictEqual(
    stats.bytes,
    entries.reduce((sum, e) => sum + e.size, 0),
  );
  assert.ok(stats.oldestMtimeMs !== undefined && stats.newestMtimeMs !== undefined);
});

test('statsFor on an empty store reports zero entries and no timestamps', async (t) => {
  const dir = await withTempDir(t);
  const stats = await statsFor(dir);
  assert.deepStrictEqual(stats, { count: 0, bytes: 0 });
});

test('pruneOlderThan removes only files past the cutoff', async (t) => {
  const dir = await withTempDir(t);
  const store = new DiskStore(dir);
  const oldKey = embedKey('ns', 'model', 'old');
  const freshKey = embedKey('ns', 'model', 'fresh');
  await store.set(oldKey, Float32Array.from([1]));
  await store.set(freshKey, Float32Array.from([2]));

  const [a, b, rest] = keyToSegments(oldKey);
  const oldPath = join(dir, a, b, `${rest}.vec`);
  const longAgo = new Date(Date.now() - 1_000_000);
  await utimes(oldPath, longAgo, longAgo);

  const removed = await pruneOlderThan(dir, 500_000);
  assert.strictEqual(removed, 1);
  assert.strictEqual(await store.get(oldKey), undefined);
  assert.notStrictEqual(await store.get(freshKey), undefined);
});

test('gcCorrupt removes a file that fails its checksum and leaves good ones alone', async (t) => {
  const dir = await withTempDir(t);
  const store = new DiskStore(dir);
  const goodKey = embedKey('ns', 'model', 'good');
  const badKey = embedKey('ns', 'model', 'bad');
  await store.set(goodKey, Float32Array.from([1, 2, 3]));
  await store.set(badKey, Float32Array.from([4, 5, 6]));

  const [a, b, rest] = keyToSegments(badKey);
  await writeFile(join(dir, a, b, `${rest}.vec`), Buffer.from('not a vector file'));

  const result = await gcCorrupt(dir);
  assert.strictEqual(result.checked, 2);
  assert.strictEqual(result.removed, 1);
  assert.notStrictEqual(await store.get(goodKey), undefined);
  assert.strictEqual(await store.get(badKey), undefined);
});
