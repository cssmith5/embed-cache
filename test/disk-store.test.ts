import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DiskStore } from '../src/disk-store.ts';
import { embedKey, keyToSegments } from '../src/key.ts';

async function withTempDir(t: import('node:test').TestContext): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'embed-cache-test-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  return dir;
}

test('set/get round-trips a vector to disk', async (t) => {
  const dir = await withTempDir(t);
  const store = new DiskStore(dir);
  const key = embedKey('ns', 'model', 'text');
  const vector = Float32Array.from([1, 2, 3, 4]);

  await store.set(key, vector);
  const loaded = await store.get(key);
  assert.deepStrictEqual(loaded, vector);
});

test('get on a missing key returns undefined', async (t) => {
  const dir = await withTempDir(t);
  const store = new DiskStore(dir);
  const missing = await store.get(embedKey('ns', 'model', 'text'));
  assert.strictEqual(missing, undefined);
});

test('files are sharded under <key[0:2]>/<key[2:4]>/<key[4:]>.vec', async (t) => {
  const dir = await withTempDir(t);
  const store = new DiskStore(dir);
  const key = embedKey('ns', 'model', 'text');
  await store.set(key, Float32Array.from([1, 2]));

  const [a, b, rest] = keyToSegments(key);
  const path = join(dir, a, b, `${rest}.vec`);
  const buf = await readFile(path);
  assert.ok(buf.length > 0);
});

test('no shard directory holds more than one file per distinct key', async (t) => {
  const dir = await withTempDir(t);
  const store = new DiskStore(dir);
  const key = embedKey('ns', 'model', 'text');
  await store.set(key, Float32Array.from([1, 2]));

  const [a, b] = keyToSegments(key);
  const entries = await readdir(join(dir, a, b));
  assert.strictEqual(entries.length, 1);
});

test('a corrupt file is treated as a miss and removed', async (t) => {
  const dir = await withTempDir(t);
  const store = new DiskStore(dir);
  const key = embedKey('ns', 'model', 'text');
  await store.set(key, Float32Array.from([1, 2, 3]));

  const [a, b, rest] = keyToSegments(key);
  const path = join(dir, a, b, `${rest}.vec`);
  await writeFile(path, Buffer.from('not a vector file'));

  const loaded = await store.get(key);
  assert.strictEqual(loaded, undefined);

  const afterCleanup = await readdir(join(dir, a, b)).catch(() => []);
  assert.strictEqual(afterCleanup.length, 0);
});

test('delete removes the file and is a no-op if it is already gone', async (t) => {
  const dir = await withTempDir(t);
  const store = new DiskStore(dir);
  const key = embedKey('ns', 'model', 'text');
  await store.set(key, Float32Array.from([1, 2, 3]));

  await store.delete(key);
  assert.strictEqual(await store.get(key), undefined);
  await store.delete(key); // already gone, must not throw
});

test('clear removes the whole store directory', async (t) => {
  const dir = await withTempDir(t);
  const store = new DiskStore(dir);
  await store.set(embedKey('ns', 'model', 'one'), Float32Array.from([1]));
  await store.set(embedKey('ns', 'model', 'two'), Float32Array.from([2]));

  await store.clear();
  const remaining = await readdir(dir).catch(() => []);
  assert.strictEqual(remaining.length, 0);
});

test('keys yields every key on disk, and nothing once cleared', async (t) => {
  const dir = await withTempDir(t);
  const store = new DiskStore(dir);
  const one = embedKey('ns', 'model', 'one');
  const two = embedKey('ns', 'model', 'two');
  await store.set(one, Float32Array.from([1]));
  await store.set(two, Float32Array.from([2]));

  const found: string[] = [];
  for await (const key of store.keys()) found.push(key);
  assert.deepStrictEqual(found.sort(), [one, two].sort());

  await store.clear();
  const afterClear: string[] = [];
  for await (const key of store.keys()) afterClear.push(key);
  assert.deepStrictEqual(afterClear, []);
});

test('keys on a directory that was never created yields nothing', async (t) => {
  const dir = join(await withTempDir(t), 'never-created');
  const store = new DiskStore(dir);
  const found: string[] = [];
  for await (const key of store.keys()) found.push(key);
  assert.deepStrictEqual(found, []);
});

test('a new DiskStore over the same directory reads back a previously written vector', async (t) => {
  const dir = await withTempDir(t);
  const key = embedKey('ns', 'model', 'text');
  const vector = Float32Array.from([5, 6, 7]);

  await new DiskStore(dir).set(key, vector);
  const loaded = await new DiskStore(dir).get(key);
  assert.deepStrictEqual(loaded, vector);
});
