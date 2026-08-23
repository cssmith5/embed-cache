import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { EmbedCache } from '../src/index.ts';

async function withTempDir(t: import('node:test').TestContext): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'embed-cache-test-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  return dir;
}

test('get on an empty cache is a miss, set makes it a hit', async () => {
  const cache = new EmbedCache();
  assert.strictEqual(await cache.get('model', 'text'), undefined);

  await cache.set('model', 'text', [1, 2, 3]);
  assert.deepStrictEqual(await cache.get('model', 'text'), Float32Array.from([1, 2, 3]));
});

test('getOrCompute calls compute once and caches the result', async () => {
  const cache = new EmbedCache();
  let calls = 0;
  const compute = async () => {
    calls++;
    return [1, 2, 3];
  };

  const first = await cache.getOrCompute('model', 'text', compute);
  const second = await cache.getOrCompute('model', 'text', compute);
  assert.deepStrictEqual(first, Float32Array.from([1, 2, 3]));
  assert.deepStrictEqual(second, first);
  assert.strictEqual(calls, 1);
});

test('concurrent getOrCompute calls for the same key share one computation', async () => {
  const cache = new EmbedCache();
  let calls = 0;
  const compute = async () => {
    calls++;
    return [1, 2, 3];
  };

  const [a, b, c] = await Promise.all([
    cache.getOrCompute('model', 'text', compute),
    cache.getOrCompute('model', 'text', compute),
    cache.getOrCompute('model', 'text', compute),
  ]);
  assert.strictEqual(calls, 1);
  assert.deepStrictEqual(a, b);
  assert.deepStrictEqual(b, c);
});

test('a rejected compute is not cached, so the next call retries', async () => {
  const cache = new EmbedCache();
  let calls = 0;
  const compute = async () => {
    calls++;
    if (calls === 1) throw new Error('boom');
    return [1, 2, 3];
  };

  await assert.rejects(cache.getOrCompute('model', 'text', compute), /boom/);
  const result = await cache.getOrCompute('model', 'text', compute);
  assert.deepStrictEqual(result, Float32Array.from([1, 2, 3]));
  assert.strictEqual(calls, 2);
});

test('getOrComputeMany calls computeBatch once with missing texts, deduplicated and in order', async () => {
  const cache = new EmbedCache();
  await cache.set('model', 'b', [0, 2]);

  const batches: string[][] = [];
  const vectors = await cache.getOrComputeMany('model', ['a', 'b', 'c', 'a'], async (missing) => {
    batches.push(missing);
    return missing.map((text) => [text.charCodeAt(0), 1]);
  });

  assert.deepStrictEqual(batches, [['a', 'c']]);
  assert.deepStrictEqual(vectors, [
    Float32Array.from(['a'.charCodeAt(0), 1]),
    Float32Array.from([0, 2]),
    Float32Array.from(['c'.charCodeAt(0), 1]),
    Float32Array.from(['a'.charCodeAt(0), 1]),
  ]);
});

test('getOrComputeMany with nothing missing never calls computeBatch', async () => {
  const cache = new EmbedCache();
  await cache.set('model', 'a', [1]);
  let called = false;
  const vectors = await cache.getOrComputeMany('model', ['a', 'a'], async () => {
    called = true;
    return [];
  });
  assert.strictEqual(called, false);
  assert.deepStrictEqual(vectors, [Float32Array.from([1]), Float32Array.from([1])]);
});

test('getOrComputeMany rejects when computeBatch returns the wrong count', async () => {
  const cache = new EmbedCache();
  await assert.rejects(
    cache.getOrComputeMany('model', ['a', 'b'], async () => [[1]]),
    RangeError,
  );
});

test('expectedDim rejects a mismatched vector on set and on stored disk data', async (t) => {
  const dir = await withTempDir(t);
  const cache = new EmbedCache({ dir, expectedDim: 3 });
  await assert.rejects(cache.set('model', 'text', [1, 2]), RangeError);

  // write a vector of the right length directly, then reopen with a
  // different expectedDim to exercise the corrupt-on-read path
  const writer = new EmbedCache({ dir, expectedDim: 2 });
  await writer.set('model', 'text', [1, 2]);

  const reader = new EmbedCache({ dir, expectedDim: 3 });
  assert.strictEqual(await reader.get('model', 'text'), undefined);
  assert.strictEqual(reader.stats().corrupt, 1);
});

test('delete removes a key from memory and disk', async (t) => {
  const dir = await withTempDir(t);
  const cache = new EmbedCache({ dir });
  await cache.set('model', 'text', [1, 2, 3]);
  await cache.delete('model', 'text');
  assert.strictEqual(await cache.get('model', 'text'), undefined);
});

test('clearMemory drops the memory tier but leaves disk intact', async (t) => {
  const dir = await withTempDir(t);
  const cache = new EmbedCache({ dir });
  await cache.set('model', 'text', [1, 2, 3]);
  cache.clearMemory();
  assert.strictEqual(cache.stats().entries, 0);
  assert.deepStrictEqual(await cache.get('model', 'text'), Float32Array.from([1, 2, 3]));
});

test('clear drops both tiers', async (t) => {
  const dir = await withTempDir(t);
  const cache = new EmbedCache({ dir });
  await cache.set('model', 'text', [1, 2, 3]);
  await cache.clear();
  assert.strictEqual(await new EmbedCache({ dir }).get('model', 'text'), undefined);
});

test('a fresh EmbedCache over the same directory reads a previously written vector from disk', async (t) => {
  const dir = await withTempDir(t);
  await new EmbedCache({ dir }).set('model', 'text', [4, 5, 6]);

  const reader = new EmbedCache({ dir });
  assert.deepStrictEqual(await reader.get('model', 'text'), Float32Array.from([4, 5, 6]));
  const stats = reader.stats();
  assert.strictEqual(stats.diskHits, 1);
  assert.strictEqual(stats.memoryHits, 0);
});

test('stats tracks hits, misses, computed, and hitRate', async () => {
  const cache = new EmbedCache();
  await cache.get('model', 'missing'); // miss
  await cache.getOrCompute('model', 'text', async () => [1]); // computed, then a miss internally
  await cache.get('model', 'text'); // memory hit

  const stats = cache.stats();
  assert.strictEqual(stats.computed, 1);
  assert.strictEqual(stats.memoryHits, 1);
  assert.strictEqual(stats.misses, 2);
  assert.strictEqual(stats.hitRate, 1 / 3);
});

test('resetStats zeroes the counters without touching cached entries', async () => {
  const cache = new EmbedCache();
  await cache.set('model', 'text', [1]);
  await cache.get('model', 'text');
  cache.resetStats();

  const stats = cache.stats();
  assert.strictEqual(stats.memoryHits, 0);
  assert.strictEqual(stats.hitRate, 0);
  assert.strictEqual(stats.entries, 1);
});

test('namespace changes the key, so the same model/text misses under a different namespace', async () => {
  const a = new EmbedCache({ namespace: 'v1' });
  const b = new EmbedCache({ namespace: 'v2' });
  await a.set('model', 'text', [1, 2, 3]);
  assert.strictEqual(await b.get('model', 'text'), undefined);
});
