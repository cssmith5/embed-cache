import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Lru } from '../src/lru.ts';

function vec(n: number): Float32Array {
  return new Float32Array(n);
}

test('set/get/has/delete round-trip a value', () => {
  const lru = new Lru();
  const value = vec(4);
  lru.set('a', value);
  assert.strictEqual(lru.has('a'), true);
  assert.strictEqual(lru.get('a'), value);
  assert.strictEqual(lru.size, 1);
  assert.strictEqual(lru.bytes, value.byteLength);

  assert.strictEqual(lru.delete('a'), true);
  assert.strictEqual(lru.has('a'), false);
  assert.strictEqual(lru.size, 0);
  assert.strictEqual(lru.bytes, 0);
});

test('get on a missing key returns undefined without throwing', () => {
  const lru = new Lru();
  assert.strictEqual(lru.get('missing'), undefined);
});

test('delete on a missing key returns false and leaves bytes untouched', () => {
  const lru = new Lru();
  lru.set('a', vec(4));
  assert.strictEqual(lru.delete('missing'), false);
  assert.strictEqual(lru.bytes, 16);
});

test('re-setting a key replaces the value and updates the byte count', () => {
  const lru = new Lru();
  lru.set('a', vec(4));
  lru.set('a', vec(8));
  assert.strictEqual(lru.size, 1);
  assert.strictEqual(lru.bytes, 32);
});

test('maxEntries evicts the least recently used entry first', () => {
  const lru = new Lru({ maxEntries: 2 });
  lru.set('a', vec(1));
  lru.set('b', vec(1));
  lru.set('c', vec(1)); // evicts 'a'

  assert.strictEqual(lru.has('a'), false);
  assert.strictEqual(lru.has('b'), true);
  assert.strictEqual(lru.has('c'), true);
  assert.strictEqual(lru.size, 2);
});

test('get() promotes a key so it survives the next eviction', () => {
  const lru = new Lru({ maxEntries: 2 });
  lru.set('a', vec(1));
  lru.set('b', vec(1));
  lru.get('a'); // 'a' is now more recently used than 'b'
  lru.set('c', vec(1)); // evicts 'b', not 'a'

  assert.strictEqual(lru.has('a'), true);
  assert.strictEqual(lru.has('b'), false);
  assert.strictEqual(lru.has('c'), true);
});

test('maxBytes evicts down to the byte budget', () => {
  const lru = new Lru({ maxBytes: 10 });
  lru.set('a', vec(1)); // 4 bytes
  lru.set('b', vec(1)); // 4 bytes, total 8
  lru.set('c', vec(1)); // 4 bytes, total 12 > 10, evicts 'a'

  assert.strictEqual(lru.has('a'), false);
  assert.strictEqual(lru.has('b'), true);
  assert.strictEqual(lru.has('c'), true);
  assert.strictEqual(lru.bytes, 8);
});

test('clear empties the map and resets the byte count', () => {
  const lru = new Lru();
  lru.set('a', vec(4));
  lru.set('b', vec(4));
  lru.clear();
  assert.strictEqual(lru.size, 0);
  assert.strictEqual(lru.bytes, 0);
  assert.strictEqual(lru.has('a'), false);
});

test('unbounded Lru never evicts', () => {
  const lru = new Lru();
  for (let i = 0; i < 100; i++) lru.set(`k${i}`, vec(16));
  assert.strictEqual(lru.size, 100);
});

function fakeClock(start = 0): { now: () => number; advance: (ms: number) => void } {
  let t = start;
  return { now: () => t, advance: (ms: number) => (t += ms) };
}

test('maxAge keeps an entry until its age passes, then treats it as a miss', () => {
  const clock = fakeClock();
  const lru = new Lru({ maxAge: 100, now: clock.now });
  lru.set('a', vec(1));

  clock.advance(99);
  assert.deepStrictEqual(lru.get('a'), vec(1));

  clock.advance(1);
  assert.strictEqual(lru.get('a'), undefined);
});

test('an expired entry is dropped from size and bytes on access', () => {
  const clock = fakeClock();
  const lru = new Lru({ maxAge: 100, now: clock.now });
  lru.set('a', vec(1));
  clock.advance(200);

  assert.strictEqual(lru.has('a'), false);
  assert.strictEqual(lru.size, 0);
  assert.strictEqual(lru.bytes, 0);
});

test('re-setting a key resets its age', () => {
  const clock = fakeClock();
  const lru = new Lru({ maxAge: 100, now: clock.now });
  lru.set('a', vec(1));
  clock.advance(60);
  lru.set('a', vec(1)); // refreshed at t=60, so it now expires at t=160

  clock.advance(60); // t=120: would have expired under the original set
  assert.strictEqual(lru.has('a'), true);
});

test('maxAge evicts an expired entry from the front even when under capacity', () => {
  const clock = fakeClock();
  const lru = new Lru({ maxAge: 100, now: clock.now });
  lru.set('a', vec(1));
  clock.advance(200);
  lru.set('b', vec(1)); // triggers a sweep; 'a' is expired and well under any budget

  assert.strictEqual(lru.size, 1);
  assert.strictEqual(lru.has('b'), true);
});

test('without maxAge, entries never expire regardless of elapsed time', () => {
  const clock = fakeClock();
  const lru = new Lru({ now: clock.now });
  lru.set('a', vec(1));
  clock.advance(1_000_000_000);
  assert.deepStrictEqual(lru.get('a'), vec(1));
});
