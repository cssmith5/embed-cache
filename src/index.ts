import { DiskStore } from './disk-store.ts';
import { decodeVector, encodeVector, toFloat32, VectorFormatError } from './float32.ts';
import { embedKey, keyToSegments } from './key.ts';
import { Lru } from './lru.ts';

export { DiskStore, Lru, decodeVector, embedKey, encodeVector, keyToSegments, toFloat32, VectorFormatError };

export interface EmbedCacheOptions {
  /** directory for the on-disk tier; omit for memory only */
  dir?: string;
  /** max vectors held in memory (default unlimited) */
  maxEntries?: number;
  /** max bytes of vector payload held in memory (default unlimited) */
  maxBytes?: number;
  /**
   * max time in ms a value stays fresh in memory (default unlimited). Only
   * governs the memory tier: an expired entry is just a memory miss, and
   * falls through to the disk tier (which never expires) or recompute.
   */
  maxAge?: number;
  /** mixed into every key; bump it to invalidate after a chunking change */
  namespace?: string;
  /** reject vectors of any other length */
  expectedDim?: number;
}

export interface EmbedCacheStats {
  memoryHits: number;
  diskHits: number;
  misses: number;
  computed: number;
  corrupt: number;
  diskWrites: number;
  entries: number;
  bytes: number;
  hitRate: number;
}

interface RawStats {
  memoryHits: number;
  diskHits: number;
  misses: number;
  computed: number;
  corrupt: number;
  diskWrites: number;
}

function freshStats(): RawStats {
  return { memoryHits: 0, diskHits: 0, misses: 0, computed: 0, corrupt: 0, diskWrites: 0 };
}

export class EmbedCache {
  readonly #memory: Lru;
  readonly #disk?: DiskStore;
  readonly #namespace: string;
  readonly #expectedDim?: number;
  readonly #inflight = new Map<string, Promise<Float32Array>>();
  #stats: RawStats = freshStats();

  constructor(options: EmbedCacheOptions = {}) {
    this.#memory = new Lru({ maxEntries: options.maxEntries, maxBytes: options.maxBytes, maxAge: options.maxAge });
    this.#disk = options.dir ? new DiskStore(options.dir) : undefined;
    this.#namespace = options.namespace ?? '';
    this.#expectedDim = options.expectedDim;
  }

  async get(model: string, text: string): Promise<Float32Array | undefined> {
    return this.#getByKey(embedKey(this.#namespace, model, text));
  }

  async set(model: string, text: string, vector: Float32Array | ArrayLike<number>): Promise<void> {
    await this.#store(embedKey(this.#namespace, model, text), vector);
  }

  async delete(model: string, text: string): Promise<void> {
    const key = embedKey(this.#namespace, model, text);
    this.#memory.delete(key);
    if (this.#disk) await this.#disk.delete(key);
  }

  /**
   * Cached vector, or compute it once and store it. Concurrent calls for the
   * same (model, text) share one in-flight computation. Rejections are not
   * cached, so a failed compute can be retried on the next call.
   */
  async getOrCompute(
    model: string,
    text: string,
    compute: (text: string, model: string) => Promise<Float32Array | ArrayLike<number>>,
  ): Promise<Float32Array> {
    const key = embedKey(this.#namespace, model, text);

    const inflight = this.#inflight.get(key);
    if (inflight) return inflight;

    // no await before this point, and none until the map is populated below,
    // so a second synchronous call for the same key always finds this entry
    const promise = (async () => {
      try {
        const cached = await this.#getByKey(key);
        if (cached !== undefined) return cached;
        const raw = await compute(text, model);
        this.#stats.computed++;
        return await this.#store(key, raw);
      } finally {
        this.#inflight.delete(key);
      }
    })();
    this.#inflight.set(key, promise);
    return promise;
  }

  /**
   * Batched getOrCompute. Looks every text up first, then calls
   * computeBatch once with only the texts that were missing, deduplicated
   * and in first-seen order. The result lines up positionally with `texts`.
   */
  async getOrComputeMany(
    model: string,
    texts: string[],
    computeBatch: (missing: string[]) => Promise<Array<Float32Array | ArrayLike<number>>>,
  ): Promise<Float32Array[]> {
    const keys = texts.map((text) => embedKey(this.#namespace, model, text));
    const lookups = await Promise.all(keys.map((key) => this.#getByKey(key)));

    const missingTexts: string[] = [];
    const seen = new Set<string>();
    for (let i = 0; i < texts.length; i++) {
      if (lookups[i] === undefined && !seen.has(texts[i])) {
        seen.add(texts[i]);
        missingTexts.push(texts[i]);
      }
    }

    const computedByText = new Map<string, Float32Array>();
    if (missingTexts.length > 0) {
      const raw = await computeBatch(missingTexts);
      if (raw.length !== missingTexts.length) {
        throw new RangeError(
          `computeBatch returned ${raw.length} vector(s) for ${missingTexts.length} text(s)`,
        );
      }
      this.#stats.computed += missingTexts.length;
      for (let i = 0; i < missingTexts.length; i++) {
        const text = missingTexts[i];
        const key = embedKey(this.#namespace, model, text);
        computedByText.set(text, await this.#store(key, raw[i]));
      }
    }

    return texts.map((text, i) => lookups[i] ?? computedByText.get(text)!);
  }

  stats(): EmbedCacheStats {
    const hits = this.#stats.memoryHits + this.#stats.diskHits;
    const total = hits + this.#stats.misses;
    return {
      ...this.#stats,
      entries: this.#memory.size,
      bytes: this.#memory.bytes,
      hitRate: total > 0 ? hits / total : 0,
    };
  }

  resetStats(): void {
    this.#stats = freshStats();
  }

  clearMemory(): void {
    this.#memory.clear();
  }

  async clear(): Promise<void> {
    this.#memory.clear();
    if (this.#disk) await this.#disk.clear();
  }

  /**
   * Every key currently held in memory and/or on disk, deduplicated. Keys
   * are opaque SHA-256 hashes, not the (model, text) pairs that produced
   * them, so this is meant for bulk work against a keep-set you already
   * computed with `embedKey`, not for browsing by model or text.
   */
  async *keys(): AsyncGenerator<string> {
    const seen = new Set<string>();
    for (const key of this.#memory.keys()) {
      seen.add(key);
      yield key;
    }
    if (this.#disk) {
      for await (const key of this.#disk.keys()) {
        if (!seen.has(key)) yield key;
      }
    }
  }

  /** deletes every key for which predicate returns true, from both tiers; returns the count removed */
  async purge(predicate: (key: string) => boolean): Promise<number> {
    const toRemove: string[] = [];
    for await (const key of this.keys()) {
      if (predicate(key)) toRemove.push(key);
    }
    for (const key of toRemove) {
      this.#memory.delete(key);
      if (this.#disk) await this.#disk.delete(key);
    }
    return toRemove.length;
  }

  async #getByKey(key: string): Promise<Float32Array | undefined> {
    const cached = this.#memory.get(key);
    if (cached !== undefined) {
      this.#stats.memoryHits++;
      return cached;
    }

    if (this.#disk) {
      const fromDisk = await this.#disk.get(key);
      if (fromDisk !== undefined) {
        if (this.#expectedDim !== undefined && fromDisk.length !== this.#expectedDim) {
          this.#stats.corrupt++;
          await this.#disk.delete(key);
        } else {
          this.#stats.diskHits++;
          this.#memory.set(key, fromDisk);
          return fromDisk;
        }
      }
    }

    this.#stats.misses++;
    return undefined;
  }

  async #store(key: string, vector: Float32Array | ArrayLike<number>): Promise<Float32Array> {
    const vec = toFloat32(vector);
    if (this.#expectedDim !== undefined && vec.length !== this.#expectedDim) {
      throw new RangeError(`expected a ${this.#expectedDim}-dimensional vector, got ${vec.length}`);
    }
    this.#memory.set(key, vec);
    if (this.#disk) {
      await this.#disk.set(key, vec);
      this.#stats.diskWrites++;
    }
    return vec;
  }
}
