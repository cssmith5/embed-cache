export interface LruOptions {
  maxEntries?: number;
  maxBytes?: number;
  /** entries older than this (ms since insertion or last set) are treated as evicted */
  maxAge?: number;
  /** injectable clock, for tests; defaults to Date.now */
  now?: () => number;
}

interface Entry {
  value: Float32Array;
  expiresAt: number;
}

/**
 * Byte-aware LRU. Insertion order in the backing Map doubles as recency
 * order: a get() re-inserts the key so it moves to the "newest" end, and
 * eviction walks from the front, which is always the least recently used.
 *
 * With maxAge set, expiresAt is non-decreasing along that same front-to-back
 * order (every entry got maxAge added to a clock that only moves forward),
 * so the eviction sweep can stop at the first live, in-budget entry without
 * missing an expired one further back.
 */
export class Lru {
  readonly #map = new Map<string, Entry>();
  readonly #maxEntries: number;
  readonly #maxBytes: number;
  readonly #maxAge: number;
  readonly #now: () => number;
  #bytes = 0;

  constructor(options: LruOptions = {}) {
    this.#maxEntries = options.maxEntries ?? Infinity;
    this.#maxBytes = options.maxBytes ?? Infinity;
    this.#maxAge = options.maxAge ?? Infinity;
    this.#now = options.now ?? Date.now;
  }

  get size(): number {
    return this.#map.size;
  }

  get bytes(): number {
    return this.#bytes;
  }

  has(key: string): boolean {
    return this.get(key) !== undefined;
  }

  get(key: string): Float32Array | undefined {
    const entry = this.#map.get(key);
    if (entry === undefined) return undefined;
    if (this.#now() >= entry.expiresAt) {
      this.#map.delete(key);
      this.#bytes -= entry.value.byteLength;
      return undefined;
    }
    this.#map.delete(key);
    this.#map.set(key, entry);
    return entry.value;
  }

  set(key: string, value: Float32Array): void {
    const existing = this.#map.get(key);
    if (existing !== undefined) {
      this.#bytes -= existing.value.byteLength;
      this.#map.delete(key);
    }
    const expiresAt = this.#maxAge === Infinity ? Infinity : this.#now() + this.#maxAge;
    this.#map.set(key, { value, expiresAt });
    this.#bytes += value.byteLength;
    this.#evict();
  }

  delete(key: string): boolean {
    const existing = this.#map.get(key);
    if (existing === undefined) return false;
    this.#bytes -= existing.value.byteLength;
    return this.#map.delete(key);
  }

  clear(): void {
    this.#map.clear();
    this.#bytes = 0;
  }

  /** live (non-expired) keys, oldest first; does not evict, so a stale entry may still show up */
  keys(): string[] {
    const now = this.#now();
    const live: string[] = [];
    for (const [key, entry] of this.#map) {
      if (now < entry.expiresAt) live.push(key);
    }
    return live;
  }

  #evict(): void {
    const now = this.#now();
    for (const [key, entry] of this.#map) {
      const expired = now >= entry.expiresAt;
      if (!expired && this.#map.size <= this.#maxEntries && this.#bytes <= this.#maxBytes) break;
      this.#map.delete(key);
      this.#bytes -= entry.value.byteLength;
    }
  }
}
