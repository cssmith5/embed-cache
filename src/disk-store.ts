import { mkdir, readdir, readFile, rename, rm, stat, unlink, writeFile } from 'node:fs/promises';
import { randomBytes } from 'node:crypto';
import { dirname, join } from 'node:path';
import { decodeVector, encodeVector, VectorFormatError } from './float32.ts';
import { keyToSegments } from './key.ts';

export interface DiskStoreOptions {
  /**
   * max time in ms a file is trusted before it's treated as stale, based on
   * its mtime rather than a field in the file (default unlimited). Unlike
   * the in-memory maxAge, this survives a process restart.
   */
  maxAge?: number;
}

/**
 * One file per vector under <dir>/<key[0:2]>/<key[2:4]>/<key[4:]>.vec, so no
 * single directory grows past ~256 children even with millions of keys.
 * Writes go to a temp file next to the target and get renamed into place,
 * so a reader never observes a partial write.
 */
export class DiskStore {
  readonly #dir: string;
  readonly #maxAge?: number;

  constructor(dir: string, options: DiskStoreOptions = {}) {
    this.#dir = dir;
    this.#maxAge = options.maxAge;
  }

  #pathFor(key: string): string {
    const [a, b, rest] = keyToSegments(key);
    return join(this.#dir, a, b, `${rest}.vec`);
  }

  async get(key: string): Promise<Float32Array | undefined> {
    const path = this.#pathFor(key);

    if (this.#maxAge !== undefined) {
      let mtimeMs: number;
      try {
        mtimeMs = (await stat(path)).mtimeMs;
      } catch (err) {
        if ((err as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
        throw err;
      }
      if (Date.now() - mtimeMs > this.#maxAge) {
        await unlink(path).catch(() => {});
        return undefined;
      }
    }

    let buf: Buffer;
    try {
      buf = await readFile(path);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
      throw err;
    }

    try {
      return decodeVector(buf);
    } catch (err) {
      if (err instanceof VectorFormatError) {
        // corrupt or truncated file: treat as a miss rather than serving
        // bad data or throwing, and clean it up so it gets recomputed
        await unlink(path).catch(() => {});
        return undefined;
      }
      throw err;
    }
  }

  async set(key: string, vector: Float32Array): Promise<void> {
    const path = this.#pathFor(key);
    await mkdir(dirname(path), { recursive: true });
    const tmpPath = `${path}.${randomBytes(6).toString('hex')}.tmp`;
    try {
      await writeFile(tmpPath, encodeVector(vector));
      await rename(tmpPath, path);
    } catch (err) {
      await unlink(tmpPath).catch(() => {});
      throw err;
    }
  }

  async delete(key: string): Promise<void> {
    try {
      await unlink(this.#pathFor(key));
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err;
    }
  }

  async clear(): Promise<void> {
    await rm(this.#dir, { recursive: true, force: true });
  }

  /** walks the shard tree and yields every key with a file on disk, in no particular order */
  async *keys(): AsyncGenerator<string> {
    const top = await readdir(this.#dir, { withFileTypes: true }).catch((err) => {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') return [];
      throw err;
    });
    for (const a of top) {
      if (!a.isDirectory()) continue;
      const mids = await readdir(join(this.#dir, a.name), { withFileTypes: true }).catch(() => []);
      for (const b of mids) {
        if (!b.isDirectory()) continue;
        const files = await readdir(join(this.#dir, a.name, b.name)).catch(() => []);
        for (const file of files) {
          if (file.endsWith('.vec')) yield `${a.name}${b.name}${file.slice(0, -'.vec'.length)}`;
        }
      }
    }
  }
}
