# embed-cache

Embeddings are pure functions of (model, text), they cost money and latency, and
in any real pipeline you end up embedding the same chunk over and over --
re-running an ingest, retrying a batch, restarting a dev server.

`embed-cache` is a cache for exactly that: an in-memory LRU in front of an
optional on-disk store, keyed by the SHA-256 of the model and the text. Zero
runtime dependencies -- `node:crypto`, `node:fs` and `node:zlib` only.

Vectors are stored as raw little-endian float32 behind a 16-byte header with a
CRC32, not as JSON. A 1536-dimension vector is 6160 bytes on disk instead of
roughly 30 kB of decimal text, and the float32 bits come back exactly as they
went in.

## Install

```bash
npm install
npm run build
```

Requires Node 22 or newer.

## Usage

```ts
import { EmbedCache } from 'embed-cache';

const cache = new EmbedCache({
  dir: '.cache/embeddings',   // omit for a memory-only cache
  maxBytes: 64 * 1024 * 1024, // in-memory budget
  expectedDim: 1536,
});

const vector = await cache.getOrCompute('text-embedding-3-small', chunk, async (text, model) => {
  const response = await openai.embeddings.create({ model, input: text });
  return response.data[0].embedding; // number[] is fine, it gets narrowed to float32
});
```

Concurrent calls for the same text share one computation, so a cold start with
fifty parallel workers still makes one request per distinct text.

### Batching

Embedding endpoints take many inputs per request, so the interesting call is the
batch one. `getOrComputeMany` looks everything up first and hands your callback
only the texts that were missing, deduplicated and in first-seen order. The
result lines up positionally with the input.

```ts
const vectors = await cache.getOrComputeMany(model, chunks, async (missing) => {
  const response = await openai.embeddings.create({ model, input: missing });
  return response.data.map((d) => d.embedding);
});
```

`examples/basic.ts` runs this end to end against a fake provider:

```
$ node examples/basic.ts
first pass:  4 vectors, 1 api call(s) for 3 text(s)
second pass: 4 vectors, 1 api call(s) total
identical:   true
after restart: dim 8, stats {"memoryHits":0,"diskHits":1,"misses":0,...,"hitRate":1}
hit rate 50.0%  (memory 4, disk 0, miss 4, computed 3)  3 entries / 96 bytes in memory
```

Four documents, one duplicate, one API call for three texts; the second pass
costs nothing, and a freshly constructed cache over the same directory reads the
vector back from disk.

## API

### `new EmbedCache(options?)`

| option | meaning |
|---|---|
| `dir` | directory for the on-disk tier; omit for memory only |
| `maxEntries` | max vectors held in memory (default unlimited) |
| `maxBytes` | max bytes of vector payload held in memory (default unlimited) |
| `maxAge` | max time in ms a value stays fresh in memory (default unlimited) |
| `namespace` | mixed into every key; bump it to invalidate after a chunking change |
| `expectedDim` | reject vectors of any other length |

- `getOrCompute(model, text, compute)` -- cached vector, or compute it once and
  store it. Rejections are not cached.
- `getOrComputeMany(model, texts, computeBatch)` -- as above for an array;
  `computeBatch` is called at most once with only the missing texts.
- `get(model, text)` / `set(model, text, vector)` / `delete(model, text)`
- `stats()` -- `memoryHits`, `diskHits`, `misses`, `computed`, `corrupt`,
  `diskWrites`, `entries`, `bytes`, `hitRate`; `resetStats()` zeroes them.
- `clearMemory()` / `clear()` -- drop the memory tier, or both tiers.

`maxAge` only bounds how long a value is trusted in memory before it's
re-fetched or recomputed; the disk format has no timestamp field, so a
value written to disk is kept until you delete it or call `clear()`. If you
need embeddings to actually expire, put the on-disk directory behind a
process that also gets torn down periodically, or bump `namespace`.

The pieces are exported separately too, if you want them on their own:
`Lru` (byte-aware LRU), `DiskStore` (atomic sharded file store),
`embedKey`/`keyToSegments` (key derivation), and
`encodeVector`/`decodeVector`/`toFloat32` (the float32 codec).

### On-disk format

One file per vector at `<dir>/<key[0:2]>/<key[2:4]>/<key[4:]>.vec`, so no single
directory grows past ~256 children. Each file is:

```
offset  size   field
0       4      magic "EMBC"
4       1      format version
5       1      dtype (1 = float32)
6       2      reserved
8       4      dim, uint32 little-endian
12      4      crc32 of the payload
16      4*dim  payload, float32 little-endian
```

Writes go to a temporary file and are renamed into place, so an interrupted
write can never be read as a valid vector. A file that fails its checksum or
header check is deleted and recomputed rather than being served or throwing.

## Test

```bash
npm test
```

## License

MIT (c) cssmith5
