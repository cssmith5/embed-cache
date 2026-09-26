#!/usr/bin/env node
import { stat, unlink } from 'node:fs/promises';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { DiskStore } from './disk-store.ts';
import { keyToSegments } from './key.ts';

const DURATION_UNITS: Record<string, number> = { ms: 1, s: 1000, m: 60_000, h: 3_600_000, d: 86_400_000 };

/** accepts a plain millisecond count or a suffixed duration like "30s", "10m", "2h", "7d" */
export function parseDuration(input: string): number {
  const match = /^(\d+(?:\.\d+)?)(ms|s|m|h|d)?$/.exec(input.trim());
  if (!match) throw new Error(`invalid duration "${input}", expected e.g. 500ms, 30s, 10m, 2h, 7d`);
  return Number(match[1]) * DURATION_UNITS[match[2] ?? 'ms'];
}

function pathFor(dir: string, key: string): string {
  const [a, b, rest] = keyToSegments(key);
  return join(dir, a, b, `${rest}.vec`);
}

export interface CacheEntryInfo {
  key: string;
  size: number;
  mtimeMs: number;
}

/** every key on disk with its file size and mtime, in no particular order */
export async function listEntries(dir: string): Promise<CacheEntryInfo[]> {
  const store = new DiskStore(dir);
  const entries: CacheEntryInfo[] = [];
  for await (const key of store.keys()) {
    const info = await stat(pathFor(dir, key)).catch(() => undefined);
    if (info) entries.push({ key, size: info.size, mtimeMs: info.mtimeMs });
  }
  return entries;
}

export interface CacheStats {
  count: number;
  bytes: number;
  oldestMtimeMs?: number;
  newestMtimeMs?: number;
}

export async function statsFor(dir: string): Promise<CacheStats> {
  const entries = await listEntries(dir);
  const stats: CacheStats = { count: entries.length, bytes: 0 };
  for (const entry of entries) {
    stats.bytes += entry.size;
    if (stats.oldestMtimeMs === undefined || entry.mtimeMs < stats.oldestMtimeMs) stats.oldestMtimeMs = entry.mtimeMs;
    if (stats.newestMtimeMs === undefined || entry.mtimeMs > stats.newestMtimeMs) stats.newestMtimeMs = entry.mtimeMs;
  }
  return stats;
}

/** deletes every file whose mtime is older than maxAgeMs; returns the count removed */
export async function pruneOlderThan(dir: string, maxAgeMs: number, now: number = Date.now()): Promise<number> {
  const cutoff = now - maxAgeMs;
  let removed = 0;
  for (const entry of await listEntries(dir)) {
    if (entry.mtimeMs < cutoff) {
      await unlink(pathFor(dir, entry.key)).catch(() => {});
      removed++;
    }
  }
  return removed;
}

/**
 * Reads every entry through DiskStore.get(), which already deletes a file
 * that fails its header or checksum check rather than serving it. Walking
 * every key this way is enough to clean up corruption without duplicating
 * that check here.
 */
export async function gcCorrupt(dir: string): Promise<{ checked: number; removed: number }> {
  const store = new DiskStore(dir);
  let checked = 0;
  let removed = 0;
  for (const entry of await listEntries(dir)) {
    checked++;
    await store.get(entry.key);
    if (await stat(pathFor(dir, entry.key)).catch(() => undefined) === undefined) removed++;
  }
  return { checked, removed };
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  return `${value.toFixed(1)} ${units[unit]}`;
}

const USAGE = 'usage: embed-cache-cli <dir> <stats|list|prune|gc> [--older-than <duration>]';

export async function run(argv: string[]): Promise<void> {
  const [dir, command, ...rest] = argv;
  if (!dir || !command) throw new Error(USAGE);

  switch (command) {
    case 'stats': {
      const s = await statsFor(dir);
      console.log(`${s.count} entries, ${formatBytes(s.bytes)}`);
      if (s.count > 0) {
        console.log(`oldest: ${new Date(s.oldestMtimeMs!).toISOString()}`);
        console.log(`newest: ${new Date(s.newestMtimeMs!).toISOString()}`);
      }
      return;
    }
    case 'list': {
      for (const entry of await listEntries(dir)) {
        console.log(`${entry.key}  ${entry.size}  ${new Date(entry.mtimeMs).toISOString()}`);
      }
      return;
    }
    case 'prune': {
      const flagIndex = rest.indexOf('--older-than');
      const value = flagIndex >= 0 ? rest[flagIndex + 1] : undefined;
      if (!value) throw new Error('prune requires --older-than <duration>, e.g. --older-than 30d');
      const removed = await pruneOlderThan(dir, parseDuration(value));
      console.log(`removed ${removed} entries older than ${value}`);
      return;
    }
    case 'gc': {
      const { checked, removed } = await gcCorrupt(dir);
      console.log(`checked ${checked} entries, removed ${removed} corrupt`);
      return;
    }
    default:
      throw new Error(`unknown command "${command}", expected stats, list, prune, or gc\n${USAGE}`);
  }
}

const isMainModule = process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMainModule) {
  run(process.argv.slice(2)).catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exitCode = 1;
  });
}
