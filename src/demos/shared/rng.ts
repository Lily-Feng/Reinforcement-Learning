/**
 * Seeded pseudo-random number generation.
 *
 * Demos must be reproducible: the same seed has to produce the same run, on any
 * machine, forever. `Math.random()` cannot do that, so every stochastic demo in
 * this book draws from a seeded generator threaded explicitly through the code.
 */

export type Rng = () => number;

/**
 * mulberry32 — a small, fast, well-distributed 32-bit PRNG.
 * Returns a function producing floats in [0, 1).
 */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return function next(): number {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** FNV-1a, so a human-typed seed like "bandit" maps to a stable 32-bit number. */
export function hashSeed(input: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < input.length; i += 1) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** Accepts either a numeric seed or a free-text seed and normalises it. */
export function toSeed(value: string | number): number {
  if (typeof value === 'number') return value >>> 0;
  const trimmed = value.trim();
  if (trimmed === '') return 0;
  if (/^\d+$/.test(trimmed)) return Number(trimmed) >>> 0;
  return hashSeed(trimmed);
}

/** A fresh seed for "give me a new environment". */
export function randomSeed(): number {
  return Math.floor(Math.random() * 0x100000000) >>> 0;
}

/** Uniform integer in [0, n). */
export function randInt(rng: Rng, n: number): number {
  return Math.floor(rng() * n);
}

/** Uniform pick from a non-empty array. */
export function pick<T>(rng: Rng, items: readonly T[]): T {
  return items[randInt(rng, items.length)]!;
}
