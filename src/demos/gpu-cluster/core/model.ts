/**
 * The bucketed exact model.
 *
 * State: busy counts per length bucket (n_1..n_B) with Σn ≤ K, the head tier c
 * and the head bucket b. Index s = (countIndex·4 + c)·B + b.
 *
 * One step follows the simulator's order: admit (or not), pay, charge κ per
 * busy GPU, each bucket loses Binomial(n_b, q_b) jobs, then a fresh head is
 * drawn: tier from the mix, bucket uniform. Because the head is drawn fresh,
 * every transition factors through the post-admission count vector (the
 * "afterstate"), which is what the solvers in exact.ts exploit.
 *
 * The model has no clock, so the continuing-task references assume A = 0 and
 * use the base mix.
 */

import { makeBuckets, type Buckets } from './buckets.ts';
import { ACCEPT, PAY, TIERS, tierMix, type ClusterConfig } from './config.ts';

export interface SparseRow {
  next: Int32Array;
  prob: Float64Array;
}

export interface Model {
  K: number;
  B: number;
  kappa: number;
  buckets: Buckets;
  /** Tier mix used by the continuing model (base mix, clock omitted). */
  mix: number[];
  counts: number[][];
  nCounts: number;
  nStates: number;
  /** Free GPUs per count vector. */
  free: Int32Array;
  /** Σ n_b q_b per count vector. */
  rho: Float64Array;
  /** Count index after admitting one job of bucket b: inc[ci·B + b], −1 when full. */
  inc: Int32Array;
  /** Distribution of the next count vector from each afterstate. */
  trans: SparseRow[];
  /** Index of the all-free count vector. */
  empty: number;
}

export const countKey = (n: readonly number[]) => n.join(',');

function enumerateCounts(K: number, B: number): number[][] {
  const out: number[][] = [];
  const cur = new Array<number>(B).fill(0);
  const rec = (b: number, left: number) => {
    if (b === B) { out.push([...cur]); return; }
    for (let k = 0; k <= left; k += 1) { cur[b] = k; rec(b + 1, left - k); }
  };
  rec(0, K);
  // Order by total busy, so index 0 is the empty cluster.
  out.sort((x, y) => x.reduce((s, v) => s + v, 0) - y.reduce((s, v) => s + v, 0));
  return out;
}

function binomialPmf(n: number, q: number): number[] {
  const out = new Array<number>(n + 1).fill(0);
  let coef = 1;
  for (let k = 0; k <= n; k += 1) {
    out[k] = coef * q ** k * (1 - q) ** (n - k);
    coef = (coef * (n - k)) / (k + 1);
  }
  return out;
}

export function buildModel(cfg: ClusterConfig, buckets: Buckets = makeBuckets({ median: cfg.median, sigma: cfg.sigma }, cfg.B)): Model {
  const K = cfg.K;
  const B = buckets.count;
  const counts = enumerateCounts(K, B);
  const index = new Map(counts.map((n, i) => [countKey(n), i]));
  const nCounts = counts.length;
  const free = new Int32Array(nCounts);
  const rho = new Float64Array(nCounts);
  const inc = new Int32Array(nCounts * B).fill(-1);
  counts.forEach((n, i) => {
    const busy = n.reduce((s, v) => s + v, 0);
    free[i] = K - busy;
    rho[i] = n.reduce((s, v, b) => s + v * buckets.q[b]!, 0);
    if (busy < K) {
      for (let b = 0; b < B; b += 1) {
        const m = [...n];
        m[b]! += 1;
        inc[i * B + b] = index.get(countKey(m))!;
      }
    }
  });

  const trans: SparseRow[] = counts.map(n => {
    const pmfs = n.map((nb, b) => binomialPmf(nb, buckets.q[b]!));
    const acc = new Map<number, number>();
    const cur = new Array<number>(B).fill(0);
    const rec = (b: number, p: number) => {
      if (p === 0) return;
      if (b === B) {
        const j = index.get(countKey(cur))!;
        acc.set(j, (acc.get(j) ?? 0) + p);
        return;
      }
      for (let k = 0; k <= n[b]!; k += 1) { cur[b] = n[b]! - k; rec(b + 1, p * pmfs[b]![k]!); }
    };
    rec(0, 1);
    const next = Int32Array.from(acc.keys());
    const prob = Float64Array.from(acc.values());
    return { next, prob };
  });

  return {
    K, B, kappa: cfg.kappa, buckets, mix: tierMix({ base: cfg.base, amp: 0 }, 0),
    counts, nCounts, nStates: nCounts * TIERS * B, free, rho, inc, trans, empty: 0,
  };
}

export const stateIndex = (model: Model, ci: number, c: number, b: number) => (ci * TIERS + c) * model.B + b;

export function decodeState(model: Model, s: number): { ci: number; c: number; b: number } {
  const b = s % model.B;
  const rest = (s - b) / model.B;
  const c = rest % TIERS;
  return { ci: (rest - c) / TIERS, c, b };
}

/** Count index after acting: accepting with no free GPU is a reject. */
export function afterstate(model: Model, s: number, a: number): number {
  const { ci, b } = decodeState(model, s);
  if (a === ACCEPT && model.free[ci]! > 0) return model.inc[ci * model.B + b]!;
  return ci;
}

/** Expected one-step reward: tier price if admitted, minus κ per busy GPU after admission. */
export function reward(model: Model, s: number, a: number): number {
  const { ci, c } = decodeState(model, s);
  const after = afterstate(model, s, a);
  const pay = after !== ci ? PAY[c]! : 0;
  return pay - model.kappa * (model.K - model.free[after]!);
}

/** What the agent observes in a bucketed state. */
export interface BucketObservation {
  g: number;
  c: number;
  len: number;
  rho: number;
  bucket: number;
}

export function observe(model: Model, s: number): BucketObservation {
  const { ci, c, b } = decodeState(model, s);
  return { g: model.free[ci]!, c, len: model.buckets.len[b]!, rho: model.rho[ci]!, bucket: b };
}

/** Deterministic policy as an accept flag per state. */
export type ModelPolicy = Uint8Array;

export function policyFrom(model: Model, decide: (obs: BucketObservation, s: number) => number): ModelPolicy {
  const out = new Uint8Array(model.nStates);
  for (let s = 0; s < model.nStates; s += 1) {
    const o = observe(model, s);
    out[s] = o.g > 0 && decide(o, s) === ACCEPT ? 1 : 0;
  }
  return out;
}
