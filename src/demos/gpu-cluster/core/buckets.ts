/**
 * Job lengths: a LogNormal with median `m` and log-std `σ`, clipped to
 * [LEN_MIN, LEN_MAX], and its split into equal-probability buckets.
 *
 * Every length is generated from a standard normal draw z as
 * ℓ(z) = clamp(m·e^{σz}). Bucket b is the z-interval between the b/B and
 * (b+1)/B normal quantiles, so the simulator and the bucketed model agree on
 * which bucket a job falls in. Bucket b's completion probability is
 * q_b = E[1/ℓ | ℓ in bucket b], integrated numerically in z.
 */

export const LEN_MIN = 2;
export const LEN_MAX = 120;

export interface LengthDist {
  median: number;
  sigma: number;
}

export interface Buckets {
  count: number;
  /** Interior z boundaries, ascending, length count − 1. */
  zCuts: number[];
  /** Completion probability per bucket, E[1/ℓ | bucket]. */
  q: number[];
  /** Representative length 1/q_b. */
  len: number[];
  /** E[1/ℓ] over the whole clipped distribution. */
  meanQ: number;
}

export const clampLength = (len: number) => Math.min(LEN_MAX, Math.max(LEN_MIN, len));

export function lengthAt(dist: LengthDist, z: number): number {
  return clampLength(dist.median * Math.exp(dist.sigma * z));
}

/** Standard normal density. */
export function phi(z: number): number {
  return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
}

/** Inverse standard normal CDF (Acklam, relative error below 1.2e-9). */
export function normalQuantile(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425;
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) / ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
  }
  if (p > 1 - lo) return -normalQuantile(1 - p);
  const q = p - 0.5;
  const r = q * q;
  return (((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r + a[5]!) * q / (((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1);
}

const Z_LIMIT = 9;

/** Simpson's rule on [a, b] with `n` (even) intervals. */
export function simpson(f: (x: number) => number, a: number, b: number, n = 2000): number {
  if (!(b > a)) return 0;
  const h = (b - a) / n;
  let sum = f(a) + f(b);
  for (let i = 1; i < n; i += 1) sum += (i % 2 ? 4 : 2) * f(a + i * h);
  return (sum * h) / 3;
}

/** ∫ φ(z)/ℓ(z) dz over [z0, z1], split at the clip points so the integrand is smooth on each piece. */
function integrateInverseLength(dist: LengthDist, z0: number, z1: number): number {
  const a = Math.max(z0, -Z_LIMIT);
  const b = Math.min(z1, Z_LIMIT);
  const breaks = [a, b];
  if (dist.sigma > 0) {
    for (const len of [LEN_MIN, LEN_MAX]) {
      const z = Math.log(len / dist.median) / dist.sigma;
      if (z > a && z < b) breaks.push(z);
    }
  }
  breaks.sort((x, y) => x - y);
  let total = 0;
  for (let i = 0; i + 1 < breaks.length; i += 1) {
    total += simpson(z => phi(z) / lengthAt(dist, z), breaks[i]!, breaks[i + 1]!);
  }
  return total;
}

/**
 * Equal-probability buckets. σ = 0 always uses one bucket: every job has the
 * same length, which is Example 10.2 when m = 1/0.06.
 */
export function makeBuckets(dist: LengthDist, requested = 3): Buckets {
  const count = dist.sigma === 0 ? 1 : Math.max(1, Math.round(requested));
  if (count === 1) {
    const q = dist.sigma === 0 ? 1 / clampLength(dist.median) : integrateInverseLength(dist, -Infinity, Infinity);
    return { count, zCuts: [], q: [q], len: [1 / q], meanQ: q };
  }
  const zCuts = Array.from({ length: count - 1 }, (_, i) => normalQuantile((i + 1) / count));
  const edges = [-Infinity, ...zCuts, Infinity];
  const q = Array.from({ length: count }, (_, b) => count * integrateInverseLength(dist, edges[b]!, edges[b + 1]!));
  return { count, zCuts, q, len: q.map(v => 1 / v), meanQ: q.reduce((s, v) => s + v, 0) / count };
}

/** Which bucket a normal draw z falls in. */
export function bucketOf(buckets: Buckets, z: number): number {
  let b = 0;
  while (b < buckets.zCuts.length && z >= buckets.zCuts[b]!) b += 1;
  return b;
}
