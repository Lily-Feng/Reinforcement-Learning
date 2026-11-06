/** Small statistics helpers shared by the references and the modules. */

export function mean(xs: readonly number[]): number {
  return xs.length ? xs.reduce((s, v) => s + v, 0) / xs.length : NaN;
}

export function std(xs: readonly number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((s, v) => s + (v - m) ** 2, 0) / (xs.length - 1));
}

/** Two-sided 97.5% Student-t quantiles for small df; 1.96 beyond. */
export function tQuantile(df: number): number {
  const table = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.16, 2.145, 2.131, 2.12, 2.11, 2.101, 2.093, 2.086];
  if (df < 1) return Infinity;
  if (df <= 20) return table[df - 1]!;
  if (df <= 30) return 2.042 + ((30 - df) / 10) * (2.086 - 2.042);
  if (df <= 60) return 2.0;
  return 1.96;
}

export interface Interval {
  mean: number;
  /** Half-width of the 95% interval. */
  half: number;
  n: number;
}

export function ci95(xs: readonly number[]): Interval {
  return { mean: mean(xs), half: xs.length > 1 ? (tQuantile(xs.length - 1) * std(xs)) / Math.sqrt(xs.length) : Infinity, n: xs.length };
}

/** Quantile by linear interpolation of the sorted sample. */
export function quantile(sorted: readonly number[], p: number): number {
  if (!sorted.length) return NaN;
  const x = p * (sorted.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  return sorted[i]! + (i + 1 < sorted.length ? f * (sorted[i + 1]! - sorted[i]!) : 0);
}

/** Histogram counts of xs over [lo, hi) in `bins` bins (values outside clamp to the edges). */
export function histogram(xs: readonly number[], lo: number, hi: number, bins: number): number[] {
  const out = new Array<number>(bins).fill(0);
  const w = (hi - lo) / bins || 1;
  for (const x of xs) out[Math.min(bins - 1, Math.max(0, Math.floor((x - lo) / w)))]! += 1;
  return out;
}

/** Parses "7/3", "0.25", "-1/4". */
export function parseNumber(text: string): number {
  const t = text.trim().replace('−', '-');
  const frac = /^(-?\d+(?:\.\d+)?)\s*\/\s*(-?\d+(?:\.\d+)?)$/.exec(t);
  if (frac) return Number(frac[1]) / Number(frac[2]);
  return t === '' ? NaN : Number(t);
}
