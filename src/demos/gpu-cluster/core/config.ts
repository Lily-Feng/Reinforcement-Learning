/**
 * The cluster's parameters, shared by the simulator, the bucketed model and
 * every module. Defaults follow the brief; σ = 0 with m = 1/0.06 is Example 10.2.
 */

export const TIERS = 4;
export const TIER_NAMES = ['Free', 'Basic', 'Pro', 'Enterprise'] as const;
export const PAY: readonly number[] = [1, 2, 4, 8];
/** Steps per day (15 minutes each). */
export const STEPS_PER_DAY = 96;
/** How each tier's share moves with the diurnal signal. */
export const DIURNAL_DIR: readonly number[] = [-1, -0.5, 0.5, 1];

export const ACCEPT = 1;
export const REJECT = 0;

export interface ClusterConfig {
  /** Number of GPUs. */
  K: number;
  /** Median job length. */
  median: number;
  /** Log-std of the job length. */
  sigma: number;
  /** Energy cost per busy GPU per step. */
  kappa: number;
  /** Diurnal amplitude of the tier mix. */
  amp: number;
  /** Base tier mix (normalized on use). */
  base: number[];
  /** Buckets for the exact model (1 when σ = 0). */
  B: number;
  /** Shift length for the episodic variant. */
  H: number;
  seed: number;
}

export const DEFAULT_CONFIG: ClusterConfig = {
  K: 10, median: 16, sigma: 0.6, kappa: 0, amp: 0,
  base: [0.25, 0.25, 0.25, 0.25], B: 3, H: 48, seed: 1,
};

/** Example 10.2: ten servers, every busy server frees with probability 0.06. */
export const BOOK_CONFIG: ClusterConfig = { ...DEFAULT_CONFIG, median: 1 / 0.06, sigma: 0, B: 1 };

/** d(φ) = −cos 2πφ: −1 at midnight, +1 at noon. */
export function diurnal(phase: number): number {
  return -Math.cos(2 * Math.PI * phase);
}

/** Tier mix w(φ) = base_c·(1 + A·dir_c·d(φ)), normalized. */
export function tierMix(cfg: Pick<ClusterConfig, 'base' | 'amp'>, phase: number): number[] {
  const d = diurnal(phase);
  const raw = cfg.base.map((b, c) => Math.max(0, b * (1 + cfg.amp * DIURNAL_DIR[c]! * d)));
  const total = raw.reduce((s, v) => s + v, 0) || 1;
  return raw.map(v => v / total);
}

/** A stable key for caching references that depend on the config. */
export function configKey(cfg: ClusterConfig, extra = ''): string {
  return [cfg.K, cfg.median, cfg.sigma, cfg.kappa, cfg.amp, cfg.base.join('/'), cfg.B, cfg.H, extra].join('|');
}
