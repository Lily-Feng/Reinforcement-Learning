/**
 * The continuous GPU-cluster simulator. Pure: no DOM, all randomness from the
 * injected RNG.
 *
 * One step, in this exact order (the bucketed model uses the same order):
 *   1. accept puts the head job in a free slot with q = 1/ℓ and pays the tier
 *      price; accepting with no free slot is a reject;
 *   2. r ← r − κ·(busy slots);
 *   3. every busy slot completes independently with its own q
 *      (a job admitted this step may complete this step);
 *   4. τ ← (τ + 1) mod P;
 *   5. draw the next head: tier from w(φ(τ)), length independently.
 *
 * In bucketed mode each sampled length snaps to its bucket's representative
 * 1/q_b, so the simulator realizes the exact model's transition law.
 */

import type { Rng } from '../../shared/rng.ts';
import { bucketOf, lengthAt, makeBuckets, type Buckets } from './buckets.ts';
import { ACCEPT, PAY, STEPS_PER_DAY, TIERS, tierMix, type ClusterConfig } from './config.ts';

export type EnvMode = 'continuous' | 'bucketed';

export interface Observation {
  /** Free GPUs. */
  g: number;
  /** Head tier. */
  c: number;
  /** Head job length. */
  len: number;
  /** Expected completions next step, Σ_busy q_i. */
  rho: number;
  /** Time of day in [0, 1). */
  phase: number;
  /** Steps left in the shift (episodic only; Infinity when continuing). */
  togo: number;
  /** Head length bucket (for mapping onto the exact model). */
  bucket: number;
}

export interface Env {
  cfg: ClusterConfig;
  buckets: Buckets;
  mode: EnvMode;
  episodic: boolean;
  rng: Rng;
  /** Per-slot completion probability; 0 = free. */
  q: Float64Array;
  /** Per-slot tier and bucket of the job it holds (−1 = free). */
  tier: Int8Array;
  bucket: Int8Array;
  /** Per-slot job length. */
  len: Float64Array;
  head: { c: number; len: number; bucket: number };
  tau: number;
  /** Steps taken since the last reset. */
  t: number;
  done: boolean;
}

export function gaussian(rng: Rng): number {
  const u1 = 1 - rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function sampleTier(rng: Rng, mix: readonly number[]): number {
  let u = rng();
  for (let c = 0; c < TIERS - 1; c += 1) {
    u -= mix[c]!;
    if (u < 0) return c;
  }
  return TIERS - 1;
}

function drawHead(env: Env): void {
  const mix = tierMix(env.cfg, env.tau / STEPS_PER_DAY);
  const c = sampleTier(env.rng, mix);
  const z = gaussian(env.rng);
  const b = bucketOf(env.buckets, z);
  const len = env.mode === 'bucketed' ? env.buckets.len[b]! : lengthAt(env.cfg, z);
  env.head = { c, len, bucket: b };
}

export function createEnv(cfg: ClusterConfig, rng: Rng, options: { mode?: EnvMode; episodic?: boolean; buckets?: Buckets } = {}): Env {
  const buckets = options.buckets ?? makeBuckets({ median: cfg.median, sigma: cfg.sigma }, cfg.B);
  const env: Env = {
    cfg, buckets, mode: options.mode ?? 'continuous', episodic: options.episodic ?? false, rng,
    q: new Float64Array(cfg.K), tier: new Int8Array(cfg.K).fill(-1), bucket: new Int8Array(cfg.K).fill(-1),
    len: new Float64Array(cfg.K), head: { c: 0, len: 1, bucket: 0 }, tau: 0, t: 0, done: false,
  };
  resetEnv(env);
  return env;
}

/** All GPUs free, clock at midnight, fresh head. */
export function resetEnv(env: Env): void {
  env.q.fill(0);
  env.tier.fill(-1);
  env.bucket.fill(-1);
  env.len.fill(0);
  env.tau = 0;
  env.t = 0;
  env.done = false;
  drawHead(env);
}

export function freeCount(env: Env): number {
  let g = 0;
  for (let i = 0; i < env.q.length; i += 1) if (env.q[i] === 0) g += 1;
  return g;
}

export function observe(env: Env): Observation {
  let g = 0;
  let rho = 0;
  for (let i = 0; i < env.q.length; i += 1) {
    if (env.q[i] === 0) g += 1;
    else rho += env.q[i]!;
  }
  return {
    g, c: env.head.c, len: env.head.len, rho, phase: env.tau / STEPS_PER_DAY,
    togo: env.episodic ? env.cfg.H - env.t : Infinity, bucket: env.head.bucket,
  };
}

export interface StepResult {
  reward: number;
  accepted: boolean;
  completed: number;
  done: boolean;
}

export function step(env: Env, action: number): StepResult {
  let reward = 0;
  let accepted = false;
  if (action === ACCEPT) {
    const slot = env.q.indexOf(0);
    if (slot >= 0) {
      env.q[slot] = 1 / env.head.len;
      env.tier[slot] = env.head.c;
      env.bucket[slot] = env.head.bucket;
      env.len[slot] = env.head.len;
      reward = PAY[env.head.c]!;
      accepted = true;
    }
  }
  let busy = 0;
  for (let i = 0; i < env.q.length; i += 1) if (env.q[i] !== 0) busy += 1;
  reward -= env.cfg.kappa * busy;
  let completed = 0;
  for (let i = 0; i < env.q.length; i += 1) {
    if (env.q[i] !== 0 && env.rng() < env.q[i]!) {
      env.q[i] = 0;
      env.tier[i] = -1;
      env.bucket[i] = -1;
      env.len[i] = 0;
      completed += 1;
    }
  }
  env.tau = (env.tau + 1) % STEPS_PER_DAY;
  env.t += 1;
  if (env.episodic && env.t >= env.cfg.H) env.done = true;
  drawHead(env);
  return { reward, accepted, completed, done: env.done };
}

/** Busy counts per bucket: the exact model's count vector for this state. */
export function bucketCounts(env: Env): number[] {
  const n = new Array<number>(env.buckets.count).fill(0);
  for (let i = 0; i < env.bucket.length; i += 1) if (env.bucket[i]! >= 0) n[env.bucket[i]!]! += 1;
  return n;
}

/** Copies the true state (not the RNG) so a rollout can branch from it. */
export function cloneEnv(env: Env, rng: Rng): Env {
  return {
    ...env, rng, q: Float64Array.from(env.q), tier: Int8Array.from(env.tier), bucket: Int8Array.from(env.bucket),
    len: Float64Array.from(env.len), head: { ...env.head },
  };
}
