/**
 * Monte Carlo references from the continuous simulator, always with a 95%
 * interval: long-run average reward by batch means, and rollout estimates of
 * q_π(s, a) over the rest of a shift.
 */

import { mulberry32, type Rng } from '../../shared/rng.ts';
import type { ClusterConfig } from './config.ts';
import { cloneEnv, createEnv, observe, step, type Env, type EnvMode } from './env.ts';
import type { Policy } from './policies.ts';
import { ci95, type Interval } from './stats.ts';

export interface LongRunOptions {
  steps?: number;
  burnIn?: number;
  batches?: number;
  seed?: number;
  mode?: EnvMode;
  onProgress?: (fraction: number) => void;
}

/** Long-run average reward with batch-means 95% CI. */
export function longRunAverage(cfg: ClusterConfig, policy: Policy, options: LongRunOptions = {}): Interval {
  const steps = options.steps ?? 400000;
  const burnIn = options.burnIn ?? 5000;
  const batches = options.batches ?? 40;
  const env = createEnv(cfg, mulberry32(options.seed ?? cfg.seed), { mode: options.mode ?? 'continuous' });
  for (let t = 0; t < burnIn; t += 1) step(env, policy(observe(env), env));
  const per = Math.floor(steps / batches);
  const means: number[] = [];
  for (let k = 0; k < batches; k += 1) {
    let sum = 0;
    for (let t = 0; t < per; t += 1) sum += step(env, policy(observe(env), env)).reward;
    means.push(sum / per);
    options.onProgress?.((k + 1) / batches);
  }
  return ci95(means);
}

/** Return of the rest of the shift from `env` after taking `action`, then following `policy`. */
export function rolloutReturn(env: Env, action: number, policy: Policy, rng: Rng): { rewards: number[] } {
  const sim = cloneEnv(env, rng);
  const rewards: number[] = [];
  let a = action;
  while (!sim.done) {
    rewards.push(step(sim, a).reward);
    if (!sim.done) a = policy(observe(sim), sim);
  }
  return { rewards };
}

/** Monte Carlo estimate of q_π(s, a) with a 95% interval. */
export function rolloutQ(env: Env, action: number, policy: Policy, runs: number, seed: number): Interval {
  const rng = mulberry32(seed);
  const returns: number[] = [];
  for (let i = 0; i < runs; i += 1) returns.push(rolloutReturn(env, action, policy, rng).rewards.reduce((s, v) => s + v, 0));
  return ci95(returns);
}
