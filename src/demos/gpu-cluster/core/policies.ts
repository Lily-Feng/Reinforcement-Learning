/**
 * Fixed policies: thresholds, baselines, and the bucketed optimum as an
 * autopilot. A policy maps an observation (and, for the reference optimum
 * only, the true simulator state) to an action.
 */

import { ACCEPT, REJECT, TIERS } from './config.ts';
import { bucketCounts, type Env, type Observation } from './env.ts';
import { countKey, stateIndex, type BucketObservation, type Model, type ModelPolicy } from './model.ts';

export type Policy = (o: Observation, env?: Env) => number;

/** θ_c ∈ 1..K+1 (K+1 = never) and Λ_c ∈ (0, ∞]: accept c iff g ≥ θ_c and ℓ ≤ Λ_c. */
export interface Threshold {
  theta: number[];
  lambda: number[];
}

export function thresholdDecide(th: Threshold, o: Pick<Observation, 'g' | 'c' | 'len'>): number {
  return o.g > 0 && o.g >= th.theta[o.c]! && o.len <= th.lambda[o.c]! ? ACCEPT : REJECT;
}

export const thresholdPolicy = (th: Threshold): Policy => o => thresholdDecide(th, o);

export const acceptAll: Policy = o => (o.g > 0 ? ACCEPT : REJECT);

export const enterpriseOnly: Policy = o => (o.g > 0 && o.c === TIERS - 1 ? ACCEPT : REJECT);

/** A reasonable hand-tuned threshold for the defaults: keep room for paying tiers, avoid long cheap jobs. */
export function handTuned(K: number): Threshold {
  return {
    theta: [Math.ceil(0.7 * K), Math.ceil(0.35 * K), 2, 1],
    lambda: [24, 48, Infinity, Infinity],
  };
}

export function acceptAllThreshold(): Threshold {
  return { theta: [1, 1, 1, 1], lambda: [Infinity, Infinity, Infinity, Infinity] };
}

/** A model policy from a threshold, applied to bucketed observations. */
export function thresholdOnModel(th: Threshold): (o: BucketObservation) => number {
  return o => thresholdDecide(th, o);
}

/**
 * The bucketed optimum as an autopilot. It reads the true per-slot buckets,
 * which agents never see: a reference, not a learnable policy.
 */
export function modelPolicyAutopilot(model: Model, policy: ModelPolicy): Policy {
  const index = new Map(model.counts.map((n, i) => [countKey(n), i]));
  return (o, env) => {
    if (!env || o.g === 0) return REJECT;
    const ci = index.get(countKey(bucketCounts(env)));
    if (ci === undefined) return REJECT;
    return policy[stateIndex(model, ci, o.c, o.bucket)]! ? ACCEPT : REJECT;
  };
}
