/**
 * Scores learned greedy policies exactly. Every bucketed state has an
 * observation (g = K − Σn, ℓ = 1/q_b, ρ = Σ n_b q_b), so a greedy policy over
 * observations is also a policy on the bucketed model, and its long-run
 * average reward can be computed exactly rather than estimated.
 */

import { greedyAction, type LinearQ } from './agents.ts';
import type { Observation } from './env.ts';
import { evaluatePolicy } from './exact.ts';
import { observe, policyFrom, type BucketObservation, type Model, type ModelPolicy } from './model.ts';

export function toObservation(o: BucketObservation, phase = 0, togo = Infinity): Observation {
  return { g: o.g, c: o.c, len: o.len, rho: o.rho, phase, togo, bucket: o.bucket };
}

/** The greedy policy of a learned q̂, as a policy on the bucketed model. */
export function greedyOnModel(model: Model, lq: LinearQ, phase = 0, togo = Infinity): ModelPolicy {
  return policyFrom(model, o => greedyAction(lq, toObservation(o, phase, togo)));
}

export interface Score {
  policy: ModelPolicy;
  /** Exact r(π) on the bucketed model. */
  avg: number;
}

export function scoreGreedy(model: Model, lq: LinearQ, phase = 0): Score {
  const policy = greedyOnModel(model, lq, phase);
  return { policy, avg: evaluatePolicy(model, policy).avg };
}

/** Fraction of states where two policies agree, weighted by μ (states with no free GPU count as agreeing). */
export function agreement(a: ModelPolicy, b: ModelPolicy, mu: Float64Array): number {
  let same = 0;
  let total = 0;
  for (let s = 0; s < a.length; s += 1) {
    total += mu[s]!;
    if (a[s] === b[s]) same += mu[s]!;
  }
  return total > 0 ? same / total : 1;
}

/** Learned q̂ on every bucketed state, for scatter plots against exact values. */
export function learnedValues(model: Model, lq: LinearQ, action: (s: number) => number): Float64Array {
  return Float64Array.from({ length: model.nStates }, (_, s) => {
    const o = toObservation(observe(model, s));
    const idx = lq.feat.indices(o, action(s));
    let q = 0;
    for (const i of idx) q += lq.w[i]!;
    return q;
  });
}
