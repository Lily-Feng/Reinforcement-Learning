/**
 * Small Markov reward processes for the ergodicity, differential-value and
 * R̄-update modules. Each chain lists P(s → s′) and the reward earned on that
 * transition; expected rewards r(s) = Σ P(s, s′)·R(s, s′).
 */

import type { Rng } from '../../shared/rng.ts';
import { solveDense } from './exact.ts';

export interface Chain {
  id: string;
  name: string;
  states: string[];
  P: number[][];
  /** Reward on the transition s → s′. */
  R: number[][];
  note: string;
}

const zeros = (n: number) => Array.from({ length: n }, () => new Array<number>(n).fill(0));

/** Exercise 10.6: A → B pays +1, B → A pays 0. */
export function alternating(): Chain {
  return { id: 'alternating', name: 'Alternating A ↔ B', states: ['A', 'B'], P: [[0, 1], [1, 0]], R: [[0, 1], [0, 0]], note: 'Rewards 1, 0, 1, 0, … from A (Exercise 10.6).' };
}

/** Exercises 10.7, 10.8: A → B → C → A with +1 on arriving at A. */
export function ring(): Chain {
  return { id: 'ring', name: 'Three-state ring', states: ['A', 'B', 'C'], P: [[0, 1, 0], [0, 0, 1], [1, 0, 0]], R: [[0, 0, 0], [0, 0, 0], [1, 0, 0]], note: '+1 on arrival at A (Exercises 10.7 and 10.8).' };
}

/** The ring, but each step stays put with probability ε. */
export function lazyRing(eps: number): Chain {
  const P = [[eps, 1 - eps, 0], [0, eps, 1 - eps], [1 - eps, 0, eps]];
  return { id: 'lazy-ring', name: 'Lazy ring', states: ['A', 'B', 'C'], P, R: [[0, 0, 0], [0, 0, 0], [1, 0, 0]], note: `Stay with probability ε = ${eps}.` };
}

/** P(A→B) = 0.3, P(B→A) = 0.6; expected reward 1 in A and 5 in B. */
export function twoState(): Chain {
  return { id: 'two-state', name: 'Two-state chain', states: ['A', 'B'], P: [[0.7, 0.3], [0.6, 0.4]], R: [[1, 1], [5, 5]], note: 'Expected rewards 1 and 5; μ = (2/3, 1/3).' };
}

/** Two disjoint rings: a 2-ring paying 1 per lap and a 3-ring paying 3 per lap. */
export function reduciblePair(): Chain {
  const P = zeros(5);
  const R = zeros(5);
  P[0]![1] = 1; P[1]![0] = 1; R[1]![0] = 1;
  P[2]![3] = 1; P[3]![4] = 1; P[4]![2] = 1; R[4]![2] = 3;
  return { id: 'reducible', name: 'Reducible pair of rings', states: ['A1', 'A2', 'B1', 'B2', 'B3'], P, R, note: 'Ring A earns 1/2 per step, ring B earns 1 per step; they never connect.' };
}

export function expectedRewards(chain: Chain): number[] {
  return chain.P.map((row, s) => row.reduce((sum, p, j) => sum + p * chain.R[s]![j]!, 0));
}

/** Stationary distribution of an irreducible chain (exact solve). */
export function stationary(chain: Chain): number[] {
  const n = chain.states.length;
  const A = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) A[j * n + i] = (i === j ? 1 : 0) - chain.P[i]![j]!;
  }
  for (let i = 0; i < n; i += 1) A[(n - 1) * n + i] = 1;
  const b = new Float64Array(n);
  b[n - 1] = 1;
  return [...solveDense(A, b, n)];
}

export function averageReward(chain: Chain, mu = stationary(chain)): number {
  const r = expectedRewards(chain);
  return mu.reduce((s, m, i) => s + m * r[i]!, 0);
}

/** Differential values normalized so Σ μ v = 0: the Abel limit of v^γ − r/(1 − γ). */
export function differentialValues(chain: Chain): number[] {
  const n = chain.states.length;
  const mu = stationary(chain);
  const avg = averageReward(chain, mu);
  const r = expectedRewards(chain);
  const A = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) A[i * n + j] = (i === j ? 1 : 0) - chain.P[i]![j]! + mu[j]!;
  }
  return [...solveDense(A, Float64Array.from(r, v => v - avg), n)];
}

export function discountedValues(chain: Chain, gamma: number): number[] {
  const n = chain.states.length;
  const A = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) A[i * n + j] = (i === j ? 1 : 0) - gamma * chain.P[i]![j]!;
  }
  return [...solveDense(A, Float64Array.from(expectedRewards(chain)), n)];
}

/** Pr{S_t = s} for t = 0..T from a start distribution. */
export function distributionOverTime(chain: Chain, start: number[], T: number): number[][] {
  const out = [start];
  let cur = start;
  for (let t = 0; t < T; t += 1) {
    const next = new Array<number>(cur.length).fill(0);
    cur.forEach((m, i) => chain.P[i]!.forEach((p, j) => { next[j]! += m * p; }));
    out.push(next);
    cur = next;
  }
  return out;
}

/** E[R_{t+1} | S_0 = s] for t = 0..T−1. */
export function expectedRewardSequence(chain: Chain, s: number, T: number): number[] {
  const r = expectedRewards(chain);
  const start = chain.states.map((_, i) => (i === s ? 1 : 0));
  return distributionOverTime(chain, start, T - 1).map(d => d.reduce((sum, m, i) => sum + m * r[i]!, 0));
}

/** One sampled transition. */
export function sampleStep(chain: Chain, s: number, rng: Rng): { next: number; reward: number } {
  let u = rng();
  const row = chain.P[s]!;
  for (let j = 0; j < row.length; j += 1) {
    u -= row[j]!;
    if (u < 0 || j === row.length - 1) return { next: j, reward: chain.R[s]![j]! };
  }
  return { next: s, reward: 0 };
}

export interface ChainProperties {
  irreducible: boolean;
  aperiodic: boolean;
  /** The limiting distribution exists (from the chosen start). */
  limitExists: boolean;
  /** The time-average reward exists (always true for a finite chain). */
  averageExists: boolean;
}

/** Period-1 test via the gcd of cycle lengths reachable in the transition graph. */
export function chainProperties(chain: Chain): ChainProperties {
  const n = chain.states.length;
  const reach = (from: number) => {
    const seen = new Set([from]);
    const stack = [from];
    while (stack.length) {
      const i = stack.pop()!;
      chain.P[i]!.forEach((p, j) => { if (p > 0 && !seen.has(j)) { seen.add(j); stack.push(j); } });
    }
    return seen;
  };
  const irreducible = chain.states.every((_, i) => reach(i).size === n);
  // Period of state 0's class: gcd of lengths k with (P^k)_{00} > 0 up to 2n.
  const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));
  let period = 0;
  let cur: number[] = chain.states.map((_, i) => (i === 0 ? 1 : 0));
  for (let k = 1; k <= 2 * n + 2; k += 1) {
    const next = new Array<number>(n).fill(0);
    cur.forEach((m, i) => chain.P[i]!.forEach((p, j) => { if (m > 0 && p > 0) next[j] = 1; }));
    cur = next;
    if (cur[0]! > 0) period = gcd(period, k);
  }
  const aperiodic = period === 1;
  return { irreducible, aperiodic, limitExists: aperiodic, averageExists: true };
}
