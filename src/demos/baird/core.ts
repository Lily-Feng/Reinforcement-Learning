/**
 * Semi-gradient TD(0) on Baird's counterexample (Sutton & Barto, Example 11.1),
 * in a six-state, seven-weight form.
 *
 * Upper states 1..5 have v̂(s_i) = w0 + 2·w_i; the bottom state 6 has
 * v̂(s_6) = 2·w0 + w6. Every upper state moves to state 6, state 6 loops on
 * itself, and every reward is 0, so the true value of every state is 0 —
 * reachable exactly with w = 0. Yet updating each transition equally often,
 * rather than as often as the target policy would visit it, drives the
 * weights to infinity.
 */

export const WEIGHTS = 7;
/** States are numbered 1..6; state 6 is the bottom state. */
export const STATES = 6;
export const BOTTOM = 6;
export const REWARD = 0;

export const DEFAULT_WEIGHTS: readonly number[] = [1, 1, 1, 1, 1, 1, 5];
export const DEFAULT_ALPHA = 0.1;
export const DEFAULT_GAMMA = 0.95;

/** Any weight or state value beyond this magnitude counts as diverged. */
export const DIVERGENCE_LIMIT = 1e6;

/** The six transitions, each used exactly once per sweep: s_i → s_6, then s_6 → s_6. */
export const TRANSITIONS: readonly (readonly [number, number])[] = [
  ...Array.from({ length: STATES - 1 }, (_, i) => [i + 1, BOTTOM] as const),
  [BOTTOM, BOTTOM],
];

/** Feature vector x(s): (1 at w0, 2 at w_i) for upper states, (2 at w0, 1 at w6) for the bottom. */
export function features(s: number): number[] {
  const x = new Array<number>(WEIGHTS).fill(0);
  if (s === BOTTOM) {
    x[0] = 2;
    x[6] = 1;
  } else {
    x[0] = 1;
    x[s] = 2;
  }
  return x;
}

/** Linear value estimate v̂(s, w) = w · x(s). */
export function value(s: number, w: readonly number[]): number {
  return features(s).reduce((sum, xj, j) => sum + xj * w[j], 0);
}

export function stateValues(w: readonly number[]): number[] {
  return Array.from({ length: STATES }, (_, i) => value(i + 1, w));
}

/** One transition's share of an update, with the numbers that produced it. */
export interface TransitionStep {
  from: number;
  to: number;
  vFrom: number;
  vTo: number;
  delta: number;
  /** α·δ·x(s), one entry per weight. */
  contribution: number[];
}

export type UpdateMode = 'batch' | 'online';

export interface UpdateRecord {
  mode: UpdateMode;
  alpha: number;
  gamma: number;
  before: number[];
  after: number[];
  steps: TransitionStep[];
  /** after − before: the sum of every step's contribution. */
  deltaW: number[];
}

function step(s: number, s2: number, w: readonly number[], alpha: number, gamma: number): TransitionStep {
  const vFrom = value(s, w);
  const vTo = value(s2, w);
  const delta = REWARD + gamma * vTo - vFrom;
  return { from: s, to: s2, vFrom, vTo, delta, contribution: features(s).map(xj => alpha * delta * xj) };
}

/**
 * Batch semi-gradient TD(0): every δ is computed from the same frozen
 * weights, the six contributions are summed, and w moves once.
 */
export function batchUpdate(w: readonly number[], alpha: number, gamma: number): UpdateRecord {
  const steps = TRANSITIONS.map(([s, s2]) => step(s, s2, w, alpha, gamma));
  const deltaW = new Array<number>(WEIGHTS).fill(0);
  for (const { contribution } of steps) contribution.forEach((c, j) => { deltaW[j] += c; });
  return { mode: 'batch', alpha, gamma, before: [...w], after: w.map((wj, j) => wj + deltaW[j]), steps, deltaW };
}

/**
 * Online semi-gradient TD(0): the same six transitions in the same order, but
 * w moves after each one, so later transitions see the earlier updates.
 */
export function onlineUpdate(w: readonly number[], alpha: number, gamma: number): UpdateRecord {
  const current = [...w];
  const steps = TRANSITIONS.map(([s, s2]) => {
    const record = step(s, s2, current, alpha, gamma);
    record.contribution.forEach((c, j) => { current[j] += c; });
    return record;
  });
  return { mode: 'online', alpha, gamma, before: [...w], after: current, steps, deltaW: current.map((wj, j) => wj - w[j]) };
}

export function update(mode: UpdateMode, w: readonly number[], alpha: number, gamma: number): UpdateRecord {
  return mode === 'batch' ? batchUpdate(w, alpha, gamma) : onlineUpdate(w, alpha, gamma);
}

export function hasDiverged(w: readonly number[]): boolean {
  return [...w, ...stateValues(w)].some(v => !Number.isFinite(v) || Math.abs(v) > DIVERGENCE_LIMIT);
}
