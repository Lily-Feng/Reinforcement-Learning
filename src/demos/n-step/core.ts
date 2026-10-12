/**
 * n-step TD prediction on the 19-state random walk (Sutton & Barto, Example 7.1).
 *
 * States 1..19 sit between two terminals. Every episode starts in the middle,
 * steps left or right with equal probability, and earns −1 at the left end and
 * +1 at the right end. With γ = 1 the true value of state i is (i − 10) / 10.
 *
 * `n = Infinity` never bootstraps, so it is constant-α Monte Carlo — the TD(1)
 * end of the spectrum. `n = 1` is TD(0).
 */
import { mulberry32, type Rng } from '../shared/rng.ts';

export const STATES = 19;
export const START = (STATES + 1) / 2;

/** True values for states 1..STATES, indexed 0..STATES-1. */
export const trueValues: readonly number[] = Array.from(
  { length: STATES },
  (_, i) => (2 * (i + 1)) / (STATES + 1) - 1,
);

/** One episode: visited non-terminal states and the reward on each transition. */
export interface Episode {
  states: number[];
  rewards: number[];
}

export function sampleEpisode(rng: Rng): Episode {
  const states = [START];
  const rewards: number[] = [];
  let s = START;
  for (;;) {
    s += rng() < 0.5 ? -1 : 1;
    if (s === 0) { rewards.push(-1); return { states, rewards }; }
    if (s === STATES + 1) { rewards.push(1); return { states, rewards }; }
    rewards.push(0);
    states.push(s);
  }
}

/**
 * Applies n-step TD to one completed episode, in the online order of the
 * textbook algorithm: the update for time τ happens at time τ + n, using the
 * value estimates as they are at that moment.
 *
 * `v` is indexed by state (0 and STATES+1 are terminals and stay 0).
 */
export function nStepEpisode(v: number[], episode: Episode, n: number, alpha: number, gamma = 1): void {
  const { states, rewards } = episode;
  const T = rewards.length;
  for (let tau = 0; tau < T; tau += 1) {
    const end = Math.min(tau + n, T);
    let g = 0;
    let discount = 1;
    for (let k = tau; k < end; k += 1) {
      g += discount * rewards[k];
      discount *= gamma;
    }
    if (tau + n < T) g += discount * v[states[tau + n]];
    const s = states[tau];
    v[s] += alpha * (g - v[s]);
  }
}

export function rmsError(v: readonly number[]): number {
  let total = 0;
  for (let i = 0; i < STATES; i += 1) total += (v[i + 1] - trueValues[i]) ** 2;
  return Math.sqrt(total / STATES);
}

export interface StreamConfig {
  n: number;
  alpha: number;
}

export interface StreamResult {
  /** Mean RMS error after each episode, averaged over runs (index 0 = before learning). */
  curve: number[];
  /** Mean estimate per state after the final episode, indexed 0..STATES-1. */
  values: number[];
  /** Mean of the curve after episode 1..episodes — the "area" summary. */
  averageError: number;
}

/**
 * Runs one stream over `runs` independent runs of `episodes` episodes.
 *
 * Run r always uses the episode sequence seeded by `seed + r`, so two streams
 * with the same seed learn from identical experience. Any gap between their
 * curves is caused by n and α, not by luck.
 */
export function runStream(config: StreamConfig, episodes: number, runs: number, seed: number): StreamResult {
  const curve = new Array<number>(episodes + 1).fill(0);
  const values = new Array<number>(STATES).fill(0);
  for (let r = 0; r < runs; r += 1) {
    const rng = mulberry32(seed + r);
    const v = new Array<number>(STATES + 2).fill(0);
    curve[0] += rmsError(v);
    for (let e = 1; e <= episodes; e += 1) {
      nStepEpisode(v, sampleEpisode(rng), config.n, config.alpha);
      curve[e] += rmsError(v);
    }
    for (let i = 0; i < STATES; i += 1) values[i] += v[i + 1];
  }
  const mean = curve.map(total => total / runs);
  return {
    curve: mean,
    values: values.map(total => total / runs),
    averageError: mean.slice(1).reduce((a, b) => a + b, 0) / episodes,
  };
}

/** Average RMS error over the first `episodes` episodes, for each α — Figure 7.2's x-axis. */
export function alphaSweep(n: number, alphas: readonly number[], episodes: number, runs: number, seed: number): number[] {
  return alphas.map(alpha => runStream({ n, alpha }, episodes, runs, seed).averageError);
}

export function nLabel(n: number): string {
  if (n === 1) return 'n = 1 · TD(0)';
  if (!Number.isFinite(n)) return 'n = ∞ · MC ≈ TD(1)';
  return `n = ${n}`;
}
