/**
 * Multi-armed bandit simulation.
 *
 * This module is pure: it never touches the DOM, never reads a global, and
 * never calls `Math.random()`. Everything stochastic comes from a seeded RNG
 * passed in via the options. That is what lets the same code drive the
 * interactive demo, reproducible single runs, and averaged comparison sweeps.
 */

import { mulberry32, pick, randInt, type Rng } from '../shared/rng';

export interface Arm {
  /** The hidden win probability. The learner never sees this. */
  readonly trueProb: number;
  pulls: number;
  wins: number;
  /** Q(a), the running estimate of this arm's value. */
  estimate: number;
}

export interface BanditOptions {
  numArms?: number;
  epsilon: number;
  seed: number;
  /**
   * Initial value for every Q(a). Zero is the neutral default; a value near 1
   * gives "optimistic initialisation", which drives early exploration on its own.
   */
  optimisticInit?: number;
}

export interface StepRecord {
  /** 1-based index of this pull. */
  step: number;
  arm: number;
  /** True when the epsilon coin flip forced a random action. */
  explored: boolean;
  /** The coin flip itself, so the UI can show why the branch was taken. */
  draw: number;
  epsilon: number;
  reward: 0 | 1;
  /** True when the chosen arm was in fact the best one. */
  optimal: boolean;
  /** Running mean reward after this step. */
  avgReward: number;
  /** Running share of pulls that hit the best arm. */
  optimalRate: number;
  /** Total expected reward given up by not always playing the best arm. */
  cumulativeRegret: number;
}

export const DEFAULT_NUM_ARMS = 5;

/** Hidden probabilities for the ordinary arms. */
const MIN_PROB = 0.1;
const MAX_PROB = 0.8;
/** The planted best arm always lands in this band, so there is something to find. */
const BEST_MIN = 0.85;
const BEST_MAX = 0.95;

/**
 * Builds the environment. Deterministic in `seed`, and drawn from its own
 * generator so that changing epsilon mid-run cannot shift the arms underneath
 * the learner.
 */
function createArms(rng: Rng, numArms: number, optimisticInit: number): { arms: Arm[]; bestArm: number } {
  const arms: Arm[] = Array.from({ length: numArms }, () => ({
    trueProb: MIN_PROB + rng() * (MAX_PROB - MIN_PROB),
    pulls: 0,
    wins: 0,
    estimate: optimisticInit,
  }));

  // Plant one clearly-best arm above MAX_PROB so the optimal action is unique.
  const bestArm = randInt(rng, numArms);
  arms[bestArm] = {
    ...arms[bestArm]!,
    trueProb: BEST_MIN + rng() * (BEST_MAX - BEST_MIN),
  };

  return { arms, bestArm };
}

/**
 * One epsilon-greedy run over a stationary Bernoulli bandit.
 *
 * Drive it a step at a time from a UI, or in a tight loop for a sweep. Both
 * paths go through `step()`, so what the demo shows and what the comparison
 * chart plots can never drift apart.
 */
export class BanditRun {
  readonly arms: Arm[];
  readonly bestArm: number;
  readonly seed: number;
  readonly optimisticInit: number;

  /** Mutable so a slider can change it mid-run. */
  epsilon: number;

  steps = 0;
  totalReward = 0;
  optimalPulls = 0;
  cumulativeRegret = 0;
  readonly history: StepRecord[] = [];

  private readonly rng: Rng;

  constructor(options: BanditOptions) {
    const numArms = options.numArms ?? DEFAULT_NUM_ARMS;
    this.seed = options.seed >>> 0;
    this.epsilon = options.epsilon;
    this.optimisticInit = options.optimisticInit ?? 0;

    // Two generators from one seed: one lays out the environment, one drives
    // the policy. Keeping them apart means an identical seed gives identical
    // arms no matter how many actions have been taken.
    const setupRng = mulberry32(this.seed);
    const { arms, bestArm } = createArms(setupRng, numArms, this.optimisticInit);
    this.arms = arms;
    this.bestArm = bestArm;
    this.rng = mulberry32(this.seed ^ 0x9e3779b9);
  }

  get numArms(): number {
    return this.arms.length;
  }

  /** The hidden value of the best arm — used for regret, never by the policy. */
  get bestProb(): number {
    return this.arms[this.bestArm]!.trueProb;
  }

  get avgReward(): number {
    return this.steps === 0 ? 0 : this.totalReward / this.steps;
  }

  get optimalRate(): number {
    return this.steps === 0 ? 0 : this.optimalPulls / this.steps;
  }

  /** Indices tied for the highest current estimate. */
  greedyArms(): number[] {
    let best = -Infinity;
    for (const arm of this.arms) {
      if (arm.estimate > best) best = arm.estimate;
    }
    const ties: number[] = [];
    this.arms.forEach((arm, i) => {
      if (arm.estimate === best) ties.push(i);
    });
    return ties;
  }

  /** Takes one action, observes one reward, and updates one estimate. */
  step(): StepRecord {
    const draw = this.rng();
    const explored = draw < this.epsilon;

    // Ties are broken uniformly at random. Without this, an all-zero start
    // would make arm 0 permanently greedy and the demo would look broken.
    const arm = explored ? randInt(this.rng, this.numArms) : pick(this.rng, this.greedyArms());

    const chosen = this.arms[arm]!;
    const reward: 0 | 1 = this.rng() < chosen.trueProb ? 1 : 0;

    chosen.pulls += 1;
    chosen.wins += reward;

    // Incremental sample average: Q <- Q + (1/N) * (R - Q).
    // Equivalent to wins/pulls when Q starts at 0, but this form is the one
    // written on the chapter page, and it is the form that survives a
    // non-zero optimistic initialisation.
    chosen.estimate += (reward - chosen.estimate) / chosen.pulls;

    this.steps += 1;
    this.totalReward += reward;
    const optimal = arm === this.bestArm;
    if (optimal) this.optimalPulls += 1;
    this.cumulativeRegret += this.bestProb - chosen.trueProb;

    const record: StepRecord = {
      step: this.steps,
      arm,
      explored,
      draw,
      epsilon: this.epsilon,
      reward,
      optimal,
      avgReward: this.avgReward,
      optimalRate: this.optimalRate,
      cumulativeRegret: this.cumulativeRegret,
    };
    this.history.push(record);
    return record;
  }

  /** Runs many steps at once, discarding the per-step records. */
  run(steps: number): void {
    for (let i = 0; i < steps; i += 1) this.step();
  }
}

export interface RunSummary {
  epsilon: number;
  steps: number;
  totalReward: number;
  avgReward: number;
  optimalRate: number;
  cumulativeRegret: number;
  /** Mean reward at each step, for plotting. */
  avgRewardCurve: number[];
  /** Share of optimal actions at each step, for plotting. */
  optimalRateCurve: number[];
}

export interface SweepOptions {
  steps: number;
  /**
   * Independent environments averaged together. A single run is dominated by
   * luck; Sutton & Barto average 2000. A few hundred is enough to separate
   * epsilon values visibly without stalling the browser.
   */
  runs?: number;
  seed: number;
  numArms?: number;
  optimisticInit?: number;
}

/**
 * Averages `runs` independent epsilon-greedy runs, each on its own environment.
 *
 * Runs headless, so the comparison panel can evaluate several epsilon values
 * without any of them being rendered.
 */
export function evaluateEpsilon(epsilon: number, options: SweepOptions): RunSummary {
  const runs = options.runs ?? 200;
  const avgRewardCurve = new Array<number>(options.steps).fill(0);
  const optimalRateCurve = new Array<number>(options.steps).fill(0);

  let totalReward = 0;
  let optimalRate = 0;
  let cumulativeRegret = 0;

  for (let r = 0; r < runs; r += 1) {
    // Each run gets its own environment, but the set of environments is fixed
    // by the caller's seed — so every epsilon faces exactly the same problems.
    const run = new BanditRun({
      epsilon,
      seed: (options.seed + r * 0x9e3779b1) >>> 0,
      numArms: options.numArms,
      optimisticInit: options.optimisticInit,
    });

    for (let t = 0; t < options.steps; t += 1) {
      const record = run.step();
      avgRewardCurve[t] += record.reward;
      optimalRateCurve[t] += record.optimal ? 1 : 0;
    }

    totalReward += run.totalReward;
    optimalRate += run.optimalRate;
    cumulativeRegret += run.cumulativeRegret;
  }

  for (let t = 0; t < options.steps; t += 1) {
    avgRewardCurve[t]! /= runs;
    optimalRateCurve[t]! /= runs;
  }

  return {
    epsilon,
    steps: options.steps,
    totalReward: totalReward / runs,
    avgReward: totalReward / runs / options.steps,
    optimalRate: optimalRate / runs,
    cumulativeRegret: cumulativeRegret / runs,
    avgRewardCurve,
    optimalRateCurve,
  };
}

/** Evaluates several epsilon values against the same set of environments. */
export function compareEpsilons(epsilons: readonly number[], options: SweepOptions): RunSummary[] {
  return epsilons.map((epsilon) => evaluateEpsilon(epsilon, options));
}

/** Boxcar smoothing, so noisy per-step curves stay readable. */
export function smooth(series: readonly number[], window: number): number[] {
  if (window <= 1) return [...series];
  const out = new Array<number>(series.length);
  let sum = 0;
  for (let i = 0; i < series.length; i += 1) {
    sum += series[i]!;
    if (i >= window) sum -= series[i - window]!;
    out[i] = sum / Math.min(i + 1, window);
  }
  return out;
}
