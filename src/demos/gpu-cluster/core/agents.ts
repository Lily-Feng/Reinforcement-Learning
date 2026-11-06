/**
 * Semi-gradient Sarsa agents with linear q̂(o, a, w) = Σ_{i active} w_i.
 *
 * - Episodic n-step Sarsa (Sections 10.1–10.2), for the one-shift task.
 * - Differential n-step Sarsa (Sections 10.3, 10.5), with a choice of R̄ rule.
 * - Discounted continuing n-step Sarsa with γ (Section 10.4's comparison).
 *
 * All three follow the chapter's n-step pseudocode: the last n + 1 feature
 * lists and rewards sit in ring buffers indexed mod n + 1, and the update for
 * time τ = t − n + 1 happens once R_{t+1}, S_{t+1}, A_{t+1} are known.
 *
 * Step sizes: `alpha` is what the reader sees, "α × active features"; the
 * per-weight step is alpha / active. Every update can return a trace record
 * so the UI shows the numbers the agent actually used.
 */

import type { Rng } from '../../shared/rng.ts';
import { ACCEPT, REJECT } from './config.ts';
import type { Observation } from './env.ts';
import type { Featurizer } from './features.ts';

export type AverageRule = 'td' | 'reward' | 'sample-average' | 'unbiased';

export const RULE_LABELS: Record<AverageRule, string> = {
  td: 'R̄ ← R̄ + βδ (TD error)',
  reward: 'R̄ ← R̄ + β(R − R̄)',
  'sample-average': 'Sample average of R',
  unbiased: 'Unbiased constant step (Ex. 10.9)',
};

export interface AverageState {
  rule: AverageRule;
  beta: number;
  rbar: number;
  /** Trace ō for the unbiased rule, ō₀ = 0. */
  obar: number;
  /** Rewards seen, for the sample average. */
  count: number;
}

export function createAverage(rule: AverageRule, beta: number, rbar0 = 0): AverageState {
  return { rule, beta, rbar: rbar0, obar: 0, count: 0 };
}

/** Applies the chosen R̄ rule after one transition with reward R and TD error δ. */
export function updateAverage(st: AverageState, R: number, delta: number): void {
  switch (st.rule) {
    case 'td':
      st.rbar += st.beta * delta;
      break;
    case 'reward':
      st.rbar += st.beta * (R - st.rbar);
      break;
    case 'sample-average':
      st.count += 1;
      st.rbar += (R - st.rbar) / st.count;
      break;
    case 'unbiased':
      st.obar += st.beta * (1 - st.obar);
      st.rbar += (st.beta / st.obar) * delta;
      break;
  }
}

export interface UpdateRecord {
  /** Time index of the pair being updated. */
  tau: number;
  obs: Observation;
  action: number;
  features: number[];
  qBefore: number;
  /** Rewards entering the target, R_{τ+1} … */
  rewards: number[];
  /** q̂(S_{τ+n}, A_{τ+n}) used to bootstrap, or null when the target is a full return. */
  bootstrap: number | null;
  target: number;
  delta: number;
  /** Change applied to each active weight (all equal for binary features). */
  dw: number;
  qAfter: number;
  rbarBefore?: number;
  rbarAfter?: number;
}

export interface LinearQ {
  feat: Featurizer;
  w: Float64Array;
}

export function createWeights(feat: Featurizer, q0 = 0): Float64Array {
  return new Float64Array(feat.size).fill(q0 / feat.active);
}

export function qValue(w: Float64Array, idx: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < idx.length; i += 1) s += w[idx[i]!]!;
  return s;
}

export function qOf(lq: LinearQ, o: Observation, a: number): number {
  return qValue(lq.w, lq.feat.indices(o, a));
}

/** Greedy action; ties go to reject (deterministic, for exact scoring). */
export function greedyAction(lq: LinearQ, o: Observation): number {
  if (o.g === 0) return REJECT;
  return qOf(lq, o, ACCEPT) > qOf(lq, o, REJECT) ? ACCEPT : REJECT;
}

interface Choice {
  action: number;
  features: number[];
  q: number;
}

/** ε-greedy among the valid actions; ties broken uniformly at random. */
function choose(lq: LinearQ, o: Observation, eps: number, rng: Rng): Choice {
  const rej = lq.feat.indices(o, REJECT);
  if (o.g === 0) return { action: REJECT, features: rej, q: qValue(lq.w, rej) };
  const acc = lq.feat.indices(o, ACCEPT);
  const qr = qValue(lq.w, rej);
  const qa = qValue(lq.w, acc);
  let action: number;
  if (rng() < eps) action = rng() < 0.5 ? ACCEPT : REJECT;
  else if (qa === qr) action = rng() < 0.5 ? ACCEPT : REJECT;
  else action = qa > qr ? ACCEPT : REJECT;
  return action === ACCEPT ? { action, features: acc, q: qa } : { action, features: rej, q: qr };
}

function applyUpdate(w: Float64Array, features: readonly number[], step: number): void {
  for (let i = 0; i < features.length; i += 1) w[features[i]!]! += step;
}

export interface AgentOptions {
  alpha: number;
  eps: number;
  n: number;
  q0?: number;
  trace?: boolean;
}

/* ------------------------------------------------------------------ */
/* Episodic n-step semi-gradient Sarsa                                 */
/* ------------------------------------------------------------------ */

export interface EpisodicAgent extends LinearQ {
  opts: AgentOptions;
  rng: Rng;
  /** Ring buffers, indexed mod n + 1. */
  obs: Observation[];
  actions: number[];
  feats: number[][];
  rewards: number[];
  t: number;
  T: number;
  updates: number;
}

export function createEpisodicAgent(feat: Featurizer, opts: AgentOptions, rng: Rng): EpisodicAgent {
  const m = opts.n + 1;
  return {
    feat, w: createWeights(feat, opts.q0 ?? 0), opts, rng,
    obs: new Array(m), actions: new Array(m).fill(0), feats: new Array(m), rewards: new Array(m).fill(0), t: 0, T: Infinity, updates: 0,
  };
}

/** S_0 is observed: choose A_0. */
export function episodicBegin(ag: EpisodicAgent, o: Observation): number {
  const ch = choose(ag, o, ag.opts.eps, ag.rng);
  ag.t = 0;
  ag.T = Infinity;
  ag.obs[0] = o;
  ag.actions[0] = ch.action;
  ag.feats[0] = ch.features;
  return ch.action;
}

function episodicUpdate(ag: EpisodicAgent, tau: number): UpdateRecord | null {
  const { n } = ag.opts;
  const m = n + 1;
  const end = Math.min(tau + n, ag.T);
  const rewards: number[] = [];
  let G = 0;
  for (let i = tau + 1; i <= end; i += 1) { G += ag.rewards[i % m]!; rewards.push(ag.rewards[i % m]!); }
  let bootstrap: number | null = null;
  if (tau + n < ag.T) {
    bootstrap = qValue(ag.w, ag.feats[(tau + n) % m]!);
    G += bootstrap;
  }
  const f = ag.feats[tau % m]!;
  const qBefore = qValue(ag.w, f);
  const delta = G - qBefore;
  const dw = (ag.opts.alpha / ag.feat.active) * delta;
  applyUpdate(ag.w, f, dw);
  ag.updates += 1;
  if (!ag.opts.trace) return null;
  return { tau, obs: ag.obs[tau % m]!, action: ag.actions[tau % m]!, features: [...f], qBefore, rewards, bootstrap, target: G, delta, dw, qAfter: qValue(ag.w, f) };
}

/**
 * R_{t+1}, S_{t+1} observed. Returns A_{t+1} (or −1 at the end of the shift)
 * and the update records produced this step (several at the end, as the
 * remaining n − 1 pairs are flushed).
 */
export function episodicStep(ag: EpisodicAgent, reward: number, o: Observation, done: boolean): { action: number; records: UpdateRecord[] } {
  const m = ag.opts.n + 1;
  const t = ag.t;
  ag.rewards[(t + 1) % m] = reward;
  let action = -1;
  if (done) {
    ag.T = t + 1;
  } else {
    const ch = choose(ag, o, ag.opts.eps, ag.rng);
    action = ch.action;
    ag.obs[(t + 1) % m] = o;
    ag.actions[(t + 1) % m] = action;
    ag.feats[(t + 1) % m] = ch.features;
  }
  const records: UpdateRecord[] = [];
  let tau = t - ag.opts.n + 1;
  if (tau >= 0) {
    const r = episodicUpdate(ag, tau);
    if (r) records.push(r);
  }
  if (done) {
    for (tau = Math.max(0, tau + 1); tau <= ag.T - 1; tau += 1) {
      const r = episodicUpdate(ag, tau);
      if (r) records.push(r);
    }
  }
  ag.t = t + 1;
  return { action, records };
}

/* ------------------------------------------------------------------ */
/* Continuing: differential or discounted n-step semi-gradient Sarsa   */
/* ------------------------------------------------------------------ */

export interface ContinuingOptions extends AgentOptions {
  kind: 'differential' | 'discounted';
  /** Discounted only. */
  gamma?: number;
  /** Differential only. */
  rule?: AverageRule;
  beta?: number;
  rbar0?: number;
}

export interface ContinuingAgent extends LinearQ {
  opts: ContinuingOptions;
  rng: Rng;
  avg: AverageState;
  obs: Observation[];
  actions: number[];
  feats: number[][];
  rewards: number[];
  t: number;
  /** The action to take now. */
  action: number;
}

export function createContinuingAgent(feat: Featurizer, opts: ContinuingOptions, rng: Rng): ContinuingAgent {
  const m = opts.n + 1;
  return {
    feat, w: createWeights(feat, opts.q0 ?? 0), opts, rng,
    avg: createAverage(opts.rule ?? 'td', opts.beta ?? 0.01, opts.rbar0 ?? 0),
    obs: new Array(m), actions: new Array(m).fill(0), feats: new Array(m), rewards: new Array(m).fill(0), t: 0, action: REJECT,
  };
}

export function continuingBegin(ag: ContinuingAgent, o: Observation): number {
  const ch = choose(ag, o, ag.opts.eps, ag.rng);
  ag.t = 0;
  ag.obs[0] = o;
  ag.actions[0] = ch.action;
  ag.feats[0] = ch.features;
  ag.action = ch.action;
  return ch.action;
}

/** R_{t+1}, S_{t+1} observed: choose A_{t+1}, update the pair from time t − n + 1. */
export function continuingStep(ag: ContinuingAgent, reward: number, o: Observation): UpdateRecord | null {
  const { n } = ag.opts;
  const m = n + 1;
  const t = ag.t;
  const ch = choose(ag, o, ag.opts.eps, ag.rng);
  ag.rewards[(t + 1) % m] = reward;
  ag.obs[(t + 1) % m] = o;
  ag.actions[(t + 1) % m] = ch.action;
  ag.feats[(t + 1) % m] = ch.features;
  ag.action = ch.action;
  ag.t = t + 1;
  const tau = t - n + 1;
  if (tau < 0) return null;

  const differential = ag.opts.kind === 'differential';
  const gamma = ag.opts.gamma ?? 1;
  const rbarBefore = ag.avg.rbar;
  const rewards: number[] = [];
  let G = 0;
  let disc = 1;
  for (let i = tau + 1; i <= tau + n; i += 1) {
    const r = ag.rewards[i % m]!;
    rewards.push(r);
    G += differential ? r - rbarBefore : disc * r;
    disc *= gamma;
  }
  const bootstrap = qValue(ag.w, ag.feats[(tau + n) % m]!);
  G += differential ? bootstrap : disc * bootstrap;
  const f = ag.feats[tau % m]!;
  const qBefore = qValue(ag.w, f);
  const delta = G - qBefore;
  if (differential) updateAverage(ag.avg, reward, delta);
  const dw = (ag.opts.alpha / ag.feat.active) * delta;
  applyUpdate(ag.w, f, dw);
  if (!ag.opts.trace) return null;
  return {
    tau, obs: ag.obs[tau % m]!, action: ag.actions[tau % m]!, features: [...f], qBefore, rewards, bootstrap, target: G, delta, dw,
    qAfter: qValue(ag.w, f), rbarBefore, rbarAfter: ag.avg.rbar,
  };
}
