/**
 * Heavy computations the lab runs in its worker. Pure functions of plain,
 * structured-cloneable inputs; progress is reported through a callback.
 */

import { mulberry32 } from '../../shared/rng.ts';
import {
  continuingBegin, continuingStep, createContinuingAgent, createEpisodicAgent, episodicBegin, episodicStep,
  greedyAction, qValue, type ContinuingOptions, type LinearQ,
} from './agents.ts';
import { ACCEPT, REJECT, TIERS, type ClusterConfig } from './config.ts';
import { bucketCounts, createEnv, observe, resetEnv, step, type Env, type Observation } from './env.ts';
import { discountedValues, evaluatePolicy, relativeValueIteration, solveShift } from './exact.ts';
import { greedyOnModel, toObservation } from './evaluate.ts';
import { createFeaturizer, defaultFeatures, type FeatureConfig } from './features.ts';
import { buildModel, countKey, observe as observeModel, policyFrom, reward as modelReward, afterstate, decodeState, stateIndex, type Model, type ModelPolicy } from './model.ts';
import { longRunAverage } from './montecarlo.ts';
import { acceptAll, enterpriseOnly, handTuned, modelPolicyAutopilot, thresholdOnModel, thresholdPolicy, type Policy, type Threshold } from './policies.ts';
import { ci95, mean, type Interval } from './stats.ts';
import { applyT } from './exact.ts';

export type Progress = (fraction: number, note?: string) => void;

/* ------------------------------------------------------------------ */

export interface Reference {
  rStar: number;
  policy: Uint8Array;
  nu: Float64Array;
  mu: Float64Array;
  U: Float64Array;
  acceptAll: number;
  enterprise: number;
  handTuned: number;
  iterations: number;
  residualSpan: number;
  B: number;
  nStates: number;
}

export function computeReference(cfg: ClusterConfig, progress?: Progress): Reference {
  const model = buildModel(cfg);
  const res = relativeValueIteration(model, { onProgress: (it, span) => progress?.(Math.min(0.95, Math.log10(1 / Math.max(span, 1e-12)) / 10), `RVI iteration ${it}`) });
  const ev = evaluatePolicy(model, res.policy);
  const score = (decide: (o: ReturnType<typeof observeModel>) => number) => evaluatePolicy(model, policyFrom(model, decide)).avg;
  return {
    rStar: res.rStar, policy: res.policy, nu: ev.nu, mu: ev.mu, U: res.U,
    acceptAll: score(() => ACCEPT), enterprise: score(o => (o.c === TIERS - 1 ? ACCEPT : REJECT)),
    handTuned: score(thresholdOnModel(handTuned(cfg.K))),
    iterations: res.iterations, residualSpan: res.residualSpan, B: model.B, nStates: model.nStates,
  };
}

export interface ShiftReference {
  decision: Uint8Array[];
  value: number;
  /** Expected remaining revenue from an empty cluster with k steps to go. */
  emptyValue: number[];
}

export function computeShift(cfg: ClusterConfig): ShiftReference {
  const model = buildModel(cfg);
  const sol = solveShift(model, cfg);
  return { decision: sol.decision, value: sol.value, emptyValue: sol.U.map(u => u[model.empty]!) };
}

/* ------------------------------------------------------------------ */

export type PolicySpec =
  | { kind: 'threshold'; th: Threshold }
  | { kind: 'acceptAll' }
  | { kind: 'enterprise' }
  | { kind: 'model'; policy: Uint8Array };

export function policyFromSpec(cfg: ClusterConfig, spec: PolicySpec): Policy {
  switch (spec.kind) {
    case 'threshold': return thresholdPolicy(spec.th);
    case 'acceptAll': return acceptAll;
    case 'enterprise': return enterpriseOnly;
    default: return modelPolicyAutopilot(buildModel(cfg), spec.policy);
  }
}

export function computeLongRun(cfg: ClusterConfig, spec: PolicySpec, steps: number, seed: number, progress?: Progress): Interval {
  return longRunAverage(cfg, policyFromSpec(cfg, spec), { steps, seed, onProgress: f => progress?.(f) });
}

/* ------------------------------------------------------------------ */
/* Training helpers                                                     */
/* ------------------------------------------------------------------ */

export function trainContinuing(cfg: ClusterConfig, spec: FeatureConfig, opts: ContinuingOptions, steps: number, seed: number, onCheckpoint?: (t: number, agent: LinearQ) => void, every = 0) {
  const env = createEnv(cfg, mulberry32(seed));
  const feat = createFeaturizer(cfg, env.buckets, spec);
  const agent = createContinuingAgent(feat, opts, mulberry32(seed + 7919));
  continuingBegin(agent, observe(env));
  for (let t = 1; t <= steps; t += 1) {
    continuingStep(agent, step(env, agent.action).reward, observe(env));
    if (every && onCheckpoint && t % every === 0) onCheckpoint(t, agent);
  }
  return { agent, env };
}

/** Free GPUs × job length grid of greedy decisions per tier, ρ at its typical value for each g. */
export interface DecisionMap {
  lengths: number[];
  /** accept[c][g][li] ∈ {0, 1}. */
  accept: number[][][];
}

export function typicalRho(cfg: ClusterConfig, meanQ: number, g: number): number {
  return (cfg.K - g) * meanQ;
}

export function decisionMap(cfg: ClusterConfig, lq: LinearQ, meanQ: number, extra: Partial<Observation> = {}, rhoScale = 1, nLen = 24): DecisionMap {
  const lengths = Array.from({ length: nLen }, (_, i) => 2 * (60 ** (i / (nLen - 1))));
  const accept = Array.from({ length: TIERS }, (_, c) => Array.from({ length: cfg.K + 1 }, (_, g) => lengths.map(len => {
    const o: Observation = { g, c, len, rho: typicalRho(cfg, meanQ, g) * rhoScale, phase: 0, togo: Infinity, bucket: 0, ...extra };
    return greedyAction(lq, o);
  })));
  return { lengths, accept };
}

/* ------------------------------------------------------------------ */
/* M07: which n learns fastest                                          */
/* ------------------------------------------------------------------ */

export interface SweepResult {
  ns: number[];
  alphas: number[];
  /** perf[ni][ai]: mean revenue per shift over the first `shifts` shifts, with CI over runs. */
  perf: Interval[][];
  /** Best-α learning curves (revenue per shift, mean over runs) per n. */
  curves: { n: number; alpha: number; points: number[] }[];
}

function runShifts(cfg: ClusterConfig, n: number, alpha: number, shifts: number, seed: number, q0 = 0, eps = 0.1): number[] {
  const env = createEnv(cfg, mulberry32(seed), { episodic: true });
  const feat = createFeaturizer(cfg, env.buckets, defaultFeatures(cfg, true));
  const ag = createEpisodicAgent(feat, { alpha, eps, n, q0 }, mulberry32(seed + 31));
  const out: number[] = [];
  for (let k = 0; k < shifts; k += 1) {
    resetEnv(env);
    let a = episodicBegin(ag, observe(env));
    let total = 0;
    while (!env.done) {
      const r = step(env, a);
      total += r.reward;
      a = episodicStep(ag, r.reward, observe(env), r.done).action;
    }
    out.push(total);
  }
  return out;
}

export function computeSweep(cfg: ClusterConfig, ns: number[], alphas: number[], runs: number, shifts: number, seed: number, progress?: Progress, q0 = 0, eps = 0.1): SweepResult {
  const perf: Interval[][] = [];
  const all: number[][][][] = [];
  const total = ns.length * alphas.length * runs;
  let done = 0;
  ns.forEach((n, ni) => {
    perf.push([]);
    all.push([]);
    alphas.forEach(alpha => {
      const curves: number[][] = [];
      for (let r = 0; r < runs; r += 1) {
        curves.push(runShifts(cfg, n, alpha, shifts, seed + 1000 * r, q0, eps));
        done += 1;
        progress?.(done / total, `n = ${n}, α = ${alpha}`);
      }
      perf[ni]!.push(ci95(curves.map(mean)));
      all[ni]!.push(curves);
    });
  });
  const curves = ns.map((n, ni) => {
    let best = 0;
    perf[ni]!.forEach((p, ai) => { if (p.mean > perf[ni]![best]!.mean) best = ai; });
    const runsCurves = all[ni]![best]!;
    return { n, alpha: alphas[best]!, points: Array.from({ length: shifts }, (_, k) => mean(runsCurves.map(c => c[k]!))) };
  });
  return { ns, alphas, perf, curves };
}

/* ------------------------------------------------------------------ */
/* M15: γ as an algorithm parameter                                     */
/* ------------------------------------------------------------------ */

export interface CompareEntry {
  label: string;
  gamma: number | null;
  /** Exact r(π) of the learned greedy policy on the bucketed model. */
  avg: number;
  policy: Uint8Array;
  map: DecisionMap;
  /** Exact r(π) at checkpoints. */
  curve: number[];
}

export function computeCompare(cfg: ClusterConfig, gammas: number[], steps: number, alpha: number, seed: number, progress?: Progress): CompareEntry[] {
  const model = buildModel(cfg);
  const spec = defaultFeatures(cfg);
  const configs: { label: string; gamma: number | null; opts: ContinuingOptions }[] = [
    ...gammas.map(g => ({ label: `Discounted γ = ${g}`, gamma: g, opts: { kind: 'discounted' as const, gamma: g, alpha, eps: 0.1, n: 1 } })),
    { label: 'Differential', gamma: null, opts: { kind: 'differential', alpha, beta: 0.01, eps: 0.1, n: 1, rule: 'td' } },
  ];
  const every = Math.max(1, Math.floor(steps / 20));
  return configs.map((c, i) => {
    const curve: number[] = [];
    const { agent } = trainContinuing(cfg, spec, c.opts, steps, seed, (t, lq) => {
      curve.push(evaluatePolicy(model, greedyOnModel(model, lq)).avg);
      progress?.((i + t / steps) / configs.length, c.label);
    }, every);
    const policy = greedyOnModel(model, agent);
    return { label: c.label, gamma: c.gamma, avg: evaluatePolicy(model, policy).avg, policy, map: decisionMap(cfg, agent, model.buckets.meanQ), curve };
  });
}

/* ------------------------------------------------------------------ */
/* M17: chattering                                                      */
/* ------------------------------------------------------------------ */

export type ChatterFeatures = 'coarse' | 'default' | 'tabular';

export interface ChatterResult {
  /** raster[k][i]: decision of the greedy policy at checkpoint k in sample state i. */
  raster: Uint8Array[];
  sample: number[];
  sampleLabels: string[];
  /** Exact r(π) of the greedy policy at each checkpoint. */
  avg: number[];
  /** μ*-weighted fraction of states whose decision changed since the previous checkpoint. */
  flips: number[];
  /** Checkpoints where r(π) fell by more than 0.02 since the previous one. */
  drops: number;
  every: number;
  rStar: number;
  nStates: number;
}

export function chatterFeatures(kind: ChatterFeatures, cfg: ClusterConfig): FeatureConfig {
  if (kind === 'default') return defaultFeatures(cfg);
  return { kind: 'tiles', groups: { g1: true, g2: false, g3: false, g4: false }, tilings: { g1: 2, g2: 4, g3: 4, g4: 4 }, tilesPerDim: 2, wrap: true, ihtSize: 8192 };
}

export function computeChatter(cfgIn: ClusterConfig, kind: ChatterFeatures, alpha: number, eps: number, steps: number, every: number, seed: number, progress?: Progress): ChatterResult {
  const cfg = kind === 'tabular' ? { ...cfgIn, sigma: 0, median: 1 / 0.06, B: 1 } : cfgIn;
  const model = buildModel(cfg);
  const opt = relativeValueIteration(model);
  const mu = evaluatePolicy(model, opt.policy).mu;
  // A fixed sample of decision states with some free GPUs, spread over (g, c, b).
  const sample: number[] = [];
  const sampleLabels: string[] = [];
  const rng = mulberry32(seed + 5);
  const candidates = Array.from({ length: model.nStates }, (_, s) => s).filter(s => mu[s]! > 1e-6 && model.free[decodeState(model, s).ci]! > 0);
  const want = Math.min(48, candidates.length);
  while (sample.length < want) {
    const s = candidates[Math.floor(rng() * candidates.length)]!;
    if (!sample.includes(s)) sample.push(s);
  }
  sample.sort((a, b) => {
    const A = observeModel(model, a);
    const Bo = observeModel(model, b);
    return A.c - Bo.c || Bo.g - A.g || A.len - Bo.len;
  });
  for (const s of sample) {
    const o = observeModel(model, s);
    sampleLabels.push(`${['Free', 'Basic', 'Pro', 'Ent'][o.c]} · g=${o.g} · ℓ≈${o.len.toFixed(0)}`);
  }
  const raster: Uint8Array[] = [];
  const avg: number[] = [];
  const flips: number[] = [];
  let prev: ModelPolicy | null = null;
  let drops = 0;
  const spec = kind === 'tabular' ? { kind: 'tabular' as const, groups: { g1: false, g2: false, g3: false, g4: false }, tilings: { g1: 0, g2: 0, g3: 0, g4: 0 }, tilesPerDim: 1, wrap: false, ihtSize: 0 } : chatterFeatures(kind, cfg);
  trainContinuing(cfg, spec, { kind: 'differential', alpha, beta: 0.01, eps, n: 1, rule: 'td' }, steps, seed, (t, lq) => {
    const pol = greedyOnModel(model, lq);
    raster.push(Uint8Array.from(sample, s => pol[s]!));
    const a = evaluatePolicy(model, pol).avg;
    if (prev) {
      let f = 0;
      for (let s = 0; s < pol.length; s += 1) if (pol[s] !== prev[s]) f += mu[s]!;
      flips.push(f);
      if (a < avg[avg.length - 1]! - 0.02) drops += 1;
    } else flips.push(0);
    avg.push(a);
    prev = pol;
    progress?.(t / steps);
  }, every);
  return { raster, sample, sampleLabels, avg, flips, drops, every, rStar: opt.rStar, nStates: model.nStates };
}

/* ------------------------------------------------------------------ */
/* M14: ranking threshold policies                                      */
/* ------------------------------------------------------------------ */

export interface FamilyEntry {
  label: string;
  th: Threshold;
  avg: number;
  /** J_γ(π) per γ in the grid. */
  J: number[];
  /** v^γ_π(s₀) per γ. */
  v0: number[];
}

export interface FamilyResult {
  gammas: number[];
  entries: FamilyEntry[];
  startLabel: string;
  /** Pairs (i, j) whose v^γ(s₀) order differs between some two γ in the grid. */
  swaps: { i: number; j: number; gammaA: number; gammaB: number }[];
}

export function familyPolicies(K: number): { label: string; th: Threshold }[] {
  const inf = Infinity;
  const out: { label: string; th: Threshold }[] = [];
  const thetas: number[][] = [[1, 1, 1, 1], [K + 1, 1, 1, 1], [K + 1, K + 1, 1, 1], [K + 1, K + 1, K + 1, 1]];
  for (const th of thetas) out.push({ label: `θ=(${th.map(v => (v > K ? '∞' : v)).join(',')})`, th: { theta: th, lambda: [inf, inf, inf, inf] } });
  for (const f0 of [0.5, 0.7, 0.9]) {
    for (const f1 of [0.2, 0.4]) {
      const th = [Math.ceil(f0 * K), Math.ceil(f1 * K), 2, 1];
      out.push({ label: `θ=(${th.join(',')})`, th: { theta: th, lambda: [inf, inf, inf, inf] } });
    }
  }
  for (const L of [12, 24, 48]) {
    const th = [Math.ceil(0.5 * K), Math.ceil(0.2 * K), 1, 1];
    out.push({ label: `θ=(${th.join(',')}) Λ≤${L}`, th: { theta: th, lambda: [L, L * 2, inf, inf] } });
  }
  for (const g of [2, 3, 4]) out.push({ label: `all θ=${g}`, th: { theta: [g, g, g, g], lambda: [inf, inf, inf, inf] } });
  out.push({ label: 'Pro+ only, θ=(∞,∞,2,1)', th: { theta: [K + 1, K + 1, 2, 1], lambda: [inf, inf, inf, inf] } });
  out.push({ label: 'short jobs only', th: { theta: [1, 1, 1, 1], lambda: [10, 10, 10, inf] } });
  return out;
}

export function computeFamily(cfg: ClusterConfig, gammas: number[], progress?: Progress): FamilyResult {
  const model = buildModel(cfg);
  const pols = familyPolicies(cfg.K);
  // s₀: a full-ish cluster (two free GPUs) facing a Free-tier long request.
  const startCount = model.counts.findIndex(n => model.K - n.reduce((s, v) => s + v, 0) === 2 && n[model.B - 1] === n.reduce((s, v) => s + v, 0));
  const s0 = stateIndex(model, Math.max(0, startCount), 0, model.B - 1);
  const entries = pols.map((p, i) => {
    const policy = policyFrom(model, thresholdOnModel(p.th));
    const ev = evaluatePolicy(model, policy);
    const J: number[] = [];
    const v0: number[] = [];
    for (const g of gammas) {
      const dv = discountedValues(model, policy, g, ev);
      J.push(dv.J);
      v0.push(dv.v[s0]!);
    }
    progress?.((i + 1) / pols.length);
    return { label: p.label, th: p.th, avg: ev.avg, J, v0 };
  });
  const swaps: FamilyResult['swaps'] = [];
  for (let i = 0; i < entries.length; i += 1) {
    for (let j = i + 1; j < entries.length; j += 1) {
      const d = gammas.map((_, k) => entries[i]!.v0[k]! - entries[j]!.v0[k]!);
      const pos = d.findIndex(v => v > 1e-9);
      const neg = d.findIndex(v => v < -1e-9);
      if (pos >= 0 && neg >= 0) swaps.push({ i, j, gammaA: gammas[Math.min(pos, neg)]!, gammaB: gammas[Math.max(pos, neg)]! });
    }
  }
  const o = observeModel(model, s0);
  return { gammas, entries, swaps, startLabel: `${o.g} free GPUs, busy jobs all long, Free-tier request with ℓ ≈ ${o.len.toFixed(0)}` };
}

/* ------------------------------------------------------------------ */
/* M06: n-step targets                                                  */
/* ------------------------------------------------------------------ */

export interface TargetsResult {
  ns: (number | null)[];
  /** targets[i]: samples of G_{t:t+n} for ns[i] (null = ∞, the full return). */
  targets: number[][];
  /** Monte Carlo estimate of q_π(s, a) (the n = ∞ samples) with CI. */
  mc: Interval;
  /** Exact q_π(s, a) on the bucketed model. */
  exact: number;
  /** One sample trajectory for the timeline. */
  example: { rewards: number[]; qhat: number[] };
  stateLabel: string;
}

/**
 * Targets from one fixed bucketed state under the hand-tuned threshold policy,
 * bootstrapping from a deliberately imperfect q̂ = 0.8·q_π + 5 (exact q_π of
 * the same policy, scaled and shifted).
 */
export function computeTargets(cfg: ClusterConfig, ns: (number | null)[], samples: number, seed: number, progress?: Progress): TargetsResult {
  const model = buildModel(cfg);
  const th = handTuned(cfg.K);
  const pol = policyFrom(model, thresholdOnModel(th));
  const fixed = solveShift(model, cfg, (_k, s) => pol[s]!);
  const H = cfg.H;
  const TUs = fixed.U.map(u => applyT(model, u));
  const qExact = (k: number, s: number, a: number) => modelReward(model, s, a) + TUs[k - 1]![afterstate(model, s, a)]!;
  const qhat = (k: number, s: number, a: number) => (k <= 0 ? 0 : 0.8 * qExact(k, s, a) + 5);
  const index = new Map(model.counts.map((n, i) => [countKey(n), i]));
  const env = createEnv(cfg, mulberry32(seed), { mode: 'bucketed', episodic: true });
  const policy = thresholdPolicy(th);
  // Start: 8 steps into the shift under the policy, then evaluate (S_t, accept).
  const t0 = 8;
  const prep = (rngSeed: number): Env => {
    env.rng = mulberry32(rngSeed);
    resetEnv(env);
    for (let t = 0; t < t0; t += 1) step(env, policy(observe(env), env));
    return env;
  };
  prep(seed);
  const startQ = Float64Array.from(env.q);
  const startTier = Int8Array.from(env.tier);
  const startBucket = Int8Array.from(env.bucket);
  const startLen = Float64Array.from(env.len);
  const startHead = { ...env.head };
  const startTau = env.tau;
  const s0 = stateIndex(model, index.get(countKey(bucketCounts(env)))!, startHead.c, startHead.bucket);
  const a0 = ACCEPT;
  const kStart = H - t0;
  const targets = ns.map(() => [] as number[]);
  let example = { rewards: [] as number[], qhat: [] as number[] };
  const rng = mulberry32(seed + 99);
  for (let i = 0; i < samples; i += 1) {
    env.q.set(startQ); env.tier.set(startTier); env.bucket.set(startBucket); env.len.set(startLen);
    env.head = { ...startHead }; env.tau = startTau; env.t = t0; env.done = false; env.rng = rng;
    const rewards: number[] = [];
    const qh: number[] = [];
    let a = a0;
    while (!env.done) {
      const k = H - env.t;
      const ci = index.get(countKey(bucketCounts(env)))!;
      qh.push(qhat(k, stateIndex(model, ci, env.head.c, env.head.bucket), a));
      rewards.push(step(env, a).reward);
      if (!env.done) a = policy(observe(env), env);
    }
    qh.push(0);
    ns.forEach((n, j) => {
      const m = n === null ? rewards.length : Math.min(n, rewards.length);
      let G = 0;
      for (let t = 0; t < m; t += 1) G += rewards[t]!;
      G += m < rewards.length ? qh[m]! : 0;
      targets[j]!.push(G);
    });
    if (i === 0) example = { rewards, qhat: qh };
    if (i % 200 === 0) progress?.(i / samples);
  }
  const full = targets[ns.indexOf(null)] ?? targets[targets.length - 1]!;
  const o = observeModel(model, s0);
  return {
    ns, targets, mc: ci95(full), exact: qExact(kStart, s0, a0), example,
    stateLabel: `${o.g} free GPUs, ${['Free', 'Basic', 'Pro', 'Enterprise'][o.c]} request, ℓ ≈ ${o.len.toFixed(0)}, ${kStart} steps left; action = accept`,
  };
}

/* ------------------------------------------------------------------ */
/* M09: replicas of the cluster                                         */
/* ------------------------------------------------------------------ */

export interface ReplicaResult {
  /** dist[t][g]: fraction of replicas with g free GPUs at step t. */
  dist: number[][];
  /** Running time-average reward of replica 0. */
  running: number[];
  /** Mean over replicas of the reward at step t. */
  meanReward: number[];
}

export function computeReplicas(cfg: ClusterConfig, replicas: number, T: number, seed: number, progress?: Progress): ReplicaResult {
  const policy = thresholdPolicy(handTuned(cfg.K));
  const envs = Array.from({ length: replicas }, (_, i) => createEnv(cfg, mulberry32(seed + i * 7)));
  const dist: number[][] = [];
  const running: number[] = [];
  const meanReward: number[] = [];
  let sum0 = 0;
  for (let t = 0; t < T; t += 1) {
    const counts = new Array<number>(cfg.K + 1).fill(0);
    let rsum = 0;
    envs.forEach((env, i) => {
      const o = observe(env);
      counts[o.g]! += 1;
      const r = step(env, policy(o, env)).reward;
      rsum += r;
      if (i === 0) { sum0 += r; running.push(sum0 / (t + 1)); }
    });
    dist.push(counts.map(c => c / replicas));
    meanReward.push(rsum / replicas);
    if (t % 20 === 0) progress?.(t / T);
  }
  return { dist, running, meanReward };
}

export { greedyAction, qValue, toObservation };
