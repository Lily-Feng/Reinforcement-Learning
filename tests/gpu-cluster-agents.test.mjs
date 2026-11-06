import test from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32 } from '../src/demos/shared/rng.ts';
import { BOOK_CONFIG, DEFAULT_CONFIG } from '../src/demos/gpu-cluster/core/config.ts';
import { createEnv, observe, step } from '../src/demos/gpu-cluster/core/env.ts';
import { createFeaturizer, defaultFeatures, TABULAR } from '../src/demos/gpu-cluster/core/features.ts';
import {
  createEpisodicAgent, episodicBegin, episodicStep, createAverage, updateAverage,
  createContinuingAgent, continuingBegin, continuingStep,
} from '../src/demos/gpu-cluster/core/agents.ts';
import { buildModel } from '../src/demos/gpu-cluster/core/model.ts';
import { relativeValueIteration } from '../src/demos/gpu-cluster/core/exact.ts';
import { scoreGreedy } from '../src/demos/gpu-cluster/core/evaluate.ts';
import { ring, sampleStep, differentialValues } from '../src/demos/gpu-cluster/core/toy-chains.ts';

test('episodic n-step Sarsa with n ≥ H uses the Monte Carlo target', () => {
  const cfg = { ...DEFAULT_CONFIG, H: 24 };
  const env = createEnv(cfg, mulberry32(2), { episodic: true });
  const feat = createFeaturizer(cfg, env.buckets, defaultFeatures(cfg, true));
  const ag = createEpisodicAgent(feat, { alpha: 0.1, eps: 0.2, n: 30, trace: true }, mulberry32(3));
  let a = episodicBegin(ag, observe(env));
  const rewards = [];
  const records = [];
  while (!env.done) {
    const r = step(env, a);
    rewards.push(r.reward);
    const res = episodicStep(ag, r.reward, observe(env), r.done);
    records.push(...res.records);
    a = res.action;
  }
  assert.equal(records.length, cfg.H);
  records.forEach(rec => {
    const G = rewards.slice(rec.tau).reduce((s, v) => s + v, 0);
    assert.equal(rec.bootstrap, null);
    assert.ok(Math.abs(rec.target - G) < 1e-12, `τ=${rec.tau}`);
  });
});

test('ring: td rule with frozen exact values gives δ = 0; reward rule cycles −1/3, −1/3, +2/3', () => {
  const chain = ring();
  const v = differentialValues(chain);
  const td = createAverage('td', 0, 1 / 3);
  const rw = createAverage('reward', 0, 1 / 3);
  const rng = mulberry32(1);
  let s = 0;
  const errors = [];
  for (let t = 0; t < 12; t += 1) {
    const { next, reward } = sampleStep(chain, s, rng);
    const delta = reward - td.rbar + v[next] - v[s];
    assert.ok(Math.abs(delta) < 1e-12, `δ = ${delta}`);
    updateAverage(td, reward, delta);
    errors.push(reward - rw.rbar);
    updateAverage(rw, reward, delta);
    s = next;
  }
  errors.forEach((e, t) => assert.ok(Math.abs(e - [-1 / 3, -1 / 3, 2 / 3][t % 3]) < 1e-12));
});

test('book preset: learned greedy policy reaches at least 97% of r*', () => {
  const model = buildModel(BOOK_CONFIG);
  const { rStar } = relativeValueIteration(model);
  const env = createEnv(BOOK_CONFIG, mulberry32(1));
  const feat = createFeaturizer(BOOK_CONFIG, env.buckets, TABULAR);
  const ag = createContinuingAgent(feat, { kind: 'differential', alpha: 0.01, beta: 0.01, eps: 0.1, n: 1, rule: 'td' }, mulberry32(2));
  continuingBegin(ag, observe(env));
  for (let t = 0; t < 2000000; t += 1) continuingStep(ag, step(env, ag.action).reward, observe(env));
  const { avg } = scoreGreedy(model, ag);
  assert.ok(avg >= 0.97 * rStar, `${avg} vs r* = ${rStar}`);
});
