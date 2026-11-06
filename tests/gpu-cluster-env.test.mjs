import test from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32 } from '../src/demos/shared/rng.ts';
import { DEFAULT_CONFIG, tierMix, STEPS_PER_DAY } from '../src/demos/gpu-cluster/core/config.ts';
import { createEnv, observe, step, bucketCounts } from '../src/demos/gpu-cluster/core/env.ts';
import { buildModel, policyFrom } from '../src/demos/gpu-cluster/core/model.ts';
import { evaluatePolicy } from '../src/demos/gpu-cluster/core/exact.ts';
import { makeBuckets } from '../src/demos/gpu-cluster/core/buckets.ts';
import { handTuned, thresholdDecide, thresholdPolicy, thresholdOnModel } from '../src/demos/gpu-cluster/core/policies.ts';
import { longRunAverage } from '../src/demos/gpu-cluster/core/montecarlo.ts';

const run = (seed, steps) => {
  const env = createEnv(DEFAULT_CONFIG, mulberry32(seed));
  const out = [];
  for (let t = 0; t < steps; t += 1) {
    const o = observe(env);
    out.push(o.g, o.c, o.len, step(env, (t % 3) ? 1 : 0).reward);
  }
  return out;
};

test('same seed, same trajectory; accept with no free GPU is a reject', () => {
  assert.deepEqual(run(7, 500), run(7, 500));
  assert.notDeepEqual(run(7, 500), run(8, 500));
  const env = createEnv({ ...DEFAULT_CONFIG, K: 4, median: 120, sigma: 0 }, mulberry32(3));
  let full = 0;
  for (let t = 0; t < 400; t += 1) {
    const g = observe(env).g;
    const res = step(env, 1);
    if (g === 0) { full += 1; assert.equal(res.accepted, false); assert.equal(res.reward, 0); }
  }
  assert.ok(full > 50, 'the cluster filled up');
});

test('bucketed mode matches the exact r(π) within 2%', () => {
  const th = handTuned(DEFAULT_CONFIG.K);
  const model = buildModel(DEFAULT_CONFIG);
  const exact = evaluatePolicy(model, policyFrom(model, thresholdOnModel(th))).avg;
  const env = createEnv(DEFAULT_CONFIG, mulberry32(11), { mode: 'bucketed' });
  let sum = 0;
  const N = 1000000;
  for (let t = 0; t < N; t += 1) sum += step(env, thresholdDecide(th, observe(env))).reward;
  assert.ok(Math.abs(sum / N - exact) / exact < 0.02, `${sum / N} vs ${exact}`);
});

test('continuous mode: exact bucketed r(π) approaches the Monte Carlo estimate as B grows', () => {
  const th = handTuned(DEFAULT_CONFIG.K);
  const mc = longRunAverage(DEFAULT_CONFIG, thresholdPolicy(th), { steps: 1200000, seed: 5 });
  const errs = [1, 2, 3, 4].map(B => {
    const model = buildModel({ ...DEFAULT_CONFIG, B });
    return Math.abs(evaluatePolicy(model, policyFrom(model, thresholdOnModel(th))).avg - mc.mean) / mc.mean;
  });
  assert.ok(errs[3] < 0.03, `B=4 error ${errs[3]}`);
  assert.ok(errs[3] < errs[0], `errors ${errs.join(', ')}`);
});

test('observation on bucketed states: ρ = Σ n_b q_b and g = K − Σ n_b', () => {
  const env = createEnv(DEFAULT_CONFIG, mulberry32(4), { mode: 'bucketed' });
  const bk = makeBuckets(DEFAULT_CONFIG, DEFAULT_CONFIG.B);
  for (let t = 0; t < 3000; t += 1) {
    const o = observe(env);
    const n = bucketCounts(env);
    assert.equal(o.g, DEFAULT_CONFIG.K - n.reduce((s, v) => s + v, 0));
    assert.ok(Math.abs(o.rho - n.reduce((s, v, b) => s + v * bk.q[b], 0)) < 1e-12);
    step(env, t % 2);
  }
});

test('empirical tier frequencies match w(φ)', () => {
  const cfg = { ...DEFAULT_CONFIG, amp: 0.8 };
  const env = createEnv(cfg, mulberry32(9));
  const counts = Array.from({ length: STEPS_PER_DAY }, () => [0, 0, 0, 0]);
  for (let t = 0; t < STEPS_PER_DAY * 6000; t += 1) {
    counts[env.tau][observe(env).c] += 1;
    step(env, 0);
  }
  for (const tau of [0, 24, 48, 72]) {
    const w = tierMix(cfg, tau / STEPS_PER_DAY);
    const total = counts[tau].reduce((s, v) => s + v, 0);
    counts[tau].forEach((k, c) => assert.ok(Math.abs(k / total - w[c]) < 0.02, `τ=${tau} c=${c}: ${k / total} vs ${w[c]}`));
  }
});
