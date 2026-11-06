import test from 'node:test';
import assert from 'node:assert/strict';
import { makeBuckets, lengthAt, normalQuantile } from '../src/demos/gpu-cluster/core/buckets.ts';
import { BOOK_CONFIG, DEFAULT_CONFIG, PAY } from '../src/demos/gpu-cluster/core/config.ts';
import { buildModel, decodeState, afterstate, reward, policyFrom } from '../src/demos/gpu-cluster/core/model.ts';
import {
  evaluatePolicy, stepFull, discountedValues, differentialValues, relativeValueIteration, applyT,
} from '../src/demos/gpu-cluster/core/exact.ts';
import { alternating, ring, twoState, stationary, averageReward, differentialValues as chainValues } from '../src/demos/gpu-cluster/core/toy-chains.ts';
import { handTuned, thresholdOnModel } from '../src/demos/gpu-cluster/core/policies.ts';

const near = (a, b, eps, msg = '') => assert.ok(Math.abs(a - b) < eps, `${msg} ${a} ≠ ${b} (±${eps})`);

test('toy chains: rates and differential values', () => {
  near(averageReward(alternating()), 0.5, 1e-9);
  chainValues(alternating()).forEach((v, i) => near(v, [0.25, -0.25][i], 1e-9));
  near(averageReward(ring()), 1 / 3, 1e-9);
  chainValues(ring()).forEach((v, i) => near(v, [-1 / 3, 0, 1 / 3][i], 1e-9));
  stationary(twoState()).forEach((v, i) => near(v, [2 / 3, 1 / 3][i], 1e-9));
  near(averageReward(twoState()), 7 / 3, 1e-9);
});

test('σ = 0 gives the 44 states and transition structure of Example 10.2', () => {
  const model = buildModel(BOOK_CONFIG);
  assert.equal(model.nStates, 44);
  assert.equal(model.B, 1);
  near(model.buckets.q[0], 0.06, 1e-12);
  // From k busy servers, the number still busy after a step is Binomial(k, 0.94).
  for (let ci = 0; ci < model.nCounts; ci += 1) {
    const k = model.counts[ci][0];
    const row = model.trans[ci];
    let total = 0;
    row.next.forEach((j, idx) => {
      const left = model.counts[j][0];
      const freed = k - left;
      let coef = 1;
      for (let i = 0; i < freed; i += 1) coef = (coef * (k - i)) / (i + 1);
      near(row.prob[idx], coef * 0.06 ** freed * 0.94 ** left, 1e-12);
      total += row.prob[idx];
    });
    near(total, 1, 1e-12);
  }
  // Accepting pays the priority; with no free server it is a reject.
  for (let s = 0; s < model.nStates; s += 1) {
    const { ci, c } = decodeState(model, s);
    if (model.free[ci] > 0) assert.equal(reward(model, s, 1), PAY[c]);
    else { assert.equal(afterstate(model, s, 1), ci); assert.equal(reward(model, s, 1), 0); }
  }
});

test('default model has C(K+B, B)·4·B states', () => {
  assert.equal(buildModel(DEFAULT_CONFIG).nStates, 3432);
});

const policies = model => [
  new Uint8Array(model.nStates).fill(1),
  policyFrom(model, thresholdOnModel(handTuned(model.K))),
  policyFrom(model, o => (o.c >= 2 ? 1 : 0)),
  policyFrom(model, o => (o.g >= 5 ? 1 : 0)),
];

test('μ_π is invariant and J_γ(1 − γ) = r(π)', () => {
  for (const cfg of [BOOK_CONFIG, DEFAULT_CONFIG, { ...DEFAULT_CONFIG, kappa: 0.2 }]) {
    const model = buildModel(cfg);
    for (const pol of policies(model)) {
      const ev = evaluatePolicy(model, pol);
      const next = stepFull(model, pol, ev.mu);
      let err = 0;
      for (let s = 0; s < model.nStates; s += 1) err = Math.max(err, Math.abs(next[s] - ev.mu[s]));
      assert.ok(err < 1e-10, `‖μP − μ‖∞ = ${err}`);
      for (const gamma of [0, 0.5, 0.9, 0.99]) {
        near(discountedValues(model, pol, gamma, ev).J * (1 - gamma), ev.avg, 1e-8, `γ=${gamma}`);
      }
      const dv = differentialValues(model, pol, ev);
      let center = 0;
      for (let s = 0; s < model.nStates; s += 1) center += ev.mu[s] * dv.h[s];
      near(center, 0, 1e-9, 'Σ μ h');
    }
  }
});

test('RVI: small span residual, shift-invariant TD errors, transform matches plain RVI', () => {
  const model = buildModel(DEFAULT_CONFIG);
  const res = relativeValueIteration(model);
  assert.ok(res.residualSpan < 1e-8, `span ${res.residualSpan}`);
  near(evaluatePolicy(model, res.policy).avg, res.rStar, 1e-8, 'r(π*) = r*');

  // TD errors of the optimal differential values are unchanged by a constant shift.
  const TU = applyT(model, res.U);
  const shifted = applyT(model, res.U.map(v => v + 17.5));
  for (let s = 0; s < model.nStates; s += 50) {
    const a = res.policy[s];
    const after = afterstate(model, s, a);
    const { ci } = decodeState(model, s);
    const d1 = reward(model, s, a) - res.rStar + TU[after] - res.U[ci];
    const d2 = reward(model, s, a) - res.rStar + shifted[after] - (res.U[ci] + 17.5);
    near(d1, d2, 1e-9);
  }

  // The chain is aperiodic (binomial departures give self-loops), so plain RVI (η = 1) converges too.
  const plain = relativeValueIteration(model, { eta: 1 });
  near(plain.rStar, res.rStar, 1e-8);
  assert.deepEqual([...plain.policy], [...res.policy]);
});

test('bucket probabilities integrate correctly', () => {
  const dist = { median: 16, sigma: 0.6 };
  for (const B of [1, 2, 3, 4]) {
    const bk = makeBuckets(dist, B);
    // High-resolution midpoint rule in probability space.
    const M = 400000;
    const q = new Array(B).fill(0);
    for (let i = 0; i < M; i += 1) {
      const u = (i + 0.5) / M;
      q[Math.min(B - 1, Math.floor(u * B))] += 1 / lengthAt(dist, normalQuantile(u));
    }
    q.forEach((v, b) => near(bk.q[b], (v * B) / M, 2e-5, `B=${B} b=${b}`));
    // Sum consistency: the bucket average equals E[1/ℓ] from the one-bucket model.
    near(bk.q.reduce((s, v) => s + v, 0) / B, makeBuckets(dist, 1).q[0], 1e-6, 'mean');
    for (let b = 1; b < B; b += 1) assert.ok(bk.q[b] < bk.q[b - 1], 'longer buckets complete less often');
  }
});
