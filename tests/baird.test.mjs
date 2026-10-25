import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_ALPHA, DEFAULT_GAMMA, DEFAULT_WEIGHTS, TRANSITIONS,
  batchUpdate, features, hasDiverged, onlineUpdate, stateValues, value,
} from '../src/demos/baird/core.ts';

const close = (actual, expected, eps = 1e-9) =>
  actual.forEach((v, i) => assert.ok(Math.abs(v - expected[i]) < eps, `index ${i}: ${v} ≠ ${expected[i]}`));

test('features and values match the problem definition', () => {
  assert.deepEqual(features(3), [1, 0, 0, 2, 0, 0, 0]);
  assert.deepEqual(features(6), [2, 0, 0, 0, 0, 0, 1]);
  assert.equal(value(1, DEFAULT_WEIGHTS), 3);
  assert.equal(value(6, DEFAULT_WEIGHTS), 7);
  assert.deepEqual(TRANSITIONS, [[1, 6], [2, 6], [3, 6], [4, 6], [5, 6], [6, 6]]);
});

test('one batch from the defaults gives the textbook sanity check', () => {
  const record = batchUpdate(DEFAULT_WEIGHTS, DEFAULT_ALPHA, DEFAULT_GAMMA);
  close(record.after, [2.755, 1.73, 1.73, 1.73, 1.73, 1.73, 4.965]);
  close(stateValues(record.after), [6.215, 6.215, 6.215, 6.215, 6.215, 10.475]);
  close(record.steps.map(s => s.delta), [3.65, 3.65, 3.65, 3.65, 3.65, -0.35]);
  // Batch deltas all read the frozen weights.
  for (const s of record.steps) assert.equal(s.vTo, 7);
  close(record.deltaW, record.after.map((w, j) => w - DEFAULT_WEIGHTS[j]));
});

test('online updates move w between transitions', () => {
  const record = onlineUpdate(DEFAULT_WEIGHTS, DEFAULT_ALPHA, DEFAULT_GAMMA);
  close([record.steps[0].delta], [3.65]);
  assert.ok(record.steps[1].vTo > record.steps[0].vTo, 'the second transition sees the first update');
  const summed = record.steps.reduce((acc, s) => acc.map((a, j) => a + s.contribution[j]), new Array(7).fill(0));
  close(record.deltaW, summed);
});

test('defaults diverge in both modes; small γ converges to the true value 0', () => {
  for (const update of [batchUpdate, onlineUpdate]) {
    let w = [...DEFAULT_WEIGHTS];
    let i = 0;
    while (!hasDiverged(w) && i < 2000) { w = update(w, DEFAULT_ALPHA, DEFAULT_GAMMA).after; i += 1; }
    assert.ok(hasDiverged(w), `${update.name} should diverge`);
  }
  let w = [...DEFAULT_WEIGHTS];
  for (let i = 0; i < 5000; i += 1) w = batchUpdate(w, DEFAULT_ALPHA, 0.8).after;
  for (const v of stateValues(w)) assert.ok(Math.abs(v) < 1e-6);
});
