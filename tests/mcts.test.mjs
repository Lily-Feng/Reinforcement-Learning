import test from 'node:test';
import assert from 'node:assert/strict';
import { greedyChild, initialCounts, snapshot } from '../src/demos/mcts/core.ts';

test('greedy selection picks the highest ratio, node 3', () => {
  assert.equal(greedyChild(), 3);
});

test('success backs up one return along the selected path, leaving siblings unchanged', () => {
  assert.deepEqual(snapshot(3).counts, initialCounts);
  assert.deepEqual(snapshot(4).child, [1, 1]);
  assert.deepEqual(snapshot(4).counts, initialCounts);
  assert.deepEqual(snapshot(5).counts, [[2, 4], [0, 1], [2, 2], [1, 2], [0, 1]]);
  const final = snapshot(6);
  assert.deepEqual(final.counts, [[3, 5], [0, 1], [2, 2], [1, 2], [0, 1]]);
  assert.deepEqual(final.counts.map(([w, n]) => w / n), [.6, 0, 1, .5, 0]);
  // Root totals remain the sum of its children's rollout counts.
  for (const i of [0, 1]) assert.equal(final.counts[0][i], final.counts.slice(1, 4).reduce((sum, pair) => sum + pair[i], 0));
});

test('failure increments visits but adds no successes; snapshots can rewind and reset', () => {
  assert.deepEqual(snapshot(6, false).counts, [[2, 5], [0, 1], [1, 2], [1, 2], [0, 1]]);
  assert.deepEqual(snapshot(6, false).child, [0, 1]);
  assert.deepEqual(snapshot(2).child, [0, 0]);
  assert.equal(snapshot(0).child, null);
  assert.deepEqual(snapshot(0).counts, initialCounts);
});
