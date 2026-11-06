import test from 'node:test';
import assert from 'node:assert/strict';
import { createIHT, tiles, tileswrap } from '../src/demos/gpu-cluster/core/tiles.ts';
import { createFeaturizer, defaultFeatures } from '../src/demos/gpu-cluster/core/features.ts';
import { makeBuckets } from '../src/demos/gpu-cluster/core/buckets.ts';
import { DEFAULT_CONFIG } from '../src/demos/gpu-cluster/core/config.ts';

test('same inputs give the same indices', () => {
  const iht = createIHT(1024);
  const a = tiles(iht, 8, [1.3, 2.7], [1, 0]);
  const b = tiles(iht, 8, [1.3, 2.7], [1, 0]);
  assert.deepEqual(a, b);
  assert.equal(new Set(a).size, 8);
  assert.notDeepEqual(tiles(iht, 8, [1.3, 2.7], [1, 1]), a);
});

test('active features equal the total tilings of enabled groups', () => {
  for (const amp of [0, 0.8]) {
    const cfg = { ...DEFAULT_CONFIG, amp };
    const spec = defaultFeatures(cfg);
    const f = createFeaturizer(cfg, makeBuckets(cfg, 3), spec);
    const o = { g: 3, c: 2, len: 20, rho: 0.4, phase: 0.3, togo: Infinity, bucket: 1 };
    assert.equal(f.active, amp > 0 ? 16 : 12);
    assert.equal(new Set(f.indices(o, 1)).size, f.active);
  }
});

test('with wrap, φ = 0.999 and φ = 0.001 share a tile', () => {
  const iht = createIHT(4096);
  const late = tileswrap(iht, 4, [2, 0.999 * 4], [null, 4]);
  const early = tileswrap(iht, 4, [2, 0.001 * 4], [null, 4]);
  assert.ok(late.some(i => early.includes(i)));
  const iht2 = createIHT(4096);
  const late2 = tiles(iht2, 4, [2, 0.999 * 4]);
  const early2 = tiles(iht2, 4, [2, 0.001 * 4]);
  assert.ok(!late2.some(i => early2.includes(i)), 'without wrap they share nothing');
});
