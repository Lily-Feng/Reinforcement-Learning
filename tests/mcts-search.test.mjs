import test from 'node:test';
import assert from 'node:assert/strict';
import { buildSearch, children, terminals } from '../src/demos/mcts/search.ts';

test('default walkthrough demonstrates every operation, both returns, and a direct terminal visit', () => {
  const frames = buildSearch();
  for (const phase of ['select', 'expand', 'rollout', 'return', 'backup', 'complete', 'choose']) assert.ok(frames.some(f => f.phase === phase), phase);
  for (const result of [0, 1]) assert.ok(frames.some(f => f.phase === 'return' && f.result === result));
  assert.ok(frames.some(f => f.phase === 'return' && f.rollout.length === 1));
  assert.equal(frames.filter(f => f.phase === 'complete').length, 18);
  const final = frames.at(-1);
  assert.equal(final.tree.find(n => n.id === 1).visits, 18);
  const rootChildren = final.tree.filter(n => [2, 3].includes(n.id));
  assert.equal(final.tree.find(n => n.id === final.active).visits, Math.max(...rootChildren.map(n => n.visits)));
});

test('only expansion adds nodes; rollout traverses model states without storing them', () => {
  const frames = buildSearch();
  for (let i = 1; i < frames.length; i++) {
    const f = frames[i], prev = frames[i-1];
    assert.equal(f.tree.length - prev.tree.length, f.phase === 'expand' ? 1 : 0);
    if (f.phase === 'expand') {
      assert.deepEqual(f.tree.find(n => n.id === f.active), {id:f.active, wins:0, visits:0});
      assert.ok(prev.tree.some(n => n.id === Math.floor(f.active / 2)));
    }
    if (f.phase === 'rollout') {
      assert.deepEqual(f.tree, prev.tree);
      assert.ok(f.rollout.slice(1).every(id => !f.tree.some(n => n.id === id)));
    }
    for (let j = 1; j < f.rollout.length; j++) assert.ok(children(f.rollout[j-1]).includes(f.rollout[j]));
    if (f.phase === 'return') assert.equal(f.result, terminals[f.rollout.at(-1)]);
  }
});

test('backup changes exactly one path node by the sampled return; final counts conserve visits', () => {
  for (const policy of ['uct', 'greedy']) for (const seed of [0, 7, 42]) {
    const frames = buildSearch(policy, seed);
    for (let i = 1; i < frames.length; i++) {
      const f = frames[i], prev = frames[i-1];
      if (f.phase !== 'backup') continue;
      assert.ok(f.path.includes(f.updated));
      for (const node of f.tree) {
        const old = prev.tree.find(n => n.id === node.id);
        assert.equal(node.visits, old.visits + Number(node.id === f.updated));
        assert.equal(node.wins, old.wins + (node.id === f.updated ? f.result : 0));
      }
    }
    for (const complete of frames.filter(f => f.phase === 'complete')) {
      assert.equal(complete.tree.find(n=>n.id===1).visits, complete.iteration);
      assert.equal(complete.tree.find(n=>n.id===1).wins, frames.filter(f=>f.phase==='return' && f.iteration<=complete.iteration).reduce((sum,f)=>sum+f.result,0));
    }
  }
});

test('UCT exposes correct score terms and selects the highest score', () => {
  for (const policy of ['uct', 'greedy']) for (const f of buildSearch(policy)) if (f.choices.length) {
    const parent = f.tree.find(n => n.id === Math.floor(f.active / 2));
    for (const c of f.choices) {
      const n = f.tree.find(n => n.id === c.id);
      assert.equal(c.mean, n.wins / n.visits);
      assert.equal(c.bonus, policy === 'uct' ? Math.SQRT2 * Math.sqrt(Math.log(parent.visits) / n.visits) : 0);
      assert.equal(c.score, c.mean + c.bonus);
    }
    assert.equal(f.choices.find(c=>c.id===f.active).score, Math.max(...f.choices.map(c=>c.score)));
  }
});

test('seeded replay is deterministic and snapshots do not share mutable statistics', () => {
  const a = buildSearch('uct',7), b = buildSearch('uct',7);
  assert.deepEqual(a, b);
  assert.notDeepEqual(a, buildSearch('uct',42));
  const rootBefore = a[0].tree[0].visits;
  a.at(-1).tree[0].visits++;
  assert.equal(a[0].tree[0].visits, rootBefore);
});
