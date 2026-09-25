import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { initial, advance, backup, transition, greedy } from '../src/demos/dynamic-programming/core.ts';
const solve = config => {
  let state = initial(config);
  for (let i = 0; i < 160000; i++) {
    if (state.phase === 'done') return state;
    state = advance(config, state);
  }
  throw new Error('Did not converge');
};
const near = (a,b,tolerance=1e-10) => assert.ok(Math.abs(a-b) < tolerance, `${a} ≠ ${b}`);
const base = { size:2, gamma:1, theta:1e-8, mode:'value' };

test('matches original Python V and Q at four discounts and two thresholds', () => {
  const fixtures=JSON.parse(readFileSync(new URL('./fixtures/dp-python.json', import.meta.url)));
  assert.equal(fixtures.length, 16);
  for (const fixture of fixtures) {
    const state = solve(fixture.config);
    state.v.forEach((v,s) => near(v,fixture.v[s]));
    state.q.forEach((row,s) => row.forEach((q,a) => near(q,fixture.q[s][a])));
  }
});
test('original optimal values and deterministic first-maximum actions', () => {
  const state=solve(base);
  assert.deepEqual(state.v,[-2,-1,-1,0]);
  assert.deepEqual(state.q.slice(0,3).map(greedy),[2,3,2]);
});
test('uniform random policy has analytic undiscounted values', () => {
  const state=solve({...base,mode:'evaluation'});
  [-8,-6,-6,0].forEach((v,s)=>near(state.v[s],v,1e-6));
  assert.deepEqual(state.policy[0],[.25,.25,.25,.25]);
});
test('policy iteration reaches a stable optimal policy on both grid sizes', () => {
  for(const size of [2,4]) {
    const state=solve({...base,size,mode:'policy'});
    assert.equal(state.changes,0);
    assert.ok(state.improvements>=2);
    state.v.forEach((v,s)=>near(v,-((size-1-Math.floor(s/size))+(size-1-s%size)),1e-6));
  }
});
test('zero discount eliminates all future reward', () => {
  for(const mode of ['value','evaluation','policy']) assert.deepEqual(solve({...base,gamma:0,mode}).v,[-1,-1,-1,0]);
});
test('wall self-loops and terminal entry preserve the step cost', () => {
  assert.deepEqual(transition(2,0,0),{prob:1,next:0,reward:-1,done:false});
  assert.deepEqual(transition(2,1,3),{prob:1,next:3,reward:-1,done:true});
  assert.deepEqual(transition(2,3,0),{prob:1,next:3,reward:0,done:true});
  const state=initial(base);state.v[3]=100;
  assert.equal(backup(base,state,1).q[3],-1);
});
test('evaluation reads newly updated values within the same sweep', () => {
  const config={...base,mode:'evaluation'};
  const first=initial(config), second=advance(config,first), third=advance(config,second);
  assert.equal(second.v[0],-1);
  assert.equal(third.v[1],-1.25);
  assert.deepEqual(first.v,[0,0,0,0]);
  assert.equal(third.sweeps,0);
});
test('policy improvement is a separate phase and does not change V', () => {
  const config={...base,mode:'policy'};
  let state=initial(config);
  while(state.phase==='evaluate')state=advance(config,state);
  assert.equal(state.phase,'improve');
  const next=advance(config,state);
  assert.deepEqual(next.v,state.v);
  assert.equal(next.phase,'evaluate');
  assert.equal(next.improvements,1);
});
test('a completed run is idempotent', () => {
  const state=solve(base);assert.equal(advance(base,state),state);
});
