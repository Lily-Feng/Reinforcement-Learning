/**
 * The 10-armed testbed numbers quoted on the UCB chapter page.
 *
 *   node experiments/ucb-testbed.mjs
 *
 * UCB (c = 2 and c = 1) against epsilon-greedy (0.1), sample-average estimates,
 * 2000 runs of 1000 steps. Also prints the diagnostics behind the step-11 spike:
 * how often step 11 picks the arm with the best first reward, how often step 12
 * is allowed to repeat it, and the true value of the arm step 11 lands on.
 *
 * Dependency-free and seeded, so the figures on the page can be reproduced
 * exactly rather than taken on trust.
 */
const RUNS = 2000, STEPS = 1000, K = 10;

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function makeNormal(rng) {
  let spare = null;
  return () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u, v, s;
    do { u = 2 * rng() - 1; v = 2 * rng() - 1; s = u * u + v * v; } while (s === 0 || s >= 1);
    const m = Math.sqrt(-2 * Math.log(s) / s);
    spare = v * m; return u * m;
  };
}

// argmax with uniform random tie-breaking (reservoir style)
function argmaxRandom(vals, rng) {
  let best = -Infinity, count = 0, idx = 0;
  for (let i = 0; i < vals.length; i++) {
    const v = vals[i];
    if (v > best + 1e-12) { best = v; count = 1; idx = i; }
    else if (Math.abs(v - best) <= 1e-12) { count++; if (rng() < 1 / count) idx = i; }
  }
  return idx;
}

function run(method, param, seed) {
  const reward = new Float64Array(STEPS);
  const optimal = new Float64Array(STEPS);
  // extra diagnostics
  let step11Optimal = 0, step12RepeatsStep11 = 0, step11SelectedQstar = 0;
  let step11IsArgmaxFirstReward = 0;

  for (let r = 0; r < RUNS; r++) {
    const rng = mulberry32(seed + r * 7919);
    const normal = makeNormal(rng);
    const q = new Float64Array(K);
    for (let a = 0; a < K; a++) q[a] = normal();
    let bestArm = 0;
    for (let a = 1; a < K; a++) if (q[a] > q[bestArm]) bestArm = a;

    const Q = new Float64Array(K), N = new Float64Array(K);
    const firstReward = new Float64Array(K).fill(NaN);
    const scratch = new Float64Array(K);
    let a11 = -1;

    for (let t = 0; t < STEPS; t++) {
      let A;
      if (method === 'ucb') {
        const lnt = Math.log(t + 1);
        for (let a = 0; a < K; a++) {
          scratch[a] = N[a] === 0 ? Infinity : Q[a] + param * Math.sqrt(lnt / N[a]);
        }
        A = argmaxRandom(scratch, rng);
      } else {
        A = rng() < param ? Math.floor(rng() * K) : argmaxRandom(Q, rng);
      }
      const R = q[A] + normal();
      N[A] += 1;
      Q[A] += (R - Q[A]) / N[A];
      if (Number.isNaN(firstReward[A])) firstReward[A] = R;
      reward[t] += R;
      if (A === bestArm) optimal[t] += 1;

      if (t === 10) {
        a11 = A;
        if (A === bestArm) step11Optimal++;
        step11SelectedQstar += q[A];
        let am = 0;
        for (let a = 1; a < K; a++) if (firstReward[a] > firstReward[am]) am = a;
        // firstReward[A] was just overwritten check: A's first reward set on its first pull (steps 1-10)
        if (A === am) step11IsArgmaxFirstReward++;
      }
      if (t === 11 && A === a11) step12RepeatsStep11++;
    }
  }
  for (let t = 0; t < STEPS; t++) { reward[t] /= RUNS; optimal[t] = 100 * optimal[t] / RUNS; }
  return {
    reward: Array.from(reward), optimal: Array.from(optimal),
    step11OptimalPct: 100 * step11Optimal / RUNS,
    step12RepeatPct: 100 * step12RepeatsStep11 / RUNS,
    step11MeanQstar: step11SelectedQstar / RUNS,
    step11ArgmaxFirstRewardPct: 100 * step11IsArgmaxFirstReward / RUNS,
  };
}

const SEED = 20260905;
const results = {
  ucb2: run('ucb', 2, SEED),
  ucb1: run('ucb', 1, SEED),
  eps: run('eps', 0.1, SEED),
};

const fmt = (x, d = 3) => x.toFixed(d);
console.log('step | ucb c=2 R | ucb c=1 R | eps R | ucb2 %opt | ucb1 %opt | eps %opt');
for (const s of [1,2,5,9,10,11,12,13,14,15,20,21,30,50,100,200,500,1000]) {
  const i = s - 1;
  console.log([s, fmt(results.ucb2.reward[i]), fmt(results.ucb1.reward[i]), fmt(results.eps.reward[i]),
    fmt(results.ucb2.optimal[i],1), fmt(results.ucb1.optimal[i],1), fmt(results.eps.optimal[i],1)].join(' | '));
}
const avg = (arr, a, b) => arr.slice(a - 1, b).reduce((x, y) => x + y, 0) / (b - a + 1);
console.log('\nmean reward steps 1-10:  ucb2', fmt(avg(results.ucb2.reward,1,10)), 'ucb1', fmt(avg(results.ucb1.reward,1,10)), 'eps', fmt(avg(results.eps.reward,1,10)));
console.log('mean reward steps 12-20: ucb2', fmt(avg(results.ucb2.reward,12,20)), 'ucb1', fmt(avg(results.ucb1.reward,12,20)));
console.log('mean reward steps 1-1000: ucb2', fmt(avg(results.ucb2.reward,1,1000)), 'ucb1', fmt(avg(results.ucb1.reward,1,1000)), 'eps', fmt(avg(results.eps.reward,1,1000)));
for (const [k, v] of Object.entries(results)) {
  console.log(k, 'step11 %optimal', fmt(v.step11OptimalPct,1), '| step11 mean q* of chosen', fmt(v.step11MeanQstar),
    '| step12 repeats step11', fmt(v.step12RepeatPct,1) + '%', '| step11 = argmax first reward', fmt(v.step11ArgmaxFirstRewardPct,1) + '%');
}
console.log('spike ratio R11/R12: ucb2', fmt(results.ucb2.reward[10]/results.ucb2.reward[11],2), 'ucb1', fmt(results.ucb1.reward[10]/results.ucb1.reward[11],2));
