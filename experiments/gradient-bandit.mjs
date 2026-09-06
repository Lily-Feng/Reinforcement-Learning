/**
 * The 10-armed testbed numbers quoted on the gradient bandit chapter page.
 *
 *   node experiments/gradient-bandit.mjs
 *
 * Gradient bandit with and without a reward baseline, at alpha = 0.1 and 0.4,
 * on two testbeds: the shifted one from Sutton & Barto figure 2.5
 * (q* ~ N(+4, 1)) and the standard one (q* ~ N(0, 1)). Running both is the
 * point: the baseline is what makes the method indifferent to that offset, so
 * the pair of runs isolates exactly what the baseline buys.
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

/**
 * @param alpha    step size
 * @param baseline subtract the running average reward from R_t
 * @param offset   mean of q*(a); 4 reproduces figure 2.5
 */
function run({ alpha, baseline, offset, seed }) {
  const optimal = new Float64Array(STEPS);
  const reward = new Float64Array(STEPS);
  // Preference of the optimal arm minus the mean preference, averaged over
  // runs: shows the separation the updates are actually building.
  const prefGap = new Float64Array(STEPS);

  for (let r = 0; r < RUNS; r++) {
    const rng = mulberry32(seed + r * 7919);
    const normal = makeNormal(rng);
    const q = new Float64Array(K);
    for (let a = 0; a < K; a++) q[a] = offset + normal();
    let bestArm = 0;
    for (let a = 1; a < K; a++) if (q[a] > q[bestArm]) bestArm = a;

    const H = new Float64Array(K);       // preferences, all zero => uniform policy
    const pi = new Float64Array(K);
    let avgReward = 0;                   // running sample average of all rewards

    for (let t = 0; t < STEPS; t++) {
      // softmax over preferences, shifted by the max for numerical stability
      let maxH = -Infinity;
      for (let a = 0; a < K; a++) if (H[a] > maxH) maxH = H[a];
      let sum = 0;
      for (let a = 0; a < K; a++) { pi[a] = Math.exp(H[a] - maxH); sum += pi[a]; }
      for (let a = 0; a < K; a++) pi[a] /= sum;

      // sample A_t from pi
      let u = rng(), A = K - 1, acc = 0;
      for (let a = 0; a < K; a++) { acc += pi[a]; if (u < acc) { A = a; break; } }

      const R = q[A] + normal();
      const B = baseline ? avgReward : 0;
      avgReward += (R - avgReward) / (t + 1);

      // H(a) <- H(a) + alpha (R - B) (1{a = A} - pi(a))
      for (let a = 0; a < K; a++) {
        H[a] += alpha * (R - B) * ((a === A ? 1 : 0) - pi[a]);
      }

      reward[t] += R;
      if (A === bestArm) optimal[t] += 1;
      let meanH = 0;
      for (let a = 0; a < K; a++) meanH += H[a];
      prefGap[t] += H[bestArm] - meanH / K;
    }
  }
  for (let t = 0; t < STEPS; t++) {
    optimal[t] = 100 * optimal[t] / RUNS;
    reward[t] /= RUNS;
    prefGap[t] /= RUNS;
  }
  return { optimal: Array.from(optimal), reward: Array.from(reward), prefGap: Array.from(prefGap) };
}

const SEED = 20260905;
const configs = [
  ['a0.1 baseline', { alpha: 0.1, baseline: true }],
  ['a0.4 baseline', { alpha: 0.4, baseline: true }],
  ['a0.1 no base ', { alpha: 0.1, baseline: false }],
  ['a0.4 no base ', { alpha: 0.4, baseline: false }],
];

const results = {};
for (const offset of [4, 0]) {
  for (const [name, cfg] of configs) {
    results[`${offset}|${name.trim()}`] = run({ ...cfg, offset, seed: SEED });
  }
}

const fmt = (x, d = 1) => x.toFixed(d);
for (const offset of [4, 0]) {
  console.log(`\n=== testbed q* ~ N(${offset}, 1) — % optimal action ===`);
  console.log('step  | ' + configs.map(([n]) => n).join(' | '));
  for (const s of [1, 10, 50, 100, 200, 500, 1000]) {
    console.log(
      String(s).padStart(5) + ' | ' +
      configs.map(([n]) => fmt(results[`${offset}|${n.trim()}`].optimal[s - 1]).padStart(13)).join(' | ')
    );
  }
  console.log('final avg reward: ' + configs
    .map(([n]) => `${n.trim()} ${fmt(results[`${offset}|${n.trim()}`].reward[999], 2)}`).join(' | '));
  console.log('preference gap of the optimal arm at step 1000: ' + configs
    .map(([n]) => `${n.trim()} ${fmt(results[`${offset}|${n.trim()}`].prefGap[999], 2)}`).join(' | '));
}

if (process.argv.includes('--json')) {
  const { writeFileSync } = await import('node:fs');
  const path = process.argv[process.argv.indexOf('--json') + 1];
  writeFileSync(path, JSON.stringify(results));
  console.log('\nwrote', path);
}
