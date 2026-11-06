/**
 * M12 · How R̄ should be updated. Ring panel (Exercise 10.8): with exact
 * values, δ is exactly 0 while R − R̄ cycles. Cluster panel: four R̄ rules
 * estimating r(π) of a fixed policy in the diurnal cluster, against a Monte
 * Carlo band, with a mid-run traffic shift.
 */

import { drawXY } from '../../shared/bars';
import { mulberry32 } from '../../shared/rng';
import { createAverage, qValue, RULE_LABELS, updateAverage, type AverageRule, type AverageState } from '../core/agents.ts';
import { createEnv, observe, step, type Env } from '../core/env.ts';
import { createFeaturizer, defaultFeatures, type Featurizer } from '../core/features.ts';
import { handTuned, thresholdDecide } from '../core/policies.ts';
import { differentialValues, ring } from '../core/toy-chains.ts';
import type { Interval } from '../core/stats.ts';
import type { ModuleContext } from '../app/context';
import { button, canvasPanel, card, check, createRunner, el, fmt, legend, moduleHeader, observeBox, range, runnerControls, theme } from '../app/ui';
import { kpi, recorder } from './common';

const RULES: AverageRule[] = ['td', 'reward', 'sample-average', 'unbiased'];

interface Local {
  ringBeta: number;
  ringR0: number;
  frozen: boolean;
  beta: number;
  rbar0: number;
  speed: number;
}

export function mount(root: HTMLElement, ctx: ModuleContext): () => void {
  const store = ctx.store;
  const L = store.local<Local>('m12', { ringBeta: 0.1, ringR0: 0, frozen: true, beta: 0.001, rbar0: 20, speed: 200 });
  const t0 = theme();

  // --- ring panel ---------------------------------------------------------
  const chain = ring();
  const v = differentialValues(chain);
  const ringTable = el('table');
  const ringChart = canvasPanel('R̄ on the ring, both rules, exact values', 200, canvas => {
    const t = theme();
    const traces = (rule: AverageRule) => {
      const st = createAverage(rule, L.ringBeta, L.ringR0);
      let s = 0;
      const pts: number[][] = [[0, st.rbar]];
      for (let k = 1; k <= 90; k += 1) {
        const next = (s + 1) % 3;
        const R = chain.R[s]![next]!;
        updateAverage(st, R, R - st.rbar + v[next]! - v[s]!);
        pts.push([k, st.rbar]);
        s = next;
      }
      return pts;
    };
    drawXY(canvas, [{ color: t.series[0]!, points: traces('td') }, { color: t.series[1]!, points: traces('reward') }], {
      xLabel: 'step', refs: [{ value: 1 / 3, color: t.ref, label: 'r(π) = 1/3' }], textColor: t.text, gridColor: t.grid,
    });
  }, { legend: legend([{ label: 'R̄ ← R̄ + βδ', color: t0.series[0]! }, { label: 'R̄ ← R̄ + β(R − R̄)', color: t0.series[1]! }, { label: 'true 1/3', color: t0.ref, dash: true }]) });

  const renderRing = () => {
    const rbar = L.frozen ? 1 / 3 : L.ringR0;
    const st = createAverage('td', L.ringBeta, rbar);
    let s = 0;
    const rows: HTMLElement[] = [];
    for (let k = 0; k < 9; k += 1) {
      const next = (s + 1) % 3;
      const R = chain.R[s]![next]!;
      const before = st.rbar;
      const delta = R - before + v[next]! - v[s]!;
      rows.push(el('tr', {}, el('td', { text: String(k) }), el('td', { class: 'l', text: `${chain.states[s]} → ${chain.states[next]}` }), el('td', { text: String(R) }),
        el('td', { text: fmt(before, 4) }), el('td', { text: fmt(R - before, 4) }), el('td', { text: `${fmt(v[next]! - v[s]!, 4)}` }), el('td', { text: fmt(delta, 4) })));
      if (!L.frozen) updateAverage(st, R, delta);
      s = next;
    }
    ringTable.replaceChildren(
      el('thead', {}, el('tr', {}, ...['t', 'transition', 'R', 'R̄', 'R − R̄', 'v(S′) − v(S)', 'δ'].map(h => el('th', { text: h })))),
      el('tbody', {}, ...rows));
    ringChart.redraw();
  };

  // --- cluster panel ------------------------------------------------------
  const cfg = { ...ctx.cfg, amp: 0.8 };
  const th = handTuned(cfg.K);
  let env: Env;
  let feat: Featurizer;
  let learners: { rule: AverageRule; w: Float64Array; avg: AverageState; series: ReturnType<typeof recorder> }[] = [];
  let prevIdx: number[] = [];
  let steps = 0;
  let shiftedAt: number | null = null;
  let mcBefore: Interval | null = null;
  let mcAfter: Interval | null = null;
  const alpha = 0.1;

  const resetCluster = () => {
    env = createEnv({ ...cfg, base: [...cfg.base] }, mulberry32(cfg.seed));
    feat = createFeaturizer(cfg, env.buckets, defaultFeatures(cfg));
    learners = RULES.map(rule => ({ rule, w: new Float64Array(feat.size), avg: createAverage(rule, L.beta, L.rbar0), series: recorder(800) }));
    const o = observe(env);
    prevIdx = feat.indices(o, thresholdDecide(th, o));
    steps = 0;
    shiftedAt = null;
    mcAfter = null;
  };
  resetCluster();

  const unit = () => {
    const o = observe(env);
    const R = step(env, thresholdDecide(th, o)).reward;
    const o2 = observe(env);
    const idx = feat.indices(o2, thresholdDecide(th, o2));
    for (const ln of learners) {
      const delta = R - ln.avg.rbar + qValue(ln.w, idx) - qValue(ln.w, prevIdx);
      updateAverage(ln.avg, R, delta);
      const dw = (alpha / feat.active) * delta;
      for (const i of prevIdx) ln.w[i]! += dw;
    }
    prevIdx = idx;
    steps += 1;
    if (steps % 100 === 0) learners.forEach(ln => ln.series.push(steps, ln.avg.rbar));
  };

  const k = RULES.map(rule => kpi(RULE_LABELS[rule]));
  const mcK = kpi('Monte Carlo r(π)', '…', '95% CI');
  const chart = canvasPanel('R̄ estimates of r(π) for a fixed threshold policy, A = 0.8', 260, canvas => {
    const t = theme();
    const refs = [];
    if (mcBefore) refs.push({ value: mcBefore.mean, band: mcBefore.half, color: t.ref, label: 'MC r(π)' });
    if (mcAfter) refs.push({ value: mcAfter.mean, band: mcAfter.half, color: t.danger, label: 'MC r(π) after shift' });
    drawXY(canvas, learners.map((ln, i) => ({ color: t.series[i]!, points: ln.series.points() })), {
      xMin: 0, xMax: Math.max(steps, 1000), xLabel: 'steps', refs, textColor: t.text, gridColor: t.grid,
      vlines: shiftedAt !== null ? [{ x: shiftedAt, color: t.danger, label: 'traffic shift' }] : [],
      formatX: x => (x >= 1000 ? `${Math.round(x / 1000)}k` : String(Math.round(x))),
    });
  }, { legend: legend([...RULES.map((r, i) => ({ label: RULE_LABELS[r], color: t0.series[i]! })), { label: 'Monte Carlo r(π) ± 95% CI', color: t0.ref, dash: true }]) });

  const render = () => {
    learners.forEach((ln, i) => k[i]!.set(fmt(ln.avg.rbar)));
    chart.redraw();
  };
  const runner = createRunner({ unit, render, perFrame: () => L.speed, onState: (p, ff) => rc.sync(p, ff) });
  const rc = runnerControls(runner, { ff: [{ label: '20k steps', units: 2e4 }, { label: '100k steps', units: 1e5 }] });

  const shiftBtn = button('Traffic shift: double the Enterprise share', () => {
    if (shiftedAt !== null) return;
    const base = [...env.cfg.base];
    base[3] = base[3]! * 2;
    env.cfg = { ...env.cfg, base };
    shiftedAt = steps;
    shiftBtn.disabled = true;
    ctx.when(store.longRun(env.cfg, { kind: 'threshold', th }, 'm12-hand'), iv => { mcAfter = iv; mcK.set(`${fmt(mcBefore?.mean ?? NaN)} → ${fmt(iv.mean)}`, `± ${fmt(iv.half)} after the shift`); render(); });
  });
  const restart = () => { runner.pause(); resetCluster(); shiftBtn.disabled = false; render(); };

  root.append(
    moduleHeader('m12', 'How R̄ should be updated', 'Differential Sarsa needs a running estimate R̄ of r(π). Updating it with the <em>TD error</em> δ, not with the reward error R − R̄, strips out reward fluctuations the values already predict.'),
    el('div', { class: 'grid2' },
      card('Ring A → B → C → A, +1 on arrival at A (Exercise 10.8)',
        check('Freeze R̄ at the true 1/3', L.frozen, x => { L.frozen = x; renderRing(); }).el,
        range('Initial R̄ (when unfrozen)', { min: 0, max: 1, step: 0.05, value: L.ringR0, format: x => fmt(x, 2) }, x => { L.ringR0 = x; renderRing(); }).el,
        range('β', { min: 0.01, max: 0.5, step: 0.01, value: L.ringBeta, format: x => fmt(x, 2) }, x => { L.ringBeta = x; renderRing(); }).el,
        el('p', { class: 'hint', text: `Exact differential values: v(A) = ${fmt(v[0]!, 4)}, v(B) = ${fmt(v[1]!, 4)}, v(C) = ${fmt(v[2]!, 4)}.` }),
        el('div', { class: 'table-scroll' }, ringTable)),
      ringChart.el),
    el('div', { class: 'mod-grid' },
      el('div', { class: 'controls' },
        card('Cluster run (A = 0.8, hand-tuned threshold)', rc.el,
          range('Steps per frame', { min: 1, max: 3000, step: 1, value: L.speed }, x => { L.speed = x; }).el,
          range('Initial R̄', { min: 0, max: 20, step: 0.5, value: L.rbar0, format: x => fmt(x, 1) }, x => { L.rbar0 = x; restart(); }).el,
          range('β', { min: 0.001, max: 0.1, step: 0.001, value: L.beta, format: x => fmt(x, 3) }, x => { L.beta = x; restart(); }).el,
          shiftBtn, button('Restart run', restart),
          el('p', { class: 'hint', text: 'Each rule has its own tile-coded q̂ (G1–G3, α × active = 0.1) evaluating the same fixed policy on the same trajectory.' }))),
      el('div', { class: 'panels' }, el('div', { class: 'kpis' }, ...k.map(x => x.el), mcK.el), chart.el)),
    observeBox([
      'Ring, frozen at 1/3: R − R̄ cycles −1/3, −1/3, +2/3 forever, but δ is exactly 0 every step. The value differences absorb the predictable part of the reward.',
      'Unfreeze it: the δ rule moves smoothly to 1/3, while the R − R̄ rule zig-zags around it with amplitude set by β.',
      'Cluster: start R̄ at 20 with a small β. The constant-β rules stay biased by the initial value for a long time. The sample average forgets the start at once.',
      'Press <strong>Traffic shift</strong>: the sample average lags far behind the new rate. The unbiased constant-step trick (Exercise 10.9) fixed the start <em>and</em> still tracks the change.',
      'Raise β to 0.01: with A = 0.8 the constant-step estimates now follow the <em>daily</em> swing in reward (a day is 96 steps), not the long-run average. β sets the averaging window.',
    ]),
  );
  renderRing();
  render();
  ctx.when(store.longRun(cfg, { kind: 'threshold', th }, 'm12-hand'), iv => { mcBefore = iv; mcK.set(`${fmt(iv.mean)} ± ${fmt(iv.half)}`, '400k-step batch means'); render(); });
  return () => { runner.destroy(); chart.destroy(); ringChart.destroy(); };
}
