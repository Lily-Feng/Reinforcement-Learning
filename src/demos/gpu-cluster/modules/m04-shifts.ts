/**
 * M04 · Training over shifts. Episodic semi-gradient Sarsa at three step
 * sizes (several runs each), learning curves of revenue per shift, and the
 * learned greedy policy at a chosen time in the shift next to the optimal
 * shift policy from finite-horizon DP.
 */

import { drawXY } from '../../shared/bars';
import { mulberry32 } from '../../shared/rng';
import { createEpisodicAgent, episodicBegin, episodicStep, greedyAction, type EpisodicAgent } from '../core/agents.ts';
import { TIER_NAMES } from '../core/config.ts';
import { createEnv, observe, resetEnv, step, type Env } from '../core/env.ts';
import { createFeaturizer, defaultFeatures } from '../core/features.ts';
import { typicalRho, type ShiftReference } from '../core/jobs.ts';
import { stateIndex } from '../core/model.ts';
import type { ModuleContext } from '../app/context';
import { canvasPanel, card, createRunner, el, fmt, legend, moduleHeader, observeBox, range, runnerControls, theme } from '../app/ui';
import { bucketEdges, bucketOfLength, decisionMap, kpi, lengthGrid, mapLegend, occupancyCounts } from './common';

const ALPHAS = [0.05, 0.2, 0.6];
const RUNS = 4;

export function mount(root: HTMLElement, ctx: ModuleContext): () => void {
  const cfg = ctx.cfg;
  const L = ctx.store.local('m04', { tau: 0, perFrame: 1 });
  if (L.tau >= cfg.H) L.tau = 0;
  const model = ctx.store.model(cfg);
  const lengths = lengthGrid(20);
  const edges = bucketEdges(model, cfg);
  const occ = occupancyCounts(model);
  let shiftRef: ShiftReference | null = null;

  interface Learner { alpha: number; env: Env; agent: EpisodicAgent; curve: number[] }
  let learners: Learner[] = [];
  let shifts = 0;
  const reset = () => {
    learners = ALPHAS.flatMap((alpha, ai) => Array.from({ length: RUNS }, (_, r) => {
      const env = createEnv(cfg, mulberry32(cfg.seed + 101 * r), { episodic: true });
      const feat = createFeaturizer(cfg, env.buckets, defaultFeatures(cfg, true));
      return { alpha, env, agent: createEpisodicAgent(feat, { alpha, eps: 0.1, n: 1 }, mulberry32(cfg.seed + 17 * r + ai)), curve: [] };
    }));
    shifts = 0;
  };
  reset();

  const unit = () => {
    for (const ln of learners) {
      resetEnv(ln.env);
      let a = episodicBegin(ln.agent, observe(ln.env));
      let total = 0;
      while (!ln.env.done) {
        const r = step(ln.env, a);
        total += r.reward;
        a = episodicStep(ln.agent, r.reward, observe(ln.env), r.done).action;
      }
      ln.curve.push(total);
    }
    shifts += 1;
  };

  const shown = () => learners[RUNS]!; // middle α, first run
  const t0 = theme();
  const curves = canvasPanel('Revenue per shift (mean over runs)', 240, canvas => {
    const t = theme();
    const series = ALPHAS.map((alpha, ai) => {
      const group = learners.filter(l => l.alpha === alpha);
      const pts: number[][] = [];
      const win = 5;
      for (let k = 0; k < shifts; k += 1) {
        let s = 0; let n = 0;
        for (let j = Math.max(0, k - win + 1); j <= k; j += 1) for (const l of group) { s += l.curve[j]!; n += 1; }
        pts.push([k + 1, s / n]);
      }
      return { color: t.series[ai]!, points: pts };
    });
    drawXY(canvas, series, { xMin: 0, xMax: Math.max(10, shifts), xLabel: 'shift', refs: shiftRef ? [{ value: shiftRef.value, color: t.ref, label: 'optimal (DP)' }] : [], textColor: t.text, gridColor: t.grid });
  }, { legend: legend([...ALPHAS.map((a, i) => ({ label: `α × active = ${a}`, color: t0.series[i]! })), { label: 'optimal expected revenue (finite-horizon DP)', color: t0.ref, dash: true }]) });

  const togo = () => cfg.H - L.tau;
  const learned = TIER_NAMES.map((name, c) => decisionMap(`Learned · ${name}`, () => cfg.K, lengths, (g, li) => {
    if (g === 0) return 0;
    return greedyAction(shown().agent, { g, c, len: lengths[li]!, rho: typicalRho(cfg, model.buckets.meanQ, g), phase: L.tau / 96, togo: togo(), bucket: 0 });
  }, { height: 140 }));
  const optimal = TIER_NAMES.map((name, c) => decisionMap(`Optimal (DP, bucket bands) · ${name}`, () => cfg.K, lengths, (g, li) => {
    if (!shiftRef || g === 0 || occ[g]! < 0) return g === 0 ? 0 : null;
    return shiftRef.decision[togo()]![stateIndex(model, occ[g]!, c, bucketOfLength(edges, lengths[li]!))]!;
  }, { height: 140 }));

  const k = { shifts: kpi('Shifts trained'), last: kpi('Last 5 shifts (α = 0.2)'), opt: kpi('Optimal expected revenue', '…', 'finite-horizon DP, bucketed') };
  const render = () => {
    k.shifts.set(String(shifts));
    const c = shown().curve;
    const recent = c.slice(-5);
    k.last.set(recent.length ? fmt(recent.reduce((a, b) => a + b, 0) / recent.length, 1) : '—');
    curves.redraw();
    [...learned, ...optimal].forEach(m => m.redraw());
  };
  const runner = createRunner({ unit, render, perFrame: () => L.perFrame, onState: (p, ff) => rc.sync(p, ff) });
  const rc = runnerControls(runner, { stepLabel: 'Train 1 shift', ff: [{ label: '50 shifts', units: 50 }, { label: '200 shifts', units: 200 }] });
  const tauCtl = range('Time in the shift τ (steps)', { min: 0, max: cfg.H - 1, step: 1, value: L.tau, format: v => `${v} (${cfg.H - v} to go)` }, v => { L.tau = v; [...learned, ...optimal].forEach(m => m.redraw()); });

  root.append(
    moduleHeader('m04', 'Training over shifts', `${RUNS} runs at each of three step sizes learn the one-shift task (H = ${cfg.H} steps, every GPU free at the start). The maps show the greedy policy of one α = 0.2 run at the chosen time in the shift, and the optimal shift policy from finite-horizon DP on the bucketed model.`),
    el('div', { class: 'mod-grid' },
      el('div', { class: 'controls' },
        card('Train', rc.el, range('Shifts per frame', { min: 1, max: 10, step: 1, value: L.perFrame }, v => { L.perFrame = v; }).el),
        card('Policy maps', tauCtl.el, mapLegend(), el('p', { class: 'hint', text: 'Busy jobs in the optimal map are split across length buckets in proportion to length (the typical occupancy).' }))),
      el('div', { class: 'panels' }, el('div', { class: 'kpis' }, k.shifts.el, k.last.el, k.opt.el), curves.el,
        el('div', { class: 'grid4' }, ...learned.map(m => m.el)), el('div', { class: 'grid4' }, ...optimal.map(m => m.el))),
    ),
    observeBox([
      'The largest α climbs fastest at first but its curve is the noisiest. The smallest α is smooth and slow.',
      'Move τ to the last few steps of the shift: the optimal policy accepts almost everything, because a GPU blocked after the shift ends costs nothing. After enough training the learned maps do the same (G4 lets them see the time to go).',
      'Early in the shift the optimal policy keeps GPUs free for Pro and Enterprise. The learned policy approaches that shape more slowly.',
    ]),
  );
  render();
  ctx.when(ctx.store.shift(cfg), s => { shiftRef = s; k.opt.set(fmt(s.value, 1)); render(); });
  return () => { runner.destroy(); curves.destroy(); [...learned, ...optimal].forEach(m => m.destroy()); };
}
