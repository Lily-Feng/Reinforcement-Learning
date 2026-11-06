/**
 * M05 · Optimism explores even with ε = 0. One episodic Sarsa agent whose
 * weights all start at q₀/active; a visitation map, learning curve and the
 * fraction of Reject decisions show what a greedy agent ever tries.
 */

import { drawXY } from '../../shared/bars';
import { drawHeatmap, mix } from '../../shared/heatmap';
import { mulberry32 } from '../../shared/rng';
import { createEpisodicAgent, episodicBegin, episodicStep, type EpisodicAgent } from '../core/agents.ts';
import { REJECT } from '../core/config.ts';
import { createEnv, observe, resetEnv, step, type Env } from '../core/env.ts';
import { createFeaturizer, defaultFeatures } from '../core/features.ts';
import type { ShiftReference } from '../core/jobs.ts';
import type { ModuleContext } from '../app/context';
import { canvasPanel, card, createRunner, el, fmt, legend, moduleHeader, observeBox, range, runnerControls, theme } from '../app/ui';
import { kpi, lengthGrid } from './common';

export function mount(root: HTMLElement, ctx: ModuleContext): () => void {
  const cfg = ctx.cfg;
  const L = ctx.store.local('m05', { q0: 0, eps: 0, alpha: 0.2, perFrame: 1 });
  const lengths = lengthGrid(16);
  let shiftRef: ShiftReference | null = null;
  let env: Env;
  let agent: EpisodicAgent;
  let visits: Int32Array;
  let revenue: number[] = [];
  let rejectFrac: number[] = [];
  const reset = () => {
    env = createEnv(cfg, mulberry32(cfg.seed), { episodic: true });
    agent = createEpisodicAgent(createFeaturizer(cfg, env.buckets, defaultFeatures(cfg, true)), { alpha: L.alpha, eps: L.eps, n: 1, q0: L.q0 }, mulberry32(cfg.seed + 5));
    visits = new Int32Array((cfg.K + 1) * lengths.length);
    revenue = [];
    rejectFrac = [];
  };
  reset();
  const lenBin = (len: number) => Math.max(0, Math.min(lengths.length - 1, Math.round((Math.log(len / 2) / Math.log(60)) * (lengths.length - 1))));

  const unit = () => {
    resetEnv(env);
    let a = episodicBegin(agent, observe(env));
    let total = 0;
    let choices = 0;
    let rejects = 0;
    while (!env.done) {
      const o = observe(env);
      visits[o.g * lengths.length + lenBin(o.len)]! += 1;
      if (o.g > 0) { choices += 1; if (a === REJECT) rejects += 1; }
      const r = step(env, a);
      total += r.reward;
      a = episodicStep(agent, r.reward, observe(env), r.done).action;
    }
    revenue.push(total);
    rejectFrac.push(choices ? rejects / choices : 0);
  };

  const t0 = theme();
  const heat = canvasPanel('Visits: free GPUs × job length (all shifts so far)', 220, canvas => {
    const t = theme();
    const m = Math.max(1, ...visits);
    drawHeatmap(canvas, {
      rows: cfg.K + 1, cols: lengths.length,
      color: (r, c) => { const v = visits[(cfg.K - r) * lengths.length + c]!; return v === 0 ? null : mix(t.seqLow, t.seqHigh, Math.log1p(v) / Math.log1p(m)); },
      rowLabel: r => String(cfg.K - r), colLabel: c => (c % 5 === 0 ? lengths[c]!.toFixed(0) : ''), colLabelEvery: 1,
      xLabel: 'job length ℓ (log)', yLabel: 'free GPUs g', legend: { min: '1', max: String(m), ramp: x => mix(t.seqLow, t.seqHigh, x) }, textColor: t.text, gridColor: t.grid,
    });
  });
  const curve = canvasPanel('Revenue per shift and fraction of Reject decisions', 220, canvas => {
    const t = theme();
    const maxR = Math.max(1, ...revenue, shiftRef?.value ?? 0);
    drawXY(canvas, [
      { color: t.series[0]!, points: revenue.map((v, i) => [i + 1, v]) },
      { color: t.series[1]!, points: rejectFrac.map((v, i) => [i + 1, v * maxR]) },
    ], { xMin: 0, xMax: Math.max(10, revenue.length), yMin: 0, yMax: maxR * 1.05, xLabel: 'shift', refs: shiftRef ? [{ value: shiftRef.value, color: t.ref, label: 'optimal (DP)' }] : [], textColor: t.text, gridColor: t.grid });
  }, { legend: legend([{ label: 'revenue per shift', color: t0.series[0]! }, { label: 'fraction of Reject decisions (scaled to the axis: top = 100%)', color: t0.series[1]! }, { label: 'optimal expected revenue', color: t0.ref, dash: true }]) });

  const k = { shifts: kpi('Shifts'), rej: kpi('Reject fraction, last shift'), rev: kpi('Revenue, last 10 shifts'), bound: kpi('DP optimum per shift', '…', 'q₀ above this is optimistic') };
  const render = () => {
    k.shifts.set(String(revenue.length));
    k.rej.set(rejectFrac.length ? `${fmt(100 * rejectFrac[rejectFrac.length - 1]!, 1)}%` : '—');
    const last = revenue.slice(-10);
    k.rev.set(last.length ? fmt(last.reduce((a, b) => a + b, 0) / last.length, 1) : '—');
    heat.redraw();
    curve.redraw();
  };
  const runner = createRunner({ unit, render, perFrame: () => L.perFrame, onState: (p, ff) => rc.sync(p, ff) });
  const rc = runnerControls(runner, { stepLabel: '1 shift', ff: [{ label: '50 shifts', units: 50 }, { label: '300 shifts', units: 300 }] });
  const restart = () => { runner.pause(); reset(); render(); };
  const q0Ctl = range('Initial value q₀ (every weight = q₀ / active)', { min: -100, max: 600, step: 10, value: L.q0 }, v => { L.q0 = v; restart(); });
  const marker = el('p', { class: 'hint' });

  root.append(
    moduleHeader('m05', 'Optimism explores even with ε = 0', 'True action values in the shift task are large and positive: up to the optimal expected revenue of a whole shift. The initial value q₀ decides whether an untried action looks better or worse than a tried one.'),
    el('div', { class: 'mod-grid' },
      el('div', { class: 'controls' },
        card('Initial values and exploration', q0Ctl.el, marker,
          range('ε', { min: 0, max: 0.3, step: 0.01, value: L.eps, format: v => fmt(v, 2) }, v => { L.eps = v; restart(); }).el,
          range('α × active features', { min: 0.02, max: 1, step: 0.02, value: L.alpha, format: v => fmt(v, 2) }, v => { L.alpha = v; restart(); }).el),
        card('Run', rc.el, range('Shifts per frame', { min: 1, max: 20, step: 1, value: L.perFrame }, v => { L.perFrame = v; }).el)),
      el('div', { class: 'panels' }, el('div', { class: 'kpis' }, k.shifts.el, k.rej.el, k.rev.el, k.bound.el), curve.el, heat.el),
    ),
    observeBox([
      'q₀ = 0 is <em>pessimistic</em> here, because the true values are positive (the opposite of Mountain Car). With ε = 0, on some seeds (try seeds 2, 5 and 8 in the settings) Reject is almost never chosen. The first accepts raise the Accept values above the untouched Reject values, and the reject fraction drops to 0: accept-all.',
      'Raise q₀ to the DP marker or a little above: on every seed we tried, the greedy agent keeps trying Reject in 30–45% of decisions. Each tried action disappoints, so the alternative looks better next time. That is optimism doing the exploring, as in Figure 10.1.',
      'Honest caveat: q₀ = −100 does <em>not</em> lock in cleanly. Forced rejects when no GPU is free (g = 0) update Reject weights whose tiles are shared with g = 1–2, so Reject gets tried through generalization. Very large q₀ (≥ 400) disappoints so slowly that revenue collapses for hundreds of shifts.',
    ]),
  );
  render();
  ctx.when(ctx.store.shift(cfg), s => { shiftRef = s; k.bound.set(fmt(s.value, 1)); marker.textContent = `DP marker: optimal expected revenue of a whole shift ≈ ${fmt(s.value, 1)}. No state-action value can exceed it.`; render(); });
  return () => { runner.destroy(); heat.destroy(); curve.destroy(); };
}
